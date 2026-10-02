"""M05: controlled 2x2 data-support x dual-awareness experiment.

Factors:
  support: original M01 train vs original + stratified M04 training-only support
  decoder: standard price/VM GNN vs dual-aware price decoder

Validation, test, and G14 continuation are frozen. M04 is selected using only
its training-only bus-31 dual severity; no validation/test/G14 targets affect
selection. Run branches independently to use separate GPUs.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from graphopf.gnn_baseline import EdgeMessageLayer,SupervisedGraphOPF

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def load(p):
 z=np.load(p);return {k:z[k] for k in z.files}
def graph(g,d):
 ef=g['edge_features'].astype(np.float32);em=ef.mean(0,keepdims=True);es=np.maximum(ef.std(0,keepdims=True),1e-6)
 return torch.tensor(g['edge_index'],dtype=torch.long,device=d),torch.tensor((ef-em)/es,dtype=torch.float32,device=d)
def regimes(data,n):
 r=np.array(['unknown']*n,dtype=object);p=data/'g08_regime_evaluation.csv'
 if p.exists():
  with p.open(newline='') as f:
   for q in csv.DictReader(f):
    i=int(q['scenario']);
    if 0<=i<n:r[i]=q['regime']
 return r
def continuation(data,ex):
 p=data/'g14_active_set_continuation.csv'
 if not p.exists():raise FileNotFoundError('G14 continuation CSV required: '+str(p))
 out=[]
 with p.open(newline='') as f:
  rr=[q for q in csv.DictReader(f) if str(q['solved']).lower() in ('true','1','yes')]
 for q in rr:
  s=int(q['active_scenario']);c=int(q['control_scenario']);lam=float(q['lambda']);x=(ex[c].astype(float)+lam*(ex[s].astype(float)-ex[c].astype(float))).astype(np.float32)
  out.append({'active_scenario':s,'lambda':lam,'true_lmp':float(q['true_lmp']),'x':x})
 return out

def stratified_m04(z,n,seedv):
 mu=z['bus31_mu_vmin'].astype(float);active=z['bus31_active'].astype(bool);near=np.where(~active)[0];pos=np.where(active & (mu>0))[0];rng=np.random.default_rng(seedv)
 # Log-dual quantiles prevent a few million-scale samples from dominating support.
 groups=[]
 if len(near):groups.append(('near',near))
 if len(pos):
  lm=np.log1p(mu[pos]);qs=np.quantile(lm,[0,.25,.5,.75,1.])
  for k in range(4):
   hi=(lm<=qs[k+1]) if k==3 else (lm<qs[k+1]);ix=pos[(lm>=qs[k])&hi]
   if len(ix):groups.append((f'active_q{k+1}',ix))
 alloc=[n//len(groups)]*len(groups)
 for k in range(n-sum(alloc)):alloc[k%len(alloc)]+=1
 chosen=[];detail={}
 for (name,ix),want in zip(groups,alloc):
  take=rng.choice(ix,size=want,replace=want>len(ix));chosen.extend(take.tolist());detail[name]={'available':int(len(ix)),'selected':int(want),'mu_min':float(mu[ix].min()),'mu_max':float(mu[ix].max())}
 rng.shuffle(chosen);return np.array(chosen,dtype=int),detail

def stats(x,y):return x.mean((0,1),keepdims=True),np.maximum(x.std((0,1),keepdims=True),1e-6),y.mean((0,1),keepdims=True),np.maximum(y.std((0,1),keepdims=True),1e-6)
class DualNet(nn.Module):
 def __init__(self,h=64,layers=3):
  super().__init__();self.enc=nn.Sequential(nn.Linear(5,h),nn.ReLU());self.mp=nn.ModuleList(EdgeMessageLayer(h,6) for _ in range(layers));self.dual=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2));self.vm=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1));self.price=nn.Sequential(nn.Linear(h+2,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x,ei,ea):
  h=self.enc(x)
  for m in self.mp:h=m(h,ei,ea)
  z=self.dual(h);return torch.cat((self.price(torch.cat((h,z),-1)),self.vm(h)),-1),z

def train(tx,ty,tz,vx,vy,vz,g,a,d,decoder):
 st=stats(tx,ty);xm,xs,ym,ys=st;TX=torch.tensor((tx-xm)/xs,device=d);TY=torch.tensor((ty-ym)/ys,device=d);VX=torch.tensor((vx-xm)/xs,device=d);VY=torch.tensor((vy-ym)/ys,device=d);ei,ea=graph(g,d);seed(a.seed)
 if decoder=='standard':m=SupervisedGraphOPF(hidden_dim=a.hidden,layers=a.layers).to(d)
 else:m=DualNet(a.hidden,a.layers).to(d)
 TZ=torch.tensor(tz,dtype=torch.float32,device=d) if tz is not None else None;VZ=torch.tensor(vz,dtype=torch.float32,device=d) if vz is not None else None
 opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 for ep in range(1,a.epochs+1):
  m.train();opt.zero_grad()
  if decoder=='standard':p=m(TX,ei,ea);loss=F.huber_loss(p,TY,delta=1.)
  else:
   p,z=m(TX,ei,ea);w=1+a.dual_positive_weight*(TZ>0).float();loss=F.huber_loss(p,TY,delta=1.)+a.dual_weight*(F.smooth_l1_loss(z,TZ,reduction='none')*w).mean()
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();m.eval()
  with torch.no_grad():
   if decoder=='standard':vl=F.huber_loss(m(VX,ei,ea),VY,delta=1.)
   else:
    vp,vz=m(VX,ei,ea);vw=1+a.dual_positive_weight*(VZ>0).float();vl=F.huber_loss(vp,VY,delta=1.)+a.dual_weight*(F.smooth_l1_loss(vz,VZ,reduction='none')*vw).mean()
  v=float(vl)
  if v<best-1e-8:best=v;state={k:q.detach().cpu().clone() for k,q in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'{decoder:8s} epoch={ep:4d} train={loss.item():.6f} val={v:.6f}',flush=True)
  if stale>=a.patience:print(f'early_stop epoch={ep} best_val={best:.6f}',flush=True);break
 m.load_state_dict(state);return m,st,ei,ea,best
def predict(m,x,st,ei,ea,d,decoder,b=256):
 xm,xs,ym,ys=st;out=[];zz=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(x),b):
   X=torch.tensor((x[k:k+b]-xm)/xs,device=d)
   if decoder=='standard':p=m(X,ei,ea);z=None
   else:p,z=m(X,ei,ea);zz.append(z.cpu().numpy())
   out.append(p.cpu().numpy())
 return np.concatenate(out)*ys+ym,(np.concatenate(zz) if zz else None)
def met(p,y,mask,j):
 e=p[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(abs(e[...,1]).mean()),'trigger_lmp_mae':float(abs(e[:,j,0]).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--branch',choices=['A','B','C','D'],required=True);ap.add_argument('--augment-n',type=int,default=200);ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=45);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--weight-decay',type=float,default=1e-5);ap.add_argument('--dual-weight',type=float,default=.15);ap.add_argument('--dual-positive-weight',type=float,default=10.);ap.add_argument('--seed',type=int,default=20271001);a=ap.parse_args()
 data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');g=np.load(data/'graph.npz');j=int(np.where(te['bus_ids']==a.trigger_bus)[0][0]);support=a.branch in ('B','D');decoder='dual' if a.branch in ('C','D') else 'standard';tx=tr['x'].astype(np.float32);ty=tr['y'].astype(np.float32);tz=np.log1p(np.stack((tr['mu_vmin'],tr['mu_vmax']),-1).astype(np.float32)) if decoder=='dual' else None;selection=None
 if support:
  m4=load(data/'m04_dual_regime'/'targeted_dual_regime.npz');ix,selection=stratified_m04(m4,a.augment_n,a.seed+5);tx=np.concatenate((tx,m4['x'][ix].astype(np.float32)));ty=np.concatenate((ty,m4['y'][ix].astype(np.float32)))
  if decoder=='dual':tz=np.concatenate((tz,np.log1p(np.stack((m4['mu_vmin'][ix],m4['mu_vmax'][ix]),-1).astype(np.float32))))
 vx=va['x'].astype(np.float32);vy=va['y'].astype(np.float32);vz=np.log1p(np.stack((va['mu_vmin'],va['mu_vmax']),-1).astype(np.float32)) if decoder=='dual' else None;d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('M05 SUPPORT x DUAL-AWARENESS FACTORIAL');print(f'branch={a.branch} support={support} decoder={decoder} device={d} train={len(tx)} frozen_val/test={len(vx)}/{len(te["x"])}')
 if selection:print('M04 stratification='+json.dumps(selection))
 m,st,ei,ea,best=train(tx,ty,tz,vx,vy,vz,g,a,d,decoder);pe,z=predict(m,te['x'].astype(np.float32),st,ei,ea,d,decoder);reg=regimes(data,len(pe));groups={};print('\nFROZEN TEST')
 for name,mask in [('all',np.ones(len(pe),bool)),('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]:
  if mask.any():groups[name]=met(pe,te['y'],mask,j);q=groups[name];print(f'{name:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
 paths=continuation(data,te['x'].astype(np.float32));px=np.stack([q['x'] for q in paths]);pp,_=predict(m,px,st,ei,ea,d,decoder);cont={};print('\nFROZEN G14 CONTINUATION')
 for s in sorted(set(q['active_scenario'] for q in paths)):
  ix=np.array([i for i,q in enumerate(paths) if q['active_scenario']==s and 0<=q['lambda']<=1]);true=np.array([paths[i]['true_lmp'] for i in ix]);pred=pp[ix,j,0];hi=true>=50;cont[str(s)]={'n':int(len(ix)),'lmp_mae':float(abs(pred-true).mean()),'high_price_n':int(hi.sum()),'high_price_mae':float(abs(pred[hi]-true[hi]).mean()) if hi.any() else None,'max_true_lmp':float(true.max()),'max_pred_lmp':float(pred.max()),'endpoint_true':float(true[-1]),'endpoint_pred':float(pred[-1])};q=cont[str(s)];print(f'path={s} MAE={q["lmp_mae"]:.4f} high_price_MAE={q["high_price_mae"]} max true/pred={q["max_true_lmp"]:.3f}/{q["max_pred_lmp"]:.3f}')
 diag=None
 if decoder=='dual':
  true=np.log1p(te['mu_vmin'][:,j].astype(float));act=te['mu_vmin'][:,j]>1e-6;diag={'bus31_logdual_mae':float(abs(z[:,j,0]-true).mean()),'true_active':int(act.sum()),'pred_active_mean':float(z[act,j,0].mean()) if act.any() else None,'pred_inactive_mean':float(z[~act,j,0].mean())};print('\nDUAL DIAGNOSTIC '+json.dumps(diag))
 out={'configuration':vars(a),'branch':a.branch,'support_augmented':support,'decoder':decoder,'m04_selection':selection,'best_val':best,'groups':groups,'continuation':cont,'dual_diagnostic':diag};fn=data/f'm05_{a.branch.lower()}_factorial_summary.json';fn.write_text(json.dumps(out,indent=2)+'\n');torch.save({'state_dict':m.state_dict(),'stats':st,'configuration':vars(a),'branch':a.branch},data/f'm05_{a.branch.lower()}_model.pt');print(f'\nSummary: {fn}');print('Decision: compare A->B for support effect, A->C for dual architecture effect, and B->D for added dual value after targeted support. Frozen test/G14 must improve without material regular-regime degradation.')
if __name__=='__main__':main()
