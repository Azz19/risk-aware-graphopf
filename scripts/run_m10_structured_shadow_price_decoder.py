"""M10: structured shadow-price decoder.

Hypothesis from M09: once the bus-31 lower-voltage shadow price is known, the
incremental LMP response is approximately path-transferable.  M10 therefore
separates (i) nominal price prediction, (ii) rare shadow-price inference, and
(iii) a constrained economic decoder.

Branches (safe to run independently / concurrently):
  oracle    : true mu31 at evaluation, diagnostic upper bound only.
  original  : predicted mu31, trained only on frozen M01 training data.
  targeted  : predicted mu31 with stratified M04 training-only support.

The structured coefficient alpha is estimated from TRAINING-ONLY OPF labels by
regressing bus-31 price residuals on mu31 after fitting the nominal predictor.
For targeted, M04 support is allowed because it is training-only.  Frozen val,
test, and G14 are never used to fit alpha, choose support, or train networks.
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
 p=data/'g14_active_set_continuation.csv';out=[]
 with p.open(newline='') as f:rr=[q for q in csv.DictReader(f) if str(q['solved']).lower() in ('true','1','yes')]
 for q in rr:
  s=int(q['active_scenario']);c=int(q['control_scenario']);lam=float(q['lambda']);x=(ex[c].astype(float)+lam*(ex[s].astype(float)-ex[c].astype(float))).astype(np.float32)
  out.append({'active_scenario':s,'lambda':lam,'true_lmp':float(q['true_lmp']),'true_mu31':float(q.get('true_lower_v_dual',0) or 0),'x':x})
 return out
def stratified(z,n,seedv):
 mu=z['bus31_mu_vmin'].astype(float);active=z['bus31_active'].astype(bool);near=np.where(~active)[0];pos=np.where(active&(mu>0))[0];rng=np.random.default_rng(seedv);groups=[]
 if len(near):groups.append(('near',near))
 if len(pos):
  lm=np.log1p(mu[pos]);qs=np.quantile(lm,[0,.25,.5,.75,1])
  for k in range(4):
   hi=lm<=qs[k+1] if k==3 else lm<qs[k+1];ix=pos[(lm>=qs[k])&hi]
   if len(ix):groups.append((f'active_q{k+1}',ix))
 alloc=[n//len(groups)]*len(groups)
 for k in range(n-sum(alloc)):alloc[k%len(alloc)]+=1
 chosen=[];detail={}
 for (name,ix),want in zip(groups,alloc):
  take=rng.choice(ix,want,replace=want>len(ix));chosen.extend(take.tolist());detail[name]={'available':int(len(ix)),'selected':int(want),'mu_min':float(mu[ix].min()),'mu_max':float(mu[ix].max())}
 rng.shuffle(chosen);return np.array(chosen,int),detail
def stats(x,y):return x.mean((0,1),keepdims=True),np.maximum(x.std((0,1),keepdims=True),1e-6),y.mean((0,1),keepdims=True),np.maximum(y.std((0,1),keepdims=True),1e-6)
def train_base(tx,ty,vx,vy,g,a,d):
 st=stats(tx,ty);xm,xs,ym,ys=st;TX=torch.tensor((tx-xm)/xs,device=d);TY=torch.tensor((ty-ym)/ys,device=d);VX=torch.tensor((vx-xm)/xs,device=d);VY=torch.tensor((vy-ym)/ys,device=d);ei,ea=graph(g,d);seed(a.seed);m=SupervisedGraphOPF(hidden_dim=a.hidden,layers=a.layers).to(d);o=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 for ep in range(1,a.epochs+1):
  m.train();o.zero_grad();p=m(TX,ei,ea);loss=F.huber_loss(p,TY,delta=1);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);o.step();m.eval()
  with torch.no_grad():vl=float(F.huber_loss(m(VX,ei,ea),VY,delta=1))
  if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'base epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
  if stale>=a.patience:break
 m.load_state_dict(state);return m,st,ei,ea,best
def pred_base(m,x,st,ei,ea,d,b=256):
 xm,xs,ym,ys=st;o=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(x),b):o.append(m(torch.tensor((x[k:k+b]-xm)/xs,device=d),ei,ea).cpu().numpy())
 return np.concatenate(o)*ys+ym
class MuNet(nn.Module):
 def __init__(self,h,layers):
  super().__init__();self.enc=nn.Sequential(nn.Linear(5,h),nn.ReLU());self.mp=nn.ModuleList(EdgeMessageLayer(h,6) for _ in range(layers));self.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
 def forward(self,x,ei,ea):
  h=self.enc(x)
  for m in self.mp:h=m(h,ei,ea)
  z=self.head(h);return z[...,0],z[...,1] # activity logit, log1p(mu)
def train_mu(tx,mu,vx,vmu,g,a,d,st):
 xm,xs,_,_=st;TX=torch.tensor((tx-xm)/xs,device=d);VX=torch.tensor((vx-xm)/xs,device=d);M=torch.tensor(np.log1p(mu),device=d);VM=torch.tensor(np.log1p(vmu),device=d);A=(M>0).float();VA=(VM>0).float();pos=float(A.mean());pw=torch.tensor((1-pos)/max(pos,1e-6),device=d).clamp(max=a.max_pos_weight);ei,ea=graph(g,d);seed(a.seed+17);m=MuNet(a.hidden,a.layers).to(d);o=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 for ep in range(1,a.mu_epochs+1):
  m.train();o.zero_grad();lg,z=m(TX,ei,ea);reg=F.smooth_l1_loss(z,M,reduction='none');loss=F.binary_cross_entropy_with_logits(lg,A,pos_weight=pw)+a.mu_weight*(reg*(1+a.mu_positive_weight*A)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);o.step();m.eval()
  with torch.no_grad():
   lg,z=m(VX,ei,ea);vl=float(F.binary_cross_entropy_with_logits(lg,VA,pos_weight=pw)+a.mu_weight*(F.smooth_l1_loss(z,VM,reduction='none')*(1+a.mu_positive_weight*VA)).mean())
  if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'mu   epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
  if stale>=a.patience:break
 m.load_state_dict(state);return m,best
def pred_mu(m,x,st,ei,ea,d,b=256):
 xm,xs,_,_=st;ps=[];zs=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(x),b):
   lg,z=m(torch.tensor((x[k:k+b]-xm)/xs,device=d),ei,ea);ps.append(torch.sigmoid(lg).cpu().numpy());zs.append(z.cpu().numpy())
 p=np.concatenate(ps);z=np.concatenate(zs);return p,np.expm1(np.maximum(z,0))*p
def metrics(p,y,mask,j):
 e=p[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(abs(e[...,1]).mean()),'trigger_lmp_mae':float(abs(e[:,j,0]).mean())}
def bootstrap_delta(base,new,mask,j,rng,B):
 e0=abs(base[mask,j,0]);e1=abs(new[mask,j,0]);d=e1-e0;n=len(d)
 if not n:return None
 vals=np.array([np.mean(rng.choice(d,n,replace=True)) for _ in range(B)]);return {'mean_delta_mae':float(d.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--branch',choices=['oracle','original','targeted'],required=True);ap.add_argument('--augment-n',type=int,default=200);ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--mu-epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=45);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--weight-decay',type=float,default=1e-5);ap.add_argument('--mu-weight',type=float,default=.25);ap.add_argument('--mu-positive-weight',type=float,default=8.);ap.add_argument('--max-pos-weight',type=float,default=50.);ap.add_argument('--alpha-cap',type=float,default=.02);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--seed',type=int,default=20271004);a=ap.parse_args();data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');g=np.load(data/'graph.npz');j=int(np.where(te['bus_ids']==a.trigger_bus)[0][0]);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('M10 STRUCTURED SHADOW-PRICE DECODER');print(f'branch={a.branch} device={d} frozen train/val/test={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}')
 # Nominal predictor always uses original frozen training only, preserving nominal comparison.
 base,st,ei,ea,bv=train_base(tr['x'].astype(np.float32),tr['y'].astype(np.float32),va['x'].astype(np.float32),va['y'].astype(np.float32),g,a,d);bt=pred_base(base,tr['x'].astype(np.float32),st,ei,ea,d);bte=pred_base(base,te['x'].astype(np.float32),st,ei,ea,d)
 # alpha calibration support: original train, optionally plus selected M04 training-only points.
 cx=tr['x'].astype(np.float32);cy=tr['y'].astype(np.float32);cmu=tr['mu_vmin'][:,j].astype(float);cb=bt[:,j,0].astype(float);selection=None
 if a.branch=='targeted':
  m4=load(data/'m04_dual_regime'/'targeted_dual_regime.npz');ix,selection=stratified(m4,a.augment_n,a.seed+5);mb=pred_base(base,m4['x'][ix].astype(np.float32),st,ei,ea,d)[:,j,0];cx=np.concatenate((cx,m4['x'][ix].astype(np.float32)));cy=np.concatenate((cy,m4['y'][ix].astype(np.float32)));cmu=np.concatenate((cmu,m4['bus31_mu_vmin'][ix].astype(float)));cb=np.concatenate((cb,mb.astype(float)));print('M04 stratification='+json.dumps(selection))
 residual=cy[:,j,0].astype(float)-cb;active=cmu>1e-6;alpha=float((cmu[active]@residual[active])/(cmu[active]@cmu[active]+1e-12)) if active.any() else 0.;alpha=float(np.clip(alpha,-a.alpha_cap,a.alpha_cap));print(f'training-only structured alpha={alpha:.9g} active_calibration_n={active.sum()}')
 munet=None;muval=None
 if a.branch!='oracle':
  mux=tr['x'].astype(np.float32);mumu=tr['mu_vmin'].astype(np.float32)
  if a.branch=='targeted':mux=np.concatenate((mux,m4['x'][ix].astype(np.float32)));mumu=np.concatenate((mumu,m4['mu_vmin'][ix].astype(np.float32)))
  munet,muval=train_mu(mux,mumu,va['x'].astype(np.float32),va['mu_vmin'].astype(np.float32),g,a,d,st)
  prob,pmu=pred_mu(munet,te['x'].astype(np.float32),st,ei,ea,d)
  usemu=pmu[:,j]
 else:prob=None;usemu=te['mu_vmin'][:,j].astype(float)
 structured=bte.copy();structured[:,j,0]=bte[:,j,0]+alpha*usemu
 reg=regimes(data,len(bte));groups={};print('\nFROZEN TEST')
 for name,mask in [('all',np.ones(len(bte),bool)),('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]:
  if mask.any():
   q0=metrics(bte,te['y'],mask,j);q1=metrics(structured,te['y'],mask,j);groups[name]={'baseline':q0,'structured':q1};print(f'{name:14s} n={q1["n"]:4d} trigger baseline/structured={q0["trigger_lmp_mae"]:.6f}/{q1["trigger_lmp_mae"]:.6f} all_LMP={q1["lmp_mae"]:.6f}')
 trueact=te['mu_vmin'][:,j]>1e-6;diag={'true_active':int(trueact.sum())}
 if prob is not None:
  pa=prob[:,j]>=.5;diag.update({'TP':int((pa&trueact).sum()),'FP':int((pa&~trueact).sum()),'FN':int((~pa&trueact).sum()),'TN':int((~pa&~trueact).sum()),'mean_prob_active':float(prob[trueact,j].mean()) if trueact.any() else None,'mean_prob_inactive':float(prob[~trueact,j].mean()),'mu31_mae':float(abs(usemu-te['mu_vmin'][:,j]).mean())})
 print('SHADOW-PRICE DIAGNOSTIC '+json.dumps(diag))
 cont=continuation(data,te['x'].astype(np.float32));px=np.stack([q['x'] for q in cont]);bp=pred_base(base,px,st,ei,ea,d)
 if a.branch=='oracle':cu=np.array([q['true_mu31'] for q in cont])
 else:_,allmu=pred_mu(munet,px,st,ei,ea,d);cu=allmu[:,j]
 sp=bp[:,j,0]+alpha*cu;cres={};print('\nFROZEN G14 CONTINUATION')
 for s in sorted(set(q['active_scenario'] for q in cont)):
  ii=np.array([k for k,q in enumerate(cont) if q['active_scenario']==s and 0<=q['lambda']<=1]);yt=np.array([cont[k]['true_lmp'] for k in ii]);pb=bp[ii,j,0];pn=sp[ii];hi=yt>=50;cres[str(s)]={'n':int(len(ii)),'baseline_mae':float(abs(pb-yt).mean()),'structured_mae':float(abs(pn-yt).mean()),'high_price_n':int(hi.sum()),'baseline_high_price_mae':float(abs(pb[hi]-yt[hi]).mean()) if hi.any() else None,'structured_high_price_mae':float(abs(pn[hi]-yt[hi]).mean()) if hi.any() else None,'max_true':float(yt.max()),'max_baseline':float(pb.max()),'max_structured':float(pn.max())};q=cres[str(s)];print(f'path={s} MAE base/structured={q["baseline_mae"]:.4f}/{q["structured_mae"]:.4f} high_price={q["baseline_high_price_mae"]}/{q["structured_high_price_mae"]} max={q["max_true"]:.3f}/{q["max_structured"]:.3f}')
 rng=np.random.default_rng(a.seed+99);unc={n:bootstrap_delta(bte,structured,m,j,rng,a.bootstrap) for n,m in [('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]};print('\nBOOTSTRAP structured-minus-baseline trigger MAE');print(json.dumps(unc,indent=2))
 out={'configuration':vars(a),'alpha':alpha,'alpha_calibration_active_n':int(active.sum()),'m04_selection':selection,'base_best_val':bv,'mu_best_val':muval,'groups':groups,'shadow_price_diagnostic':diag,'continuation':cres,'bootstrap_delta':unc};fn=data/f'm10_{a.branch}_structured_shadow_price_summary.json';fn.write_text(json.dumps(out,indent=2)+'\n');torch.save({'base_state':base.state_dict(),'mu_state':None if munet is None else munet.state_dict(),'stats':st,'alpha':alpha,'configuration':vars(a)},data/f'm10_{a.branch}_structured_shadow_price_model.pt');print(f'\nSummary: {fn}');print('Decision: oracle success establishes decoder sufficiency. Original-vs-targeted isolates whether deployable failure is shadow-price support. A paper-level deployable success requires active/G14 high-price improvement with regular-regime trigger MAE confidence interval showing no material degradation.')
if __name__=='__main__':main()
