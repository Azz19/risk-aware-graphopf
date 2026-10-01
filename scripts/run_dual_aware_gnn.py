"""M03: dual-aware GNN with deployable predicted-dual and oracle-dual diagnostics."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from graphopf.gnn_baseline import EdgeMessageLayer

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def load(p):
 z=np.load(p);return {k:z[k] for k in z.files}
def graph(g,d):
 ef=g['edge_features'].astype(np.float32);em=ef.mean(0,keepdims=True);es=np.maximum(ef.std(0,keepdims=True),1e-6);return torch.tensor(g['edge_index'],dtype=torch.long,device=d),torch.tensor((ef-em)/es,dtype=torch.float32,device=d)
def regimes(data,n):
 r=np.array(['unknown']*n,dtype=object);p=data/'g08_regime_evaluation.csv'
 if p.exists():
  with p.open(newline='') as f:
   for q in csv.DictReader(f):
    i=int(q['scenario']);
    if 0<=i<n:r[i]=q['regime']
 return r
class Net(nn.Module):
 def __init__(self,h=64,layers=3):
  super().__init__();self.enc=nn.Sequential(nn.Linear(5,h),nn.ReLU());self.mp=nn.ModuleList(EdgeMessageLayer(h,6) for _ in range(layers));self.dual=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2));self.vm=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1));self.price=nn.Sequential(nn.Linear(h+2,h),nn.ReLU(),nn.Linear(h,1));self.oracle_price=nn.Sequential(nn.Linear(h+2,h),nn.ReLU(),nn.Linear(h,1))
 def encode(self,x,ei,ea):
  h=self.enc(x)
  for m in self.mp:h=m(h,ei,ea)
  return h
 def forward(self,x,ei,ea,true_z=None):
  h=self.encode(x,ei,ea);z=self.dual(h);p=self.price(torch.cat((h,z),-1));vm=self.vm(h);o=None if true_z is None else self.oracle_price(torch.cat((h,true_z),-1));return torch.cat((p,vm),-1),z,o

def stat(tr):
 x=tr['x'].astype(np.float32);y=tr['y'].astype(np.float32);return x.mean((0,1),keepdims=True),np.maximum(x.std((0,1),keepdims=True),1e-6),y.mean((0,1),keepdims=True),np.maximum(y.std((0,1),keepdims=True),1e-6)
def tens(z,st,d):
 xm,xs,ym,ys=st;x=torch.tensor((z['x']-xm)/xs,dtype=torch.float32,device=d);y=torch.tensor((z['y']-ym)/ys,dtype=torch.float32,device=d);dz=np.log1p(np.stack((z['mu_vmin'],z['mu_vmax']),-1).astype(np.float32));return x,y,torch.tensor(dz,dtype=torch.float32,device=d)
def train(tr,va,g,a,d):
 st=stat(tr);TX,TY,TZ=tens(tr,st,d);VX,VY,VZ=tens(va,st,d);ei,ea=graph(g,d);seed(a.seed);m=Net(a.hidden,a.layers).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 tw=1+a.dual_positive_weight*(TZ>0).float();vw=1+a.dual_positive_weight*(VZ>0).float()
 for ep in range(1,a.epochs+1):
  m.train();opt.zero_grad();p,z,o=m(TX,ei,ea,TZ);main=F.huber_loss(p,TY,delta=1.);dl=(F.smooth_l1_loss(z,TZ,reduction='none')*tw).mean();ol=F.huber_loss(o[...,0],TY[...,0],delta=1.);loss=main+a.dual_weight*dl+a.oracle_weight*ol;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();m.eval()
  with torch.no_grad():vp,vz,vo=m(VX,ei,ea,VZ);vl=F.huber_loss(vp,VY,delta=1.)+a.dual_weight*(F.smooth_l1_loss(vz,VZ,reduction='none')*vw).mean()+a.oracle_weight*F.huber_loss(vo[...,0],VY[...,0],delta=1.)
  v=float(vl)
  if v<best:best=v;state={k:q.detach().cpu().clone() for k,q in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'epoch={ep:4d} total={loss.item():.6f} main={main.item():.6f} dual={dl.item():.6f} oracle={ol.item():.6f} val={v:.6f}',flush=True)
  if stale>=a.patience:print(f'early_stop epoch={ep} best_val={best:.6f}');break
 m.load_state_dict(state);return m,st,ei,ea,best
def pred(m,z,st,ei,ea,d,b=256):
 X,Y,Z=tens(z,st,d);ps=[];zs=[];os=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(X),b):
   p,q,o=m(X[k:k+b],ei,ea,Z[k:k+b]);ps.append(p.cpu().numpy());zs.append(q.cpu().numpy());os.append(o.cpu().numpy())
 ym,ys=st[2],st[3];oracle=np.concatenate(os)[...,0]*float(ys[0,0,0])+float(ym[0,0,0]);return np.concatenate(ps)*ys+ym,np.concatenate(zs),oracle
def met(p,y,mask,tr):
 e=p[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(abs(e[...,1]).mean()),'trigger_lmp_mae':float(abs(e[:,tr,0]).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=45);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--weight-decay',type=float,default=1e-5);ap.add_argument('--dual-weight',type=float,default=.15);ap.add_argument('--dual-positive-weight',type=float,default=10.);ap.add_argument('--oracle-weight',type=float,default=.25);ap.add_argument('--seed',type=int,default=20271001);a=ap.parse_args();data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');g=np.load(data/'graph.npz');d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');j=int(np.where(te['bus_ids']==a.trigger_bus)[0][0])
 print('M03 DUAL-AWARE GNN');print(f'device={d} split={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}');print('Deployable price uses predicted log-duals; oracle price uses true duals only as a diagnostic.')
 m,st,ei,ea,best=train(tr,va,g,a,d);p,z,o=pred(m,te,st,ei,ea,d);r=regimes(data,len(p));groups={};print('\nDEPLOYABLE TEST')
 for name,mask in [('all',np.ones(len(p),bool)),('regular',r=='regular'),('near_boundary',r=='near_boundary'),('active',r=='active')]:
  if mask.any():groups[name]=met(p,te['y'],mask,j);q=groups[name];print(f'{name:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
 truez=np.log1p(np.stack((te['mu_vmin'],te['mu_vmax']),-1));trueact=te['mu_vmin'][:,j]>1e-6;print('\nBUS31 DUAL DIAGNOSTIC');print(f'true active={trueact.sum()} predicted log1p(mu_vmin) active mean={z[trueact,j,0].mean() if trueact.any() else float("nan"):.4f} inactive mean={z[~trueact,j,0].mean():.4f}');print(f'log-dual MAE bus31={abs(z[:,j,0]-truez[:,j,0]).mean():.6f}')
 print('\nORACLE-DUAL TEST DIAGNOSTIC')
 for name,mask in [('all',np.ones(len(p),bool)),('active',r=='active')]:
  if mask.any():print(f'{name:14s} LMP_MAE={abs(o[mask]-te["y"][mask,:,0]).mean():.6f} trigger_LMP_MAE={abs(o[mask,j]-te["y"][mask,j,0]).mean():.6f}')
 out={'configuration':vars(a),'best_val':best,'deployable_groups':groups,'bus31_logdual_mae':float(abs(z[:,j,0]-truez[:,j,0]).mean()),'bus31_true_active':int(trueact.sum()),'oracle_all_lmp_mae':float(abs(o-te['y'][...,0]).mean()),'oracle_active_trigger_lmp_mae':float(abs(o[r=='active',j]-te['y'][r=='active',j,0]).mean()) if (r=='active').any() else None};fn=data/'m03_dual_aware_summary.json';fn.write_text(json.dumps(out,indent=2)+'\n');torch.save({'state_dict':m.state_dict(),'stats':st,'configuration':vars(a)},data/'m03_dual_aware_model.pt');print(f'\nSummary: {fn}');print('Decision: oracle success + deployable failure isolates dual prediction/data coverage; both failing means dual conditioning alone is insufficient.')
if __name__=='__main__':main()
