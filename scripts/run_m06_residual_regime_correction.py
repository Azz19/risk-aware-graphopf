"""M06: residual rare-regime correction on top of a frozen nominal GNN.

The nominal predictor is trained only on the frozen M01 train split. A separate
residual network is trained on stratified M04 training-only support to predict
the missing LMP correction. Two branches are provided:
  oracle: apply the correction using the true M04/test bus-31 active label
          (diagnostic only; tests whether residual representation has capacity)
  gated:  learn a deployable gate from input x using original train negatives
          plus M04 near/active examples.
Validation/test/G14 identities remain frozen. Test/G14 targets are evaluation only.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from graphopf.gnn_baseline import SupervisedGraphOPF
from run_m05_support_dual_factorial import load,graph,regimes,continuation,stratified_m04,train as train_nominal,predict as predict_nominal

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
class ResidualNet(nn.Module):
 def __init__(self,n,h=128):super().__init__();self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x):return self.net(x.flatten(1)).squeeze(-1)
class GateNet(nn.Module):
 def __init__(self,n,h=128):super().__init__();self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x):return self.net(x.flatten(1)).squeeze(-1)
def fit_residual(x,r,a,d):
 xm=x.mean((0,1),keepdims=True);xs=np.maximum(x.std((0,1),keepdims=True),1e-6);scale=max(float(np.median(np.abs(r))),1.0);X=torch.tensor((x-xm)/xs,device=d);Y=torch.tensor(np.arcsinh(r/scale),dtype=torch.float32,device=d);seed(a.seed+31);m=ResidualNet(X.shape[1]*X.shape[2],a.residual_hidden).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 for ep in range(1,a.residual_epochs+1):
  m.train();opt.zero_grad();p=m(X);loss=F.huber_loss(p,Y,delta=1.0);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
  if ep==1 or ep%50==0:print(f'residual epoch={ep:4d} loss={loss.item():.6f}',flush=True)
 return m,(xm,xs,scale)
def residual_predict(m,x,st,d):
 xm,xs,scale=st;m.eval()
 with torch.no_grad():z=m(torch.tensor((x-xm)/xs,device=d)).cpu().numpy()
 return np.sinh(np.clip(z,-10,10))*scale
def fit_gate(x,y,a,d):
 xm=x.mean((0,1),keepdims=True);xs=np.maximum(x.std((0,1),keepdims=True),1e-6);X=torch.tensor((x-xm)/xs,device=d);Y=torch.tensor(y.astype(np.float32),device=d);pos=max(int(y.sum()),1);neg=max(len(y)-pos,1);pw=torch.tensor(neg/pos,device=d);seed(a.seed+47);m=GateNet(X.shape[1]*X.shape[2],a.gate_hidden).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 for ep in range(1,a.gate_epochs+1):
  m.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(X),Y,pos_weight=pw);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
  if ep==1 or ep%50==0:print(f'gate     epoch={ep:4d} loss={loss.item():.6f}',flush=True)
 return m,(xm,xs)
def gate_prob(m,x,st,d):
 xm,xs=st;m.eval()
 with torch.no_grad():return torch.sigmoid(m(torch.tensor((x-xm)/xs,device=d))).cpu().numpy()
def met(p,y,mask,j):
 e=p[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(abs(e[...,1]).mean()),'trigger_lmp_mae':float(abs(e[:,j,0]).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--branch',choices=['oracle','gated'],required=True);ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--support-n',type=int,default=200);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=45);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--weight-decay',type=float,default=1e-5);ap.add_argument('--dual-weight',type=float,default=.15);ap.add_argument('--dual-positive-weight',type=float,default=10.);ap.add_argument('--residual-hidden',type=int,default=128);ap.add_argument('--residual-epochs',type=int,default=400);ap.add_argument('--gate-hidden',type=int,default=128);ap.add_argument('--gate-epochs',type=int,default=300);ap.add_argument('--gate-threshold',type=float,default=.5);ap.add_argument('--seed',type=int,default=20271001);a=ap.parse_args()
 data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');m4=load(data/'m04_dual_regime'/'targeted_dual_regime.npz');g=np.load(data/'graph.npz');j=int(np.where(te['bus_ids']==a.trigger_bus)[0][0]);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('M06 RESIDUAL REGIME CORRECTION');print(f'branch={a.branch} device={d} frozen train/val/test={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}')
 # Frozen nominal architecture: M05-A recipe, retrained deterministically on original support only.
 nom,st,ei,ea,best=train_nominal(tr['x'].astype(np.float32),tr['y'].astype(np.float32),None,va['x'].astype(np.float32),va['y'].astype(np.float32),None,g,a,d,'standard');base_m4,_=predict_nominal(nom,m4['x'].astype(np.float32),st,ei,ea,d,'standard');ix,selection=stratified_m04(m4,a.support_n,a.seed+5);sx=m4['x'][ix].astype(np.float32);true=m4['y'][ix,j,0].astype(float);res=true-base_m4[ix,j,0].astype(float);print('M04 residual support='+json.dumps(selection));print(f'residual target n={len(ix)} median_abs={np.median(abs(res)):.3f} max_abs={max(abs(res)):.3f}')
 rnet,rst=fit_residual(sx,res,a,d)
 gate=None;gst=None
 if a.branch=='gated':
  # Original train provides abundant inactive examples; selected M04 provides near/active boundary support.
  gx=np.concatenate((tr['x'].astype(np.float32),sx));gy=np.concatenate(((tr['mu_vmin'][:,j]>1e-6),m4['bus31_active'][ix].astype(bool)));gate,gst=fit_gate(gx,gy,a,d)
 base,_=predict_nominal(nom,te['x'].astype(np.float32),st,ei,ea,d,'standard');corr=residual_predict(rnet,te['x'].astype(np.float32),rst,d);true_active=te['mu_vmin'][:,j]>1e-6
 if a.branch=='oracle':w=true_active.astype(float)
 else:w=(gate_prob(gate,te['x'].astype(np.float32),gst,d)>=a.gate_threshold).astype(float)
 pred=base.copy();pred[:,j,0]+=w*corr;reg=regimes(data,len(pred));groups={};print('\nFROZEN TEST')
 for name,mask in [('all',np.ones(len(pred),bool)),('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]:
  if mask.any():groups[name]=met(pred,te['y'],mask,j);q=groups[name];print(f'{name:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
 if a.branch=='gated':
  hard=w.astype(bool);print(f'GATE test: TP={int((hard&true_active).sum())} FP={int((hard&~true_active).sum())} FN={int((~hard&true_active).sum())} TN={int((~hard&~true_active).sum())}')
 paths=continuation(data,te['x'].astype(np.float32));px=np.stack([q['x'] for q in paths]);bp,_=predict_nominal(nom,px,st,ei,ea,d,'standard');rc=residual_predict(rnet,px,rst,d)
 if a.branch=='oracle':
  # Diagnostic oracle uses physical G14 voltage contact/dual, never for fitting.
  with (data/'g14_active_set_continuation.csv').open(newline='') as f:rows=[q for q in csv.DictReader(f) if str(q['solved']).lower() in ('true','1','yes')]
  ww=np.array([(float(q['true_lower_v_dual'])>1e-6 or float(q['true_margin_pu'])<=1e-5) for q in rows],float)
 else:ww=(gate_prob(gate,px,gst,d)>=a.gate_threshold).astype(float)
 pp=bp.copy();pp[:,j,0]+=ww*rc;cont={};print('\nFROZEN G14 CONTINUATION')
 for s in sorted(set(q['active_scenario'] for q in paths)):
  ii=np.array([i for i,q in enumerate(paths) if q['active_scenario']==s and 0<=q['lambda']<=1]);tv=np.array([paths[i]['true_lmp'] for i in ii]);pv=pp[ii,j,0];hi=tv>=50;cont[str(s)]={'n':int(len(ii)),'lmp_mae':float(abs(pv-tv).mean()),'high_price_n':int(hi.sum()),'high_price_mae':float(abs(pv[hi]-tv[hi]).mean()) if hi.any() else None,'max_true_lmp':float(tv.max()),'max_pred_lmp':float(pv.max())};q=cont[str(s)];print(f'path={s} MAE={q["lmp_mae"]:.4f} high_price_MAE={q["high_price_mae"]} max true/pred={q["max_true_lmp"]:.3f}/{q["max_pred_lmp"]:.3f}')
 out={'configuration':vars(a),'nominal_best_val':best,'m04_selection':selection,'groups':groups,'continuation':cont};fn=data/f'm06_{a.branch}_residual_regime_summary.json';fn.write_text(json.dumps(out,indent=2)+'\n');torch.save({'residual_state_dict':rnet.state_dict(),'residual_stats':rst,'gate_state_dict':gate.state_dict() if gate else None,'gate_stats':gst,'configuration':vars(a)},data/f'm06_{a.branch}_residual_regime_model.pt');print(f'\nSummary: {fn}');print('Decision: oracle improvement tests residual representational sufficiency; gated improvement tests deployability. Require nominal preservation plus material active/G14 high-price improvement.')
if __name__=='__main__':main()
