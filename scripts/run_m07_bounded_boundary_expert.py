"""M07: bounded boundary expert on top of a frozen nominal GNN.

Tests whether a regime-specialized expert can improve the rare bus-31 price branch
without the unbounded residual explosions seen in M06. The nominal M05-A-style
predictor is trained only on frozen M01 train. The specialist is trained only on
stratified M04 support and predicts a bounded positive correction at bus 31.

Branches:
  oracle: physical bus-31 activity/contact chooses specialist (diagnostic only)
  soft:   deployable learned soft gate blends nominal and specialist
Frozen validation/test/G14 are evaluation only.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from run_m05_support_dual_factorial import load,regimes,continuation,stratified_m04,train as train_nominal,predict as predict_nominal

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

class BoundaryExpert(nn.Module):
 def __init__(self,n,h=128):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x):return self.net(x.flatten(1)).squeeze(-1)
class Gate(nn.Module):
 def __init__(self,n,h=128):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x):return self.net(x.flatten(1)).squeeze(-1)

def norm_stats(x):
 return x.mean((0,1),keepdims=True),np.maximum(x.std((0,1),keepdims=True),1e-6)
def fit_expert(x,res,a,d):
 xm,xs=norm_stats(x);X=torch.tensor((x-xm)/xs,dtype=torch.float32,device=d)
 # Robust cap is intentionally learned from training-only M04 support. This prevents M06-style explosions.
 positive=np.maximum(res,0.0);cap=float(np.quantile(positive,a.cap_quantile));cap=max(cap,a.min_cap)
 target=np.clip(positive/cap,0,1);Y=torch.tensor(target,dtype=torch.float32,device=d)
 seed(a.seed+71);m=BoundaryExpert(X.shape[1]*X.shape[2],a.expert_hidden).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 for ep in range(1,a.expert_epochs+1):
  m.train();opt.zero_grad();p=torch.sigmoid(m(X));loss=F.huber_loss(p,Y,delta=.2);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if ep==1 or ep%50==0:print(f'expert epoch={ep:4d} loss={loss.item():.6f}',flush=True)
 return m,(xm,xs,cap)
def expert_corr(m,x,st,d):
 xm,xs,cap=st;m.eval()
 with torch.no_grad():p=torch.sigmoid(m(torch.tensor((x-xm)/xs,dtype=torch.float32,device=d))).cpu().numpy()
 return p*cap

def fit_gate(x,y,a,d):
 xm,xs=norm_stats(x);X=torch.tensor((x-xm)/xs,dtype=torch.float32,device=d);Y=torch.tensor(y.astype(np.float32),device=d)
 pos=max(int(y.sum()),1);neg=max(len(y)-pos,1);pw=torch.tensor(neg/pos,dtype=torch.float32,device=d)
 seed(a.seed+89);m=Gate(X.shape[1]*X.shape[2],a.gate_hidden).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 for ep in range(1,a.gate_epochs+1):
  m.train();opt.zero_grad();logit=m(X);loss=F.binary_cross_entropy_with_logits(logit,Y,pos_weight=pw);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if ep==1 or ep%50==0:print(f'gate   epoch={ep:4d} loss={loss.item():.6f}',flush=True)
 return m,(xm,xs)
def gate_prob(m,x,st,d,temp):
 xm,xs=st;m.eval()
 with torch.no_grad():return torch.sigmoid(m(torch.tensor((x-xm)/xs,dtype=torch.float32,device=d))/temp).cpu().numpy()
def met(p,y,mask,j):
 e=p[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(abs(e[...,1]).mean()),'trigger_lmp_mae':float(abs(e[:,j,0]).mean())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--branch',choices=['oracle','soft'],required=True);ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--support-n',type=int,default=200);ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=45);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--weight-decay',type=float,default=1e-5);ap.add_argument('--dual-weight',type=float,default=.15);ap.add_argument('--dual-positive-weight',type=float,default=10.);ap.add_argument('--expert-hidden',type=int,default=128);ap.add_argument('--expert-epochs',type=int,default=400);ap.add_argument('--gate-hidden',type=int,default=128);ap.add_argument('--gate-epochs',type=int,default=300);ap.add_argument('--gate-temperature',type=float,default=1.5);ap.add_argument('--cap-quantile',type=float,default=.75);ap.add_argument('--min-cap',type=float,default=25.);ap.add_argument('--seed',type=int,default=20271001);a=ap.parse_args()
 data=Path(a.data);md=data/'m01_constraint_dual';tr=load(md/'train_constraint_dual.npz');va=load(md/'val_constraint_dual.npz');te=load(md/'test_constraint_dual.npz');m4=load(data/'m04_dual_regime'/'targeted_dual_regime.npz');g=np.load(data/'graph.npz');j=int(np.where(te['bus_ids']==a.trigger_bus)[0][0]);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print('M07 BOUNDED BOUNDARY EXPERT');print(f'branch={a.branch} device={d} frozen train/val/test={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}')
 nom,st,ei,ea,best=train_nominal(tr['x'].astype(np.float32),tr['y'].astype(np.float32),None,va['x'].astype(np.float32),va['y'].astype(np.float32),None,g,a,d,'standard')
 bm4,_=predict_nominal(nom,m4['x'].astype(np.float32),st,ei,ea,d,'standard');ix,selection=stratified_m04(m4,a.support_n,a.seed+5);sx=m4['x'][ix].astype(np.float32);res=m4['y'][ix,j,0].astype(float)-bm4[ix,j,0].astype(float)
 print('M04 specialist support='+json.dumps(selection));print(f'raw residual median_abs={np.median(abs(res)):.3f} max_abs={max(abs(res)):.3f}')
 expert,est=fit_expert(sx,res,a,d);print(f'bounded correction cap={est[2]:.6f} (training-only q={a.cap_quantile})')
 gate=gst=None
 if a.branch=='soft':
  # Learn boundary likelihood from original inactive support plus M04 selected support.
  gx=np.concatenate((tr['x'].astype(np.float32),sx));gy=np.concatenate(((tr['mu_vmin'][:,j]>1e-6),m4['bus31_active'][ix].astype(bool)));gate,gst=fit_gate(gx,gy,a,d)
 base,_=predict_nominal(nom,te['x'].astype(np.float32),st,ei,ea,d,'standard');corr=expert_corr(expert,te['x'].astype(np.float32),est,d);ta=te['mu_vmin'][:,j]>1e-6
 if a.branch=='oracle':w=ta.astype(float)
 else:w=gate_prob(gate,te['x'].astype(np.float32),gst,d,a.gate_temperature)
 pred=base.copy();pred[:,j,0]+=w*corr;reg=regimes(data,len(pred));groups={};print('\nFROZEN TEST')
 for name,mask in [('all',np.ones(len(pred),bool)),('regular',reg=='regular'),('near_boundary',reg=='near_boundary'),('active',reg=='active')]:
  if mask.any():groups[name]=met(pred,te['y'],mask,j);q=groups[name];print(f'{name:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
 if a.branch=='soft':
  hard=w>=.5;print(f'GATE@0.5 test: TP={int((hard&ta).sum())} FP={int((hard&~ta).sum())} FN={int((~hard&ta).sum())} TN={int((~hard&~ta).sum())} mean_active={w[ta].mean():.4f} mean_inactive={w[~ta].mean():.4f}')
 paths=continuation(data,te['x'].astype(np.float32));px=np.stack([q['x'] for q in paths]);bp,_=predict_nominal(nom,px,st,ei,ea,d,'standard');rc=expert_corr(expert,px,est,d)
 if a.branch=='oracle':
  with (data/'g14_active_set_continuation.csv').open(newline='') as f:rows=[q for q in csv.DictReader(f) if str(q['solved']).lower() in ('true','1','yes')]
  ww=np.array([(float(q['true_lower_v_dual'])>1e-6 or float(q['true_margin_pu'])<=1e-5) for q in rows],float)
 else:ww=gate_prob(gate,px,gst,d,a.gate_temperature)
 pp=bp.copy();pp[:,j,0]+=ww*rc;cont={};print('\nFROZEN G14 CONTINUATION')
 for s in sorted(set(q['active_scenario'] for q in paths)):
  ii=np.array([i for i,q in enumerate(paths) if q['active_scenario']==s and 0<=q['lambda']<=1]);tv=np.array([paths[i]['true_lmp'] for i in ii]);pv=pp[ii,j,0];hi=tv>=50;cont[str(s)]={'n':int(len(ii)),'lmp_mae':float(abs(pv-tv).mean()),'high_price_n':int(hi.sum()),'high_price_mae':float(abs(pv[hi]-tv[hi]).mean()) if hi.any() else None,'max_true_lmp':float(tv.max()),'max_pred_lmp':float(pv.max()),'max_weight':float(ww[ii].max())};q=cont[str(s)];print(f'path={s} MAE={q["lmp_mae"]:.4f} high_price_MAE={q["high_price_mae"]} max true/pred={q["max_true_lmp"]:.3f}/{q["max_pred_lmp"]:.3f} max_w={q["max_weight"]:.3f}')
 out={'configuration':vars(a),'nominal_best_val':best,'m04_selection':selection,'correction_cap':est[2],'groups':groups,'continuation':cont};fn=data/f'm07_{a.branch}_bounded_boundary_summary.json';fn.write_text(json.dumps(out,indent=2)+'\n');torch.save({'expert_state_dict':expert.state_dict(),'expert_stats':est,'gate_state_dict':gate.state_dict() if gate else None,'gate_stats':gst,'configuration':vars(a)},data/f'm07_{a.branch}_bounded_boundary_model.pt');print(f'\nSummary: {fn}');print('Decision: oracle tests bounded specialist sufficiency; soft tests deployability. Success requires G14 high-price improvement without material regular-regime degradation.')
if __name__=='__main__':main()
