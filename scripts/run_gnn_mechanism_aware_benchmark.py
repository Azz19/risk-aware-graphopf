"""G18: mechanism-aware model benchmark.

Compare three hypotheses on the frozen G03 split and the frozen G14 continuation
paths: (1) ordinary Huber GNN, (2) voltage-margin auxiliary supervision, and
(3) voltage-margin + LMP-gradient supervision.  No test targets are used for
training/model selection.  Success requires preserving nominal accuracy while
improving the high-price continuation branch.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from graphopf.gnn_baseline import SupervisedGraphOPF


def load_split(p):
 z=np.load(p);return z['x'].astype(np.float32),z['y'].astype(np.float32)
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def graph_tensors(g,d):
 ef=g['edge_features'].astype(np.float32);em=ef.mean(0,keepdims=True);es=np.maximum(ef.std(0,keepdims=True),1e-6);return torch.tensor(g['edge_index'],dtype=torch.long,device=d),torch.tensor((ef-em)/es,dtype=torch.float32,device=d)
def regimes(data,n):
 r=np.array(['unknown']*n,dtype=object);p=data/'g08_regime_evaluation.csv'
 if p.exists():
  with p.open(newline='') as f:
   for q in csv.DictReader(f):
    i=int(q['scenario']);
    if 0<=i<n:r[i]=q['regime']
 return r
def continuation(data):
 p=data/'g14_active_set_continuation.csv'
 if not p.exists():raise FileNotFoundError('G14 continuation CSV required: '+str(p))
 rr=[]
 with p.open(newline='') as f:
  for q in csv.DictReader(f):
   if str(q['solved']).lower() not in ('true','1','yes'):continue
   q={**q,'active_scenario':int(q['active_scenario']),'lambda':float(q['lambda']),'true_lmp':float(q['true_lmp']),'true_vm_pu':float(q['true_vm_pu'])};rr.append(q)
 return rr
def nearest_index(x,query):
 # exact G14 path points need not occur in train; used only for evaluation inputs reconstructed from endpoints
 return None
def make_paths(ex,rows):
 out=[]
 for s in sorted(set(q['active_scenario'] for q in rows)):
  pr=[q for q in rows if q['active_scenario']==s];c=int(pr[0]['control_scenario']);x0=ex[c].astype(float);d=ex[s].astype(float)-x0
  for q in pr:
   q=dict(q);q['x']=(x0+q['lambda']*d).astype(np.float32);out.append(q)
 return out

def train(tx,ty,vx,vy,g,a,d,mode):
 xm=tx.mean((0,1),keepdims=True);xs=np.maximum(tx.std((0,1),keepdims=True),1e-6);ym=ty.mean((0,1),keepdims=True);ys=np.maximum(ty.std((0,1),keepdims=True),1e-6)
 TX=torch.tensor((tx-xm)/xs,device=d);TY=torch.tensor((ty-ym)/ys,device=d);VX=torch.tensor((vx-xm)/xs,device=d);VY=torch.tensor((vy-ym)/ys,device=d);ei,ea=graph_tensors(g,d);tr=a.trigger_bus-1
 seed(a.seed);m=SupervisedGraphOPF(hidden_dim=a.hidden,layers=a.layers).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 for ep in range(1,a.epochs+1):
  m.train();opt.zero_grad();TXr=TX.detach().clone().requires_grad_(mode=='margin_grad');pred=m(TXr,ei,ea);loss=F.huber_loss(pred,TY,delta=a.huber_delta)
  if mode in ('margin','margin_grad'):
   # Explicitly emphasize the trigger voltage coordinate, a deployable physical proximity signal.
   loss=loss+a.margin_weight*F.huber_loss(pred[:,tr,1],TY[:,tr,1],delta=a.huber_delta)
  if mode=='margin_grad':
   # Match local sensitivity of normalized trigger LMP to input features using finite target neighbors
   # approximated by autograd target sensitivity proxy: encourage non-flat learned price response near low VM.
   low=(TY[:,tr,1] < torch.quantile(TY[:,tr,1].detach(),a.low_vm_quantile)).float()
   grad=torch.autograd.grad(pred[:,tr,0].sum(),TXr,create_graph=True)[0];gn=grad.pow(2).mean((1,2)).sqrt();loss=loss-a.gradient_weight*(low*gn).mean()
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip);opt.step();m.eval()
  with torch.no_grad():vl=F.huber_loss(m(VX,ei,ea),VY,delta=a.huber_delta).item()
  if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'{mode:12s} epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
  if stale>=a.patience:break
 m.load_state_dict(state);return m,(xm,xs,ym,ys),ei,ea,best
def predict(m,x,st,ei,ea,d,b=256):
 xm,xs,ym,ys=st;out=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(x),b):out.append(m(torch.tensor((x[k:k+b]-xm)/xs,device=d),ei,ea).cpu().numpy())
 return np.concatenate(out)*ys+ym
def metrics(pred,y,mask,tr):
 e=pred[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(np.abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(np.abs(e[...,1]).mean()),'trigger_lmp_mae':float(np.abs(e[:,tr,0]).mean())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--branch',choices=['baseline','margin','margin_grad','all'],default='all');p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=3);p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=40);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--huber-delta',type=float,default=1.0);p.add_argument('--margin-weight',type=float,default=1.0);p.add_argument('--gradient-weight',type=float,default=1e-3);p.add_argument('--low-vm-quantile',type=float,default=.10);p.add_argument('--grad-clip',type=float,default=10.);p.add_argument('--seed',type=int,default=20271001);a=p.parse_args()
 data=Path(a.data);tx,ty=load_split(data/'train.npz');vx,vy=load_split(data/'val.npz');ex,ey=load_split(data/'test.npz');g=np.load(data/'graph.npz');reg=regimes(data,len(ex));tr=a.trigger_bus-1;cr=continuation(data);paths=make_paths(ex,cr);px=np.stack([q['x'] for q in paths]);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');modes=['baseline','margin','margin_grad'] if a.branch=='all' else [a.branch];summary={'configuration':vars(a),'models':{}};flat=[]
 print('G18 MECHANISM-AWARE MODEL BENCHMARK');print(f'device={d} branch={a.branch} split={len(tx)}/{len(vx)}/{len(ex)} continuation_points={len(px)}')
 for mode in modes:
  trainmode='baseline' if mode=='baseline' else mode;m,st,ei,ea,best=train(tx,ty,vx,vy,g,a,d,trainmode);pe=predict(m,ex,st,ei,ea,d);pp=predict(m,px,st,ei,ea,d)
  masks={'all':np.ones(len(ex),bool),'regular':reg=='regular','near_boundary':reg=='near_boundary','active':reg=='active'};groups={k:metrics(pe,ey,v,tr) for k,v in masks.items() if np.any(v)}
  cont={}
  for s in sorted(set(q['active_scenario'] for q in paths)):
   ix=np.array([i for i,q in enumerate(paths) if q['active_scenario']==s and 0<=q['lambda']<=1]);true=np.array([paths[i]['true_lmp'] for i in ix]);pred=pp[ix,tr,0];hi=true>=50;cont[str(s)]={'n':len(ix),'lmp_mae':float(np.abs(pred-true).mean()),'max_true_lmp':float(true.max()),'max_pred_lmp':float(pred.max()),'high_price_n':int(hi.sum()),'high_price_mae':float(np.abs(pred[hi]-true[hi]).mean()) if hi.any() else None,'endpoint_true':float(true[-1]),'endpoint_pred':float(pred[-1])}
  summary['models'][mode]={'best_val':best,'groups':groups,'continuation':cont};print('\n'+mode.upper())
  for k,q in groups.items():print(f'{k:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
  for s,q in cont.items():print(f' path={s} MAE={q["lmp_mae"]:.4f} high_price_n={q["high_price_n"]} high_price_MAE={q["high_price_mae"]} max true/pred={q["max_true_lmp"]:.3f}/{q["max_pred_lmp"]:.3f}')
  for k,q in groups.items():flat.append({'model':mode,'group':k,**q})
 out=data/f'g18_{a.branch}_mechanism_aware_benchmark.csv';js=data/f'g18_{a.branch}_mechanism_aware_benchmark_summary.json'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nCSV output: {out}\nSummary: {js}');print('Decision rule: improvement must appear on frozen high-price continuation points without a material loss of regular/non-extreme test accuracy. Treat margin_grad as exploratory: its gradient regularizer rewards non-flat sensitivity near low-voltage training states; it does not encode OPF duals or use test labels.')
if __name__=='__main__':main()
