"""M02: deployable constraint-aware GNN.

First model-design experiment after M01. A shared edge-aware GNN encoder predicts
(1) voltage/LMP outputs and (2) per-bus voltage constraint geometry. The price
head is explicitly conditioned on the model's OWN predicted voltage margins and
activity probability, so inference never uses oracle/test constraint labels.

M01 train/val targets supervise the auxiliary heads. Frozen G03 test and frozen
G14 continuation paths are evaluation only.
"""
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
def load_m01(p):
 z=np.load(p);return {k:z[k] for k in z.files}
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
   rr.append({**q,'active_scenario':int(q['active_scenario']),'control_scenario':int(q['control_scenario']),'lambda':float(q['lambda']),'true_lmp':float(q['true_lmp']),'true_vm_pu':float(q['true_vm_pu'])})
 return rr
def make_paths(ex,rows):
 out=[]
 for s in sorted(set(q['active_scenario'] for q in rows)):
  pr=[q for q in rows if q['active_scenario']==s];c=pr[0]['control_scenario'];x0=ex[c].astype(float);delta=ex[s].astype(float)-x0
  for q in pr:
   z=dict(q);z['x']=(x0+z['lambda']*delta).astype(np.float32);out.append(z)
 return out

class ConstraintAwareGNN(nn.Module):
 def __init__(self,node_dim=5,edge_dim=6,hidden=64,layers=3):
  super().__init__();self.enc=nn.Sequential(nn.Linear(node_dim,hidden),nn.ReLU());self.mp=nn.ModuleList(EdgeMessageLayer(hidden,edge_dim) for _ in range(layers))
  self.geom=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,3)) # standardized vmin margin, vmax margin, activity logit
  self.vm=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
  self.price=nn.Sequential(nn.Linear(hidden+3,hidden),nn.ReLU(),nn.Linear(hidden,1))
 def forward(self,x,ei,ea):
  h=self.enc(x)
  for l in self.mp:h=l(h,ei,ea)
  geom=self.geom(h);vm=self.vm(h);cond=torch.cat((h,geom[...,:2],torch.sigmoid(geom[...,2:3])),dim=-1);lmp=self.price(cond)
  return torch.cat((lmp,vm),-1),geom

def stats(tr):
 x=tr['x'].astype(np.float32);y=tr['y'].astype(np.float32);xm=x.mean((0,1),keepdims=True);xs=np.maximum(x.std((0,1),keepdims=True),1e-6);ym=y.mean((0,1),keepdims=True);ys=np.maximum(y.std((0,1),keepdims=True),1e-6)
 margins=np.stack((tr['vmin_margin'],tr['vmax_margin']),-1).astype(np.float32);mm=margins.mean((0,1),keepdims=True);ms=np.maximum(margins.std((0,1),keepdims=True),1e-6);return xm,xs,ym,ys,mm,ms
def tensors(z,st,d):
 xm,xs,ym,ys,mm,ms=st;x=torch.tensor((z['x']-xm)/xs,dtype=torch.float32,device=d);y=torch.tensor((z['y']-ym)/ys,dtype=torch.float32,device=d);m=np.stack((z['vmin_margin'],z['vmax_margin']),-1).astype(np.float32);m=torch.tensor((m-mm)/ms,dtype=torch.float32,device=d);a=torch.tensor(z['voltage_active'].astype(np.float32),device=d);return x,y,m,a
def train(tr,va,g,a,d):
 st=stats(tr);TX,TY,TM,TA=tensors(tr,st,d);VX,VY,VM,VA=tensors(va,st,d);ei,ea=graph_tensors(g,d);seed(a.seed);m=ConstraintAwareGNN(hidden=a.hidden,layers=a.layers).to(d);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);best=1e99;state=None;stale=0
 pos=float(TA.numel()-TA.sum().item())/max(float(TA.sum().item()),1.);pw=torch.tensor(min(pos,a.max_pos_weight),device=d)
 print(f'voltage activity positive fraction={TA.mean().item():.6f} BCE pos_weight={pw.item():.3f}')
 for ep in range(1,a.epochs+1):
  m.train();opt.zero_grad();pred,geom=m(TX,ei,ea);main=F.huber_loss(pred,TY,delta=a.huber_delta);ml=F.huber_loss(geom[...,:2],TM,delta=a.huber_delta);cl=F.binary_cross_entropy_with_logits(geom[...,2],TA,pos_weight=pw);loss=main+a.margin_weight*ml+a.activity_weight*cl;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip);opt.step();m.eval()
  with torch.no_grad():vp,vg=m(VX,ei,ea);vl=F.huber_loss(vp,VY,delta=a.huber_delta)+a.margin_weight*F.huber_loss(vg[...,:2],VM,delta=a.huber_delta)+a.activity_weight*F.binary_cross_entropy_with_logits(vg[...,2],VA,pos_weight=pw)
  v=float(vl.item())
  if v<best-1e-8:best=v;state={k:q.detach().cpu().clone() for k,q in m.state_dict().items()};stale=0
  else:stale+=1
  if ep==1 or ep%25==0:print(f'epoch={ep:4d} total={loss.item():.6f} main={main.item():.6f} margin={ml.item():.6f} activity={cl.item():.6f} val={v:.6f}',flush=True)
  if stale>=a.patience:print(f'early_stop epoch={ep} best_val={best:.6f}');break
 m.load_state_dict(state);return m,st,ei,ea,best
def predict(m,x,st,ei,ea,d,b=256):
 xm,xs,ym,ys,mm,ms=st;po=[];go=[];m.eval()
 with torch.no_grad():
  for k in range(0,len(x),b):
   p,g=m(torch.tensor((x[k:k+b]-xm)/xs,dtype=torch.float32,device=d),ei,ea);po.append(p.cpu().numpy());go.append(g.cpu().numpy())
 return np.concatenate(po)*ys+ym,np.concatenate(go)
def metrics(pred,y,mask,tr):
 e=pred[mask]-y[mask];return {'n':int(mask.sum()),'lmp_mae':float(np.abs(e[...,0]).mean()),'lmp_rmse':float(np.sqrt((e[...,0]**2).mean())),'vm_mae':float(np.abs(e[...,1]).mean()),'trigger_lmp_mae':float(np.abs(e[:,tr,0]).mean())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--m01',default=None);p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=3);p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=45);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--huber-delta',type=float,default=1.0);p.add_argument('--margin-weight',type=float,default=.25);p.add_argument('--activity-weight',type=float,default=.10);p.add_argument('--max-pos-weight',type=float,default=25.);p.add_argument('--grad-clip',type=float,default=10.);p.add_argument('--seed',type=int,default=20271001);a=p.parse_args()
 data=Path(a.data);md=Path(a.m01) if a.m01 else data/'m01_constraint_dual';tr=load_m01(md/'train_constraint_dual.npz');va=load_m01(md/'val_constraint_dual.npz');te=load_m01(md/'test_constraint_dual.npz');g=np.load(data/'graph.npz');d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');trig=a.trigger_bus-1
 print('M02 CONSTRAINT-AWARE GNN');print(f'device={d} split={len(tr["x"])}/{len(va["x"])}/{len(te["x"])} trigger_bus={a.trigger_bus}');print('Price decoder is conditioned only on predicted constraint geometry/activity at inference.')
 m,st,ei,ea,best=train(tr,va,g,a,d);pe,ge=predict(m,te['x'],st,ei,ea,d);reg=regimes(data,len(te['x']));masks={'all':np.ones(len(pe),bool),'regular':reg=='regular','near_boundary':reg=='near_boundary','active':reg=='active'};groups={k:metrics(pe,te['y'],v,trig) for k,v in masks.items() if v.any()}
 print('\nTEST RESULTS')
 for k,q in groups.items():print(f'{k:14s} n={q["n"]:4d} LMP_MAE={q["lmp_mae"]:.6f} LMP_RMSE={q["lmp_rmse"]:.6f} VM_MAE={q["vm_mae"]:.7f} trigger_LMP_MAE={q["trigger_lmp_mae"]:.6f}')
 # bus31 auxiliary quality; denormalize predicted margins
 mm,ms=st[4],st[5];gm=ge[...,:2]*ms+mm;gp=1/(1+np.exp(-ge[...,2]));busids=te['bus_ids'];j=int(np.where(busids==a.trigger_bus)[0][0]);trueact=te['voltage_active'][:,j].astype(bool);predact=gp[:,j]>=.5;tp=int((trueact&predact).sum());fp=int((~trueact&predact).sum());fn=int((trueact&~predact).sum());tn=int((~trueact&~predact).sum())
 print(f'\nBUS {a.trigger_bus} AUXILIARY: true_active={trueact.sum()} predicted_active={predact.sum()} TP={tp} FP={fp} FN={fn} TN={tn}');print(f'bus31 vmin-margin MAE={np.abs(gm[:,j,0]-te["vmin_margin"][:,j]).mean():.7f} pu')
 cr=continuation(data);paths=make_paths(te['x'],cr);px=np.stack([q['x'] for q in paths]);pp,pg=predict(m,px,st,ei,ea,d);pgm=pg[...,:2]*ms+mm;pgp=1/(1+np.exp(-pg[...,2]));cont={};print('\nFROZEN G14 CONTINUATION')
 for s in sorted(set(q['active_scenario'] for q in paths)):
  ix=np.array([i for i,q in enumerate(paths) if q['active_scenario']==s and 0<=q['lambda']<=1]);true=np.array([paths[i]['true_lmp'] for i in ix]);pred=pp[ix,trig,0];hi=true>=50;q={'n':len(ix),'lmp_mae':float(np.abs(pred-true).mean()),'high_price_n':int(hi.sum()),'high_price_mae':float(np.abs(pred[hi]-true[hi]).mean()) if hi.any() else None,'max_true_lmp':float(true.max()),'max_pred_lmp':float(pred.max()),'max_pred_active_prob':float(pgp[ix,trig].max()),'min_pred_vmin_margin':float(pgm[ix,trig,0].min())};cont[str(s)]=q;print(f'path={s} MAE={q["lmp_mae"]:.4f} high_price_MAE={q["high_price_mae"]} max true/pred={q["max_true_lmp"]:.3f}/{q["max_pred_lmp"]:.3f} max P(active)={q["max_pred_active_prob"]:.3f} min pred margin={q["min_pred_vmin_margin"]:.6f}')
 summary={'configuration':vars(a),'best_val':best,'groups':groups,'bus31_auxiliary':{'true_active':int(trueact.sum()),'predicted_active':int(predact.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'vmin_margin_mae':float(np.abs(gm[:,j,0]-te['vmin_margin'][:,j]).mean())},'continuation':cont};js=data/'m02_constraint_aware_summary.json';js.write_text(json.dumps(summary,indent=2)+'\n');torch.save({'state_dict':m.state_dict(),'stats':st,'configuration':vars(a)},data/'m02_constraint_aware_model.pt');print(f'\nSummary: {js}\nModel: {data/"m02_constraint_aware_model.pt"}');print('Decision target: constraint conditioning is useful only if it preserves nominal accuracy AND either detects the rare bus-31 regime or bends the frozen continuation price trajectory toward the physical high-price branch.')
if __name__=='__main__':main()
