"""G13: active-set mixture-of-experts experiment on the fixed G03 split.

Tests whether explicit regime separation can represent the discontinuous bus-31
LMP response better than one smooth GNN.  The deployable gate is trained only on
training inputs with labels derived from training targets (bus-31 VM at VMIN).
Validation/test targets are never used as gate inputs.  An oracle-gated result is
reported strictly as a diagnostic upper-bound on routing, not as deployable
performance.  A capacity-matched wider single GNN is included as a control.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphopf.gnn_baseline import SupervisedGraphOPF


def load_split(p):
    z=np.load(p); return z['x'].astype(np.float32),z['y'].astype(np.float32)

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def met(pred,y,mask,trig):
    if not np.any(mask):return {'n':0}
    e=pred[mask]-y[mask]
    return {'n':int(mask.sum()),'lmp_mae':float(np.mean(np.abs(e[...,0]))),'lmp_rmse':float(np.sqrt(np.mean(e[...,0]**2))),'vm_mae':float(np.mean(np.abs(e[...,1]))),'trigger_lmp_mae':float(np.mean(np.abs(e[:,trig,0])))}

def load_test_regimes(data,n):
    r=np.array(['unknown']*n,dtype=object);p=data/'g08_regime_evaluation.csv'
    if p.exists():
        with p.open(newline='') as f:
            for q in csv.DictReader(f):
                i=int(q['scenario']);
                if 0<=i<n:r[i]=q['regime']
    return r

def active_from_y(y,trig,vmin,tol):return y[:,trig,1]<=vmin+tol

class Gate(nn.Module):
    def __init__(self,n_bus,n_features,hidden):
        super().__init__();self.net=nn.Sequential(nn.Linear(n_bus*n_features,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x):return self.net(x.flatten(1)).squeeze(-1)

def tensors(x,y,xm,xs,ym,ys,device):
    return torch.tensor((x-xm)/xs,dtype=torch.float32,device=device),torch.tensor((y-ym)/ys,dtype=torch.float32,device=device)

def graph_tensors(g,device):
    ef=g['edge_features'].astype(np.float32);em=ef.mean(0,keepdims=True);es=np.maximum(ef.std(0,keepdims=True),1e-6)
    return torch.tensor(g['edge_index'],dtype=torch.long,device=device),torch.tensor((ef-em)/es,dtype=torch.float32,device=device)

def train_gnn(tx,ty,vx,vy,g,args,device,hidden=None,weights=None,label='model'):
    hidden=hidden or args.hidden;xm=tx.mean((0,1),keepdims=True);xs=np.maximum(tx.std((0,1),keepdims=True),1e-6);ym=ty.mean((0,1),keepdims=True);ys=np.maximum(ty.std((0,1),keepdims=True),1e-6)
    TX,TY=tensors(tx,ty,xm,xs,ym,ys,device);VX,VY=tensors(vx,vy,xm,xs,ym,ys,device);ei,ea=graph_tensors(g,device)
    W=torch.ones(len(tx),device=device) if weights is None else torch.tensor(weights,dtype=torch.float32,device=device)
    seed_all(args.seed);m=SupervisedGraphOPF(hidden_dim=hidden,layers=args.layers).to(device);opt=torch.optim.Adam(m.parameters(),lr=args.lr,weight_decay=args.weight_decay);best=float('inf');state=None;stale=0
    for ep in range(1,args.epochs+1):
        m.train();opt.zero_grad();per=F.huber_loss(m(TX,ei,ea),TY,reduction='none',delta=args.huber_delta).mean((1,2));loss=(per*W).sum()/W.sum();loss.backward();opt.step();m.eval()
        with torch.no_grad():vl=F.huber_loss(m(VX,ei,ea),VY,delta=args.huber_delta).item()
        if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if ep==1 or ep%25==0:print(f'{label:18s} epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
        if stale>=args.patience:break
    m.load_state_dict(state);return m,(xm,xs,ym,ys),ei,ea,best

def predict(m,x,stats,ei,ea,device,batch=128):
    xm,xs,ym,ys=stats;X=torch.tensor((x-xm)/xs,dtype=torch.float32,device=device);out=[];m.eval()
    with torch.no_grad():
        for s in range(0,len(X),batch):out.append(m(X[s:s+batch],ei,ea).cpu().numpy())
    return np.concatenate(out)*ys+ym

def train_gate(tx,lab,vx,vlab,args,device):
    xm=tx.mean((0,1),keepdims=True);xs=np.maximum(tx.std((0,1),keepdims=True),1e-6);TX=torch.tensor((tx-xm)/xs,dtype=torch.float32,device=device);VX=torch.tensor((vx-xm)/xs,dtype=torch.float32,device=device);Y=torch.tensor(lab.astype(np.float32),device=device);VY=torch.tensor(vlab.astype(np.float32),device=device)
    pos=max(int(lab.sum()),1);pw=torch.tensor([(len(lab)-pos)/pos],device=device);seed_all(args.seed+17);m=Gate(tx.shape[1],tx.shape[2],args.gate_hidden).to(device);opt=torch.optim.Adam(m.parameters(),lr=args.gate_lr,weight_decay=args.weight_decay);best=float('inf');state=None;stale=0
    for ep in range(1,args.gate_epochs+1):
        m.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(TX),Y,pos_weight=pw);loss.backward();opt.step();m.eval()
        with torch.no_grad():vl=F.binary_cross_entropy_with_logits(m(VX),VY,pos_weight=pw).item()
        if vl<best-1e-7:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if ep==1 or ep%50==0:print(f'gate               epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
        if stale>=args.patience:break
    m.load_state_dict(state);return m,(xm,xs),best

def gate_prob(m,x,stats,device):
    xm,xs=stats;X=torch.tensor((x-xm)/xs,dtype=torch.float32,device=device);m.eval()
    with torch.no_grad():return torch.sigmoid(m(X)).cpu().numpy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--branch',choices=['baseline','moe','all'],default='all');p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--vmin',type=float,default=.94);p.add_argument('--active-tol',type=float,default=1e-5);p.add_argument('--near-margin',type=float,default=.01);p.add_argument('--hidden',type=int,default=64);p.add_argument('--wide-hidden',type=int,default=96);p.add_argument('--layers',type=int,default=3);p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=40);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--huber-delta',type=float,default=1.);p.add_argument('--active-weight',type=float,default=50.);p.add_argument('--gate-hidden',type=int,default=128);p.add_argument('--gate-lr',type=float,default=1e-3);p.add_argument('--gate-epochs',type=int,default=500);p.add_argument('--gate-threshold',type=float,default=.5);p.add_argument('--seed',type=int,default=20271001);a=p.parse_args()
    data=Path(a.data);tx,ty=load_split(data/'train.npz');vx,vy=load_split(data/'val.npz');ex,ey=load_split(data/'test.npz');g=np.load(data/'graph.npz');trig=a.trigger_bus-1;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ta=active_from_y(ty,trig,a.vmin,a.active_tol);va=active_from_y(vy,trig,a.vmin,a.active_tol);reg=load_test_regimes(data,len(ex));ea=(reg=='active') if np.any(reg!='unknown') else active_from_y(ey,trig,a.vmin,a.active_tol);near=(reg=='near_boundary') if np.any(reg!='unknown') else ((~ea)&(ey[:,trig,1]<=a.vmin+a.near_margin));regular=(reg=='regular') if np.any(reg!='unknown') else ~(ea|near)
    print('G13 ACTIVE-SET MIXTURE-OF-EXPERTS');print(f'device={device} branch={a.branch} split={len(tx)}/{len(vx)}/{len(ex)}');print(f'active labels train/val/test={ta.sum()}/{va.sum()}/{ea.sum()} (target-derived labels used only for training/diagnostic oracle)')
    rows=[];summary={'configuration':vars(a),'counts':{'train_active':int(ta.sum()),'val_active':int(va.sum()),'test_active':int(ea.sum())},'models':{}}
    def report(name,pred):
        masks={'all':np.ones(len(ex),bool),'regular':regular,'near_boundary':near,'active':ea};rr={k:met(pred,ey,m,trig) for k,m in masks.items()};summary['models'][name]={'groups':rr};print('\n'+name.upper())
        for k,q in rr.items():print(f"{k:14s} n={q['n']:4d}"+(f" LMP_MAE={q['lmp_mae']:.6f} LMP_RMSE={q['lmp_rmse']:.6f} VM_MAE={q['vm_mae']:.7f} trigger_LMP_MAE={q['trigger_lmp_mae']:.6f}" if q['n'] else ''))
        print('stress cases:')
        for i in (575,889,984):
            if i<len(ex):print(f' scenario={i:4d} true/pred={ey[i,trig,0]:.4f}/{pred[i,trig,0]:.4f} abs_err={abs(ey[i,trig,0]-pred[i,trig,0]):.4f}')
        for i in range(len(ex)):rows.append({'model':name,'scenario':i,'regime':str(reg[i]),'active':bool(ea[i]),'trigger_true_lmp':float(ey[i,trig,0]),'trigger_pred_lmp':float(pred[i,trig,0]),'trigger_abs_error':float(abs(ey[i,trig,0]-pred[i,trig,0]))})
    if a.branch in ('baseline','all'):
        m,st,ei,eattr,b=train_gnn(tx,ty,vx,vy,g,a,device,hidden=a.wide_hidden,label='capacity_baseline');report('capacity_baseline',predict(m,ex,st,ei,eattr,device));summary['models']['capacity_baseline']['best_val']=b
    if a.branch in ('moe','all'):
        # Regular expert sees ordinary distribution; active expert uses all data but strongly upweights scarce active samples.
        mr,sr,ei,et,br=train_gnn(tx,ty,vx,vy,g,a,device,hidden=a.hidden,label='regular_expert');w=np.where(ta,a.active_weight,1.).astype(np.float32);ma,sa,eia,eta,ba=train_gnn(tx,ty,vx,vy,g,a,device,hidden=a.hidden,weights=w,label='active_expert');pr=predict(mr,ex,sr,ei,et,device);pa=predict(ma,ex,sa,eia,eta,device)
        gate,gs,bg=train_gate(tx,ta,vx,va,a,device);gp=gate_prob(gate,ex,gs,device);hard=gp>=a.gate_threshold;deploy=np.where(hard[:,None,None],pa,pr);oracle=np.where(ea[:,None,None],pa,pr);report('moe_deployable',deploy);report('moe_oracle_gate',oracle)
        tp=int(np.sum(hard&ea));fp=int(np.sum(hard&~ea));fn=int(np.sum(~hard&ea));tn=int(np.sum(~hard&~ea));summary['gate']={'best_val':bg,'threshold':a.gate_threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'test_predicted_active':int(hard.sum()),'test_true_active':int(ea.sum()),'active_probabilities':{str(i):float(gp[i]) for i in np.where(ea)[0]}};summary['models']['moe_deployable']['expert_best_val']={'regular':br,'active':ba};print(f'\nGATE test: TP={tp} FP={fp} FN={fn} TN={tn} predicted_active={hard.sum()} true_active={ea.sum()}')
        for i in np.where(ea)[0]:print(f' active scenario={i:4d} gate_p={gp[i]:.6f} routed={"active" if hard[i] else "regular"}')
    out=data/f'g13_{a.branch}_mixture_of_experts.csv';js=data/f'g13_{a.branch}_mixture_of_experts_summary.json'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nCSV output: {out}\nSummary: {js}')
if __name__=='__main__':main()
