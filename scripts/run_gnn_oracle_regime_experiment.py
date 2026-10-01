"""G11 oracle active-regime/data-sufficiency experiment.

Purpose: determine whether the G08/G09 active-set LMP failure is mainly a
regime-identification problem or a rare-data problem.  We deliberately give an
oracle binary regime indicator derived from the *target LMP tail* using the
train-only normalization.  This is diagnostic only, not a deployable model.

Two tests are reported:
  1) Oracle-feature GNN: append the oracle regime flag to every node feature.
  2) Tail oversampling sweep: repeat rare train scenarios by fixed factors.

If oracle information plus oversampling still cannot fit held-out tail cases,
there is insufficient support for the active mapping in the present dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphopf.gnn_baseline import SupervisedGraphOPF


def load(path):
    z=np.load(path); return z['x'].astype(np.float32),z['y'].astype(np.float32)


def tail_mask(y,mu,sd,thr):
    z=np.max(np.abs((y[...,0]-mu)/sd),axis=1); return z>=thr,z


def augment_oracle(x,mask):
    flag=np.broadcast_to(mask[:,None,None].astype(np.float32),(len(x),x.shape[1],1))
    return np.concatenate([x,flag],axis=-1)


def met(pred,y,mask,trig):
    if not np.any(mask): return {'n':0}
    e=pred[mask]-y[mask]
    return {'n':int(mask.sum()),'lmp_mae':float(np.mean(np.abs(e[...,0]))),
            'lmp_rmse':float(np.sqrt(np.mean(e[...,0]**2))),
            'vm_mae':float(np.mean(np.abs(e[...,1]))),
            'trigger_lmp_mae':float(np.mean(np.abs(e[:,trig,0])))}


def train_model(tx0,ty0,vx0,vy0,ex0,graph,args,repeat,train_tail,device):
    # Oversampling occurs before normalization statistics so the training
    # objective genuinely sees repeated rare regimes. Input stats remain based
    # on the original, non-repeated training split to keep comparisons clean.
    xmean=tx0.mean((0,1),keepdims=True); xstd=np.maximum(tx0.std((0,1),keepdims=True),1e-6)
    ymean=ty0.mean((0,1),keepdims=True); ystd=np.maximum(ty0.std((0,1),keepdims=True),1e-6)
    idx=np.arange(len(tx0))
    if repeat>1 and np.any(train_tail):
        idx=np.concatenate([idx,np.repeat(np.where(train_tail)[0],repeat-1)])
    tx=torch.tensor((tx0[idx]-xmean)/xstd,dtype=torch.float32,device=device)
    ty=torch.tensor((ty0[idx]-ymean)/ystd,dtype=torch.float32,device=device)
    vx=torch.tensor((vx0-xmean)/xstd,dtype=torch.float32,device=device)
    vy=torch.tensor((vy0-ymean)/ystd,dtype=torch.float32,device=device)
    ex=torch.tensor((ex0-xmean)/xstd,dtype=torch.float32,device=device)
    ef=graph['edge_features'].astype(np.float32); em=ef.mean(0,keepdims=True); es=np.maximum(ef.std(0,keepdims=True),1e-6)
    ei=torch.tensor(graph['edge_index'],dtype=torch.long,device=device)
    ea=torch.tensor((ef-em)/es,dtype=torch.float32,device=device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    # Baseline class currently assumes 5 input channels. Oracle data has 6;
    # replace only the first linear layer when needed while preserving the rest
    # of the repository architecture.
    model=SupervisedGraphOPF(hidden_dim=args.hidden,layers=args.layers).to(device)
    if tx.shape[-1] != 5:
        first=None
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and module.in_features==5:
                first=(name,module); break
        if first is None: raise RuntimeError('Could not locate 5-channel input Linear layer')
        name,old=first
        parent=model
        parts=name.split('.')
        for p in parts[:-1]: parent=getattr(parent,p)
        new=torch.nn.Linear(tx.shape[-1],old.out_features,bias=old.bias is not None).to(device)
        setattr(parent,parts[-1],new)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); state=None; stale=0
    for ep in range(1,args.epochs+1):
        model.train(); opt.zero_grad(); pr=model(tx,ei,ea)
        loss=F.huber_loss(pr,ty,delta=args.huber_delta); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=F.huber_loss(model(vx,ei,ea),vy,delta=args.huber_delta).item()
        if vl<best-1e-8:
            best=vl; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if ep==1 or ep%25==0: print(f'repeat={repeat:3d} epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
        if stale>=args.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pz=model(ex,ei,ea).cpu().numpy()
    return pz*ystd+ymean,best,len(idx)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='results/G03_gnn_batch'); p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--hidden',type=int,default=64); p.add_argument('--layers',type=int,default=3)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--patience',type=int,default=40); p.add_argument('--seed',type=int,default=20271001)
    p.add_argument('--z-threshold',type=float,default=10.0); p.add_argument('--huber-delta',type=float,default=1.0)
    p.add_argument('--trigger-bus',type=int,default=31); p.add_argument('--repeats',default='1,10,50,100')
    a=p.parse_args(); data=Path(a.data)
    tx,ty=load(data/'train.npz'); vx,vy=load(data/'val.npz'); ex,ey=load(data/'test.npz'); graph=np.load(data/'graph.npz')
    mu=float(ty[...,0].mean()); sd=max(float(ty[...,0].std()),1e-12)
    tm,tz=tail_mask(ty,mu,sd,a.z_threshold); vm,vz=tail_mask(vy,mu,sd,a.z_threshold); em,ez=tail_mask(ey,mu,sd,a.z_threshold)
    # Oracle feature is target-derived on every split: intentionally privileged information.
    txo=augment_oracle(tx,tm); vxo=augment_oracle(vx,vm); exo=augment_oracle(ex,em)
    trig=a.trigger_bus-1; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reps=[int(x) for x in a.repeats.split(',') if x.strip()]
    print('G11 ORACLE ACTIVE-REGIME / DATA-SUFFICIENCY EXPERIMENT')
    print(f'device={device} split={len(tx)}/{len(vx)}/{len(ex)} tail counts={tm.sum()}/{vm.sum()}/{em.sum()}')
    print('IMPORTANT: oracle flag is target-derived and diagnostic; results are not deployable predictor metrics.')
    rows=[]; summary={'configuration':vars(a),'tail_counts':{'train':int(tm.sum()),'val':int(vm.sum()),'test':int(em.sum())},'runs':{}}
    for r in reps:
        print(f'\nORACLE + TAIL REPEAT {r}x')
        pred,best,nfit=train_model(txo,ty,vxo,vy,exo,graph,a,r,tm,device)
        groups={'all':np.ones(len(ex),bool),'tail':em,'non_tail':~em}
        res={k:met(pred,ey,m,trig) for k,m in groups.items()}
        summary['runs'][str(r)]={'best_val_huber':best,'effective_train_n':nfit,'groups':res}
        for k,q in res.items():
            print(f"{k:10s} n={q['n']:4d}" + (f" LMP_MAE={q['lmp_mae']:.6f} LMP_RMSE={q['lmp_rmse']:.6f} trigger_LMP_MAE={q['trigger_lmp_mae']:.6f}" if q['n'] else ''))
        print('stress cases:')
        for i in (575,889,984):
            if i<len(ex): print(f' scenario={i:4d} z={ez[i]:7.3f} true/pred={ey[i,trig,0]:.4f}/{pred[i,trig,0]:.4f} abs_err={abs(ey[i,trig,0]-pred[i,trig,0]):.4f}')
        for i in range(len(ex)):
            rows.append({'repeat':r,'scenario':i,'tail':bool(em[i]),'max_abs_z_lmp':float(ez[i]),'trigger_true_lmp':float(ey[i,trig,0]),'trigger_pred_lmp':float(pred[i,trig,0]),'trigger_abs_error':float(abs(ey[i,trig,0]-pred[i,trig,0]))})
    out=data/'g11_oracle_regime_experiment.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    js=data/'g11_oracle_regime_summary.json'; js.write_text(json.dumps(summary,indent=2)+'\n')
    print(f'\nCSV output: {out}\nSummary: {js}')

if __name__=='__main__': main()
