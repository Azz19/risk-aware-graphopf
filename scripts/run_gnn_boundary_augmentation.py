"""G12: targeted active-set boundary augmentation experiment.

Final intervention for the G03--G11 diagnostic branch.  Generate genuinely new
AC-OPF samples around the bus-31 lower-voltage active-set transition by making
zero-sum spatial redistributions of the original training loads, then retrain
with increasing numbers of accepted boundary samples.  The original validation
and test sets remain untouched.

This distinguishes new physical support from G11's duplication of three rare
points.  Diagnostic experiment; generated samples are training augmentation only.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pypower.idx_bus import BUS_I, PD, QD, VM, VMIN, MU_VMIN, LAM_P
from graphopf.gnn_baseline import SupervisedGraphOPF
from graphopf.powerflow import load_case, solve_ac_opf


def load_split(p):
    z=np.load(p); return z['x'].astype(np.float32),z['y'].astype(np.float32)

def reconstruct(base,x):
    c={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}
    c['bus'][:,PD]=x[:,0]; c['bus'][:,QD]=x[:,1]; return c

def target_from_solution(sol,template):
    y=template.copy(); bus=sol['bus']; y[:,0]=bus[:,LAM_P]; y[:,1]=bus[:,VM]; return y.astype(np.float32)

def metrics(pred,y,mask,trig):
    if not np.any(mask): return {'n':0}
    e=pred[mask]-y[mask]
    return {'n':int(mask.sum()),'lmp_mae':float(np.mean(np.abs(e[...,0]))),'lmp_rmse':float(np.sqrt(np.mean(e[...,0]**2))),'vm_mae':float(np.mean(np.abs(e[...,1]))),'trigger_lmp_mae':float(np.mean(np.abs(e[:,trig,0])))}

def tail_mask(y,mu,sd,thr):
    z=np.max(np.abs((y[...,0]-mu)/sd),axis=1); return z>=thr,z

def spatial_perturb(x,rng,scale):
    """Redistribute load spatially while preserving total P/Q to roundoff."""
    z=x.copy(); load=np.where(x[:,0]>1e-8)[0]
    if len(load)<2:return z
    d=rng.normal(size=len(load)); d-=d.mean()
    # Scale relative to mean positive load, then enforce nonnegative demand.
    amp=scale*max(float(x[load,0].mean()),1.0); dp=d*amp
    # Reactive perturbation follows local Q/P ratio where defined.
    ratio=np.divide(x[load,1],x[load,0],out=np.zeros(len(load)),where=np.abs(x[load,0])>1e-8)
    dq=dp*ratio
    # Remove residual sums independently.
    dp-=dp.mean(); dq-=dq.mean()
    alpha=1.0
    neg=dp<0
    if np.any(neg): alpha=min(alpha,float(np.min(x[load[neg],0]/(-dp[neg]+1e-12)))*0.95)
    negq=dq<0
    if np.any(negq): alpha=min(alpha,float(np.min(np.maximum(x[load[negq],1],0)/(-dq[negq]+1e-12)))*0.95)
    alpha=max(0.0,min(1.0,alpha)); z[load,0]+=alpha*dp; z[load,1]+=alpha*dq
    return z

def train(tx0,ty0,vx,vy,ex,graph,args,device):
    xm=tx0.mean((0,1),keepdims=True); xs=np.maximum(tx0.std((0,1),keepdims=True),1e-6)
    ym=ty0.mean((0,1),keepdims=True); ys=np.maximum(ty0.std((0,1),keepdims=True),1e-6)
    TX=torch.tensor((tx0-xm)/xs,dtype=torch.float32,device=device); TY=torch.tensor((ty0-ym)/ys,dtype=torch.float32,device=device)
    VX=torch.tensor((vx-xm)/xs,dtype=torch.float32,device=device); VY=torch.tensor((vy-ym)/ys,dtype=torch.float32,device=device)
    EX=torch.tensor((ex-xm)/xs,dtype=torch.float32,device=device)
    ef=graph['edge_features'].astype(np.float32); em=ef.mean(0,keepdims=True); es=np.maximum(ef.std(0,keepdims=True),1e-6)
    ei=torch.tensor(graph['edge_index'],dtype=torch.long,device=device); ea=torch.tensor((ef-em)/es,dtype=torch.float32,device=device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    m=SupervisedGraphOPF(hidden_dim=args.hidden,layers=args.layers).to(device); opt=torch.optim.Adam(m.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); state=None; stale=0
    for ep in range(1,args.epochs+1):
        m.train(); opt.zero_grad(); loss=F.huber_loss(m(TX,ei,ea),TY,delta=args.huber_delta); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():vl=F.huber_loss(m(VX,ei,ea),VY,delta=args.huber_delta).item()
        if vl<best-1e-8:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if ep==1 or ep%25==0:print(f'epoch={ep:4d} train={loss.item():.6f} val={vl:.6f}',flush=True)
        if stale>=args.patience:break
    m.load_state_dict(state);m.eval()
    with torch.no_grad():pz=m(EX,ei,ea).cpu().numpy()
    return pz*ys+ym,best

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--config',default='configs/case57.yaml')
    p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--pool',type=int,default=300);p.add_argument('--max-attempts',type=int,default=12000)
    p.add_argument('--near-margin',type=float,default=.01);p.add_argument('--active-tol',type=float,default=1e-5);p.add_argument('--dual-tol',type=float,default=1e-6)
    p.add_argument('--perturb-scale',type=float,default=.12);p.add_argument('--sizes',default='0,25,50,100,250');p.add_argument('--z-threshold',type=float,default=10.)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=3);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--patience',type=int,default=40);p.add_argument('--huber-delta',type=float,default=1.);p.add_argument('--seed',type=int,default=20271001)
    a=p.parse_args();data=Path(a.data);tx,ty=load_split(data/'train.npz');vx,vy=load_split(data/'val.npz');ex,ey=load_split(data/'test.npz');graph=np.load(data/'graph.npz')
    cfg=yaml.safe_load(Path(a.config).read_text());base=load_case(cfg['case']['path']);ids=base['bus'][:,BUS_I].astype(int);loc=np.where(ids==a.trigger_bus)[0]
    if not len(loc):raise ValueError(f'trigger bus {a.trigger_bus} not found');
    trig=int(loc[0]);rng=np.random.default_rng(a.seed);mu=float(ty[...,0].mean());sd=max(float(ty[...,0].std()),1e-12);test_tail,test_z=tail_mask(ey,mu,sd,a.z_threshold)
    print('G12 TARGETED ACTIVE-SET BOUNDARY AUGMENTATION');print(f'original split={len(tx)}/{len(vx)}/{len(ex)} trigger_bus={a.trigger_bus} target_pool={a.pool}')
    print('Generating new zero-sum spatial load redistributions; validation/test remain untouched.',flush=True)
    # Prefer seeds already near the transition according to their stored VM target.
    margins=np.abs(ty[:,trig,1]-float(base['bus'][trig,VMIN]));seed_idx=np.argsort(margins)[:min(300,len(tx))]
    ax=[];ay=[];meta=[];active_n=near_n=0;attempt=0
    while len(ax)<a.pool and attempt<a.max_attempts:
        attempt+=1;s=int(rng.choice(seed_idx));scale=a.perturb_scale*float(rng.uniform(.25,1.25));xx=spatial_perturb(tx[s],rng,scale)
        try:sol=solve_ac_opf(reconstruct(base,xx))
        except Exception:continue
        bus=sol['bus'];j=int(np.where(bus[:,BUS_I].astype(int)==a.trigger_bus)[0][0]);vm=float(bus[j,VM]);vmin=float(bus[j,VMIN]);dual=float(bus[j,MU_VMIN]);margin=vm-vmin
        active=bool(margin<=a.active_tol or dual>a.dual_tol);near=bool((not active) and margin<=a.near_margin)
        # Balance active and near-boundary support when possible.
        want=(active and active_n<a.pool//2) or (near and near_n<a.pool-a.pool//2)
        if not want:continue
        yy=target_from_solution(sol,ty[s]);ax.append(xx);ay.append(yy);active_n+=int(active);near_n+=int(near)
        meta.append({'aug_index':len(ax)-1,'seed_train_scenario':s,'active':active,'margin_pu':margin,'lower_v_dual':dual,'trigger_lmp':float(yy[trig,0]),'total_pd_mw':float(xx[:,0].sum()),'total_qd_mvar':float(xx[:,1].sum())})
        if len(ax)%25==0:print(f' accepted {len(ax)}/{a.pool} after {attempt} attempts (active={active_n}, near={near_n})',flush=True)
    if not ax:raise RuntimeError('No boundary augmentation samples accepted; increase --max-attempts or --perturb-scale.')
    ax=np.stack(ax);ay=np.stack(ay);print(f'Generated {len(ax)} unique solved samples: active={active_n} near_boundary={near_n} attempts={attempt}')
    sizes=[int(x) for x in a.sizes.split(',') if x.strip()];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];summary={'configuration':vars(a),'generated':{'n':len(ax),'active':active_n,'near_boundary':near_n,'attempts':attempt},'runs':{}}
    for n in sizes:
        n=min(n,len(ax));print(f'\nAUGMENTATION n={n}')
        TX=np.concatenate([tx,ax[:n]],axis=0) if n else tx;TY=np.concatenate([ty,ay[:n]],axis=0) if n else ty
        pred,best=train(TX,TY,vx,vy,ex,graph,a,device);groups={'all':np.ones(len(ex),bool),'tail':test_tail,'non_tail':~test_tail};res={k:metrics(pred,ey,m,trig) for k,m in groups.items()};summary['runs'][str(n)]={'best_val_huber':best,'groups':res}
        for k,q in res.items():print(f"{k:10s} n={q['n']:4d}"+(f" LMP_MAE={q['lmp_mae']:.6f} LMP_RMSE={q['lmp_rmse']:.6f} VM_MAE={q['vm_mae']:.7f} trigger_LMP_MAE={q['trigger_lmp_mae']:.6f}" if q['n'] else ''))
        print('stress cases:')
        for i in (575,889,984):
            if i<len(ex):print(f' scenario={i:4d} z={test_z[i]:7.3f} true/pred={ey[i,trig,0]:.4f}/{pred[i,trig,0]:.4f} abs_err={abs(ey[i,trig,0]-pred[i,trig,0]):.4f}')
        for i in range(len(ex)):rows.append({'augmentation_n':n,'scenario':i,'tail':bool(test_tail[i]),'max_abs_z_lmp':float(test_z[i]),'trigger_true_lmp':float(ey[i,trig,0]),'trigger_pred_lmp':float(pred[i,trig,0]),'trigger_abs_error':float(abs(ey[i,trig,0]-pred[i,trig,0]))})
    with (data/'g12_boundary_pool.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
    with (data/'g12_boundary_augmentation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (data/'g12_boundary_augmentation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"\nPool: {data/'g12_boundary_pool.csv'}\nCSV output: {data/'g12_boundary_augmentation.csv'}\nSummary: {data/'g12_boundary_augmentation_summary.json'}")
if __name__=='__main__':main()
