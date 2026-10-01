"""G14: controlled active-set continuation audit.

Construct one-dimensional load-space paths from held-out active scenarios toward
nearby inactive controls, re-solve AC-OPF densely along each path, and compare the
physical bus-31 voltage/dual/LMP transition with the fixed G02 GNN prediction.

This is diagnostic, not training. It tests whether the large GNN LMP errors
observed in G07--G13 coincide with a reproducible OPF active-set transition.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import yaml
from pypower.idx_bus import BUS_I, PD, QD, VM, VMIN, MU_VMIN, LAM_P
from graphopf.gnn_baseline import SupervisedGraphOPF
from graphopf.powerflow import load_case, solve_ac_opf


def load_split(path):
    z=np.load(path); return z['x'].astype(np.float32),z['y'].astype(np.float32)

def reconstruct(base,x):
    c={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}
    c['bus'][:,PD]=x[:,0]; c['bus'][:,QD]=x[:,1]; return c

def load_model(model_path,graph,device):
    ck=torch.load(model_path,map_location='cpu',weights_only=False)
    m=SupervisedGraphOPF(hidden_dim=int(ck['hidden']),layers=int(ck['layers'])).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    xm=np.asarray(ck['x_mean'],dtype=np.float32); xs=np.asarray(ck['x_std'],dtype=np.float32)
    ym=np.asarray(ck['y_mean'],dtype=np.float32); ys=np.asarray(ck['y_std'],dtype=np.float32)
    em=np.asarray(ck['edge_mean'],dtype=np.float32); es=np.asarray(ck['edge_std'],dtype=np.float32)
    ei=torch.tensor(graph['edge_index'],dtype=torch.long,device=device)
    ea=torch.tensor((graph['edge_features'].astype(np.float32)-em)/es,dtype=torch.float32,device=device)
    return m,(xm,xs,ym,ys),ei,ea

def predict(m,x,stats,ei,ea,device):
    xm,xs,ym,ys=stats
    X=torch.tensor((x-xm)/xs,dtype=torch.float32,device=device)
    with torch.no_grad(): z=m(X,ei,ea).cpu().numpy()
    return z*ys+ym

def nearest_control(x,idx,candidates):
    totals=np.column_stack((x[:,:,0].sum(1),x[:,:,1].sum(1)))
    scale=np.maximum(totals.std(0),1e-9)
    d=np.linalg.norm((totals[candidates]-totals[idx])/scale,axis=1)
    return int(candidates[int(np.argmin(d))])

def solve_point(base,x,bus_id):
    sol=solve_ac_opf(reconstruct(base,x)); bus=sol['bus']; ids=bus[:,BUS_I].astype(int)
    j=int(np.where(ids==bus_id)[0][0]); vm=float(bus[j,VM]); vmin=float(bus[j,VMIN]); dual=float(bus[j,MU_VMIN]); lmp=float(bus[j,LAM_P])
    return {'vm':vm,'vmin':vmin,'margin':vm-vmin,'dual':dual,'lmp':lmp,'objective':float(sol.get('f',np.nan))}

def crossing_summary(rows,active_tol,dual_tol):
    rr=sorted(rows,key=lambda q:q['lambda'])
    active=np.array([(r['true_margin_pu']<=active_tol or r['true_lower_v_dual']>dual_tol) for r in rr])
    first=int(np.where(active)[0][0]) if np.any(active) else None
    if first is None: return {'crossing_found':False}
    r=rr[first]; prev=rr[first-1] if first>0 else None
    return {'crossing_found':True,'first_active_lambda':r['lambda'],'first_active_true_lmp':r['true_lmp'],'first_active_true_vm_pu':r['true_vm_pu'],'first_active_dual':r['true_lower_v_dual'],'previous_lambda':None if prev is None else prev['lambda'],'previous_true_lmp':None if prev is None else prev['true_lmp'],'lmp_jump_from_previous':None if prev is None else r['true_lmp']-prev['true_lmp']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='results/G03_gnn_batch'); p.add_argument('--config',default='configs/case57.yaml'); p.add_argument('--model',default=None)
    p.add_argument('--trigger-bus',type=int,default=31); p.add_argument('--scenarios',default='575,889,984'); p.add_argument('--points',type=int,default=101); p.add_argument('--extend',type=float,default=.15)
    p.add_argument('--active-tol',type=float,default=1e-5); p.add_argument('--dual-tol',type=float,default=1e-6); p.add_argument('--near-margin',type=float,default=.01); p.add_argument('--batch-size',type=int,default=256); a=p.parse_args()
    data=Path(a.data); model_path=Path(a.model) if a.model else data/'g02_model.pt'; tx,ty=load_split(data/'test.npz'); graph=np.load(data/'graph.npz')
    cfg=yaml.safe_load(Path(a.config).read_text()); base=load_case(cfg['case']['path']); ids=base['bus'][:,BUS_I].astype(int); loc=np.where(ids==a.trigger_bus)[0]
    if not len(loc):
        raise ValueError(f'trigger bus {a.trigger_bus} not found')
    trig=int(loc[0])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m,stats,ei,ea=load_model(model_path,graph,device)
    regimes=np.array(['unknown']*len(tx),dtype=object); g08=data/'g08_regime_evaluation.csv'
    if g08.exists():
        with g08.open(newline='') as f:
            for r in csv.DictReader(f):
                i=int(r['scenario'])
                if 0<=i<len(tx): regimes[i]=r['regime']
    if np.all(regimes=='unknown'):
        print('G08 regime table unavailable; classifying test scenarios first...',flush=True)
        for i in range(len(tx)):
            q=solve_point(base,tx[i],a.trigger_bus); regimes[i]='active' if (q['margin']<=a.active_tol or q['dual']>a.dual_tol) else ('near_boundary' if q['margin']<=a.near_margin else 'regular')
    inactive=np.where(regimes!='active')[0]
    requested=[int(s) for s in a.scenarios.split(',') if s.strip()]; active_cases=[i for i in requested if 0<=i<len(tx)]
    print('G14 CONTROLLED ACTIVE-SET CONTINUATION AUDIT'); print(f'model={model_path} device={device} trigger_bus={a.trigger_bus} points={a.points} extend={a.extend:g}'); print(f'paths={active_cases}')
    rows=[]; summary={'configuration':vars(a),'model':str(model_path),'paths':{}}; lambdas=np.linspace(-a.extend,1.0+a.extend,a.points)
    for s in active_cases:
        c=nearest_control(tx,s,inactive); x0=tx[c].astype(np.float64); x1=tx[s].astype(np.float64); direction=x1-x0
        path=np.stack([(x0+lam*direction).astype(np.float32) for lam in lambdas]); pred=[]
        for k in range(0,len(path),a.batch_size): pred.append(predict(m,path[k:k+a.batch_size],stats,ei,ea,device))
        pred=np.concatenate(pred); print(f'\npath active={s} control={c} regime(control)={regimes[c]}',flush=True); path_rows=[]; solved=0
        for k,(lam,x) in enumerate(zip(lambdas,path)):
            try: q=solve_point(base,x,a.trigger_bus); ok=True; solved+=1
            except Exception: q={'vm':np.nan,'vmin':np.nan,'margin':np.nan,'dual':np.nan,'lmp':np.nan,'objective':np.nan}; ok=False
            r={'active_scenario':s,'control_scenario':c,'lambda':float(lam),'solved':ok,'total_pd_mw':float(x[:,0].sum()),'total_qd_mvar':float(x[:,1].sum()),'true_vm_pu':q['vm'],'vmin_pu':q['vmin'],'true_margin_pu':q['margin'],'true_lower_v_dual':q['dual'],'true_lmp':q['lmp'],'objective':q['objective'],'pred_vm_pu':float(pred[k,trig,1]),'pred_lmp':float(pred[k,trig,0]),'abs_vm_error_pu':float(abs(pred[k,trig,1]-q['vm'])) if ok else np.nan,'abs_lmp_error':float(abs(pred[k,trig,0]-q['lmp'])) if ok else np.nan}
            rows.append(r); path_rows.append(r)
            if (k+1)%25==0 or k+1==len(path): print(f' solved {k+1}/{len(path)}',flush=True)
        valid=[r for r in path_rows if r['solved']]; cs=crossing_summary(valid,a.active_tol,a.dual_tol); cs.update({'active_scenario':s,'control_scenario':c,'n_solved':solved,'n_points':len(path)})
        if valid:
            cs['max_abs_lmp_error']=float(max(r['abs_lmp_error'] for r in valid)); cs['max_true_lmp']=float(max(r['true_lmp'] for r in valid)); cs['max_pred_lmp']=float(max(r['pred_lmp'] for r in valid))
        summary['paths'][str(s)]=cs; print('CROSSING',json.dumps(cs,indent=2))
    if not rows: raise RuntimeError('No continuation rows were generated.')
    out=data/'g14_active_set_continuation.csv'; js=data/'g14_active_set_continuation_summary.json'
    with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    js.write_text(json.dumps(summary,indent=2)+'\n'); print(f'\nCSV output: {out}\nSummary: {js}'); print('Interpretation target: physical VM/dual/LMP should reveal whether a repeatable active-set crossing coincides with the GNN LMP error spike.')
if __name__=='__main__': main()
