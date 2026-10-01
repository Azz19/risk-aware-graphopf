"""G16: solver-wide active-constraint continuation audit.

Re-solve the three G14 continuation paths and inventory all economically relevant
AC-OPF constraints.  The goal is to test whether the bus-31 lower-voltage dual is
the only material dual that grows with the LMP escalation, or whether generator,
other-voltage, or branch-flow constraints enter along the same path.

This is a physical mechanism diagnostic; it does not train a GNN.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import yaml
from pypower.idx_bus import BUS_I,PD,QD,VM,VMIN,VMAX,LAM_P,MU_VMIN,MU_VMAX
from pypower.idx_gen import GEN_BUS,PG,QG,PMIN,PMAX,QMIN,QMAX,MU_PMIN,MU_PMAX,MU_QMIN,MU_QMAX
from pypower.idx_brch import F_BUS,T_BUS,RATE_A,PF,QF,PT,QT,MU_SF,MU_ST
from graphopf.powerflow import load_case,solve_ac_opf


def load_split(path):
    z=np.load(path);return z['x'].astype(np.float32),z['y'].astype(np.float32)

def reconstruct(base,x):
    c={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()};c['bus'][:,PD]=x[:,0];c['bus'][:,QD]=x[:,1];return c

def nearest_control(x,idx,candidates):
    totals=np.column_stack((x[:,:,0].sum(1),x[:,:,1].sum(1)));scale=np.maximum(totals.std(0),1e-9);d=np.linalg.norm((totals[candidates]-totals[idx])/scale,axis=1);return int(candidates[int(np.argmin(d))])
def col(a,j):return a[:,j] if a.shape[1]>j else np.zeros(len(a))
def top(items,n=8):return sorted(items,key=lambda z:abs(z['dual']),reverse=True)[:n]

def audit(sol,trigger_bus,dual_tol,margin_tol):
    bus,gen,br=sol['bus'],sol['gen'],sol['branch'];ids=bus[:,BUS_I].astype(int);j=int(np.where(ids==trigger_bus)[0][0])
    constraints=[]
    for k,r in enumerate(bus):
        constraints += [dict(kind='vmin',element=int(r[BUS_I]),dual=float(col(bus,MU_VMIN)[k]),margin=float(r[VM]-r[VMIN])),dict(kind='vmax',element=int(r[BUS_I]),dual=float(col(bus,MU_VMAX)[k]),margin=float(r[VMAX]-r[VM]))]
    for k,r in enumerate(gen):
        gb=int(r[GEN_BUS]);constraints += [dict(kind='pmin',element=gb,dual=float(col(gen,MU_PMIN)[k]),margin=float(r[PG]-r[PMIN])),dict(kind='pmax',element=gb,dual=float(col(gen,MU_PMAX)[k]),margin=float(r[PMAX]-r[PG])),dict(kind='qmin',element=gb,dual=float(col(gen,MU_QMIN)[k]),margin=float(r[QG]-r[QMIN])),dict(kind='qmax',element=gb,dual=float(col(gen,MU_QMAX)[k]),margin=float(r[QMAX]-r[QG]))]
    for k,r in enumerate(br):
        rate=float(r[RATE_A]);
        if rate>0:
            sf=float(np.hypot(r[PF],r[QF]));st=float(np.hypot(r[PT],r[QT]));constraints += [dict(kind='branch_from',element=f'{int(r[F_BUS])}-{int(r[T_BUS])}',dual=float(col(br,MU_SF)[k]),margin=rate-sf),dict(kind='branch_to',element=f'{int(r[F_BUS])}-{int(r[T_BUS])}',dual=float(col(br,MU_ST)[k]),margin=rate-st)]
    active=[q for q in constraints if abs(q['dual'])>dual_tol or q['margin']<=margin_tol]
    material=[q for q in constraints if abs(q['dual'])>dual_tol]
    return {'trigger_vm':float(bus[j,VM]),'trigger_vmin':float(bus[j,VMIN]),'trigger_margin':float(bus[j,VM]-bus[j,VMIN]),'trigger_vmin_dual':float(col(bus,MU_VMIN)[j]),'trigger_lmp':float(bus[j,LAM_P]),'objective':float(sol.get('f',np.nan)),'active':active,'material':material,'top_duals':top(material)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--config',default='configs/case57.yaml');p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--scenarios',default='575,889,984');p.add_argument('--points',type=int,default=101);p.add_argument('--extend',type=float,default=.15);p.add_argument('--dual-tol',type=float,default=1e-6);p.add_argument('--margin-tol',type=float,default=1e-5);a=p.parse_args()
    data=Path(a.data);tx,_=load_split(data/'test.npz');cfg=yaml.safe_load(Path(a.config).read_text());base=load_case(cfg['case']['path'])
    # Reuse G08 regime labels so controls exactly match the G14 construction.
    regimes=np.array(['unknown']*len(tx),dtype=object);g08=data/'g08_regime_evaluation.csv'
    if not g08.exists():raise FileNotFoundError('G08 regime CSV required to reproduce G14 controls: '+str(g08))
    with g08.open(newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['scenario']);regimes[i]=r['regime']
    inactive=np.where(regimes!='active')[0];scenarios=[int(x) for x in a.scenarios.split(',') if x.strip()];lambdas=np.linspace(-a.extend,1+a.extend,a.points)
    print('G16 SOLVER-WIDE ACTIVE-CONSTRAINT AUDIT');print(f'paths={scenarios} points={a.points} extend={a.extend:g} trigger_bus={a.trigger_bus}');print('Tracking voltage, generator P/Q, and branch apparent-power constraint duals.')
    rows=[];summary={'configuration':vars(a),'paths':{}}
    for s in scenarios:
        c=nearest_control(tx,s,inactive);x0=tx[c].astype(float);d=tx[s].astype(float)-x0;path=[];prev_sig=None;events=[]
        print(f'\nPATH {s} control={c} regime(control)={regimes[c]}',flush=True)
        for k,lam in enumerate(lambdas):
            x=(x0+lam*d).astype(np.float32)
            try:
                q=audit(solve_ac_opf(reconstruct(base,x)),a.trigger_bus,a.dual_tol,a.margin_tol);ok=True
            except Exception as e:
                q=None;ok=False
            if ok:
                sig=tuple(sorted((z['kind'],str(z['element'])) for z in q['material']))
                if sig!=prev_sig:
                    added=sorted(set(sig)-set(prev_sig or ()));removed=sorted(set(prev_sig or ())-set(sig));events.append({'lambda':float(lam),'added':added,'removed':removed,'trigger_lmp':q['trigger_lmp'],'trigger_dual':q['trigger_vmin_dual']});print(f' active-set change lambda={lam:.6f} LMP={q["trigger_lmp"]:.6f} bus31dual={q["trigger_vmin_dual"]:.6g} +{added} -{removed}',flush=True);prev_sig=sig
                others=[z for z in q['material'] if not (z['kind']=='vmin' and str(z['element'])==str(a.trigger_bus))];largest=max((abs(z['dual']) for z in others),default=0.0);top_other=max(others,key=lambda z:abs(z['dual'])) if others else None
                rows.append({'active_scenario':s,'control_scenario':c,'lambda':float(lam),'solved':True,'trigger_lmp':q['trigger_lmp'],'trigger_vm_pu':q['trigger_vm'],'trigger_margin_pu':q['trigger_margin'],'trigger_vmin_dual':q['trigger_vmin_dual'],'objective':q['objective'],'n_material_constraints':len(q['material']),'largest_other_dual':largest,'largest_other_kind':'' if top_other is None else top_other['kind'],'largest_other_element':'' if top_other is None else top_other['element'],'material_constraints':json.dumps(q['material'],separators=(',',':'))})
                path.append((float(lam),q,largest,top_other))
            else:rows.append({'active_scenario':s,'control_scenario':c,'lambda':float(lam),'solved':False})
            if (k+1)%25==0 or k+1==len(lambdas):print(f' solved {k+1}/{len(lambdas)}',flush=True)
        obs=[z for z in path if 0<=z[0]<=1];peak=max(obs,key=lambda z:z[1]['trigger_lmp']) if obs else None
        ps={'control_scenario':c,'events':events,'n_solved':len(path)}
        if peak:
            lam,q,largest,to=peak;ps['observed_peak']={'lambda':lam,'trigger_lmp':q['trigger_lmp'],'trigger_vmin_dual':q['trigger_vmin_dual'],'largest_other_dual':largest,'largest_other_constraint':to,'top_material_duals':q['top_duals']};print(f' observed peak lambda={lam:.6f} LMP={q["trigger_lmp"]:.6f} bus31dual={q["trigger_vmin_dual"]:.6g} largest_other_dual={largest:.6g} other={to}')
        summary['paths'][str(s)]=ps
    out=data/'g16_solverwide_constraint_audit.csv';js=data/'g16_solverwide_constraint_summary.json'
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nCSV output: {out}\nSummary: {js}');print('Interpretation: compare active-set change events with G15 acceleration lambdas. If no new material constraint enters near acceleration and bus-31 MU_VMIN dominates, the voltage shadow-price mechanism is strongly isolated; otherwise report the coupled active-set sequence.')
if __name__=='__main__':main()
