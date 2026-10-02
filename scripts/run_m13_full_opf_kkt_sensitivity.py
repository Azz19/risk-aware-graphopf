"""M13: full OPF-KKT solution-map sensitivity / active-set transition audit.

M12 established an important separation: the Newton AC power-flow Jacobian gives
excellent one-step voltage predictions, but a single PF Jacobian does not locate
the OPF active-set transition reliably. M13 therefore differentiates the *solved
AC-OPF solution map* rather than the PF equations alone.

For every frozen G14 continuation point in lambda in [0,1], the script solves the
base OPF and two infinitesimally perturbed OPFs along the known load direction.
Central differences (with adaptive one-sided fallbacks) estimate derivatives of
primal voltage margin, bus-31 inequality multiplier, and LMP. Because each
perturbed point is itself a converged AC-OPF satisfying stationarity, feasibility,
complementarity and dual feasibility to solver tolerance, these derivatives are
numerical implicit sensitivities of the full OPF KKT solution map. This is more
robust across PYPOWER versions than reconstructing solver-internal Hessian/KKT
blocks, while testing the same local solution-map mechanism.

Important research semantics:
* No neural model is fit.
* No future continuation point, future LMP, or future dual is used to construct
  a local prediction.
* The dual at the current/epsilon-perturbed OPF is part of the KKT state and is
  allowed as a local physical feature; future frozen duals are evaluation-only.
* Predictions are reported only while the current point is pre-activation.
* Active-set stability of the epsilon stencil, solver failures, conditioning
  proxies, and epsilon sensitivity are explicitly audited.
* Path-block bootstrap is used because continuation points within a path are not
  independent experimental units.

This is a mechanism/falsification experiment, not evidence of causality.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from pypower.idx_bus import BUS_I, PD, QD, VM, VMIN, LAM_P, MU_VMIN, MU_VMAX
from pypower.idx_gen import MU_PMIN, MU_PMAX, MU_QMIN, MU_QMAX
from pypower.idx_brch import MU_SF, MU_ST
from graphopf.powerflow import load_case, solve_ac_opf


def ff(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def load_npz(path):
    z=np.load(path); return {k:z[k] for k in z.files}


def load_config_case(config):
    cfg=yaml.safe_load(Path(config).read_text()); p=Path(cfg['case']['path'])
    if not p.is_absolute(): p=Path.cwd()/p
    if not p.exists(): raise FileNotFoundError(f'MATPOWER/PGLib case not found: {p}')
    return p


def continuation_rows(data,test_x):
    rows=[]; src=data/'g14_active_set_continuation.csv'
    with src.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if str(r.get('solved','')).lower() not in {'true','1','yes'}: continue
            la=ff(r.get('lambda'))
            if not (0 <= la <= 1): continue
            s=int(r['active_scenario']); c=int(r['control_scenario'])
            d=test_x[s].astype(float)-test_x[c].astype(float)
            x=test_x[c].astype(float)+la*d
            rows.append(dict(path=s,control=c,lambda_=la,x=x,direction=d,true_lmp=ff(r.get('true_lmp'))))
    return rows


def clone_case(base):
    return {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}


def set_loads(base,x,bus_ids):
    out=clone_case(base); pos={int(b):i for i,b in enumerate(bus_ids)}
    for i in range(len(out['bus'])):
        b=int(round(out['bus'][i,BUS_I]))
        if b in pos:
            k=pos[b]; out['bus'][i,PD]=float(x[k,0]); out['bus'][i,QD]=float(x[k,1])
    return out


def col(a,j):
    return a[:,j] if a.shape[1] > j else np.zeros(len(a))


def solve_state(base,x,bus_ids,trigger_bus,dual_tol):
    try: s=solve_ac_opf(set_loads(base,x,bus_ids))
    except Exception as e: return None,repr(e)
    if not isinstance(s,dict): return None,f'non-dict solver result {type(s).__name__}'
    b=s['bus']; hit=np.where(np.rint(b[:,BUS_I]).astype(int)==int(trigger_bus))[0]
    if len(hit)!=1: return None,f'trigger bus {trigger_bus} not unique'
    j=int(hit[0]); vm=float(b[j,VM]); vmin=float(b[j,VMIN]); mu=float(col(b,MU_VMIN)[j]); lmp=float(col(b,LAM_P)[j])
    # Full inequality active-set fingerprint: voltage, generator and branch duals.
    vals=[]
    for arr,idxs in [(b,[MU_VMIN,MU_VMAX]),(s['gen'],[MU_PMIN,MU_PMAX,MU_QMIN,MU_QMAX]),(s['branch'],[MU_SF,MU_ST])]:
        for idx in idxs: vals.extend((col(arr,idx)>dual_tol).astype(np.uint8).tolist())
    fp=np.asarray(vals,np.uint8)
    return dict(vm=vm,vmin=vmin,margin=vm-vmin,mu=mu,lmp=lmp,objective=float(s.get('f',np.nan)),fingerprint=fp),None


def deriv(center,minus,plus,h,field):
    if minus is not None and plus is not None:
        return (plus[field]-minus[field])/(2*h),'central'
    if plus is not None: return (plus[field]-center[field])/h,'forward'
    if minus is not None: return (center[field]-minus[field])/h,'backward'
    return np.nan,'failed'


def fp_distance(a,b):
    if a is None or b is None: return np.nan
    return int(np.count_nonzero(a['fingerprint']!=b['fingerprint']))


def bootstrap_path(values,rng,B):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if not len(v): return [np.nan,np.nan]
    z=np.array([np.mean(rng.choice(v,len(v),replace=True)) for _ in range(B)])
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]


def first_active(rr,tol):
    for r in sorted(rr,key=lambda q:q['lambda']):
        if r['mu']>tol: return float(r['lambda'])
    return np.nan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='results/G03_gnn_batch'); p.add_argument('--config',default='configs/case57.yaml')
    p.add_argument('--trigger-bus',type=int,default=31); p.add_argument('--dual-tol',type=float,default=1e-6)
    p.add_argument('--eps',type=float,default=1e-4); p.add_argument('--eps-large',type=float,default=5e-4)
    p.add_argument('--approach-eps',type=float,default=1e-10); p.add_argument('--bootstrap',type=int,default=5000); p.add_argument('--seed',type=int,default=20271007)
    a=p.parse_args(); data=Path(a.data)
    te=load_npz(data/'m01_constraint_dual'/'test_constraint_dual.npz'); bus_ids=te['bus_ids'].astype(int)
    cont=continuation_rows(data,te['x'].astype(np.float64)); paths=sorted(set(q['path'] for q in cont))
    case_path=load_config_case(a.config); base=load_case(str(case_path))
    if not isinstance(base,dict): raise RuntimeError(f'Case loader returned {type(base).__name__}, expected dict: {case_path}')
    print('M13 FULL OPF-KKT SOLUTION-MAP SENSITIVITY AUDIT')
    print(f'source={data/"g14_active_set_continuation.csv"} paths={paths} observed_lambda=[0,1] trigger=vmin@{a.trigger_bus}')
    print(f'central epsilon={a.eps:g}; robustness epsilon={a.eps_large:g}')
    print('Each stencil point is a separately converged AC-OPF; derivatives therefore include optimizer/dual response, not PF response alone.')
    print('Future continuation labels/duals/LMPs are evaluation-only.')

    out=[]; failures=[]
    for n,q in enumerate(cont,1):
        c,err=solve_state(base,q['x'],bus_ids,a.trigger_bus,a.dual_tol)
        if c is None: failures.append(dict(path=q['path'],lambda_=q['lambda_'],where='center',error=err)); continue
        derivs={}; stencil={}
        for tag,h in [('small',a.eps),('large',a.eps_large)]:
            xm=q['x']-h*q['direction']; xp=q['x']+h*q['direction']
            m,em=solve_state(base,xm,bus_ids,a.trigger_bus,a.dual_tol); z,ep=solve_state(base,xp,bus_ids,a.trigger_bus,a.dual_tol)
            for side,e in [('minus',em),('plus',ep)]:
                if e is not None: failures.append(dict(path=q['path'],lambda_=q['lambda_'],where=f'{tag}_{side}',error=e))
            dm,mode=deriv(c,m,z,h,'margin'); dmu,_=deriv(c,m,z,h,'mu'); dlmp,_=deriv(c,m,z,h,'lmp'); dobj,_=deriv(c,m,z,h,'objective')
            derivs[tag]=(dm,dmu,dlmp,dobj,mode); stencil[tag]=(m,z)
        dm,dmu,dlmp,dobj,mode=derivs['small']; dmL,dmuL,dlmpL,_,_=derivs['large']
        if np.isfinite(dm) and dm < -a.approach_eps: pred_margin=q['lambda_']+max(c['margin']/(-dm),0.0)
        else: pred_margin=np.inf
        # Before activation mu is exactly zero under complementarity, so a local
        # dual derivative need not anticipate first contact. We still record its
        # implied zero-crossing where mathematically defined, but do not promote
        # it as the primary transition estimator.
        if np.isfinite(dmu) and dmu > a.approach_eps: pred_dual=q['lambda_']+max((a.dual_tol-c['mu'])/dmu,0.0)
        else: pred_dual=np.inf
        rel_dm=abs(dm-dmL)/(abs(dmL)+1e-12) if np.isfinite(dm) and np.isfinite(dmL) else np.nan
        rel_dl=abs(dlmp-dlmpL)/(abs(dlmpL)+1e-12) if np.isfinite(dlmp) and np.isfinite(dlmpL) else np.nan
        ms,ps=stencil['small']; ml,pl=stencil['large']
        out.append(dict(path=q['path'],control=q['control'],lambda_=q['lambda_'],margin=c['margin'],mu=c['mu'],lmp=c['lmp'],objective=c['objective'],
            dmargin_dlambda=dm,dmu_dlambda=dmu,dlmp_dlambda=dlmp,dobjective_dlambda=dobj,derivative_mode=mode,
            predicted_hit_margin=pred_margin,predicted_hit_dual=pred_dual,small_vs_large_rel_dmargin=rel_dm,small_vs_large_rel_dlmp=rel_dl,
            active_set_changes_small_minus=fp_distance(c,ms),active_set_changes_small_plus=fp_distance(c,ps),
            active_set_changes_large_minus=fp_distance(c,ml),active_set_changes_large_plus=fp_distance(c,pl),true_lmp_eval_only=q['true_lmp']))
        if n%25==0 or n==len(cont): print(f' solved {n}/{len(cont)} center_failures={sum(x["where"]=="center" for x in failures)} stencil_failures={sum(x["where"]!="center" for x in failures)}',flush=True)
    if not out: raise RuntimeError('M13 produced no solved KKT sensitivity points')

    print('\nPATHWISE PROSPECTIVE TRANSITION AUDIT')
    summary={'configuration':vars(a),'case':str(case_path),'n_requested':len(cont),'n_solved':len(out),'failures':failures,'paths':{}}
    last_errors=[]; all_pre_errors=[]; increment_errors=[]
    for path in paths:
        rr=sorted([r for r in out if r['path']==path],key=lambda z:z['lambda_']); true_hit=first_active([dict(lambda=r['lambda_'],mu=r['mu']) for r in rr],a.dual_tol)
        pre=[r for r in rr if np.isfinite(true_hit) and r['lambda_']<true_hit]
        finite=[r for r in pre if np.isfinite(r['predicted_hit_margin'])]
        last=finite[-1] if finite else None; pred=float(last['predicted_hit_margin']) if last else np.nan
        err=abs(pred-true_hit) if np.isfinite(pred) and np.isfinite(true_hit) else np.nan; last_errors.append(err)
        pe=[abs(r['predicted_hit_margin']-true_hit) for r in finite]; all_pre_errors.extend(pe)
        # Prospective one-step LMP increments use only the left point's local KKT
        # derivative; right point is evaluation-only.
        ie=[]
        for left,right in zip(rr[:-1],rr[1:]):
            dl=right['lambda_']-left['lambda_']; pred_inc=left['dlmp_dlambda']*dl; true_inc=right['lmp']-left['lmp']
            if np.isfinite(pred_inc): ie.append(abs(pred_inc-true_inc)); increment_errors.append(abs(pred_inc-true_inc))
        stable=sum((r['active_set_changes_small_minus']==0 and r['active_set_changes_small_plus']==0) for r in pre)
        summary['paths'][str(path)]={'n':len(rr),'true_activation_lambda':true_hit,'last_preactive_predicted_lambda':pred,'last_preactive_abs_error':err,
            'median_all_preactive_abs_error':float(np.median(pe)) if pe else None,'preactive_n':len(pre),'epsilon_stable_fraction':stable/max(len(pre),1),
            'median_small_large_rel_dmargin':float(np.nanmedian([r['small_vs_large_rel_dmargin'] for r in pre])) if pre else None,
            'median_small_large_rel_dlmp':float(np.nanmedian([r['small_vs_large_rel_dlmp'] for r in pre])) if pre else None,
            'one_step_lmp_increment_mae':float(np.mean(ie)) if ie else None}
        print(f'path={path} true_hit={true_hit:.6f} last_pre_pred={pred:.6f} abs_err={err:.6f} stable_stencil={stable}/{len(pre)} LMP_increment_MAE={np.mean(ie) if ie else np.nan:.6f}')

    rng=np.random.default_rng(a.seed)
    summary['aggregate']={'mean_last_preactive_activation_abs_error':float(np.nanmean(last_errors)),'median_all_preactive_activation_abs_error':float(np.median(all_pre_errors)) if all_pre_errors else None,
        'mean_one_step_lmp_increment_abs_error':float(np.mean(increment_errors)) if increment_errors else None,
        'path_block_bootstrap_last_preactive_error_ci95':bootstrap_path(last_errors,rng,a.bootstrap),'bootstrap_unit':'continuation path',
        'interpretation':'Numerical implicit sensitivity of the converged AC-OPF KKT solution map; not an explicit assembled KKT-matrix inverse.'}
    csvout=data/'m13_full_opf_kkt_sensitivity.csv'; jsout=data/'m13_full_opf_kkt_sensitivity_summary.json'
    with csvout.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    jsout.write_text(json.dumps(summary,indent=2)+'\n')
    ag=summary['aggregate']; print('\nAGGREGATE')
    print(f"mean last-preactivation lambda error={ag['mean_last_preactive_activation_abs_error']:.6f}")
    print(f"median all-preactivation lambda error={ag['median_all_preactive_activation_abs_error']}")
    print(f"mean one-step LMP-increment abs error={ag['mean_one_step_lmp_increment_abs_error']}")
    print(f"path-block bootstrap 95% CI={ag['path_block_bootstrap_last_preactive_error_ci95']}")
    print(f'CSV output: {csvout}'); print(f'Summary: {jsout}')
    print('Decision: if full-OPF solution-map sensitivities materially reduce M12 boundary error and predict LMP increments across all frozen paths with epsilon-stable stencils, the paper can attribute the missing transition information to optimization/KKT geometry. If boundary anticipation remains path-specific despite accurate local increments, the defensible conclusion is that local differential information alone cannot prospectively identify the discrete active-set switch; a continuation-aware or explicit regime model is required.')

if __name__=='__main__': main()
