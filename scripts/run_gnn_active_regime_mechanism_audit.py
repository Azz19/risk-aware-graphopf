"""G15: active-regime mechanism audit.

Post-process G14 continuation paths to separate the first active-set contact from
the subsequent constrained-regime price escalation.  No OPF re-solves or model
training are performed.  The audit reports observed-path (0<=lambda<=1) and
extended-path behavior separately, detects LMP acceleration, and measures the
association between the bus-31 lower-voltage dual and LMP.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def f(v):
    try:return float(v)
    except (TypeError,ValueError):return np.nan

def b(v):return str(v).strip().lower() in {'1','true','yes','y'}

def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3 or np.std(x[m])==0 or np.std(y[m])==0:return np.nan
    return float(np.corrcoef(x[m],y[m])[0,1])

def summarize(rr,active_tol,dual_tol,accel_factor):
    rr=sorted([r for r in rr if r['solved']],key=lambda r:r['lambda'])
    lam=np.array([r['lambda'] for r in rr]);lmp=np.array([r['true_lmp'] for r in rr]);pred=np.array([r['pred_lmp'] for r in rr]);dual=np.array([r['true_lower_v_dual'] for r in rr]);margin=np.array([r['true_margin_pu'] for r in rr])
    active=(margin<=active_tol)|(dual>dual_tol)
    ia=int(np.where(active)[0][0]) if active.any() else None
    # Local finite-difference slope. Define acceleration relative to robust inactive slope.
    dl=np.diff(lam);slope=np.abs(np.diff(lmp)/dl)
    inactive_steps=np.where((~active[:-1])&(~active[1:])&np.isfinite(slope))[0]
    base=float(np.median(slope[inactive_steps])) if len(inactive_steps) else float(np.nanmedian(slope))
    threshold=max(accel_factor*base,1.0)
    post=np.arange(len(slope));post=post[(post>=max((ia or 0)-1,0))&(slope>=threshold)]
    accel_i=int(post[0]+1) if len(post) else None
    def region(mask):
        if not np.any(mask):return {'n':0}
        ii=np.where(mask)[0];err=np.abs(pred[ii]-lmp[ii]);j=ii[int(np.nanargmax(lmp[ii]))];e=ii[int(np.nanargmax(err))]
        return {'n':int(len(ii)),'max_true_lmp':float(lmp[j]),'lambda_at_max_true_lmp':float(lam[j]),'pred_lmp_at_max_true_lmp':float(pred[j]),'max_abs_lmp_error':float(err.max()),'lambda_at_max_error':float(lam[e]),'max_dual':float(np.nanmax(dual[ii])),'min_margin_pu':float(np.nanmin(margin[ii])),'dual_lmp_pearson':pearson(dual[ii],lmp[ii])}
    observed=(lam>=0)&(lam<=1);extended=(lam>1);pre=(~active)&observed;act=active&observed
    out={'crossing_found':ia is not None,'inactive_lmp_slope_median':base,'acceleration_slope_threshold':threshold,'observed_0_to_1':region(observed),'observed_inactive':region(pre),'observed_active':region(act),'extension_gt_1':region(extended),'full_path_dual_lmp_pearson':pearson(dual,lmp),'active_path_dual_lmp_pearson':pearson(dual[active],lmp[active])}
    if ia is not None:
        out.update({'first_active_lambda':float(lam[ia]),'first_active_lmp':float(lmp[ia]),'first_active_dual':float(dual[ia]),'first_active_margin_pu':float(margin[ia]),'lmp_change_at_contact':None if ia==0 else float(lmp[ia]-lmp[ia-1])})
    if accel_i is not None:out.update({'acceleration_found':True,'acceleration_lambda':float(lam[accel_i]),'acceleration_true_lmp':float(lmp[accel_i]),'acceleration_dual':float(dual[accel_i]),'acceleration_margin_pu':float(margin[accel_i]),'local_abs_lmp_slope':float(slope[accel_i-1])})
    else:out['acceleration_found']=False
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--input',default=None);p.add_argument('--active-tol',type=float,default=1e-5);p.add_argument('--dual-tol',type=float,default=1e-6);p.add_argument('--accel-factor',type=float,default=10.0);a=p.parse_args()
    data=Path(a.data);src=Path(a.input) if a.input else data/'g14_active_set_continuation.csv'
    if not src.exists():raise FileNotFoundError(f'G14 CSV not found: {src}')
    rows=[]
    with src.open(newline='') as fh:
        for q in csv.DictReader(fh):
            rows.append({'active_scenario':int(q['active_scenario']),'control_scenario':int(q['control_scenario']),'lambda':f(q['lambda']),'solved':b(q['solved']),'true_vm_pu':f(q['true_vm_pu']),'true_margin_pu':f(q['true_margin_pu']),'true_lower_v_dual':f(q['true_lower_v_dual']),'true_lmp':f(q['true_lmp']),'pred_vm_pu':f(q['pred_vm_pu']),'pred_lmp':f(q['pred_lmp'])})
    scenarios=sorted(set(r['active_scenario'] for r in rows));summary={'configuration':vars(a),'source':str(src),'paths':{}}
    flat=[]
    print('G15 ACTIVE-REGIME MECHANISM AUDIT');print(f'source={src} paths={scenarios} acceleration_factor={a.accel_factor:g}')
    print('Observed path means 0<=lambda<=1; extrapolated stress points are reported separately.')
    for s in scenarios:
        rr=[r for r in rows if r['active_scenario']==s];z=summarize(rr,a.active_tol,a.dual_tol,a.accel_factor);summary['paths'][str(s)]=z
        o=z['observed_0_to_1'];e=z['extension_gt_1']
        print(f'\nPATH {s} control={rr[0]["control_scenario"]}')
        print(f' contact: lambda={z.get("first_active_lambda",np.nan):.6f} LMP={z.get("first_active_lmp",np.nan):.6f} dual={z.get("first_active_dual",np.nan):.6g} dLMP={z.get("lmp_change_at_contact",np.nan):.6f}')
        if z['acceleration_found']:print(f' acceleration: lambda={z["acceleration_lambda"]:.6f} LMP={z["acceleration_true_lmp"]:.6f} dual={z["acceleration_dual"]:.6g} |dLMP/dlambda|={z["local_abs_lmp_slope"]:.3f}')
        else:print(' acceleration: not detected')
        print(f' observed [0,1]: max true LMP={o.get("max_true_lmp",np.nan):.6f} at lambda={o.get("lambda_at_max_true_lmp",np.nan):.6f}; pred={o.get("pred_lmp_at_max_true_lmp",np.nan):.6f}; max error={o.get("max_abs_lmp_error",np.nan):.6f}; max dual={o.get("max_dual",np.nan):.6g}')
        print(f' dual-LMP Pearson: observed={o.get("dual_lmp_pearson",np.nan):.6f} active_full={z["active_path_dual_lmp_pearson"]:.6f}')
        if e['n']:print(f' extension >1: max true LMP={e.get("max_true_lmp",np.nan):.6f}; max error={e.get("max_abs_lmp_error",np.nan):.6f}; max dual={e.get("max_dual",np.nan):.6g}')
        flat.append({'active_scenario':s,'control_scenario':rr[0]['control_scenario'],'first_active_lambda':z.get('first_active_lambda',np.nan),'contact_lmp_change':z.get('lmp_change_at_contact',np.nan),'acceleration_lambda':z.get('acceleration_lambda',np.nan),'observed_max_true_lmp':o.get('max_true_lmp',np.nan),'observed_lambda_at_max_lmp':o.get('lambda_at_max_true_lmp',np.nan),'observed_pred_at_max_lmp':o.get('pred_lmp_at_max_true_lmp',np.nan),'observed_max_abs_lmp_error':o.get('max_abs_lmp_error',np.nan),'observed_max_dual':o.get('max_dual',np.nan),'observed_dual_lmp_pearson':o.get('dual_lmp_pearson',np.nan),'extension_max_true_lmp':e.get('max_true_lmp',np.nan),'extension_max_abs_lmp_error':e.get('max_abs_lmp_error',np.nan)})
    out=data/'g15_active_regime_mechanism_audit.csv';js=data/'g15_active_regime_mechanism_summary.json'
    with out.open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    js.write_text(json.dumps(summary,indent=2)+'\n')
    print(f'\nCSV output: {out}\nSummary: {js}')
    print('NOTE: G14 records the bus-31 lower-voltage dual only. G15 can establish association with that dual, but cannot identify unrecorded generator/Q/thermal/other-bus active-set changes. A solver-wide constraint audit is required before assigning causality.')
if __name__=='__main__':main()
