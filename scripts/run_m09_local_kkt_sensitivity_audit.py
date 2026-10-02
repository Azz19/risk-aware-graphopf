"""M09: local KKT sensitivity / regime-transition audit.

Paper-oriented mechanism test using the frozen G16 continuation audit.  No GNN
is trained and no test point is used to fit its own derivative.  The experiment
asks whether LMP escalation is explained by a stable local shadow-price
sensitivity or by a changing KKT regime.

Primary tests
-------------
1. Finite-difference dLMP/dmu and dLMP/dlambda along each observed path.
2. Pre-registered early/mid/tail windows to quantify sensitivity drift.
3. One-step causal-order diagnostic: dual increments at k predict LMP increments
   at k+1 versus the reverse ordering.  This is predictive timing evidence only,
   not a causal claim.
4. Leave-one-path-out local-slope transfer: slopes learned on the other paths
   reconstruct held-path LMP increments.  This is the key generalization test.
5. Active-set-event alignment: compare largest sensitivity changes with G16
   material-constraint signature changes.

Because continuation points are path-correlated, uncertainty is reported with a
path/block bootstrap rather than treating all 303 points as IID.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def ff(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan

def material(s):
    try:return json.loads(s) if s else []
    except Exception:return []
def mu(items,kind,element):
    z=[ff(q.get('dual')) for q in items if q.get('kind')==kind and str(q.get('element'))==str(element)]
    return max(z,key=abs) if z else 0.0
def sig(items,tol=1e-6):return tuple(sorted((q.get('kind'),str(q.get('element'))) for q in items if abs(ff(q.get('dual')))>tol))
def safe_corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)<1e-14 or np.std(b)<1e-14:return np.nan
    return float(np.corrcoef(a,b)[0,1])
def met(y,p):
    e=np.asarray(p)-np.asarray(y);return {'n':int(len(e)),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'max_abs':float(np.max(abs(e)))} if len(e) else None
def slope(dx,dy,ridge=1e-12):
    dx=np.asarray(dx,float);dy=np.asarray(dy,float);return float((dx@dy)/(dx@dx+ridge))
def boot_ci(vals,rng,B=5000):
    vals=np.asarray(vals,float);vals=vals[np.isfinite(vals)]
    if not len(vals):return [np.nan,np.nan]
    b=np.array([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(B)])
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--input',default=None);p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--companion-bus',type=int,default=46);p.add_argument('--high-price',type=float,default=45.0);p.add_argument('--dual-tol',type=float,default=1e-6);p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=20271003);a=p.parse_args()
    data=Path(a.data);src=Path(a.input) if a.input else data/'g16_solverwide_constraint_audit.csv'
    if not src.exists():raise FileNotFoundError(src)
    rows=[]
    with src.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if str(r.get('solved','')).lower() not in {'true','1','yes'}:continue
            la=ff(r.get('lambda'))
            if not 0<=la<=1:continue
            m=material(r.get('material_constraints',''))
            rows.append({'path':int(r['active_scenario']),'lambda':la,'lmp':ff(r['trigger_lmp']),'mu31':ff(r['trigger_vmin_dual']),'mu46':mu(m,'vmax',a.companion_bus),'sig':sig(m,a.dual_tol)})
    paths=sorted(set(r['path'] for r in rows));print('M09 LOCAL KKT SENSITIVITY / REGIME-TRANSITION AUDIT');print(f'source={src} paths={paths} observed_lambda=[0,1] trigger=vmin@{a.trigger_bus} companion=vmax@{a.companion_bus}');print('Frozen physical continuation only. No neural training and no extrapolated lambda>1 points.')
    summary={'configuration':vars(a),'source':str(src),'paths':{},'leave_one_path_out':{}};out=[];path_slopes={}
    for s in paths:
        rr=sorted([r for r in rows if r['path']==s],key=lambda z:z['lambda']);la=np.array([r['lambda'] for r in rr]);y=np.array([r['lmp'] for r in rr]);u=np.array([r['mu31'] for r in rr]);v=np.array([r['mu46'] for r in rr]);dl=np.diff(la);dy=np.diff(y);du=np.diff(u);dv=np.diff(v);mid=(la[1:]+la[:-1])/2
        sy=slope(du,dy);path_slopes[s]=sy
        windows={'early':mid<.75,'transition':(mid>=.75)&(mid<.95),'tail':mid>=.95}
        wr={}
        for n,mask in windows.items():
            wr[n]={'n':int(mask.sum()),'dLMP_dlambda':slope(dl[mask],dy[mask]) if mask.any() else np.nan,'dLMP_dmu31':slope(du[mask],dy[mask]) if mask.any() else np.nan,'corr_dLMP_dmu31':safe_corr(dy[mask],du[mask]) if mask.sum()>2 else np.nan}
        # local secant sensitivity and active-set change indicator
        loc=np.divide(dy,du,out=np.full_like(dy,np.nan),where=abs(du)>1e-10);events=np.array([rr[k]['sig']!=rr[k+1]['sig'] for k in range(len(rr)-1)])
        finite=np.isfinite(loc);event_abs=np.nanmedian(abs(loc[events&finite])) if np.any(events&finite) else np.nan;stable_abs=np.nanmedian(abs(loc[(~events)&finite])) if np.any((~events)&finite) else np.nan
        # timing: increment correlations at same step and one-step lead/lag
        same=safe_corr(du,dy);lead=safe_corr(du[:-1],dy[1:]);reverse=safe_corr(dy[:-1],du[1:])
        topidx=np.argsort(np.nan_to_num(abs(np.diff(np.divide(dy,dl,out=np.zeros_like(dy),where=abs(dl)>0))),nan=-1))[-3:][::-1]+1 if len(dy)>1 else []
        top=[{'lambda':float(mid[i]),'dLMP_dlambda':float(dy[i]/dl[i]),'active_set_change':bool(events[i]),'mu31':float(u[i+1]),'lmp':float(y[i+1])} for i in topidx]
        summary['paths'][str(s)]={'global_dLMP_dmu31':sy,'windows':wr,'same_step_corr':same,'dual_leads_price_corr':lead,'price_leads_dual_corr':reverse,'median_abs_local_sensitivity_event':float(event_abs),'median_abs_local_sensitivity_stable':float(stable_abs),'top_sensitivity_changes':top,'n_active_set_events':int(events.sum())}
        print(f'\nPATH {s} global dLMP/dMU31={sy:.8g} same_corr={same:.6f} dual->next_price={lead:.6f} price->next_dual={reverse:.6f}')
        for n,z in wr.items():print(f' {n:10s} n={z["n"]:2d} dLMP/dlambda={z["dLMP_dlambda"]:.6g} dLMP/dMU31={z["dLMP_dmu31"]:.8g} corr={z["corr_dLMP_dmu31"]:.6f}')
        print(f' active-set events={events.sum()} median |local sensitivity| event/stable={event_abs:.6g}/{stable_abs:.6g}')
        for k in range(len(dy)):out.append({'path':s,'lambda_left':la[k],'lambda_right':la[k+1],'lambda_mid':mid[k],'lmp_left':y[k],'lmp_right':y[k+1],'mu31_left':u[k],'mu31_right':u[k+1],'mu46_left':v[k],'mu46_right':v[k+1],'delta_lmp':dy[k],'delta_mu31':du[k],'delta_mu46':dv[k],'dLMP_dlambda':dy[k]/dl[k],'dLMP_dmu31':loc[k],'active_set_change':int(events[k])})
    print('\nLEAVE-ONE-PATH-OUT INCREMENT TRANSFER')
    transfer_mae=[]
    for held in paths:
        tr=[r for r in out if r['path']!=held];te=[r for r in out if r['path']==held];b=slope([r['delta_mu31'] for r in tr],[r['delta_lmp'] for r in tr]);pred=np.array([b*r['delta_mu31'] for r in te]);true=np.array([r['delta_lmp'] for r in te]);m=met(true,pred);hi=np.array([r['lmp_right']>=a.high_price for r in te]);mh=met(true[hi],pred[hi]);summary['leave_one_path_out'][str(held)]={'slope':b,'all_increment':m,'high_price_increment':mh};transfer_mae.append(m['mae']);print(f' held={held} slope={b:.8g} increment_MAE={m["mae"]:.6f} high_price_n={hi.sum()} high_price_increment_MAE={np.nan if mh is None else mh["mae"]:.6f}')
    rng=np.random.default_rng(a.seed);summary['path_block_bootstrap']={'mean_LOPO_increment_MAE':float(np.mean(transfer_mae)),'95pct_CI':boot_ci(transfer_mae,rng,a.bootstrap),'unit':'path'}
    csvout=data/'m09_local_kkt_sensitivity_audit.csv';js=data/'m09_local_kkt_sensitivity_audit_summary.json'
    with csvout.open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nPATH-BLOCK BOOTSTRAP mean LOPO increment MAE={np.mean(transfer_mae):.6f} 95% CI={summary["path_block_bootstrap"]["95pct_CI"]}');print(f'CSV output: {csvout}\nSummary: {js}');print('Decision: stable held-path increment transfer supports a reusable KKT-sensitivity decoder. Large path-specific drift or error concentrated at active-set changes supports a regime-conditioned/local-Jacobian decoder instead. Timing correlations are descriptive and must not be reported as causal evidence.')
if __name__=='__main__':main()
