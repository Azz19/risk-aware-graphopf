"""G17: voltage-dual / LMP sensitivity decomposition.

Post-process G16's solver-wide continuation audit.  Quantify how much of the
trigger-bus LMP escalation is statistically associated with the bus-31 lower-V
dual and the strongest concurrent non-trigger dual (expected bus-46 upper-V).
This is a descriptive sensitivity decomposition, not a causal proof.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def f(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan

def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3 or np.std(x[m])==0 or np.std(y[m])==0:return np.nan
    return float(np.corrcoef(x[m],y[m])[0,1])
def fit(X,y):
    X=np.asarray(X,float);y=np.asarray(y,float);m=np.isfinite(y)&np.all(np.isfinite(X),axis=1);X=X[m];y=y[m]
    if len(y)<X.shape[1]+2:return None
    A=np.column_stack([np.ones(len(X)),X]);coef=np.linalg.lstsq(A,y,rcond=None)[0];pred=A@coef;ssr=float(np.sum((y-pred)**2));sst=float(np.sum((y-y.mean())**2));r2=1-ssr/sst if sst>0 else np.nan
    return {'n':len(y),'intercept':float(coef[0]),'coefficients':[float(v) for v in coef[1:]],'r2':float(r2),'rmse':float(np.sqrt(np.mean((y-pred)**2)))}
def parse_material(s):
    try:return json.loads(s) if s else []
    except json.JSONDecodeError:return []
def dual_of(items,kind,element):
    vals=[f(q.get('dual')) for q in items if q.get('kind')==kind and str(q.get('element'))==str(element)]
    return max(vals,key=abs) if vals else 0.0

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--input',default=None);p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--companion-bus',type=int,default=46);p.add_argument('--acceleration-lambdas',default='575:0.968,889:0.929,984:0.942');a=p.parse_args()
    data=Path(a.data);src=Path(a.input) if a.input else data/'g16_solverwide_constraint_audit.csv'
    if not src.exists():raise FileNotFoundError(src)
    accel={int(k):float(v) for k,v in (z.split(':') for z in a.acceleration_lambdas.split(','))}
    rows=[]
    with src.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if str(r.get('solved','')).lower() not in {'true','1','yes'}:continue
            mats=parse_material(r.get('material_constraints',''));r['_lambda']=f(r['lambda']);r['_lmp']=f(r['trigger_lmp']);r['_d31']=f(r['trigger_vmin_dual']);r['_d46']=dual_of(mats,'vmax',a.companion_bus);rows.append(r)
    paths=sorted(set(int(r['active_scenario']) for r in rows));summary={'configuration':vars(a),'source':str(src),'paths':{}};flat=[]
    print('G17 VOLTAGE-DUAL / LMP SENSITIVITY DECOMPOSITION');print(f'source={src} trigger=vmin@{a.trigger_bus} companion=vmax@{a.companion_bus}');print('Fits are descriptive associations along controlled OPF paths; they are not causal decompositions.')
    for s in paths:
        rr=[r for r in rows if int(r['active_scenario'])==s and 0<=r['_lambda']<=1];lam=np.array([r['_lambda'] for r in rr]);y=np.array([r['_lmp'] for r in rr]);d31=np.array([r['_d31'] for r in rr]);d46=np.array([r['_d46'] for r in rr]);la=accel.get(s,np.nan);post=lam>=la if np.isfinite(la) else np.ones(len(lam),bool)
        # Center price on the last pre-acceleration point to focus on escalation.
        pre=np.where(lam<la)[0] if np.isfinite(la) else np.array([],int);base=float(y[pre[-1]]) if len(pre) else float(y[0]);dy=y-base
        one31=fit(d31[:,None],dy);one46=fit(d46[:,None],dy);both=fit(np.column_stack([d31,d46]),dy);postfit=fit(np.column_stack([d31[post],d46[post]]),dy[post])
        z={'acceleration_lambda':la,'baseline_lmp':base,'n_observed':len(rr),'n_post_acceleration':int(post.sum()),'corr_lmp_bus31_vmin_dual':pearson(y,d31),'corr_lmp_bus46_vmax_dual':pearson(y,d46),'corr_bus31_bus46_duals':pearson(d31,d46),'delta_lmp_vs_bus31':one31,'delta_lmp_vs_bus46':one46,'delta_lmp_vs_both':both,'post_acceleration_delta_lmp_vs_both':postfit,'peak':{}}
        if len(y):
            j=int(np.argmax(y));z['peak']={'lambda':float(lam[j]),'lmp':float(y[j]),'delta_lmp':float(dy[j]),'bus31_vmin_dual':float(d31[j]),'bus46_vmax_dual':float(d46[j]),'dual_ratio_31_to_46':float(d31[j]/d46[j]) if d46[j]!=0 else None}
        summary['paths'][str(s)]=z
        print(f'\nPATH {s}: acceleration lambda={la:.6f} baseline LMP={base:.6f}')
        print(f' correlations: LMP~MU_VMIN31={z["corr_lmp_bus31_vmin_dual"]:.6f} LMP~MU_VMAX46={z["corr_lmp_bus46_vmax_dual"]:.6f} dual31~dual46={z["corr_bus31_bus46_duals"]:.6f}')
        if both:print(f' two-dual fit: deltaLMP={both["intercept"]:.6g} + ({both["coefficients"][0]:.6g})*MU31 + ({both["coefficients"][1]:.6g})*MU46; R2={both["r2"]:.8f} RMSE={both["rmse"]:.6g}')
        if postfit:print(f' post-acceleration two-dual R2={postfit["r2"]:.8f} RMSE={postfit["rmse"]:.6g}')
        q=z['peak'];print(f' peak observed: lambda={q.get("lambda",np.nan):.6f} LMP={q.get("lmp",np.nan):.6f} deltaLMP={q.get("delta_lmp",np.nan):.6f} MU31={q.get("bus31_vmin_dual",np.nan):.6g} MU46={q.get("bus46_vmax_dual",np.nan):.6g} ratio={q.get("dual_ratio_31_to_46",np.nan):.6f}')
        flat.append({'active_scenario':s,'acceleration_lambda':la,'baseline_lmp':base,'corr_lmp_mu31':z['corr_lmp_bus31_vmin_dual'],'corr_lmp_mu46':z['corr_lmp_bus46_vmax_dual'],'corr_mu31_mu46':z['corr_bus31_bus46_duals'],'r2_mu31':None if one31 is None else one31['r2'],'r2_mu46':None if one46 is None else one46['r2'],'r2_both':None if both is None else both['r2'],'rmse_both':None if both is None else both['rmse'],'post_r2_both':None if postfit is None else postfit['r2'],'post_rmse_both':None if postfit is None else postfit['rmse'],'peak_lmp':q.get('lmp'),'peak_mu31':q.get('bus31_vmin_dual'),'peak_mu46':q.get('bus46_vmax_dual'),'peak_dual_ratio':q.get('dual_ratio_31_to_46')})
    # pooled fit with path fixed effects plus both duals
    obs=[r for r in rows if 0<=r['_lambda']<=1];ps=paths;Y=np.array([r['_lmp'] for r in obs]);D31=np.array([r['_d31'] for r in obs]);D46=np.array([r['_d46'] for r in obs]);dummy=np.column_stack([[1.0 if int(r['active_scenario'])==s else 0.0 for r in obs] for s in ps[1:]]) if len(ps)>1 else np.empty((len(obs),0));pooled=fit(np.column_stack([D31,D46,dummy]),Y);summary['pooled_observed_fit']=pooled
    if pooled:print(f'\nPOOLED observed-path fit with path fixed effects: R2={pooled["r2"]:.8f} RMSE={pooled["rmse"]:.6g}; dual coefficients MU31={pooled["coefficients"][0]:.6g}, MU46={pooled["coefficients"][1]:.6g}')
    out=data/'g17_voltage_dual_sensitivity.csv';js=data/'g17_voltage_dual_sensitivity_summary.json'
    with out.open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nCSV output: {out}\nSummary: {js}');print('Decision target: if duals are highly collinear, do NOT interpret regression coefficients as separate causal contributions. High fit then supports a coupled voltage-constrained regime, not unique attribution to either bus.')
if __name__=='__main__':main()
