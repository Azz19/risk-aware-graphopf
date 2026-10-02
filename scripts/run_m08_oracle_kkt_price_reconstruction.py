"""M08: oracle KKT/active-set price reconstruction diagnostic.

Uses the frozen G16 continuation audit only.  It asks whether the observed
high-price branch can be reconstructed from solver dual/active-set information
without feeding target LMP into the pointwise reconstruction.  Fits coefficients
on pre-specified calibration paths/regions and reports held-path errors.

This is an oracle mechanism diagnostic, not a deployable predictor.  Because
MU_VMIN@31 and MU_VMAX@46 are highly collinear, coefficients must not be read as
separate causal effects.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def f(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan

def parse_material(s):
    try:return json.loads(s) if s else []
    except Exception:return []
def dual_of(items,kind,element):
    vals=[f(q.get('dual')) for q in items if q.get('kind')==kind and str(q.get('element'))==str(element)]
    return max(vals,key=abs) if vals else 0.0

def fit_ridge(X,y,ridge=1e-8):
    X=np.asarray(X,float);y=np.asarray(y,float);A=np.column_stack([np.ones(len(X)),X]);R=np.eye(A.shape[1])*ridge;R[0,0]=0
    return np.linalg.solve(A.T@A+R,A.T@y)
def pred(coef,X):return np.column_stack([np.ones(len(X)),np.asarray(X,float)])@coef
def metrics(y,p):
    e=np.asarray(p)-np.asarray(y);return {'n':int(len(e)),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'max_abs_error':float(np.max(abs(e)))} if len(e) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='results/G03_gnn_batch');ap.add_argument('--input',default=None);ap.add_argument('--trigger-bus',type=int,default=31);ap.add_argument('--companion-bus',type=int,default=46);ap.add_argument('--high-price',type=float,default=45.0);ap.add_argument('--ridge',type=float,default=1e-8);a=ap.parse_args()
    data=Path(a.data);src=Path(a.input) if a.input else data/'g16_solverwide_constraint_audit.csv'
    if not src.exists():raise FileNotFoundError(src)
    rows=[]
    with src.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if str(r.get('solved','')).lower() not in {'true','1','yes'}:continue
            lam=f(r['lambda'])
            if not (0<=lam<=1):continue
            mats=parse_material(r.get('material_constraints',''))
            rows.append({'path':int(r['active_scenario']),'lambda':lam,'lmp':f(r['trigger_lmp']),'mu31':f(r['trigger_vmin_dual']),'mu46':dual_of(mats,'vmax',a.companion_bus)})
    paths=sorted(set(r['path'] for r in rows));print('M08 ORACLE KKT / ACTIVE-SET PRICE RECONSTRUCTION');print(f'source={src} paths={paths} trigger=vmin@{a.trigger_bus} companion=vmax@{a.companion_bus}');print('Oracle diagnostic: solver duals are used as features. Target LMP is used only to fit calibration coefficients, never as a pointwise input.')
    # Feature choices test whether one dual suffices or the coupled voltage regime is needed.
    specs={'mu31':['mu31'],'mu46':['mu46'],'coupled':['mu31','mu46']};summary={'configuration':vars(a),'source':str(src),'models':{}};flat=[]
    for name,cols in specs.items():
        print(f'\nMODEL {name} features={cols}');summary['models'][name]={}
        for held in paths:
            tr=[r for r in rows if r['path']!=held];te=[r for r in rows if r['path']==held];X=np.array([[r[c] for c in cols] for r in tr]);y=np.array([r['lmp'] for r in tr]);Xt=np.array([[r[c] for c in cols] for r in te]);yt=np.array([r['lmp'] for r in te]);coef=fit_ridge(X,y,a.ridge);yp=pred(coef,Xt);hi=yt>=a.high_price
            m=metrics(yt,yp);mh=metrics(yt[hi],yp[hi]);peak=int(np.argmax(yt));rec={'held_path':held,'coefficients':[float(v) for v in coef],'all':m,'high_price':mh,'peak_true':float(yt[peak]),'peak_pred':float(yp[peak]),'peak_lambda':float(te[peak]['lambda'])};summary['models'][name][str(held)]=rec
            print(f' held={held} all_MAE={m["mae"]:.6f} high_price_n={int(hi.sum())} high_price_MAE={np.nan if mh is None else mh["mae"]:.6f} peak true/pred={yt[peak]:.3f}/{yp[peak]:.3f}')
            flat.append({'model':name,'held_path':held,'all_mae':m['mae'],'all_rmse':m['rmse'],'high_price_n':int(hi.sum()),'high_price_mae':np.nan if mh is None else mh['mae'],'peak_true':yt[peak],'peak_pred':yp[peak],'peak_lambda':te[peak]['lambda'],'coefficients':json.dumps([float(v) for v in coef])})
    # Same-path prefix calibration: fit only through lambda<=0.95, predict tail >0.95.
    summary['prefix_tail']={};print('\nCOUPLED SAME-PATH PREFIX -> TAIL')
    for s in paths:
        rr=[r for r in rows if r['path']==s];tr=[r for r in rr if r['lambda']<=.95];te=[r for r in rr if r['lambda']>.95];X=np.array([[r['mu31'],r['mu46']] for r in tr]);y=np.array([r['lmp'] for r in tr]);Xt=np.array([[r['mu31'],r['mu46']] for r in te]);yt=np.array([r['lmp'] for r in te]);coef=fit_ridge(X,y,a.ridge);yp=pred(coef,Xt);m=metrics(yt,yp);summary['prefix_tail'][str(s)]={'coefficients':[float(v) for v in coef],'tail':m};print(f' path={s} tail_n={len(te)} tail_MAE={m["mae"]:.6f} tail_RMSE={m["rmse"]:.6f} max_err={m["max_abs_error"]:.6f}')
    out=data/'m08_oracle_kkt_price_reconstruction.csv';js=data/'m08_oracle_kkt_price_reconstruction_summary.json'
    with out.open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    js.write_text(json.dumps(summary,indent=2)+'\n');print(f'\nCSV output: {out}\nSummary: {js}');print('Decision: strong coupled-dual reconstruction, especially on held-path high-price points, supports a structured KKT/active-set decoder. Poor held-path reconstruction means path-specific sensitivity information is still missing and M09 should compute local KKT/Jacobian sensitivities explicitly.')
if __name__=='__main__':main()
