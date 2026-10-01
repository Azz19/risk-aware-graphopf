"""M01: build solver-supervised constraint/dual dataset for model development.

Re-solves the frozen G03 train/validation/test scenarios and records OPF primal,
economic, margin, active-set, and dual targets. This does NOT augment or alter
the frozen split. G14 continuation trajectories are deliberately excluded.

Outputs one compressed NPZ per split plus a JSON manifest. Targets include:
  bus: VM, LAM_P, voltage margins, MU_VMIN, MU_VMAX
  gen: PG/QG margins and MU_PMIN/MU_PMAX/MU_QMIN/MU_QMAX
  branch: apparent-power margins and MU_SF/MU_ST
The original x/y arrays and scenario indices are retained for exact alignment.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from pypower.idx_bus import BUS_I,PD,QD,VM,VMIN,VMAX,LAM_P,MU_VMIN,MU_VMAX
from pypower.idx_gen import GEN_BUS,PG,QG,PMIN,PMAX,QMIN,QMAX,MU_PMIN,MU_PMAX,MU_QMIN,MU_QMAX
from pypower.idx_brch import F_BUS,T_BUS,RATE_A,PF,QF,PT,QT,MU_SF,MU_ST
from graphopf.powerflow import load_case,solve_ac_opf


def load_split(p):
 z=np.load(p);return z['x'].astype(np.float32),z['y'].astype(np.float32)
def reconstruct(base,x):
 c={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()};c['bus'][:,PD]=x[:,0];c['bus'][:,QD]=x[:,1];return c
def col(a,j):return a[:,j] if a.shape[1]>j else np.zeros(len(a))
def solve_targets(base,x,dual_tol,margin_tol):
 s=solve_ac_opf(reconstruct(base,x));b,g,r=s['bus'],s['gen'],s['branch']
 vm=b[:,VM];vmin_margin=vm-b[:,VMIN];vmax_margin=b[:,VMAX]-vm;mu_vmin=col(b,MU_VMIN);mu_vmax=col(b,MU_VMAX)
 pg,qg=g[:,PG],g[:,QG];pmin_margin=pg-g[:,PMIN];pmax_margin=g[:,PMAX]-pg;qmin_margin=qg-g[:,QMIN];qmax_margin=g[:,QMAX]-qg
 rate=r[:,RATE_A];sf=np.hypot(r[:,PF],r[:,QF]);st=np.hypot(r[:,PT],r[:,QT]);valid=rate>0
 bfm=np.where(valid,rate-sf,np.inf);btm=np.where(valid,rate-st,np.inf)
 # activity is solver/economic activity OR numerical contact; useful as a supervised auxiliary target
 va=((mu_vmin>dual_tol)|(vmin_margin<=margin_tol)|(mu_vmax>dual_tol)|(vmax_margin<=margin_tol)).astype(np.uint8)
 ga=((col(g,MU_PMIN)>dual_tol)|(pmin_margin<=margin_tol)|(col(g,MU_PMAX)>dual_tol)|(pmax_margin<=margin_tol)|(col(g,MU_QMIN)>dual_tol)|(qmin_margin<=margin_tol)|(col(g,MU_QMAX)>dual_tol)|(qmax_margin<=margin_tol)).astype(np.uint8)
 ba=((valid)&((col(r,MU_SF)>dual_tol)|(bfm<=margin_tol)|(col(r,MU_ST)>dual_tol)|(btm<=margin_tol))).astype(np.uint8)
 return dict(vm=vm.astype(np.float32),lmp=b[:,LAM_P].astype(np.float32),vmin_margin=vmin_margin.astype(np.float32),vmax_margin=vmax_margin.astype(np.float32),mu_vmin=mu_vmin.astype(np.float32),mu_vmax=mu_vmax.astype(np.float32),voltage_active=va,pg=pg.astype(np.float32),qg=qg.astype(np.float32),pmin_margin=pmin_margin.astype(np.float32),pmax_margin=pmax_margin.astype(np.float32),qmin_margin=qmin_margin.astype(np.float32),qmax_margin=qmax_margin.astype(np.float32),mu_pmin=col(g,MU_PMIN).astype(np.float32),mu_pmax=col(g,MU_PMAX).astype(np.float32),mu_qmin=col(g,MU_QMIN).astype(np.float32),mu_qmax=col(g,MU_QMAX).astype(np.float32),gen_active=ga,branch_from_margin=bfm.astype(np.float32),branch_to_margin=btm.astype(np.float32),mu_sf=col(r,MU_SF).astype(np.float32),mu_st=col(r,MU_ST).astype(np.float32),branch_active=ba,objective=np.float64(s.get('f',np.nan)))
def stack(rows,key):return np.stack([q[key] for q in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--config',default='configs/case57.yaml');p.add_argument('--output',default=None);p.add_argument('--splits',default='train,val,test');p.add_argument('--dual-tol',type=float,default=1e-6);p.add_argument('--margin-tol',type=float,default=1e-5);p.add_argument('--progress',type=int,default=100);a=p.parse_args()
 data=Path(a.data);out=Path(a.output) if a.output else data/'m01_constraint_dual';out.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load(Path(a.config).read_text());base=load_case(cfg['case']['path']);splits=[z.strip() for z in a.splits.split(',') if z.strip()]
 print('M01 CONSTRAINT / DUAL DATASET BUILDER');print(f'source={data} output={out} splits={splits}');print('Frozen G03 scenarios only; no G14 continuation samples are added.')
 manifest={'configuration':vars(a),'case':cfg['case'],'splits':{},'target_semantics':{'dual_tol':a.dual_tol,'margin_tol':a.margin_tol}}
 for name in splits:
  x,y=load_split(data/f'{name}.npz');rows=[];failed=[];t=time.time();print(f'\n{name.upper()} n={len(x)}',flush=True)
  for i,xi in enumerate(x):
   try:rows.append(solve_targets(base,xi,a.dual_tol,a.margin_tol))
   except Exception as e:failed.append({'scenario':i,'error':repr(e)});rows.append(None)
   if (i+1)%a.progress==0 or i+1==len(x):print(f' solved {i+1}/{len(x)} failures={len(failed)}',flush=True)
  if failed:raise RuntimeError(f'{name}: {len(failed)} OPF failures; refusing to create a misaligned dataset. First failures: {failed[:3]}')
  keys=[k for k in rows[0] if k!='objective'];payload={'x':x,'y':y,'scenario':np.arange(len(x),dtype=np.int32),'objective':np.array([q['objective'] for q in rows],dtype=np.float64)}
  for k in keys:payload[k]=stack(rows,k)
  payload['bus_ids']=base['bus'][:,BUS_I].astype(np.int32);payload['gen_bus_ids']=base['gen'][:,GEN_BUS].astype(np.int32);payload['branch_from_bus']=base['branch'][:,F_BUS].astype(np.int32);payload['branch_to_bus']=base['branch'][:,T_BUS].astype(np.int32);payload['branch_rate_a']=base['branch'][:,RATE_A].astype(np.float32)
  dest=out/f'{name}_constraint_dual.npz';np.savez_compressed(dest,**payload)
  trigger=np.where(payload['bus_ids']==31)[0];tj=int(trigger[0]) if len(trigger) else None
  info={'n':len(x),'file':str(dest),'seconds':time.time()-t,'failures':0,'voltage_active_samples':int(payload['voltage_active'].any(1).sum()),'generator_active_samples':int(payload['gen_active'].any(1).sum()),'branch_active_samples':int(payload['branch_active'].any(1).sum()),'max_lmp':float(payload['lmp'].max()),'max_voltage_dual':float(max(payload['mu_vmin'].max(),payload['mu_vmax'].max()))}
  if tj is not None:info.update({'bus31_vmin_active_samples':int(((payload['mu_vmin'][:,tj]>a.dual_tol)|(payload['vmin_margin'][:,tj]<=a.margin_tol)).sum()),'bus31_max_mu_vmin':float(payload['mu_vmin'][:,tj].max()),'bus31_max_lmp':float(payload['lmp'][:,tj].max())})
  manifest['splits'][name]=info;print(json.dumps(info,indent=2))
 mf=out/'manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n');print(f'\nManifest: {mf}');print('M01 complete. Keep train/val/test identities frozen. Use train targets for fitting, val for model selection, and test only for final evaluation.')
if __name__=='__main__':main()
