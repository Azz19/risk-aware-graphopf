"""M04: independent targeted dual-regime data generation.

Runs concurrently with M03. Generates NEW training-only OPF samples near/inside
the bus-31 lower-voltage regime and records continuous voltage dual targets.
Frozen G03 val/test and G14 continuation sets are never modified or added.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from pypower.idx_bus import BUS_I,PD,QD,VM,VMIN,VMAX,LAM_P,MU_VMIN,MU_VMAX
from graphopf.powerflow import load_case,solve_ac_opf

def reconstruct(base,x):
 c={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()};c['bus'][:,PD]=x[:,0];c['bus'][:,QD]=x[:,1];return c
def perturb(x,rng,scale):
 z=x.copy();ii=np.where(x[:,0]>1e-8)[0];d=rng.normal(size=len(ii));d-=d.mean();dp=d*scale*max(float(x[ii,0].mean()),1.);ratio=np.divide(x[ii,1],x[ii,0],out=np.zeros(len(ii)),where=x[ii,0]>1e-8);dq=dp*ratio;dp-=dp.mean();dq-=dq.mean();alpha=1.
 neg=dp<0
 if neg.any():alpha=min(alpha,float(np.min(x[ii[neg],0]/(-dp[neg]+1e-12)))*.95)
 neg=dq<0
 if neg.any():alpha=min(alpha,float(np.min(np.maximum(x[ii[neg],1],0)/(-dq[neg]+1e-12)))*.95)
 alpha=max(0.,min(1.,alpha));z[ii,0]+=alpha*dp;z[ii,1]+=alpha*dq;return z.astype(np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='results/G03_gnn_batch');p.add_argument('--config',default='configs/case57.yaml');p.add_argument('--output',default=None);p.add_argument('--trigger-bus',type=int,default=31);p.add_argument('--pool',type=int,default=500);p.add_argument('--active-fraction',type=float,default=.67);p.add_argument('--near-margin',type=float,default=.01);p.add_argument('--active-tol',type=float,default=1e-5);p.add_argument('--dual-tol',type=float,default=1e-6);p.add_argument('--perturb-scale',type=float,default=.12);p.add_argument('--max-attempts',type=int,default=20000);p.add_argument('--seed',type=int,default=20271002);a=p.parse_args();data=Path(a.data);out=Path(a.output) if a.output else data/'m04_dual_regime';out.mkdir(parents=True,exist_ok=True);tr=np.load(data/'train.npz');tx=tr['x'].astype(np.float32);ty=tr['y'].astype(np.float32);cfg=yaml.safe_load(Path(a.config).read_text());base=load_case(cfg['case']['path']);ids=base['bus'][:,BUS_I].astype(int);j=int(np.where(ids==a.trigger_bus)[0][0]);rng=np.random.default_rng(a.seed);marg=np.abs(ty[:,j,1]-base['bus'][j,VMIN]);seeds=np.argsort(marg)[:min(400,len(tx))];want_active=int(round(a.pool*a.active_fraction));want_near=a.pool-want_active;rows=[];na=nn=0;attempt=0;t=time.time()
 print('M04 TARGETED DUAL-REGIME DATASET BUILDER');print(f'target={a.pool} active={want_active} near={want_near} trigger_bus={a.trigger_bus}');print('Training-only generation; frozen validation/test/G14 remain untouched.')
 while len(rows)<a.pool and attempt<a.max_attempts:
  attempt+=1;s=int(rng.choice(seeds));x=perturb(tx[s],rng,a.perturb_scale*float(rng.uniform(.2,1.5)))
  try:sol=solve_ac_opf(reconstruct(base,x))
  except Exception:continue
  b=sol['bus'];vm=float(b[j,VM]);vmin=float(b[j,VMIN]);mu=float(b[j,MU_VMIN]);margin=vm-vmin;active=margin<=a.active_tol or mu>a.dual_tol;near=(not active) and margin<=a.near_margin
  if active and na>=want_active:continue
  if near and nn>=want_near:continue
  if not (active or near):continue
  y=ty[s].copy();y[:,0]=b[:,LAM_P];y[:,1]=b[:,VM];rows.append((x,y,b[:,MU_VMIN].astype(np.float32),b[:,MU_VMAX].astype(np.float32),(b[:,VM]-b[:,VMIN]).astype(np.float32),(b[:,VMAX]-b[:,VM]).astype(np.float32),s,active,margin,mu,float(b[j,LAM_P])));na+=int(active);nn+=int(near)
  if len(rows)%25==0:print(f' accepted {len(rows)}/{a.pool} attempts={attempt} active={na} near={nn} max_mu31={max(q[9] for q in rows):.3f} max_lmp31={max(q[10] for q in rows):.3f}',flush=True)
 if len(rows)<a.pool:raise RuntimeError(f'Only generated {len(rows)}/{a.pool} after {attempt} attempts. Increase --max-attempts or adjust --perturb-scale.')
 X=np.stack([q[0] for q in rows]);Y=np.stack([q[1] for q in rows]);mumin=np.stack([q[2] for q in rows]);mumax=np.stack([q[3] for q in rows]);vminm=np.stack([q[4] for q in rows]);vmaxm=np.stack([q[5] for q in rows]);active=np.array([q[7] for q in rows],dtype=np.uint8);mu31=np.array([q[9] for q in rows]);lmp31=np.array([q[10] for q in rows]);dest=out/'targeted_dual_regime.npz';np.savez_compressed(dest,x=X,y=Y,mu_vmin=mumin,mu_vmax=mumax,vmin_margin=vminm,vmax_margin=vmaxm,bus_ids=ids.astype(np.int32),seed_train_scenario=np.array([q[6] for q in rows],dtype=np.int32),bus31_active=active,bus31_mu_vmin=mu31.astype(np.float32),bus31_lmp=lmp31.astype(np.float32))
 pos=mu31[mu31>0];summary={'configuration':vars(a),'n':len(rows),'active':na,'near':nn,'attempts':attempt,'seconds':time.time()-t,'max_bus31_mu_vmin':float(mu31.max()),'median_positive_bus31_mu_vmin':float(np.median(pos)) if len(pos) else 0.,'max_bus31_lmp':float(lmp31.max()),'n_mu_gt_1':int((mu31>1).sum()),'n_mu_gt_100':int((mu31>100).sum()),'n_mu_gt_1000':int((mu31>1000).sum()),'file':str(dest)};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'+json.dumps(summary,indent=2));print(f'M04 dataset: {dest}');print('Do not merge with training until M03 and this dataset summary are reviewed.')
if __name__=='__main__':main()
