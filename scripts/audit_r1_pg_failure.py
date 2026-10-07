"""Diagnose R1 PG/AGC failures without touching calibration/test data."""
import json,sys
from pathlib import Path
import numpy as np,torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup,errors,policy_controls,to_pf_base
from graphopf.risk_aware_model import RiskAwareGraphOPF
from graphopf.experiments import apply_scenario,run_pf_scenario
from pypower.idx_gen import GEN_BUS,PG,PMIN,PMAX

cfg=yaml.safe_load(Path("configs/R1_risk_aware_calibration.yaml").read_text())
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_,case,_,rb,fc,corr,_,t=setup(cfg,dev)
ck=torch.load("results/R1_risk_aware_calibration/r1_pretrain_risk_continuation.pt",map_location=dev,weights_only=False)
model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),layers=int(cfg["model"]["message_passing_layers"])).to(dev).double()
model.load_state_dict(ck["state"]); model.eval()
ctl=policy_controls(model,t,case); base,alpha=to_pf_base(case,ctl)
seed=int(cfg["uncertainty"]["train_seed"])+2991
E=errors(cfg,cfg["uncertainty"]["train_family"],seed,1000,fc,corr)
g=base["gen"]; tol=float(cfg["experiment"]["feasibility_tolerance"])
hits=np.zeros(len(g),int); prehit=posthit=0; presev=[]; postsev=[]; slack=[]; mismatch=[]
for e in E:
    eff=np.maximum(fc+e,0)-fc; mismatch.append(float(eff.sum()))
    pre=apply_scenario(base,rb,fc,e,alpha); p0=pre["gen"][:,PG].copy()
    pv=np.maximum(p0-g[:,PMAX],g[:,PMIN]-p0); ps=float(np.maximum(pv,0).max())
    presev.append(ps); prehit+=ps>tol
    res,ok=run_pf_scenario(pre)
    if not ok: continue
    p1=res["gen"][:,PG]; q=np.maximum(p1-g[:,PMAX],g[:,PMIN]-p1); qs=float(np.maximum(q,0).max())
    postsev.append(qs); posthit+=qs>tol; hits+=(q>tol); slack.append(float(p1[0]-p0[0]))
active=np.where(g[:,PMAX]>g[:,PMIN]+1e-9)[0]
gens=[{"gen":int(i),"bus":int(g[i,GEN_BUS]),"pg0":float(g[i,PG]),"pmin":float(g[i,PMIN]),"pmax":float(g[i,PMAX]),"down_reserve":float(g[i,PG]-g[i,PMIN]),"up_reserve":float(g[i,PMAX]-g[i,PG]),"alpha":float(alpha[i]),"post_violation_rate":float(hits[i]/len(E))} for i in active]
s={"seed":seed,"n":len(E),"mismatch_quantiles_mw":{str(q):float(np.quantile(mismatch,q)) for q in [.01,.05,.5,.95,.99]},"pre_pf_pg_violation_rate":float(prehit/len(E)),"post_pf_pg_violation_rate":float(posthit/len(E)),"mean_pre_pf_pg_violation_mw":float(np.mean(presev)),"mean_post_pf_pg_violation_mw":float(np.mean(postsev)),"mean_abs_slack_adjustment_mw":float(np.mean(np.abs(slack))),"generators":gens,"note":"Independent diagnostic seed; frozen calibration/test untouched."}
out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
(out/"r1_pg_failure_summary.json").write_text(json.dumps(s,indent=2)+"\n")
print("\nR1 PG FAILURE DIAGNOSTIC\n"+json.dumps(s,indent=2))
