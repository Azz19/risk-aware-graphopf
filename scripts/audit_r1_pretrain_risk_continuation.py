"""R1 feasibility-pretrain -> stochastic-risk continuation diagnostic.

Loads the label-free nominally feasible checkpoint, then continues training on
correlated Student-t renewable scenarios. Uses only training-distribution data
plus an independent diagnostic validation seed; frozen calibration/test seeds
remain untouched.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import (setup,errors,policy_controls,generation_cost,
    zero_error_audit,authoritative_eval)
from graphopf.differentiable_pf import solve_power_flow,constraint_violations
from graphopf.risk import empirical_cvar
from graphopf.risk_aware_model import RiskAwareGraphOPF

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--checkpoint",default="results/R1_risk_aware_calibration/r1_nominal_feasibility_pretrain.pt")
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--scenarios",type=int,default=512)
    ap.add_argument("--risk-weight",type=float,default=100.0)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False); seed=int(ck["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    model.load_state_dict(ck["state"])
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"])*0.25,
                         weight_decay=float(cfg["training"]["weight_decay"]))
    E=errors(cfg,cfg["uncertainty"]["train_family"],int(cfg["uncertainty"]["train_seed"]),args.scenarios,fc,corr)
    rng=np.random.default_rng(int(cfg["uncertainty"]["train_seed"])); bs=int(cfg["training"]["batch_size"])
    bM=float(case["baseMVA"]); temp=float(cfg["training"]["softplus_temperature"]); a=float(cfg["training"]["cvar_alpha"])
    ft=torch.tensor(fc,dtype=torch.float64,device=device)
    print("initial_zero_error",json.dumps(zero_error_audit(cfg,model,t,case,rb,fc)))
    for ep in range(1,args.epochs+1):
        order=rng.permutation(len(E)); vals=[]
        for start in range(0,len(E),bs):
            e=torch.tensor(E[order[start:start+bs]],dtype=torch.float64,device=device)
            ctl=policy_controls(model,t,case); eff=torch.clamp(ft+e,min=0)-ft; mismatch=eff.sum(1)
            pg=ctl["pg_bus"].unsqueeze(0)-mismatch.unsqueeze(1)*ctl["alpha_bus"].unsqueeze(0)
            st=solve_power_flow(case,pg,ctl["vg_bus"].unsqueeze(0).expand(len(e),-1),eff,rb)
            v=constraint_violations(case,st,temp)
            risk=(empirical_cvar(v["voltage"],a)+empirical_cvar(v["pg"]/bM,a)+
                  empirical_cvar(v["qg"]/bM,a)+empirical_cvar(v["thermal"],a))
            cost=generation_cost(case,st["pg_bus_mw"]).mean()/base
            loss=cost+args.risk_weight*risk+100.0*empirical_cvar(v["balance"],a)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            vals.append((float(loss.detach()),float(risk.detach())))
        if ep==1 or ep%10==0:
            z=zero_error_audit(cfg,model,t,case,rb,fc)
            print(f"epoch={ep:3d} loss={np.mean([x[0] for x in vals]):.6f} risk={np.mean([x[1] for x in vals]):.6g} "
                  f"zero[V,PG,Q]=({z['max_voltage_violation_pu']:.5g},{z['max_pg_violation_mw']:.5g},{z['max_qg_violation_mvar']:.5g})",flush=True)
    out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    torch.save({"state":model.state_dict(),"seed":seed},out/"r1_pretrain_risk_continuation.pt")
    diag_seed=int(cfg["uncertainty"]["train_seed"])+1991
    z=zero_error_audit(cfg,model,t,case,rb,fc)
    val=authoritative_eval(cfg,model,t,case,rb,fc,corr,diag_seed,1000,cfg["uncertainty"]["train_family"],
                           out/"r1_pretrain_risk_validation_scenarios.csv")
    summary={"zero_error_audit":z,"diagnostic_validation":val,"diagnostic_seed":diag_seed,
             "risk_weight":args.risk_weight,"epochs":args.epochs,
             "note":"Diagnostic only; frozen calibration/test remain untouched."}
    (out/"r1_pretrain_risk_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("\nR1 PRETRAIN -> RISK FINAL\n"+json.dumps(summary,indent=2))
if __name__=="__main__": main()
