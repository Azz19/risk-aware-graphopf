"""R1 nominal-feasibility pretraining diagnostic.

Tests whether the current GNN + differentiable PF parameterization can learn a
zero-error feasible operating point without OPF labels. No calibration/test
scenarios are used. This isolates representational/optimization feasibility
before stochastic risk training.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, policy_controls, generation_cost, zero_error_audit
from graphopf.differentiable_pf import solve_power_flow, constraint_violations
from graphopf.risk_aware_model import RiskAwareGraphOPF

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--feas-weight",type=float,default=100.0)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)
    seed=int(cfg["model"]["seeds"][0]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"]),
                         weight_decay=float(cfg["training"]["weight_decay"]))
    bM=float(case["baseMVA"]); temp=float(cfg["training"]["softplus_temperature"])
    z=torch.zeros((1,len(rb)),dtype=torch.float64,device=device); rows=[]

    for ep in range(1,args.epochs+1):
        ctl=policy_controls(model,t,case)
        st=solve_power_flow(case,ctl["pg_bus"].unsqueeze(0),ctl["vg_bus"].unsqueeze(0),z,rb)
        v=constraint_violations(case,st,temp)
        # Normalize MW/MVAr to p.u.; voltage and thermal are already p.u.
        feas=v["voltage"].mean()+v["pg"].mean()/bM+v["qg"].mean()/bM+v["thermal"].mean()+100.0*v["balance"].mean()
        cost=generation_cost(case,st["pg_bus_mw"]).mean()/base
        loss=cost+args.feas_weight*feas
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if ep==1 or ep%10==0 or ep==args.epochs:
            audit=zero_error_audit(cfg,model,t,case,rb,fc)
            r={"epoch":ep,"loss":float(loss.detach()),"cost":float(cost.detach()),"smooth_feas":float(feas.detach()),**audit}
            rows.append(r)
            print(f"epoch={ep:3d} loss={r['loss']:.6f} V={r['max_voltage_violation_pu']:.6g} "
                  f"PG={r['max_pg_violation_mw']:.6g} QG={r['max_qg_violation_mvar']:.6g} "
                  f"feasible={r['operational_feasible']}",flush=True)
            if r["operational_feasible"]:
                break

    out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"r1_nominal_feasibility_pretrain.csv",index=False)
    torch.save({"state":model.state_dict(),"seed":seed},out/"r1_nominal_feasibility_pretrain.pt")
    summary={"epochs_run":rows[-1]["epoch"],"feas_weight":args.feas_weight,
             "final":rows[-1],"ever_feasible":any(x["operational_feasible"] for x in rows),
             "note":"Diagnostic only; no OPF labels and no calibration/test scenarios used."}
    (out/"r1_nominal_feasibility_pretrain_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("\nR1 NOMINAL FEASIBILITY PRETRAIN FINAL\n"+json.dumps(summary,indent=2))

if __name__=="__main__": main()
