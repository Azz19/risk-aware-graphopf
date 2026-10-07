"""R1 diagnostic: compare differentiable PF and authoritative PYPOWER at frozen controls.

This is an implementation audit only. It does not alter R1 protocol, tune on
test data, or use OPF labels for training. It checks whether the differentiable
training model and PYPOWER evaluate the same nominal control consistently.
"""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path
import numpy as np, torch, yaml
from pypower.idx_bus import BUS_I, VM, VA
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, VG
from graphopf.powerflow import load_case, solve_ac_opf, evaluate_constraints
from graphopf.experiments import select_renewable_buses, renewable_forecast, apply_scenario, run_pf_scenario
from graphopf.differentiable_pf import solve_power_flow, constraint_violations, bus_generator_limits
from graphopf.risk_aware_model import RiskAwareGraphOPF
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, policy_controls, to_pf_base


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--checkpoint",default="results/R1_risk_aware_calibration/seed20271101_risk10.pt")
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)
    ckpt=torch.load(args.checkpoint,map_location=device)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    model.load_state_dict(ckpt["state"]); model.eval()
    ctl=policy_controls(model,t,case)
    pfbase,alpha=to_pf_base(case,ctl)

    # Differentiable nominal case: base already includes forecast; zero deviation.
    zero=torch.zeros((1,len(rb)),dtype=torch.float64,device=device)
    st=solve_power_flow(case,ctl["pg_bus"].unsqueeze(0),
                        ctl["vg_bus"].unsqueeze(0),zero,rb)
    dv=constraint_violations(case,st,float(cfg["training"]["softplus_temperature"]))

    # Authoritative nominal case.
    auth,ok=run_pf_scenario(apply_scenario(pfbase,rb,fc,np.zeros_like(fc),alpha))
    if not ok:
        raise SystemExit("PYPOWER nominal PF did not converge")
    am=evaluate_constraints(auth,float(cfg["experiment"]["feasibility_tolerance"]))

    ids=case["bus"][:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}
    gen_buses=sorted({lu[int(g[GEN_BUS])] for g in case["gen"] if g[GEN_STATUS]>0})
    diff_vm=st["vm"][0].detach().cpu().numpy()-auth["bus"][:,VM]
    diff_va=np.rad2deg(st["va"][0].detach().cpu().numpy())-auth["bus"][:,VA]

    report={
      "checkpoint":args.checkpoint,
      "differentiable":{
        "max_balance_residual_pu":float(st["max_balance_residual_pu"][0]),
        "max_voltage_violation_proxy":float(dv["voltage"][0]),
        "max_pg_violation_proxy_mw":float(dv["pg"][0]),
        "max_qg_violation_proxy_mvar":float(dv["qg"][0]),
        "max_thermal_violation_proxy_pu":float(dv["thermal"][0]),
      },
      "authoritative":am,
      "state_agreement":{
        "max_abs_vm_diff_pu":float(np.max(np.abs(diff_vm))),
        "max_abs_va_diff_deg":float(np.max(np.abs(diff_va))),
        "gen_bus_vm_diff_pu":{str(int(case["bus"][i,BUS_I])):float(diff_vm[i]) for i in gen_buses},
      },
      "controls":{
        "pg_bus_mw":ctl["pg_bus"].detach().cpu().numpy().tolist(),
        "vg_bus_pu":ctl["vg_bus"].detach().cpu().numpy().tolist(),
        "alpha_bus":ctl["alpha_bus"].detach().cpu().numpy().tolist(),
      }
    }
    out=Path(cfg["experiment"]["output_dir"])/"r1_pf_agreement_audit.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
