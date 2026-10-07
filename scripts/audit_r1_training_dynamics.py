"""R1 training-dynamics audit.

Diagnostic only: trains one small R1 model and records objective components,
physical violations, controls, and gradient norms by epoch. It does not alter
the frozen paper protocol or touch the held-out test set.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, errors, policy_controls, generation_cost

from graphopf.differentiable_pf import solve_power_flow, constraint_violations
from graphopf.risk import risk_objective


def hard_metrics(case, st):
    from pypower.idx_bus import VMIN, VMAX
    from graphopf.differentiable_pf import bus_generator_limits
    device=st["vm"].device; dtype=st["vm"].dtype
    lim=bus_generator_limits(case,device,dtype); mask=lim["count"]>0
    vmin=torch.tensor(case["bus"][:,VMIN],dtype=dtype,device=device)
    vmax=torch.tensor(case["bus"][:,VMAX],dtype=dtype,device=device)
    vm=st["vm"]; pg=st["pg_bus_mw"]; qg=st["qg_bus_mvar"]
    v=torch.maximum(vm-vmax,vmin-vm).clamp_min(0).amax(1)
    p=torch.maximum(pg-lim["pmax"],lim["pmin"]-pg)[:,mask].clamp_min(0).amax(1)
    q=torch.maximum(qg-lim["qmax"],lim["qmin"]-qg)[:,mask].clamp_min(0).amax(1)
    return v,p,q


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--risk",type=float,default=10.0)
    ap.add_argument("--scenarios",type=int,default=512)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)

    from graphopf.risk_aware_model import RiskAwareGraphOPF
    seed=int(cfg["model"]["seeds"][0])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"]),
                         weight_decay=float(cfg["training"]["weight_decay"]))
    # Keep the declared training distribution/seed; model seed affects initialization only.
    E=errors(cfg,cfg["uncertainty"]["train_family"],int(cfg["uncertainty"]["train_seed"]),
             args.scenarios,fc,corr)
    rng=np.random.default_rng(int(cfg["uncertainty"]["train_seed"]))
    bs=int(cfg["training"]["batch_size"]); bM=float(case["baseMVA"])
    temp=float(cfg["training"]["softplus_temperature"]); rows=[]
    forecast_t=torch.tensor(fc,dtype=torch.float64,device=device)

    for ep in range(1,args.epochs+1):
        order=rng.permutation(len(E)); accum=[]
        for start in range(0,len(E),bs):
            e=torch.tensor(E[order[start:start+bs]],dtype=torch.float64,device=device)
            ctl=policy_controls(model,t,case)
            realized=torch.clamp(forecast_t+e,min=0); eff=realized-forecast_t
            mismatch=eff.sum(1)
            pg=ctl["pg_bus"].unsqueeze(0)-mismatch.unsqueeze(1)*ctl["alpha_bus"].unsqueeze(0)
            vg=ctl["vg_bus"].unsqueeze(0).expand(len(e),-1)
            st=solve_power_flow(case,pg,vg,eff,rb)
            v=constraint_violations(case,st,temp)
            vn={"voltage":v["voltage"],"pg":v["pg"]/bM,"qg":v["qg"]/bM,
                "thermal":v["thermal"],"balance":v["balance"]}
            cost=generation_cost(case,st["pg_bus_mw"])/base
            loss,pieces=risk_objective(cost,vn,float(cfg["training"]["cvar_alpha"]),args.risk,
                {"voltage":cfg["training"]["voltage_weight"],"pg":cfg["training"]["pg_weight"],
                 "qg":cfg["training"]["qg_weight"],"thermal":cfg["training"]["thermal_weight"],
                 "balance":100.0})
            opt.zero_grad(); loss.backward()
            total_grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)))
            head_grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.head.parameters() if p.grad is not None)))
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            hv,hp,hq=hard_metrics(case,st)
            accum.append({
                "loss":float(loss.detach()),"cost":float(cost.mean().detach()),
                "cvar_voltage":float(pieces["voltage"].detach()),
                "cvar_pg_pu":float(pieces["pg"].detach()),
                "cvar_qg_pu":float(pieces["qg"].detach()),
                "cvar_thermal":float(pieces["thermal"].detach()),
                "cvar_balance":float(pieces["balance"].detach()),
                "hard_v_max_pu":float(hv.max().detach()),
                "hard_pg_max_mw":float(hp.max().detach()),
                "hard_qg_max_mvar":float(hq.max().detach()),
                "balance_max_pu":float(st["max_balance_residual_pu"].max().detach()),
                "grad_total":total_grad,"grad_head":head_grad})
        r={k:float(np.mean([a[k] for a in accum])) for k in accum[0]}
        ctl=policy_controls(model,t,case)
        gen=t["gen_mask"].detach().cpu().numpy().astype(bool)
        vg=ctl["vg_bus"].detach().cpu().numpy()[gen]
        pgc=ctl["pg_bus"].detach().cpu().numpy()[gen]
        alpha=ctl["alpha_bus"].detach().cpu().numpy()[gen]
        r.update(epoch=ep,vg_min=float(vg.min()),vg_max=float(vg.max()),
                 pg_min=float(pgc.min()),pg_max=float(pgc.max()),
                 alpha_min=float(alpha.min()),alpha_max=float(alpha.max()))
        rows.append(r)
        if ep==1 or ep%10==0:
            print(f"epoch={ep:3d} loss={r['loss']:.6f} V={r['hard_v_max_pu']:.6g} "
                  f"PG={r['hard_pg_max_mw']:.6g} QG={r['hard_qg_max_mvar']:.6g} "
                  f"cvarV={r['cvar_voltage']:.6g} cvarQ={r['cvar_qg_pu']:.6g} "
                  f"grad={r['grad_total']:.6g}",flush=True)

    out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/"r1_training_dynamics.csv",index=False)
    summary={"epochs":args.epochs,"risk_multiplier":args.risk,"scenarios":args.scenarios,
             "first":rows[0],"best_loss":rows[int(df["loss"].argmin())],"final":rows[-1]}
    (out/"r1_training_dynamics_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("\nR1 TRAINING DYNAMICS FINAL\n"+json.dumps(summary,indent=2))


if __name__=="__main__": main()
