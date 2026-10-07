"""R1 adaptive dual-weight diagnostic.

Uses train-time CVaR constraint residuals to update nonnegative multipliers for
voltage, PG, QG and thermal terms. This is a diagnostic bridge toward the
paper's intended risk-aware Lagrangian-dual formulation; it does not alter the
frozen paper protocol or touch held-out test data.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, errors, policy_controls, generation_cost
from graphopf.differentiable_pf import solve_power_flow, constraint_violations
from graphopf.risk import empirical_cvar
from graphopf.risk_aware_model import RiskAwareGraphOPF

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--scenarios",type=int,default=512)
    ap.add_argument("--dual-lr",type=float,default=0.5)
    args=ap.parse_args()

    cfg=yaml.safe_load(Path(args.config).read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)

    seed=int(cfg["model"]["seeds"][0])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"]),
                         weight_decay=float(cfg["training"]["weight_decay"]))

    E=errors(cfg,cfg["uncertainty"]["train_family"],int(cfg["uncertainty"]["train_seed"]),
             args.scenarios,fc,corr)
    rng=np.random.default_rng(int(cfg["uncertainty"]["train_seed"]))
    bs=int(cfg["training"]["batch_size"]); bM=float(case["baseMVA"])
    temp=float(cfg["training"]["softplus_temperature"])
    alpha=float(cfg["training"]["cvar_alpha"])
    eps=float(cfg["experiment"]["target_epsilon"])
    ft=torch.tensor(fc,dtype=torch.float64,device=device)

    # Multipliers correspond to CVaR penalties. Start uniformly and adapt only
    # from training batches. Target is zero violation severity, while epsilon
    # remains the evaluation chance-constraint target.
    lam={k:1.0 for k in ["voltage","pg","qg","thermal"]}
    rows=[]

    for ep in range(1,args.epochs+1):
        order=rng.permutation(len(E)); epoch_rows=[]
        for start in range(0,len(E),bs):
            e=torch.tensor(E[order[start:start+bs]],dtype=torch.float64,device=device)
            ctl=policy_controls(model,t,case)
            realized=torch.clamp(ft+e,min=0); eff=realized-ft
            mismatch=eff.sum(1)
            pg=ctl["pg_bus"].unsqueeze(0)-mismatch.unsqueeze(1)*ctl["alpha_bus"].unsqueeze(0)
            vg=ctl["vg_bus"].unsqueeze(0).expand(len(e),-1)
            st=solve_power_flow(case,pg,vg,eff,rb)
            v=constraint_violations(case,st,temp)
            cvars={
                "voltage":empirical_cvar(v["voltage"],alpha),
                "pg":empirical_cvar(v["pg"]/bM,alpha),
                "qg":empirical_cvar(v["qg"]/bM,alpha),
                "thermal":empirical_cvar(v["thermal"],alpha),
            }
            cost=generation_cost(case,st["pg_bus_mw"]).mean()/base
            balance=empirical_cvar(v["balance"],alpha)
            loss=cost + sum(lam[k]*cvars[k] for k in lam) + 100.0*balance

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()

            # Projected dual ascent on training CVaR severities.
            for k in lam:
                lam[k]=max(0.0, lam[k] + args.dual_lr*float(cvars[k].detach()))

            epoch_rows.append({
                "loss":float(loss.detach()),"cost":float(cost.detach()),
                **{f"cvar_{k}":float(cvars[k].detach()) for k in cvars},
                **{f"lambda_{k}":float(lam[k]) for k in lam},
                "balance":float(balance.detach())
            })

        r={k:float(np.mean([x[k] for x in epoch_rows])) for k in epoch_rows[0]}
        r["epoch"]=ep; rows.append(r)
        if ep==1 or ep%10==0:
            print(f"epoch={ep:3d} loss={r['loss']:.6f} "
                  f"CVaR[V,P,Q]=({r['cvar_voltage']:.5g},{r['cvar_pg']:.5g},{r['cvar_qg']:.5g}) "
                  f"lambda[V,P,Q]=({r['lambda_voltage']:.3f},{r['lambda_pg']:.3f},{r['lambda_qg']:.3f})",
                  flush=True)

    out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/"r1_adaptive_dual_dynamics.csv",index=False)
    torch.save({"state":model.state_dict(),"lambdas":lam,"seed":seed},
               out/"r1_adaptive_dual_model.pt")
    print("\nR1 ADAPTIVE DUAL FINAL")
    print(df.tail(1).to_string(index=False))

if __name__=="__main__": main()
