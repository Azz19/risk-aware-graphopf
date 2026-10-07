"""R1 loss-gradient attribution audit.

One diagnostic training run that measures which objective terms push the policy,
with special attention to generator-voltage controls. No held-out test data are
used and the frozen R1 paper protocol is unchanged.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, errors, policy_controls, generation_cost
from graphopf.differentiable_pf import solve_power_flow,constraint_violations
from graphopf.risk import empirical_cvar
from graphopf.risk_aware_model import RiskAwareGraphOPF

def norm(g):
    return float(torch.linalg.vector_norm(g).detach()) if g is not None else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--risk",type=float,default=10.0)
    ap.add_argument("--scenarios",type=int,default=512)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)
    seed=int(cfg["model"]["seeds"][0]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
    E=errors(cfg,cfg["uncertainty"]["train_family"],int(cfg["uncertainty"]["train_seed"]),args.scenarios,fc,corr)
    rng=np.random.default_rng(int(cfg["uncertainty"]["train_seed"])); bs=int(cfg["training"]["batch_size"])
    bM=float(case["baseMVA"]); temp=float(cfg["training"]["softplus_temperature"]); alpha=float(cfg["training"]["cvar_alpha"])
    ft=torch.tensor(fc,dtype=torch.float64,device=device); rows=[]
    for ep in range(1,args.epochs+1):
        order=rng.permutation(len(E)); batch_rows=[]
        for start in range(0,len(E),bs):
            e=torch.tensor(E[order[start:start+bs]],dtype=torch.float64,device=device)
            ctl=policy_controls(model,t,case); ctl["vg_bus"].retain_grad(); ctl["pg_bus"].retain_grad(); ctl["alpha_bus"].retain_grad()
            realized=torch.clamp(ft+e,min=0); eff=realized-ft; mismatch=eff.sum(1)
            pg=ctl["pg_bus"].unsqueeze(0)-mismatch.unsqueeze(1)*ctl["alpha_bus"].unsqueeze(0)
            vg=ctl["vg_bus"].unsqueeze(0).expand(len(e),-1)
            st=solve_power_flow(case,pg,vg,eff,rb); v=constraint_violations(case,st,temp)
            terms={
              "cost":generation_cost(case,st["pg_bus_mw"]).mean()/base,
              "voltage":args.risk*empirical_cvar(v["voltage"],alpha),
              "pg":args.risk*empirical_cvar(v["pg"]/bM,alpha),
              "qg":args.risk*empirical_cvar(v["qg"]/bM,alpha),
              "thermal":args.risk*empirical_cvar(v["thermal"],alpha),
              "balance":args.risk*100.0*empirical_cvar(v["balance"],alpha)}
            total=sum(terms.values())
            grads={}
            for name,term in terms.items():
                gs=torch.autograd.grad(term,(ctl["vg_bus"],ctl["pg_bus"],ctl["alpha_bus"]),retain_graph=True,allow_unused=True)
                grads[f"{name}_grad_vg"]=norm(gs[0]); grads[f"{name}_grad_pg"]=norm(gs[1]); grads[f"{name}_grad_alpha"]=norm(gs[2])
            opt.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            row={"loss":float(total.detach()),**{f"term_{k}":float(x.detach()) for k,x in terms.items()},**grads}
            batch_rows.append(row)
        r={k:float(np.mean([x[k] for x in batch_rows])) for k in batch_rows[0]}; r["epoch"]=ep; rows.append(r)
        if ep==1 or ep%10==0:
            print(f"epoch={ep:3d} loss={r['loss']:.6f} Vterm={r['term_voltage']:.4g} Qterm={r['term_qg']:.4g} "
                  f"|dV/dVG|={r['voltage_grad_vg']:.4g} |dQ/dVG|={r['qg_grad_vg']:.4g} "
                  f"|dCost/dVG|={r['cost_grad_vg']:.4g}",flush=True)
    out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/"r1_gradient_attribution.csv",index=False)
    cols=["epoch","loss","term_cost","term_voltage","term_pg","term_qg","voltage_grad_vg","qg_grad_vg","cost_grad_vg",
          "voltage_grad_pg","qg_grad_pg","cost_grad_pg","voltage_grad_alpha","qg_grad_alpha","cost_grad_alpha"]
    print("\nR1 GRADIENT ATTRIBUTION FINAL")
    print(df.loc[[0,int(df.loss.argmin()),len(df)-1],cols].to_string(index=False))

if __name__=="__main__": main()
