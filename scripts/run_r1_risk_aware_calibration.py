"""R1: research-grade calibration block for the original Risk-Aware GraphOPF.

Protocol is fixed in configs/R1_risk_aware_calibration.yaml.  TRAIN learns without
OPF labels. CALIBRATION selects seed/risk multiplier. TEST is touched once after
selection and is evaluated with the independent PYPOWER AC-PF implementation.
"""
from __future__ import annotations
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from pypower.idx_bus import BUS_I,PD,QD,VMIN,VMAX
from pypower.idx_gen import GEN_BUS,GEN_STATUS,PG,VG,PMIN,PMAX,QMIN,QMAX
from graphopf.powerflow import load_case,solve_ac_opf,evaluate_constraints,topological_distance
from graphopf.experiments import select_renewable_buses,renewable_forecast,generate_errors,apply_scenario,run_pf_scenario
from graphopf.uncertainty import exponential_correlation
from graphopf.supervised_data import electrical_graph
from graphopf.risk_aware_model import RiskAwareGraphOPF
from graphopf.differentiable_pf import solve_power_flow,constraint_violations,bus_generator_limits
from graphopf.risk import risk_objective
from graphopf.metrics import wilson_interval


def setup(cfg,device):
    raw=load_case(cfg["case"]["path"]); rb=select_renewable_buses(raw,int(cfg["renewables"]["n_sites"]))
    fc=renewable_forecast(raw,rb,float(cfg["renewables"]["penetration"]))
    case=copy.deepcopy(raw); lu={int(r[BUS_I]):i for i,r in enumerate(case["bus"])}
    for b,p in zip(rb,fc): case["bus"][lu[int(b)],PD]-=p
    ref=solve_ac_opf(case); base=float(ref["f"]); n=len(case["bus"]); bM=float(case["baseMVA"])
    lim=bus_generator_limits(case,device,torch.float64); gm=lim["count"]>0
    ren=np.zeros(n)
    for b,p in zip(rb,fc): ren[lu[int(b)]]=p
    x=np.column_stack((case["bus"][:,PD]/bM,case["bus"][:,QD]/bM,ren/bM,
        lim["count"].cpu().numpy(),lim["pmin"].cpu().numpy()/bM,lim["pmax"].cpu().numpy()/bM,
        lim["qmin"].cpu().numpy()/bM,lim["qmax"].cpu().numpy()/bM))
    ei,ea=electrical_graph(case); ea=(ea-ea.mean(0))/np.maximum(ea.std(0),1e-6)
    tensors=dict(x=torch.tensor(x[None],dtype=torch.float64,device=device),
        edge_index=torch.tensor(ei,dtype=torch.long,device=device),
        edge_attr=torch.tensor(ea,dtype=torch.float64,device=device),gen_mask=gm,
        pmin=lim["pmin"],pmax=lim["pmax"],
        vmin=torch.tensor(case["bus"][:,VMIN],dtype=torch.float64,device=device),
        vmax=torch.tensor(case["bus"][:,VMAX],dtype=torch.float64,device=device))
    dist=topological_distance(raw,rb); corr=exponential_correlation(dist,float(cfg["uncertainty"]["correlation_length"]))
    return raw,case,ref,rb,fc,corr,base,tensors


def errors(cfg,kind,seed,n,fc,corr):
    std=float(cfg["uncertainty"]["relative_sigma"])*fc
    return generate_errors(kind,np.random.default_rng(seed),n,std,corr,float(cfg["uncertainty"]["student_df"]))


def generation_cost(case,pg_bus):
    ids=case["bus"][:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}
    vals=[]
    active=[g for g in range(len(case["gen"])) if case["gen"][g,GEN_STATUS]>0]
    # R1 IEEE57 guardrail: aggregated bus control is unambiguous only with one active gen/bus.
    buses=[int(case["gen"][g,GEN_BUS]) for g in active]
    if len(set(buses))!=len(buses): raise ValueError("R1 currently requires <=1 active generator per bus")
    for row,g in enumerate(active):
        gc=case["gencost"][g]; p=pg_bus[:,lu[int(case["gen"][g,GEN_BUS])]]
        ncoef=int(gc[3]); coef=gc[4:4+ncoef]
        y=torch.zeros_like(p)
        for a in coef: y=y*p+float(a)
        vals.append(y)
    return torch.stack(vals,1).sum(1)


def policy_controls(model,t,case):
    out=model(t["x"],t["edge_index"],t["edge_attr"],t["gen_mask"],t["pmin"],t["pmax"],t["vmin"],t["vmax"])
    return {k:v[0] for k,v in out.items()}


def train_one(cfg,mult,seed,case,rb,fc,corr,base,t,outdir):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=t["x"].device
    model=RiskAwareGraphOPF(hidden_dim=int(cfg["model"]["hidden_dim"]),layers=int(cfg["model"]["message_passing_layers"])).to(device).double()
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
    E=errors(cfg,cfg["uncertainty"]["train_family"],int(cfg["uncertainty"]["train_seed"])+seed,
             int(cfg["uncertainty"]["train_scenarios"]),fc,corr)
    bs=int(cfg["training"]["batch_size"]); patience=int(cfg["training"]["patience"]); best=1e99; stale=0; state=None; hist=[]
    rng=np.random.default_rng(seed); bM=float(case["baseMVA"]); temp=float(cfg["training"]["softplus_temperature"])
    for ep in range(1,int(cfg["training"]["epochs"])+1):
        order=rng.permutation(len(E)); losses=[]
        for start in range(0,len(E),bs):
            e=torch.tensor(E[order[start:start+bs]],dtype=torch.float64,device=device)
            ctl=policy_controls(model,t,case)
            forecast_t=torch.tensor(fc,dtype=torch.float64,device=device)
            realized=torch.clamp(forecast_t+e,min=0)
            effective_error=realized-forecast_t
            mismatch=effective_error.sum(1)
            pg=ctl["pg_bus"].unsqueeze(0)-mismatch.unsqueeze(1)*ctl["alpha_bus"].unsqueeze(0)
            vg=ctl["vg_bus"].unsqueeze(0).expand(len(e),-1)
            # 'case' already has the renewable forecast subtracted from PD.
            # The differentiable PF therefore receives only the realized
            # forecast error, exactly matching experiments.apply_scenario.
            st=solve_power_flow(case,pg,vg,effective_error,rb)
            v=constraint_violations(case,st,temp)
            vn={"voltage":v["voltage"],"pg":v["pg"]/bM,"qg":v["qg"]/bM,
                "thermal":v["thermal"],"balance":v["balance"]}
            cost=generation_cost(case,st["pg_bus_mw"])/base
            loss,pieces=risk_objective(cost,vn,float(cfg["training"]["cvar_alpha"]),float(mult),
                {"voltage":cfg["training"]["voltage_weight"],"pg":cfg["training"]["pg_weight"],
                 "qg":cfg["training"]["qg_weight"],"thermal":cfg["training"]["thermal_weight"],"balance":100.0})
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            losses.append(float(loss.detach()))
        score=float(np.mean(losses)); hist.append({"epoch":ep,"loss":score})
        if score<best-1e-7: best=score; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if ep==1 or ep%10==0: print(f"seed={seed} risk={mult:g} epoch={ep} loss={score:.6f}",flush=True)
        if stale>=patience: break
    model.load_state_dict(state)
    tag=f"seed{seed}_risk{mult:g}".replace(".","p"); torch.save({"state":state,"seed":seed,"risk_multiplier":mult},outdir/f"{tag}.pt")
    pd.DataFrame(hist).to_csv(outdir/f"{tag}_history.csv",index=False)
    return model,tag


def to_pf_base(case,controls):
    out=copy.deepcopy(case); ids=out["bus"][:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}
    pg=controls["pg_bus"].detach().cpu().numpy(); vg=controls["vg_bus"].detach().cpu().numpy()
    alpha=controls["alpha_bus"].detach().cpu().numpy(); ga=np.zeros(len(out["gen"]))
    for i,g in enumerate(out["gen"]):
        if g[GEN_STATUS]<=0: continue
        b=lu[int(g[GEN_BUS])]; out["gen"][i,PG]=pg[b]; out["gen"][i,VG]=vg[b]; ga[i]=alpha[b]
    s=ga.sum(); ga=ga/s if s>0 else ga
    return out,ga


def zero_error_audit(cfg,model,t,case,rb,fc):
    """Authoritative guardrail: learned nominal controls must be feasible before stochastic evaluation."""
    tol=float(cfg["experiment"]["feasibility_tolerance"])
    ctl=policy_controls(model,t,case); base,alpha=to_pf_base(case,ctl)
    res,ok=run_pf_scenario(apply_scenario(base,rb,fc,np.zeros_like(fc),alpha))
    if not ok:
        return {"pf_converged":False,"operational_feasible":False}
    m=evaluate_constraints(res,tol)
    return {"pf_converged":True,**m}


def controls_audit(model,t,case):
    ctl=policy_controls(model,t,case)
    pg=ctl["pg_bus"].detach().cpu().numpy(); vg=ctl["vg_bus"].detach().cpu().numpy()
    alpha=ctl["alpha_bus"].detach().cpu().numpy()
    return {"pg_bus_mw":pg.tolist(),"vg_bus_pu":vg.tolist(),"alpha_bus":alpha.tolist()}


def authoritative_eval(cfg,model,t,case,rb,fc,corr,seed,n,family,save_path=None):
    tol=float(cfg["experiment"]["feasibility_tolerance"]); conf=float(cfg["experiment"]["confidence_level"])
    ctl=policy_controls(model,t,case); base,alpha=to_pf_base(case,ctl)
    E=errors(cfg,family,seed,n,fc,corr); rows=[]; counts={k:0 for k in ["joint","voltage","pg","qg","thermal","nonconv"]}
    for i,e in enumerate(E):
        res,ok=run_pf_scenario(apply_scenario(base,rb,fc,e,alpha))
        if not ok:
            counts["joint"]+=1; counts["nonconv"]+=1
            row={"scenario_id":i,"pf_converged":False,"joint_violation":True}
        else:
            m=evaluate_constraints(res,tol); joint=not m["operational_feasible"]; counts["joint"]+=int(joint)
            for k,col in [("voltage","max_voltage_violation_pu"),("pg","max_pg_violation_mw"),("qg","max_qg_violation_mvar"),("thermal","max_thermal_overload_pu")]:
                counts[k]+=int(m[col]>tol)
            row={"scenario_id":i,"pf_converged":True,"joint_violation":joint,**m}
        rows.append(row)
        if (i+1)%1000==0: print(f"  authoritative {i+1}/{n}",flush=True)
    lo,hi=wilson_interval(counts["joint"],n,conf)
    summary={"n":n,"joint_violation_rate":counts["joint"]/n,"joint_ci_low":lo,"joint_ci_high":hi,
        "meets_5pct_point_estimate":counts["joint"]/n<=float(cfg["experiment"]["target_epsilon"]),
        "meets_5pct_upper_ci":hi<=float(cfg["experiment"]["target_epsilon"]),
        **{f"{k}_violation_rate":counts[k]/n for k in ["voltage","pg","qg","thermal"]},
        "pf_nonconvergence_rate":counts["nonconv"]/n}
    if save_path is not None: pd.DataFrame(rows).to_csv(save_path,index=False)
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/R1_risk_aware_calibration.yaml")
    ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); out=Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw,case,ref,rb,fc,corr,base,t=setup(cfg,device)
    if args.smoke:
        cfg["uncertainty"]["train_scenarios"]=256; cfg["uncertainty"]["calibration_scenarios"]=200; cfg["uncertainty"]["test_scenarios"]=200
        # Smoke is a pipeline check, but it must train long enough to reveal
        # whether the unsupervised objective can move toward nominal feasibility.
        cfg["training"]["epochs"]=40; cfg["training"]["patience"]=15
        cfg["training"]["risk_multipliers"]=[10.0]; cfg["model"]["seeds"]=cfg["model"]["seeds"][:1]
    records=[]; models={}
    for seed in cfg["model"]["seeds"]:
        for mult in cfg["training"]["risk_multipliers"]:
            model,tag=train_one(cfg,mult,int(seed),case,rb,fc,corr,base,t,out); models[tag]=model
            zero=zero_error_audit(cfg,model,t,case,rb,fc)
            (out/f"{tag}_controls.json").write_text(json.dumps({
                "zero_error_audit":zero,"controls":controls_audit(model,t,case)},indent=2)+"\n")
            cal=authoritative_eval(cfg,model,t,case,rb,fc,corr,int(cfg["uncertainty"]["calibration_seed"]),
                int(cfg["uncertainty"]["calibration_scenarios"]),cfg["uncertainty"]["train_family"])
            records.append({"tag":tag,"seed":seed,"risk_multiplier":mult,
                "zero_error_feasible":bool(zero.get("operational_feasible",False)),**cal})
            pd.DataFrame(records).to_csv(out/"calibration.csv",index=False)
    frame=pd.DataFrame(records)
    eligible=frame[frame["meets_5pct_upper_ci"] & frame["zero_error_feasible"]]
    if eligible.empty:
        selected=frame.sort_values(["zero_error_feasible","joint_ci_high","risk_multiplier"],
                                   ascending=[False,True,True]).iloc[0]
        status="CALIBRATION_TARGET_NOT_ATTAINED"
    else:
        selected=eligible.sort_values(["risk_multiplier","joint_ci_high"]).iloc[0]; status="CALIBRATION_TARGET_ATTAINED"
    tag=str(selected["tag"]); model=models[tag]
    test=authoritative_eval(cfg,model,t,case,rb,fc,corr,int(cfg["uncertainty"]["test_seed"]),
        int(cfg["uncertainty"]["test_scenarios"]),cfg["uncertainty"]["train_family"],out/"test_scenarios.csv")
    summary={"status":status,"selected":selected.to_dict(),"test":test,"device":str(device),
        "base_opf_cost":base,"renewable_buses":rb.tolist(),"renewable_forecast_mw":fc.tolist(),
        "protocol":"selection uses calibration only; test touched once after frozen selection",
        "selected_zero_error_audit":zero_error_audit(cfg,model,t,case,rb,fc),
        "claim_supported":bool(test["meets_5pct_upper_ci"])}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    print("\nR1 FINAL\n"+json.dumps(summary,indent=2,default=str))


if __name__=="__main__": main()
