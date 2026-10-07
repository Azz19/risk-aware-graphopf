"""Differentiable batched polar AC power-flow layer for R1.

The forward pass uses Newton iterations with an analytic Jacobian and
torch.linalg.solve (LU-backed on supported backends).  The dependent voltage
state remains in the autograd graph.  Equality residuals are returned and must
be audited; paper language should say "to numerical tolerance", never exact
arithmetic.
"""
from __future__ import annotations
import numpy as np
import torch
from pypower.idx_bus import BUS_I, BUS_TYPE, REF, PV, PD, QD, GS, BS, VM, VA, VMIN, VMAX
from pypower.idx_brch import F_BUS,T_BUS,BR_R,BR_X,BR_B,RATE_A,TAP,SHIFT,BR_STATUS
from pypower.idx_gen import GEN_BUS,GEN_STATUS,PMIN,PMAX,QMIN,QMAX


def build_admittance(ppc, device=None, dtype=torch.float64):
    bus=ppc["bus"]; n=len(bus); ids=bus[:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}
    Y=np.zeros((n,n),complex); base=float(ppc["baseMVA"])
    Y[np.arange(n),np.arange(n)] += (bus[:,GS] + 1j*bus[:,BS]) / base
    for br in ppc["branch"]:
        if br[BR_STATUS] <= 0: continue
        i,j=lu[int(br[F_BUS])],lu[int(br[T_BUS])]
        y=1/complex(br[BR_R],br[BR_X]); bc=1j*br[BR_B]/2
        tap=br[TAP] if br[TAP] else 1.0
        t=tap*np.exp(1j*np.deg2rad(br[SHIFT]))
        Y[i,i]+=(y+bc)/(abs(t)**2); Y[j,j]+=y+bc
        Y[i,j]+=-y/np.conj(t); Y[j,i]+=-y/t
    return (torch.tensor(Y.real,dtype=dtype,device=device),
            torch.tensor(Y.imag,dtype=dtype,device=device))


def bus_generator_limits(ppc, device=None, dtype=torch.float64):
    ids=ppc["bus"][:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}; n=len(ids)
    arr={k:np.zeros(n) for k in ("pmin","pmax","qmin","qmax","count")}
    for g in ppc["gen"]:
        if g[GEN_STATUS] <= 0: continue
        i=lu[int(g[GEN_BUS])]; arr["count"][i]+=1
        for k,col in (("pmin",PMIN),("pmax",PMAX),("qmin",QMIN),("qmax",QMAX)): arr[k][i]+=g[col]
    return {k:torch.tensor(v,dtype=dtype,device=device) for k,v in arr.items()}


def _calc(Vm,Va,G,B):
    d=Va.unsqueeze(2)-Va.unsqueeze(1); c=torch.cos(d); s=torch.sin(d)
    P=Vm*( (Vm.unsqueeze(1)*(G*c+B*s)).sum(2) )
    Q=Vm*( (Vm.unsqueeze(1)*(G*s-B*c)).sum(2) )
    return P,Q


def _jac(Vm,Va,G,B,P,Q,nonref,pq):
    d=Va.unsqueeze(2)-Va.unsqueeze(1); c=torch.cos(d); s=torch.sin(d)
    vv=Vm.unsqueeze(2)*Vm.unsqueeze(1)
    H=vv*(G*s-B*c); N=Vm.unsqueeze(2)*(G*c+B*s)
    M=-vv*(G*c+B*s); L=Vm.unsqueeze(2)*(G*s-B*c)
    diag=torch.arange(Vm.shape[1],device=Vm.device)
    H[:,diag,diag]=-Q-B.diag()*Vm**2
    N[:,diag,diag]=P/Vm+G.diag()*Vm
    M[:,diag,diag]=P-G.diag()*Vm**2
    L[:,diag,diag]=Q/Vm-B.diag()*Vm
    H=H[:,nonref][:,:,nonref]; N=N[:,nonref][:,:,pq]
    M=M[:,pq][:,:,nonref]; L=L[:,pq][:,:,pq]
    return torch.cat((torch.cat((H,N),2),torch.cat((M,L),2)),1)


def solve_power_flow(ppc, pg_bus_mw, vg_bus, renewable_mw, renewable_buses,
                     max_iter=12, tolerance=1e-8):
    """Solve batched PF. pg_bus_mw includes scheduled generation at all gen buses.

    Slack P and PV/slack Q are dependent outputs. renewable_mw is [batch,n_sites].
    """
    device=pg_bus_mw.device; dtype=pg_bus_mw.dtype; base=float(ppc["baseMVA"])
    bus=ppc["bus"]; n=len(bus); ids=bus[:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}
    typ=bus[:,BUS_TYPE].astype(int); ref=np.where(typ==REF)[0]; pq=np.where(typ!=REF)[0]
    if len(ref)!=1: raise ValueError("R1 currently requires one reference bus")
    nonref=np.where(np.arange(n)!=ref[0])[0]
    nonref=torch.tensor(nonref,dtype=torch.long,device=device); pq=torch.tensor(pq,dtype=torch.long,device=device)
    G,B=build_admittance(ppc,device,dtype); batch=pg_bus_mw.shape[0]
    pd=torch.tensor(bus[:,PD],dtype=dtype,device=device).expand(batch,-1).clone()
    qd=torch.tensor(bus[:,QD],dtype=dtype,device=device).expand(batch,-1)
    for k,b in enumerate(renewable_buses): pd[:,lu[int(b)]]-=renewable_mw[:,k]
    Psp=(pg_bus_mw-pd)/base; Qsp=-qd/base
    Vm=torch.tensor(bus[:,VM],dtype=dtype,device=device).expand(batch,-1).clone()
    Va=torch.deg2rad(torch.tensor(bus[:,VA],dtype=dtype,device=device)).expand(batch,-1).clone()
    genmask=torch.tensor(np.isin(np.arange(n),np.where(np.bincount([lu[int(g[GEN_BUS])] for g in ppc["gen"] if g[GEN_STATUS]>0],minlength=n)>0)[0]),device=device)
    Vm=torch.where(genmask.unsqueeze(0),vg_bus,Vm)
    for _ in range(max_iter):
        P,Q=_calc(Vm,Va,G,B)
        mis=torch.cat(((Psp-P)[:,nonref],(Qsp-Q)[:,pq]),1)
        J=_jac(Vm,Va,G,B,P,Q,nonref,pq)
        dx=torch.linalg.solve(J,mis.unsqueeze(-1)).squeeze(-1)
        na=len(nonref); Va=Va.index_add(1,nonref,dx[:,:na])
        Vm=Vm.index_add(1,pq,dx[:,na:])
    P,Q=_calc(Vm,Va,G,B)
    residual=torch.cat(((Psp-P)[:,nonref],(Qsp-Q)[:,pq]),1)
    pg_dep=pg_bus_mw.clone(); pg_dep[:,ref[0]]=P[:,ref[0]]*base+pd[:,ref[0]]
    qg=Q*base+qd
    return {"vm":Vm,"va":Va,"p_inj_pu":P,"q_inj_pu":Q,"pg_bus_mw":pg_dep,
            "qg_bus_mvar":qg,"max_balance_residual_pu":residual.abs().amax(1),
            "converged":residual.abs().amax(1)<=tolerance}


def constraint_violations(ppc, state, temperature=40.0):
    from .risk import smooth_positive
    device=state["vm"].device; dtype=state["vm"].dtype; bus=ppc["bus"]
    lim=bus_generator_limits(ppc,device,dtype)
    vmin=torch.tensor(bus[:,VMIN],dtype=dtype,device=device); vmax=torch.tensor(bus[:,VMAX],dtype=dtype,device=device)
    vm=state["vm"]; pg=state["pg_bus_mw"]; qg=state["qg_bus_mvar"]
    v=torch.maximum(smooth_positive(vm-vmax,temperature),smooth_positive(vmin-vm,temperature)).amax(1)
    p=torch.maximum(smooth_positive(pg-lim["pmax"],temperature),smooth_positive(lim["pmin"]-pg,temperature))
    q=torch.maximum(smooth_positive(qg-lim["qmax"],temperature),smooth_positive(lim["qmin"]-qg,temperature))
    mask=lim["count"]>0
    return {"voltage":v,"pg":p[:,mask].amax(1),"qg":q[:,mask].amax(1),
            "balance":state["max_balance_residual_pu"]}
