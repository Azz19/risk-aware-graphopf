from pathlib import Path
import numpy as np
import torch
from graphopf.risk import empirical_cvar
from graphopf.differentiable_pf import solve_power_flow, bus_generator_limits
from graphopf.powerflow import load_case, solve_ac_opf
from pypower.idx_bus import BUS_I
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, VG


def test_empirical_cvar_tail_exceeds_mean():
    x=torch.tensor([0.,0.,1.,3.],requires_grad=True)
    z=empirical_cvar(x,0.75)
    assert z >= x.mean()
    z.backward()
    assert x.grad is not None


def test_differentiable_pf_reproduces_opf_state_if_case_available():
    path=Path("pglib-opf/pglib_opf_case57_ieee.m")
    if not path.exists():
        return
    raw=load_case(str(path)); opf=solve_ac_opf(raw)
    ids=raw["bus"][:,BUS_I].astype(int); lu={b:i for i,b in enumerate(ids)}; n=len(ids)
    pg=np.zeros(n); vg=opf["bus"][:,7].copy()
    for g in opf["gen"]:
        if g[GEN_STATUS]>0:
            pg[lu[int(g[GEN_BUS])]]+=g[PG]; vg[lu[int(g[GEN_BUS])]]=g[VG]
    state=solve_power_flow(raw,torch.tensor(pg[None],dtype=torch.float64),
                           torch.tensor(vg[None],dtype=torch.float64),
                           torch.zeros((1,0),dtype=torch.float64),np.array([],dtype=int))
    assert state["max_balance_residual_pu"].item() < 1e-6
    assert torch.max(torch.abs(state["vm"][0]-torch.tensor(opf["bus"][:,7]))).item() < 1e-4
