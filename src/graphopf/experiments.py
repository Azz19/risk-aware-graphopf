from __future__ import annotations

import copy

import networkx as nx
import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I, PD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX

from graphopf.powerflow import evaluate_constraints, topological_distance
from graphopf.uncertainty import exponential_correlation, sample_gaussian, sample_student_t


def select_renewable_buses(ppc: dict, n_sites: int) -> np.ndarray:
    order = np.argsort(ppc["bus"][:, PD])[::-1]
    return ppc["bus"][order[:n_sites], BUS_I].astype(int)


def renewable_forecast(ppc: dict, bus_ids: np.ndarray, penetration: float) -> np.ndarray:
    total = penetration * float(ppc["bus"][:, PD].sum())
    load = {int(row[BUS_I]): max(float(row[PD]), 0.0) for row in ppc["bus"]}
    weights = np.array([load[int(b)] for b in bus_ids], dtype=float)
    if weights.sum() <= 0:
        weights[:] = 1.0
    return total * weights / weights.sum()


def headroom_participation(opf: dict) -> np.ndarray:
    gen = opf["gen"]
    active = gen[:, GEN_STATUS] > 0
    headroom = np.maximum(gen[:, PMAX] - gen[:, PG], 0.0) * active
    if headroom.sum() <= 0:
        headroom = active.astype(float)
    return headroom / headroom.sum()


def generate_errors(kind, rng, n_samples, std, correlation, df):
    if kind == "iid_gaussian":
        return sample_gaussian(rng, n_samples, std)
    if kind == "corr_gaussian":
        return sample_gaussian(rng, n_samples, std, correlation)
    if kind == "iid_student_t":
        return sample_student_t(rng, n_samples, std, df)
    if kind == "corr_student_t":
        return sample_student_t(rng, n_samples, std, df, correlation)
    raise ValueError(f"Unknown uncertainty family: {kind}")


def apply_scenario(base_opf, renewable_buses, forecast, error, participation):
    ppc = copy.deepcopy(base_opf)
    bus_lookup = {int(row[BUS_I]): i for i, row in enumerate(ppc["bus"])}

    # base_opf already contains the forecast renewable injection.
    # Therefore each stochastic scenario must apply ONLY the forecast error.
    # Subtracting the full realized renewable injection here would double-count
    # the forecast and make even the zero-error scenario infeasible.
    realized = np.maximum(forecast + error, 0.0)
    effective_error = realized - forecast
    for bus_id, delta in zip(renewable_buses, effective_error):
        ppc["bus"][bus_lookup[int(bus_id)], PD] -= delta

    # AGC balances the realized net renewable error around forecast dispatch.
    mismatch = float(effective_error.sum())
    ppc["gen"][:, PG] -= participation * mismatch
    return ppc


def run_pf_scenario(ppc):
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    result, success = runpf(ppc, opt)
    return result, bool(success)
