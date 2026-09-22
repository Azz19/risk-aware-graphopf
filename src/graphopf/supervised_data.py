"""Reproducible supervised AC-OPF dataset and electrical graph for GNN baselines.

This is a data-generation foundation, NOT a reproduction of any published model.
No renewable recourse or scenario-wise corrective OPF is used for model inputs.
"""
from __future__ import annotations

import copy

import numpy as np
from pypower.idx_brch import BR_STATUS, F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT
from pypower.idx_bus import BUS_I, PD, QD, VM, LAM_P
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG

from graphopf.powerflow import evaluate_constraints, solve_ac_opf


def electrical_graph(ppc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return directed edge_index [2, 2E] and edge features [2E, 6].

    Features: resistance, reactance, charging susceptance, MVA rating,
    transformer tap (1 if zero/unspecified), phase shift in degrees.
    Parallel branches are preserved; out-of-service branches are excluded.
    """
    bus_ids = ppc["bus"][:, BUS_I].astype(int)
    if len(set(bus_ids)) != len(bus_ids):
        raise ValueError("Duplicate bus IDs")
    lookup = {int(bus): i for i, bus in enumerate(bus_ids)}
    edges, attrs = [], []
    for branch in ppc["branch"]:
        if branch[BR_STATUS] <= 0:
            continue
        u, v = lookup[int(branch[F_BUS])], lookup[int(branch[T_BUS])]
        tap = float(branch[TAP]) if branch[TAP] != 0 else 1.0
        attr = [float(branch[BR_R]), float(branch[BR_X]),
                float(branch[BR_B]), float(branch[RATE_A]), tap,
                float(branch[SHIFT])]
        edges.extend(((u, v), (v, u)))
        attrs.extend((attr, attr))
    if not edges:
        raise ValueError("Case has no in-service branches")
    return np.asarray(edges, dtype=np.int64).T, np.asarray(attrs, dtype=np.float64)


def generator_bus_features(ppc: dict) -> np.ndarray:
    """Aggregate scheduled generation and generator-voltage setpoints by bus."""
    bus_ids = ppc["bus"][:, BUS_I].astype(int)
    lookup = {int(bus): i for i, bus in enumerate(bus_ids)}
    result = np.zeros((len(bus_ids), 4), dtype=np.float64)
    for gen in ppc["gen"]:
        if gen[GEN_STATUS] <= 0:
            continue
        row = lookup[int(gen[GEN_BUS])]
        result[row, 0] += 1.0
        result[row, 1] += gen[PG]
        result[row, 2] += gen[QG]
        result[row, 3] += gen[VG]
    mask = result[:, 0] > 0
    result[mask, 3] /= result[mask, 0]
    return result


def sample_operating_case(base: dict, rng: np.random.Generator,
                          load_sigma: float) -> dict:
    """Draw one nonnegative bus-wise load multiplier; no test-label leakage."""
    if load_sigma < 0:
        raise ValueError("load_sigma must be nonnegative")
    case = copy.deepcopy(base)
    factor = np.maximum(0.0, 1.0 + rng.normal(0, load_sigma, len(case["bus"])))
    case["bus"][:, PD] *= factor
    case["bus"][:, QD] *= factor
    return case


def build_example(case: dict, solved: dict, tolerance: float = 1e-6) -> dict:
    """Construct one sample. Inputs are PRE-OPF loads and fixed case attributes.

    Do not include OPF PG/QG/VG, voltage, or LMP in model inputs.
    The supervised targets (LMP, VM) follow the paper's broad target choice;
    architecture and training objective still need paper-specific verification.
    """
    if not bool(solved.get("success", False)):
        raise ValueError("OPF did not solve")
    if not evaluate_constraints(solved, tolerance)["operational_feasible"]:
        raise ValueError("OPF solution failed physical constraint checks")
    if solved["bus"].shape[1] <= LAM_P:
        raise ValueError("OPF result does not contain nodal LMP labels")
    if not np.array_equal(case["bus"][:, BUS_I], solved["bus"][:, BUS_I]):
        raise ValueError("Input and output bus order differ")
    bus = case["bus"]
    # Static generator capacities and generator counts are legitimate inputs;
    # realized OPF setpoints are not.
    lookup = {int(b): i for i, b in enumerate(bus[:, BUS_I].astype(int))}
    gen_static = np.zeros((len(bus), 3), dtype=np.float64)
    from pypower.idx_gen import PMIN, PMAX, QMIN, QMAX
    for gen in case["gen"]:
        if gen[GEN_STATUS] <= 0:
            continue
        row = lookup[int(gen[GEN_BUS])]
        gen_static[row] += [1.0, float(gen[PMIN]), float(gen[PMAX])]
    x = np.column_stack((bus[:, PD], bus[:, QD], gen_static))
    y = np.column_stack((solved["bus"][:, LAM_P], solved["bus"][:, VM]))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite feature or target")
    return {"x": x, "y": y, "objective": float(solved["f"])}


def generate_split(base: dict, n: int, seed: int, load_sigma: float,
                   tolerance: float = 1e-6, max_attempt_factor: int = 10) -> dict:
    """Generate n feasible OPF-labeled samples, tracking unsuccessful attempts."""
    if n < 1 or max_attempt_factor < 1:
        raise ValueError("n and max_attempt_factor must be positive")
    rng = np.random.default_rng(seed)
    examples, attempts = [], 0
    while len(examples) < n and attempts < n * max_attempt_factor:
        attempts += 1
        case = sample_operating_case(base, rng, load_sigma)
        try:
            solved = solve_ac_opf(case)
            examples.append(build_example(case, solved, tolerance))
        except (ValueError, RuntimeError, ArithmeticError):
            continue
    if len(examples) < n:
        raise RuntimeError(f"Only {len(examples)}/{n} feasible labels after {attempts} attempts")
    return {"x": np.stack([e["x"] for e in examples]),
            "y": np.stack([e["y"] for e in examples]),
            "objective": np.array([e["objective"] for e in examples]),
            "attempts": attempts, "seed": seed}
