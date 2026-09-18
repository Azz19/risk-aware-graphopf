from __future__ import annotations

import numpy as np
from pypower.idx_brch import PF, PT, QF, QT, RATE_A
from pypower.idx_bus import VM, VMAX, VMIN
from pypower.idx_gen import GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN


def evaluate_constraints(result: dict, tolerance: float = 1e-6) -> dict:
    bus = result["bus"]
    gen = result["gen"]
    branch = result["branch"]

    v_hi = np.maximum(bus[:, VM] - bus[:, VMAX], 0.0)
    v_lo = np.maximum(bus[:, VMIN] - bus[:, VM], 0.0)
    max_voltage = float(np.maximum(v_hi, v_lo).max(initial=0.0))

    active = gen[:, GEN_STATUS] > 0
    p_violation = np.maximum.reduce(
        [gen[:, PG] - gen[:, PMAX], gen[:, PMIN] - gen[:, PG], np.zeros(gen.shape[0])]
    )
    q_violation = np.maximum.reduce(
        [gen[:, QG] - gen[:, QMAX], gen[:, QMIN] - gen[:, QG], np.zeros(gen.shape[0])]
    )
    max_p = float(p_violation[active].max(initial=0.0))
    max_q = float(q_violation[active].max(initial=0.0))

    rate = branch[:, RATE_A]
    constrained = rate > 0
    sf = np.hypot(branch[:, PF], branch[:, QF])
    st = np.hypot(branch[:, PT], branch[:, QT])
    overload = np.zeros(branch.shape[0])
    overload[constrained] = np.maximum(
        np.maximum(sf[constrained], st[constrained]) / rate[constrained] - 1.0, 0.0
    )
    max_thermal = float(overload.max(initial=0.0))

    joint = max(max_voltage, max_p, max_q, max_thermal) <= tolerance
    return {
        "max_voltage_violation_pu": max_voltage,
        "max_pg_violation_mw": max_p,
        "max_qg_violation_mvar": max_q,
        "max_thermal_overload_pu": max_thermal,
        "operational_feasible": bool(joint),
    }
