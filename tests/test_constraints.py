import numpy as np
from pypower.idx_brch import RATE_A
from pypower.idx_bus import VM, VMAX

from graphopf.powerflow.constraints import evaluate_constraints


def _result():
    bus = np.zeros((1, 13))
    bus[0, VM] = 1.0
    bus[0, VMAX] = 1.1
    bus[0, 12] = 0.9

    gen = np.zeros((1, 21))
    gen[0, 7] = 1
    gen[0, 1] = 50
    gen[0, 8] = 100
    gen[0, 9] = 0
    gen[0, 2] = 0
    gen[0, 3] = 50
    gen[0, 4] = -50

    branch = np.zeros((1, 17))
    branch[0, RATE_A] = 100
    return {"bus": bus, "gen": gen, "branch": branch}


def test_detects_voltage_violation():
    r = _result()
    r["bus"][0, VM] = 1.2
    m = evaluate_constraints(r)
    assert m["max_voltage_violation_pu"] > 0
    assert not m["operational_feasible"]
