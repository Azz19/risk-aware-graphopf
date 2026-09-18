from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, VMAX, VMIN
from pypower.idx_gen import GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN

from graphopf.experiments import renewable_forecast, select_renewable_buses
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf

BETA_P = 0.025
BETA_Q_VALUES = [0.0, 0.01, 0.025, 0.05, 0.10]
DELTA_V_VALUES = [0.0, 0.0025, 0.005, 0.010, 0.015]


def apply_pqv_backoff(ppc: dict, beta_p: float, beta_q: float, delta_v: float) -> dict:
    out = copy.deepcopy(ppc)
    gen = out["gen"]
    active = gen[:, GEN_STATUS] > 0

    p_width = gen[:, PMAX] - gen[:, PMIN]
    q_width = gen[:, QMAX] - gen[:, QMIN]
    p_flexible = active & (p_width > 1e-9)
    q_flexible = active & (q_width > 1e-9)

    gen[p_flexible, PMIN] += beta_p * p_width[p_flexible]
    gen[p_flexible, PMAX] -= beta_p * p_width[p_flexible]
    gen[q_flexible, QMIN] += beta_q * q_width[q_flexible]
    gen[q_flexible, QMAX] -= beta_q * q_width[q_flexible]

    bus = out["bus"]
    if np.any(bus[:, VMIN] + delta_v >= bus[:, VMAX] - delta_v):
        raise ValueError("Voltage backoff collapses a bus voltage interval")
    bus[:, VMIN] += delta_v
    bus[:, VMAX] -= delta_v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = load_case(cfg["case"]["path"])
    renewable_buses = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, renewable_buses, float(cfg["renewables"]["penetration"]))
    forecast_case = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(forecast_case["bus"])}
    for b, p in zip(renewable_buses, forecast):
        forecast_case["bus"][lookup[int(b)], 2] -= p

    base = solve_ac_opf(forecast_case)
    base_cost = float(base["f"])
    rows = []

    for beta_q, delta_v in itertools.product(BETA_Q_VALUES, DELTA_V_VALUES):
        row = {"beta_p": BETA_P, "beta_q": beta_q, "delta_v_pu": delta_v}
        try:
            tightened = apply_pqv_backoff(forecast_case, BETA_P, beta_q, delta_v)
            result = solve_ac_opf(tightened)
        except (RuntimeError, ValueError) as exc:
            row.update({"solved": False, "failure": str(exc)})
            rows.append(row)
            continue

        active = result["gen"][:, GEN_STATUS] > 0
        p_up = forecast_case["gen"][:, PMAX] - result["gen"][:, PG]
        p_down = result["gen"][:, PG] - forecast_case["gen"][:, PMIN]
        q_up = forecast_case["gen"][:, QMAX] - result["gen"][:, QG]
        q_down = result["gen"][:, QG] - forecast_case["gen"][:, QMIN]
        qflex = active & ((forecast_case["gen"][:, QMAX] - forecast_case["gen"][:, QMIN]) > 1e-9)

        vm = result["bus"][:, 7]
        physical_v_low = vm - forecast_case["bus"][:, VMIN]
        physical_v_high = forecast_case["bus"][:, VMAX] - vm
        physical_eval = copy.deepcopy(result)
        physical_eval["gen"][:, PMIN] = forecast_case["gen"][:, PMIN]
        physical_eval["gen"][:, PMAX] = forecast_case["gen"][:, PMAX]
        physical_eval["gen"][:, QMIN] = forecast_case["gen"][:, QMIN]
        physical_eval["gen"][:, QMAX] = forecast_case["gen"][:, QMAX]
        physical_eval["bus"][:, VMIN] = forecast_case["bus"][:, VMIN]
        physical_eval["bus"][:, VMAX] = forecast_case["bus"][:, VMAX]
        metrics = evaluate_constraints(physical_eval)

        row.update({
            "solved": True,
            "failure": "",
            "objective": float(result["f"]),
            "cost_increase_pct": 100.0 * (float(result["f"]) - base_cost) / base_cost,
            "min_physical_p_up_mw": float(p_up[active].min()),
            "min_physical_p_down_mw": float(p_down[active].min()),
            "min_physical_q_up_mvar": float(q_up[qflex].min()) if qflex.any() else np.nan,
            "min_physical_q_down_mvar": float(q_down[qflex].min()) if qflex.any() else np.nan,
            "min_physical_v_margin_pu": float(np.minimum(physical_v_low, physical_v_high).min()),
            "operational_feasible_physical": bool(metrics["operational_feasible"]),
        })
        rows.append(row)

    frame = pd.DataFrame(rows)
    out = Path("results/E04_pqv_backoff")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "feasibility_cost_sweep.csv", index=False)
    print(f"base_objective={base_cost:.6f}, beta_p_frozen={BETA_P}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
