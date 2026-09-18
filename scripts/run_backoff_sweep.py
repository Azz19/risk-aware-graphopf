from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_gen import GEN_STATUS, PG, PMAX, PMIN

from graphopf.experiments import (
    apply_symmetric_generator_backoff,
    renewable_forecast,
    select_renewable_buses,
)
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf

BETAS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = load_case(cfg["case"]["path"])
    buses = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, buses, float(cfg["renewables"]["penetration"]))
    forecast_case = raw.copy()
    forecast_case["bus"] = raw["bus"].copy()
    lookup = {int(row[0]): i for i, row in enumerate(forecast_case["bus"])}
    for b, p in zip(buses, forecast):
        forecast_case["bus"][lookup[int(b)], 2] -= p

    base = solve_ac_opf(forecast_case)
    base_cost = float(base["f"])
    rows = []

    for beta in BETAS:
        tightened = apply_symmetric_generator_backoff(forecast_case, beta)
        try:
            result = solve_ac_opf(tightened)
            solved = True
        except RuntimeError:
            rows.append({"beta": beta, "solved": False})
            continue

        active = result["gen"][:, GEN_STATUS] > 0
        physical_width = forecast_case["gen"][:, PMAX] - forecast_case["gen"][:, PMIN]
        flexible = active & (physical_width > 1e-6)
        pg = result["gen"][:, PG]
        tightened_pmin = result["gen"][:, PMIN]
        tightened_pmax = result["gen"][:, PMAX]
        physical_pmin = forecast_case["gen"][:, PMIN]
        physical_pmax = forecast_case["gen"][:, PMAX]

        tight_up = tightened_pmax - pg
        tight_down = pg - tightened_pmin
        physical_up = physical_pmax - pg
        physical_down = pg - physical_pmin

        # Evaluate the forecast dispatch against the real equipment limits.
        physical_result = result.copy()
        physical_result["gen"] = result["gen"].copy()
        physical_result["gen"][:, PMIN] = physical_pmin
        physical_result["gen"][:, PMAX] = physical_pmax
        metrics = evaluate_constraints(physical_result)

        rows.append({
            "beta": beta,
            "solved": solved,
            "objective": float(result["f"]),
            "cost_increase_pct": 100.0 * (float(result["f"]) - base_cost) / base_cost,
            "active_generators": int(active.sum()),
            "flexible_generators": int(flexible.sum()),
            "fixed_range_generators": int((active & ~flexible).sum()),
            "min_physical_up_reserve_mw": float(physical_up[flexible].min()) if flexible.any() else np.nan,
            "min_physical_down_reserve_mw": float(physical_down[flexible].min()) if flexible.any() else np.nan,
            "total_physical_up_reserve_mw": float(physical_up[flexible].sum()),
            "total_physical_down_reserve_mw": float(physical_down[flexible].sum()),
            "min_tightened_up_headroom_mw": float(tight_up[active].min()),
            "min_tightened_down_headroom_mw": float(tight_down[active].min()),
            "generators_at_tightened_pmin": int(np.sum(active & (tight_down <= 1e-6))),
            "generators_at_tightened_pmax": int(np.sum(active & (tight_up <= 1e-6))),
            "operational_feasible": bool(metrics["operational_feasible"]),
        })

    out = Path("results/E03_backoff")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "backoff_sweep.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
