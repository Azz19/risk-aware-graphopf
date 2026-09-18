from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_gen import PMAX, PMIN

from graphopf.experiments import (
    apply_scenario,
    apply_symmetric_generator_backoff,
    generate_errors,
    renewable_forecast,
    run_pf_scenario,
    select_renewable_buses,
)
from graphopf.metrics import wilson_interval
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf

BETAS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]
VALIDATION_SEED = 20261019


def restore_physical_limits(opf: dict, physical_case: dict) -> dict:
    out = copy.deepcopy(opf)
    out["gen"][:, PMIN] = physical_case["gen"][:, PMIN]
    out["gen"][:, PMAX] = physical_case["gen"][:, PMAX]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n = args.n_samples
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])

    raw = load_case(cfg["case"]["path"])
    buses = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, buses, float(cfg["renewables"]["penetration"]))
    forecast_case = copy.deepcopy(raw)
    lookup = {int(row[0]): i for i, row in enumerate(forecast_case["bus"])}
    for b, p in zip(buses, forecast):
        forecast_case["bus"][lookup[int(b)], 2] -= p

    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    # Common random numbers: every beta sees exactly the same IID Gaussian errors.
    errors = generate_errors(
        "iid_gaussian",
        np.random.default_rng(VALIDATION_SEED),
        n,
        std,
        None,
        float(cfg["uncertainty"]["student_df"]),
    )

    base_cost = float(solve_ac_opf(forecast_case)["f"])
    rows = []
    for beta in BETAS:
        try:
            tightened = apply_symmetric_generator_backoff(forecast_case, beta)
            tightened_opf = solve_ac_opf(tightened)
        except RuntimeError:
            rows.append({"beta": beta, "opf_solved": False})
            continue

        opf = restore_physical_limits(tightened_opf, forecast_case)
        zero, ok = run_pf_scenario(apply_scenario(
            opf, buses, forecast, np.zeros_like(forecast), participation=None
        ))
        if not ok or not evaluate_constraints(zero, tol)["operational_feasible"]:
            raise RuntimeError(f"Zero-error invariant failed for beta={beta}")

        joint = voltage = pg = qg = thermal = nonconv = 0
        for error in errors:
            scenario = apply_scenario(opf, buses, forecast, error, participation=None)
            result, converged = run_pf_scenario(scenario)
            if not converged:
                nonconv += 1
                joint += 1
                continue
            m = evaluate_constraints(result, tol)
            joint += int(not m["operational_feasible"])
            voltage += int(m["max_voltage_violation_pu"] > tol)
            pg += int(m["max_pg_violation_mw"] > tol)
            qg += int(m["max_qg_violation_mvar"] > tol)
            thermal += int(m["max_thermal_overload_pu"] > tol)

        lo, hi = wilson_interval(joint, n, confidence)
        rows.append({
            "beta": beta,
            "opf_solved": True,
            "cost_increase_pct": 100.0 * (float(tightened_opf["f"]) - base_cost) / base_cost,
            "joint_violation_rate": joint / n,
            "joint_ci_low": lo,
            "joint_ci_high": hi,
            "meets_5pct_point_estimate": joint / n <= 0.05,
            "meets_5pct_upper_ci": hi <= 0.05,
            "voltage_violation_rate": voltage / n,
            "pg_violation_rate": pg / n,
            "qg_violation_rate": qg / n,
            "thermal_violation_rate": thermal / n,
            "pf_nonconvergence_rate": nonconv / n,
        })

    frame = pd.DataFrame(rows)
    out = Path("results/E03_backoff_validation")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "iid_gaussian_validation.csv", index=False)
    print(f"validation_seed={VALIDATION_SEED}, n={n}, relative_sigma={cfg['uncertainty']['relative_sigma']}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
