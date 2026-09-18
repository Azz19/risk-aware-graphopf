from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from graphopf.experiments import (
    apply_scenario,
    generate_errors,
    headroom_participation,
    renewable_forecast,
    run_pf_scenario,
    select_renewable_buses,
)
from graphopf.metrics import wilson_interval
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n = args.n_samples or int(cfg["experiment"]["n_samples"])
    seed = int(cfg["experiment"]["seed"])
    outdir = Path(cfg["experiment"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_case(cfg["case"]["path"])
    # Baseline OPF includes the forecast renewable injection.
    buses = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, buses, float(cfg["renewables"]["penetration"]))
    forecast_case = raw.copy()
    forecast_case["bus"] = raw["bus"].copy()
    lookup = {int(row[0]): i for i, row in enumerate(forecast_case["bus"])}
    for b, p in zip(buses, forecast):
        forecast_case["bus"][lookup[int(b)], 2] -= p

    opf = solve_ac_opf(forecast_case)
    participation = headroom_participation(opf)
    distance = topological_distance(raw, buses)
    corr = exponential_correlation(distance, float(cfg["uncertainty"]["correlation_length"]))
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    df = float(cfg["uncertainty"]["student_df"])

    # Invariant: zero renewable forecast error must reproduce a feasible
    # forecast operating point (up to the configured numerical tolerance).
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    zero_scenario = apply_scenario(
        opf, buses, forecast, np.zeros_like(forecast), participation
    )
    zero_result, zero_converged = run_pf_scenario(zero_scenario)
    if not zero_converged:
        raise RuntimeError("Zero-error invariant failed: AC power flow did not converge")
    zero_metrics = evaluate_constraints(zero_result, tol)
    print("Zero-error invariant:")
    print(json.dumps(zero_metrics, indent=2))
    if not zero_metrics["operational_feasible"]:
        raise RuntimeError(
            "Zero-error invariant failed: forecast operating point is classified "
            "as infeasible after AC power flow. Do not run Monte Carlo until fixed."
        )

    all_rows = []
    summaries = []
    for family_index, family in enumerate(cfg["uncertainty"]["distributions"]):
        rng = np.random.default_rng(seed + family_index)
        errors = generate_errors(family, rng, n, std, corr, df)
        violations = 0
        nonconverged = 0
        rows = []
        for s, error in enumerate(errors):
            scenario = apply_scenario(opf, buses, forecast, error, participation)
            result, converged = run_pf_scenario(scenario)
            if converged:
                m = evaluate_constraints(result, tol)
            else:
                nonconverged += 1
                m = {
                    "max_voltage_violation_pu": np.nan,
                    "max_pg_violation_mw": np.nan,
                    "max_qg_violation_mvar": np.nan,
                    "max_thermal_overload_pu": np.nan,
                    "operational_feasible": False,
                }
            joint_violation = not (converged and m["operational_feasible"])
            violations += int(joint_violation)
            rows.append({
                "family": family,
                "scenario_id": s,
                "pf_converged": converged,
                "joint_violation": joint_violation,
                **m,
            })
        all_rows.extend(rows)
        rate = violations / n
        lo, hi = wilson_interval(violations, n, float(cfg["evaluation"]["confidence_level"]))
        frame = pd.DataFrame(rows)
        summaries.append({
            "family": family,
            "n": n,
            "joint_violation_rate": rate,
            "ci_low": lo,
            "ci_high": hi,
            "voltage_violation_rate": float((frame["max_voltage_violation_pu"] > tol).mean()),
            "pg_violation_rate": float((frame["max_pg_violation_mw"] > tol).mean()),
            "qg_violation_rate": float((frame["max_qg_violation_mvar"] > tol).mean()),
            "thermal_violation_rate": float((frame["max_thermal_overload_pu"] > tol).mean()),
            "mean_voltage_violation_pu": float(frame["max_voltage_violation_pu"].mean()),
            "mean_pg_violation_mw": float(frame["max_pg_violation_mw"].mean()),
            "mean_qg_violation_mvar": float(frame["max_qg_violation_mvar"].mean()),
            "mean_thermal_overload_pu": float(frame["max_thermal_overload_pu"].mean()),
            "pf_nonconvergence_rate": nonconverged / n,
        })

    pd.DataFrame(all_rows).to_csv(outdir / "scenarios.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
    metadata = {
        "seed": seed,
        "renewable_buses": buses.tolist(),
        "renewable_forecast_mw": forecast.tolist(),
        "participation_factors": participation.tolist(),
        "zero_error_invariant": zero_metrics,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
