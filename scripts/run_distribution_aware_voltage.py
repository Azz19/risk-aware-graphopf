"""E09: distribution-aware conventional voltage baseline, with independent testing.

Select a nominally feasible fixed voltage policy on four uncertainty families
by minimizing the worst empirical joint violation rate; freeze before testing.
No GNN, adaptive recourse, or optimization on test scenarios.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD
from graphopf.experiments import (
    apply_scenario, generate_errors, renewable_forecast, run_pf_scenario,
    select_renewable_buses,
)
from graphopf.metrics import wilson_interval
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, evaluate, BETA_P, BETA_Q, DELTA_V
from run_distribution_shift import FAMILIES, FROZEN_OFFSETS

OFFSETS = (0.0, 0.00125, 0.0025)
TRAIN_SEED = 20270221
TEST_SEED = 20270321


def nominal_feasible(policy, rb, forecast, tol):
    result, ok = run_pf_scenario(
        apply_scenario(policy, rb, forecast, np.zeros_like(forecast)))
    return bool(ok and evaluate_constraints(result, tol)["operational_feasible"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--n-train", type=int, default=300,
                    help="Number of tuning scenarios PER uncertainty family")
    ap.add_argument("--n-test", type=int, default=2000,
                    help="Number of independent test scenarios PER family")
    args = ap.parse_args()
    if args.n_train < 1 or args.n_test < 1:
        ap.error("Sample counts must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    df = float(cfg["uncertainty"]["student_df"])
    corr = exponential_correlation(
        topological_distance(raw, rb),
        float(cfg["uncertainty"]["correlation_length"]))
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for bus_id, p in zip(rb, forecast):
        physical["bus"][lookup[int(bus_id)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    baseline = restore_physical_limits(tight, physical)
    frozen_e07 = adjusted(baseline, physical, FROZEN_OFFSETS)
    if not nominal_feasible(baseline, rb, forecast, tol):
        raise RuntimeError("E04 baseline fails zero-error physical feasibility")
    if not nominal_feasible(frozen_e07, rb, forecast, tol):
        raise RuntimeError("Frozen E07 fails zero-error physical feasibility")

    # All candidates see the SAME training scenarios within each family.
    train = {
        family: generate_errors(
            family, np.random.default_rng(TRAIN_SEED + j),
            args.n_train, std, corr, df)
        for j, family in enumerate(FAMILIES)
    }
    grid = list(itertools.product(OFFSETS, OFFSETS))
    records = []
    policies = {}
    print(f"E09: {len(grid)} fixed policies; {args.n_train} tuning scenarios "
          f"per family; training seeds {TRAIN_SEED}..{TRAIN_SEED+3}", flush=True)
    for i, (d8, d12) in enumerate(grid, start=1):
        key = (d8, d12)
        try:
            policy = adjusted(baseline, physical, {8: d8, 12: d12})
            ok = nominal_feasible(policy, rb, forecast, tol)
        except (ValueError, RuntimeError):
            ok = False
        row = {"delta_v8": d8, "delta_v12": d12, "nominal_feasible": ok}
        if ok:
            policies[key] = policy
            rates = []
            for family in FAMILIES:
                rate = evaluate(policy, rb, forecast, train[family], tol)["joint_rate"]
                row[f"{family}_train_joint_rate"] = rate
                rates.append(rate)
            row["worst_train_joint_rate"] = max(rates)
            row["mean_train_joint_rate"] = float(np.mean(rates))
            print(f"candidate {i}/{len(grid)} offsets={key} "
                  f"worst_train={max(rates):.4f} mean_train={np.mean(rates):.4f} "
                  f"per_family={[round(x, 4) for x in rates]}", flush=True)
        else:
            row["worst_train_joint_rate"] = np.nan
            row["mean_train_joint_rate"] = np.nan
            print(f"candidate {i}/{len(grid)} offsets={key} nominal infeasible", flush=True)
        records.append(row)

    out = Path("results/E09_distribution_aware_voltage")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "tuning.csv", index=False)
    eligible = frame[frame["nominal_feasible"]].sort_values(
        ["worst_train_joint_rate", "mean_train_joint_rate",
         "delta_v8", "delta_v12"])
    if eligible.empty:
        raise RuntimeError("No nominally feasible E09 candidate; test not run")
    chosen = eligible.iloc[0]
    key = (float(chosen["delta_v8"]), float(chosen["delta_v12"]))
    print(f"SELECTED ON TRAIN ONLY: offsets={key} "
          f"worst_train={chosen['worst_train_joint_rate']:.4f} "
          f"mean_train={chosen['mean_train_joint_rate']:.4f}", flush=True)

    # Generate fresh held-out scenarios only AFTER the tuning choice is frozen.
    # Compare all three policies on the same draws within each family.
    results = []
    for j, family in enumerate(FAMILIES):
        seed = TEST_SEED + j
        errors = generate_errors(
            family, np.random.default_rng(seed), args.n_test, std, corr, df)
        print(f"HELD-OUT {family}: n={args.n_test} seed={seed}", flush=True)
        for name, policy in (
            ("e04_baseline", baseline),
            ("frozen_e07", frozen_e07),
            ("selected_e09", policies[key]),
        ):
            rates = evaluate(policy, rb, forecast, errors, tol)
            count = int(round(rates["joint_rate"] * args.n_test))
            lo, hi = wilson_interval(count, args.n_test, confidence)
            results.append({
                "family": family, "policy": name, "seed": seed,
                "n": args.n_test, "joint_count": count,
                "delta_v8": 0.0 if name == "e04_baseline" else (
                    FROZEN_OFFSETS[8] if name == "frozen_e07" else key[0]),
                "delta_v12": 0.0 if name == "e04_baseline" else (
                    FROZEN_OFFSETS[12] if name == "frozen_e07" else key[1]),
                **rates, "joint_ci_low": lo, "joint_ci_high": hi,
                "meets_5pct_upper_ci": bool(hi <= 0.05),
            })
            print(f"  {name}: joint={rates['joint_rate']:.4f} "
                  f"CI=[{lo:.4f}, {hi:.4f}] q={rates['qg_rate']:.4f} "
                  f"v={rates['voltage_rate']:.4f} pg={rates['pg_rate']:.4f} "
                  f"thermal={rates['thermal_rate']:.4f} "
                  f"pf_fail={rates['pf_fail_rate']:.4f}", flush=True)
        pd.DataFrame(results).to_csv(out / "held_out_test.csv", index=False)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
