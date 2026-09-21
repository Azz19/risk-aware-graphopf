"""E08: frozen E07 policy under IID/correlated Gaussian and Student-t errors.

No policy selection. Same scenarios per family for E04 and E07; fresh seeds.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD
from graphopf.experiments import generate_errors, renewable_forecast, select_renewable_buses
from graphopf.metrics import wilson_interval
from graphopf.powerflow import load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, evaluate, BETA_P, BETA_Q, DELTA_V

SEED = 20270121
FAMILIES = ("iid_gaussian", "corr_gaussian", "iid_student_t", "corr_student_t")
FROZEN_OFFSETS = {8: 0.00125, 12: 0.00125}


def sanity(std, corr, df, n=12000):
    print("E08 sampler sanity (independent diagnostic draws; not evaluation scenarios)", flush=True)
    for j, family in enumerate(FAMILIES):
        e = generate_errors(family, np.random.default_rng(20270101 + j), n, std, corr, df)
        normalized = e / std
        sd = normalized.std(axis=0, ddof=1)
        cc = np.corrcoef(normalized, rowvar=False)
        target = corr if family.startswith("corr_") else np.eye(len(std))
        max_sd_error = float(np.max(np.abs(sd - 1)))
        max_corr_error = float(np.max(np.abs(cc - target)))
        print(f"  {family}: max marginal SD error={max_sd_error:.3f}; "
              f"max Pearson correlation error={max_corr_error:.3f}; "
              f"excess_kurtosis_site0={float(np.mean((normalized[:, 0]-normalized[:, 0].mean())**4) / np.var(normalized[:, 0])**2 - 3):.2f}",
              flush=True)
        if max_sd_error > 0.15 or max_corr_error > 0.15:
            raise RuntimeError(f"Sampler sanity check failed for {family}; inspect draws before evaluation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    if args.n_samples < 1:
        ap.error("--n-samples must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    df = float(cfg["uncertainty"]["student_df"])
    corr = exponential_correlation(
        topological_distance(raw, rb), float(cfg["uncertainty"]["correlation_length"]))
    if np.any(std <= 0):
        raise RuntimeError("Expected positive forecast-error standard deviations")
    sanity(std, corr, df)
    if args.sanity_only:
        return
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for bus_id, p in zip(rb, forecast):
        physical["bus"][lookup[int(bus_id)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    baseline = restore_physical_limits(tight, physical)
    selected = adjusted(baseline, physical, FROZEN_OFFSETS)
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])
    from graphopf.experiments import apply_scenario, run_pf_scenario
    from graphopf.powerflow import evaluate_constraints
    for name, policy in (("baseline", baseline), ("frozen_e07", selected)):
        result, ok = run_pf_scenario(apply_scenario(policy, rb, forecast, np.zeros_like(forecast)))
        if not ok or not evaluate_constraints(result, tol)["operational_feasible"]:
            raise RuntimeError(f"{name} fails zero-error physical feasibility")
    out = Path("results/E08_distribution_shift")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, family in enumerate(FAMILIES):
        seed = SEED + j
        errors = generate_errors(family, np.random.default_rng(seed), args.n_samples, std, corr, df)
        print(f"E08 {family}: n={args.n_samples} seed={seed}", flush=True)
        for name, policy in (("baseline", baseline), ("frozen_e07", selected)):
            rates = evaluate(policy, rb, forecast, errors, tol)
            count = int(round(rates["joint_rate"] * args.n_samples))
            lo, hi = wilson_interval(count, args.n_samples, confidence)
            rows.append({"family": family, "policy": name, "seed": seed,
                         "n": args.n_samples, "joint_count": count, **rates,
                         "joint_ci_low": lo, "joint_ci_high": hi,
                         "meets_5pct_upper_ci": bool(hi <= 0.05)})
            print(f"  {name}: joint={rates['joint_rate']:.4f} "
                  f"CI=[{lo:.4f}, {hi:.4f}] q={rates['qg_rate']:.4f} "
                  f"v={rates['voltage_rate']:.4f} pg={rates['pg_rate']:.4f} "
                  f"thermal={rates['thermal_rate']:.4f} pf_fail={rates['pf_fail_rate']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"CSV output: {out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
