"""E13: diagnose E12 voltage failures without tuning on held-out data.

Reconstruct E12 coefficients and the same E12 held-out draws. Report the
zero-error intercept effect, predicted setpoint changes, and bus-level
voltage violations. No new controller is selected in this audit.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from graphopf.experiments import apply_scenario, generate_errors, renewable_forecast, run_pf_scenario, select_renewable_buses
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, BETA_P, BETA_Q, DELTA_V
from run_distribution_shift import FAMILIES, FROZEN_OFFSETS
from run_linear_voltage import TEST_SEED, effective_error, fit_labels, fit_ridge, predict_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--labels", default="results/E11_control_ablation/generator_controls.csv")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    if args.n_test < 1 or args.alpha <= 0:
        ap.error("n-test must be positive and alpha > 0")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for b, p in zip(rb, forecast):
        physical["bus"][lookup[int(b)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    frozen = adjusted(restore_physical_limits(tight, physical), physical, FROZEN_OFFSETS)
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    corr = exponential_correlation(topological_distance(raw, rb),
                                   float(cfg["uncertainty"]["correlation_length"]))
    df = float(cfg["uncertainty"]["student_df"])
    active = np.flatnonzero(physical["gen"][:, GEN_STATUS] > 0)
    x, y, _ = fit_labels(args.labels, forecast, std, corr, df, active)
    coeff = fit_ridge(x, y, args.alpha)
    out = Path("results/E13_voltage_audit")
    out.mkdir(parents=True, exist_ok=True)
    # Zero-error scenario has no renewable mismatch. Any predicted correction
    # comes from the fitted intercept and may alter nominal feasibility.
    zero = np.zeros_like(forecast)
    zero_policy = predict_policy(frozen, physical, active, coeff, zero)
    zero_pf, zero_ok = run_pf_scenario(apply_scenario(zero_policy, rb, forecast, zero))
    zero_metrics = evaluate_constraints(zero_pf, tol) if zero_ok else {}
    zero_row = dict(pf_converged=bool(zero_ok),
                    operational_feasible=bool(zero_ok and zero_metrics["operational_feasible"]),
                    **{k: float(v) for k, v in zero_metrics.items()
                       if k.startswith("max_") and np.isscalar(v)})
    pd.DataFrame([zero_row]).to_csv(out / "zero_error.csv", index=False)
    print(f"E13 zero-error learned policy: {zero_row}", flush=True)
    bus_ids = physical["bus"][:, BUS_I].astype(int)
    bus_rows, gen_rows, family_rows = [], [], []
    for j, family in enumerate(FAMILIES):
        errors = generate_errors(family, np.random.default_rng(TEST_SEED+j),
                                 args.n_test, std, corr, df)
        voltage_counts = np.zeros(len(bus_ids), dtype=int)
        max_excess = np.zeros(len(bus_ids))
        sum_excess = np.zeros(len(bus_ids))
        q_count = 0
        voltage_scenario_count = 0
        gen_delta = []
        for i, error in enumerate(errors):
            policy = predict_policy(frozen, physical, active, coeff,
                                    effective_error(error, forecast) / std)
            gen_delta.append(policy["gen"][active, VG] - frozen["gen"][active, VG])
            result, ok = run_pf_scenario(apply_scenario(policy, rb, forecast, error))
            if not ok:
                continue
            vm = result["bus"][:, VM]
            excess = np.maximum(vm - physical["bus"][:, VMAX], 0) + np.maximum(
                physical["bus"][:, VMIN] - vm, 0)
            violation = excess > tol
            voltage_counts += violation.astype(int)
            max_excess = np.maximum(max_excess, excess)
            sum_excess += excess
            voltage_scenario_count += int(np.any(violation))
            q_count += int(evaluate_constraints(result, tol)["max_qg_violation_mvar"] > tol)
        for b, n, mx, sm in zip(bus_ids, voltage_counts, max_excess, sum_excess):
            bus_rows.append(dict(family=family, bus_id=int(b),
                                 violation_count=int(n), violation_rate=n/args.n_test,
                                 max_excess_pu=float(mx),
                                 mean_excess_pu=float(sm/args.n_test)))
        changes = np.asarray(gen_delta)
        for idx, g in enumerate(active):
            gen_rows.append(dict(family=family, generator_row=int(g),
                                 bus_id=int(physical["gen"][g, GEN_BUS]),
                                 intercept_pu=float(coeff[0, idx]),
                                 zero_error_delta_pu=float(zero_policy["gen"][g, VG] -
                                                           frozen["gen"][g, VG]),
                                 mean_abs_delta_pu=float(np.mean(np.abs(changes[:, idx]))),
                                 max_abs_delta_pu=float(np.max(np.abs(changes[:, idx]))),
                                 clip_fraction=float(np.mean(
                                     np.abs(changes[:, idx] -
                                            (np.column_stack((np.ones(args.n_test),
                                             np.asarray([effective_error(e, forecast)/std
                                                         for e in errors]))) @ coeff)[:, idx]) > 1e-9))))
        family_rows.append(dict(family=family, n=args.n_test,
                                voltage_scenario_count=voltage_scenario_count,
                                q_scenario_count=q_count))
        print(f"E13 {family}: voltage_scenarios={voltage_scenario_count}/{args.n_test}; "
              f"Q_scenarios={q_count}/{args.n_test}", flush=True)
    pd.DataFrame(bus_rows).to_csv(out / "bus_voltage.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(out / "generator_adjustments.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out / "summary.csv", index=False)
    print("Top voltage-violation buses by family:", flush=True)
    frame = pd.DataFrame(bus_rows)
    for family in FAMILIES:
        top = frame[frame.family == family].nlargest(5, "violation_count")
        print(top[["bus_id", "violation_count", "max_excess_pu"]].to_string(index=False),
              flush=True)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
