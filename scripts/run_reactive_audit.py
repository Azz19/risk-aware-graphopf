"""E05: reactive-limit audit and perfect-information corrective-OPF diagnostic.

Run from repository root. Corrective OPF is an offline feasibility diagnostic,
NOT a deployable recourse policy or a cost-comparable baseline.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.api import ppoption, runpf
from pypower.idx_bus import BUS_I, PD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, QG, QMAX, QMIN
from graphopf.experiments import (
    apply_scenario, generate_errors, renewable_forecast,
    run_pf_scenario, select_renewable_buses,
)
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits, VALIDATION_SEED

BETA_P = 0.025
BETA_Q = 0.10
DELTA_V = 0.0025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--max-corrective", type=int, default=30,
                        help="Number of violating scenarios to test with full-information AC-OPF.")
    args = parser.parse_args()
    if args.n_samples < 1 or args.max_corrective < 0:
        parser.error("n-samples must be positive and max-corrective nonnegative")
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
    opf = restore_physical_limits(tight, physical)
    zero, success = run_pf_scenario(apply_scenario(opf, rb, forecast, np.zeros_like(forecast)))
    if not success or not evaluate_constraints(zero, tol)["operational_feasible"]:
        raise RuntimeError("Zero-error invariant failed; E05 aborted.")

    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    errors = generate_errors("iid_gaussian", np.random.default_rng(VALIDATION_SEED),
                             args.n_samples, std, None, float(cfg["uncertainty"]["student_df"]))
    active = physical["gen"][:, GEN_STATUS] > 0
    gen_ids = np.flatnonzero(active)
    upper_count = np.zeros(len(gen_ids), dtype=int)
    lower_count = np.zeros(len(gen_ids), dtype=int)
    upper_sum = np.zeros(len(gen_ids))
    lower_sum = np.zeros(len(gen_ids))
    upper_max = np.zeros(len(gen_ids))
    lower_max = np.zeros(len(gen_ids))
    scenarios = []
    corrective = []
    attempted = 0
    joint = q_events = failed = 0
    print(f"E05 start: n={args.n_samples}, seed={VALIDATION_SEED}, "
          f"beta_p={BETA_P}, beta_q={BETA_Q}, delta_v={DELTA_V}", flush=True)
    print("Standard runpf uses ENFORCE_Q_LIMS=False (default); E05 does not "
          "switch PV buses to PQ. Corrective OPF is a separate diagnostic.", flush=True)

    for k, e in enumerate(errors):
        scenario = apply_scenario(opf, rb, forecast, e)
        result, converged = run_pf_scenario(scenario)
        if not converged:
            failed += 1
            joint += 1
            scenarios.append({"scenario": k, "pf_converged": False,
                              "joint_violation": True, "q_violation": False,
                              "max_q_upper_mvar": np.nan, "max_q_lower_mvar": np.nan})
            continue
        m = evaluate_constraints(result, tol)
        joint += int(not m["operational_feasible"])
        q = result["gen"][gen_ids, QG]
        upper = np.maximum(q - physical["gen"][gen_ids, QMAX], 0.0)
        lower = np.maximum(physical["gen"][gen_ids, QMIN] - q, 0.0)
        hi = upper > tol
        lo = lower > tol
        q_event = bool(hi.any() or lo.any())
        q_events += int(q_event)
        upper_count += hi
        lower_count += lo
        upper_sum += upper
        lower_sum += lower
        upper_max = np.maximum(upper_max, upper)
        lower_max = np.maximum(lower_max, lower)
        scenarios.append({
            "scenario": k, "pf_converged": True,
            "joint_violation": not m["operational_feasible"],
            "q_violation": q_event, "max_q_upper_mvar": upper.max(initial=0.0),
            "max_q_lower_mvar": lower.max(initial=0.0),
            "max_voltage_violation_pu": m["max_voltage_violation_pu"],
            "max_pg_violation_mw": m["max_pg_violation_mw"],
            "max_thermal_overload_pu": m["max_thermal_overload_pu"],
        })

        if q_event and attempted < args.max_corrective:
            attempted += 1
            # Scenario contains realized net injections. Remove OPF result fields
            # and use ORIGINAL physical constraints; no backoff or fixed AGC.
            # This allows perfect-information redispatch and voltage controls.
            diagnostic = copy.deepcopy(scenario)
            for key in ("f", "success", "et", "order", "raw", "mu"):
                diagnostic.pop(key, None)
            diagnostic["gen"][:, QMIN] = physical["gen"][:, QMIN]
            diagnostic["gen"][:, QMAX] = physical["gen"][:, QMAX]
            try:
                corrected = solve_ac_opf(diagnostic)
                feasible = evaluate_constraints(corrected, tol)["operational_feasible"]
                status = "solved_physical_feasible" if feasible else "solved_limits_failed"
                corrected_cost = float(corrected["f"])
            except (RuntimeError, ValueError) as exc:
                status = "solver_failed_not_proven_infeasible"
                corrected_cost = np.nan
            corrective.append({"scenario": k, "status": status,
                               "corrective_objective": corrected_cost})
            print(f"corrective {attempted}/{args.max_corrective}: "
                  f"scenario={k} status={status}", flush=True)
        if (k + 1) % 250 == 0 or k + 1 == args.n_samples:
            print(f"progress {k+1}/{args.n_samples}, joint={joint}, "
                  f"q_events={q_events}, pf_failures={failed}", flush=True)

    per_gen = pd.DataFrame({
        "generator_row_zero_based": gen_ids,
        "bus_id": physical["gen"][gen_ids, GEN_BUS].astype(int),
        "upper_violation_count": upper_count,
        "lower_violation_count": lower_count,
        "upper_violation_rate": upper_count / args.n_samples,
        "lower_violation_rate": lower_count / args.n_samples,
        "mean_upper_excess_mvar_all_scenarios": upper_sum / args.n_samples,
        "mean_lower_excess_mvar_all_scenarios": lower_sum / args.n_samples,
        "max_upper_excess_mvar": upper_max,
        "max_lower_excess_mvar": lower_max,
    }).sort_values(["upper_violation_count", "lower_violation_count"], ascending=False)
    out = Path("results/E05_reactive_audit")
    out.mkdir(parents=True, exist_ok=True)
    per_gen.to_csv(out / "per_generator.csv", index=False)
    pd.DataFrame(scenarios).to_csv(out / "scenario_audit.csv", index=False)
    pd.DataFrame(corrective, columns=["scenario", "status", "corrective_objective"]).to_csv(
        out / "corrective_diagnostic.csv", index=False)
    print("\nE05 SUMMARY", flush=True)
    print(f"joint_rate={joint/args.n_samples:.6f}, "
          f"q_violation_rate={q_events/args.n_samples:.6f}, "
          f"pf_nonconvergence_rate={failed/args.n_samples:.6f}", flush=True)
    print(per_gen.to_string(index=False), flush=True)
    print("\nCORRECTIVE OPF: perfect-information diagnostic, NOT operational recourse", flush=True)
    print(pd.DataFrame(corrective).groupby("status").size().to_string() if corrective
          else "No corrective cases attempted.", flush=True)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
