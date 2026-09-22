"""E10: random corrective AC-OPF feasibility diagnostic on fresh scenarios.

Perfect-information OPF is NOT deployable recourse; solver failure does not
prove infeasibility. Sample violating scenarios uniformly without replacement.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG
from graphopf.experiments import (
    apply_scenario, generate_errors, renewable_forecast, run_pf_scenario,
    select_renewable_buses,
)
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, BETA_P, BETA_Q, DELTA_V
from run_distribution_shift import FAMILIES, FROZEN_OFFSETS

SEED = 20270421
SELECTION_SEED = 20270491


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Fresh scenarios per family")
    ap.add_argument("--max-corrective", type=int, default=30,
                    help="Random violating scenarios per family")
    args = ap.parse_args()
    if args.n_samples < 1 or args.max_corrective < 1:
        ap.error("n-samples and max-corrective must be positive")
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
    baseline = restore_physical_limits(tight, physical)
    policy = adjusted(baseline, physical, FROZEN_OFFSETS)
    zero, ok = run_pf_scenario(apply_scenario(policy, rb, forecast, np.zeros_like(forecast)))
    if not ok or not evaluate_constraints(zero, tol)["operational_feasible"]:
        raise RuntimeError("Frozen E07 fails zero-error feasibility")
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    corr = exponential_correlation(
        topological_distance(raw, rb), float(cfg["uncertainty"]["correlation_length"]))
    df = float(cfg["uncertainty"]["student_df"])
    active = np.flatnonzero(physical["gen"][:, GEN_STATUS] > 0)
    out = Path("results/E10_corrective_feasibility")
    out.mkdir(parents=True, exist_ok=True)
    rows, generator_rows, family_rows = [], [], []
    for j, family in enumerate(FAMILIES):
        errors = generate_errors(family, np.random.default_rng(SEED+j),
                                 args.n_samples, std, corr, df)
        violating = []
        pf_fail = 0
        for i, error in enumerate(errors):
            scenario = apply_scenario(policy, rb, forecast, error)
            result, success = run_pf_scenario(scenario)
            if not success:
                pf_fail += 1
                violating.append(i)
            elif not evaluate_constraints(result, tol)["operational_feasible"]:
                violating.append(i)
        n_select = min(args.max_corrective, len(violating))
        selected = np.random.default_rng(SELECTION_SEED+j).choice(
            violating, size=n_select, replace=False) if n_select else []
        print(f"E10 {family}: seed={SEED+j}, n={args.n_samples}, "
              f"violating={len(violating)}, pf_fail={pf_fail}, "
              f"randomly_selected={n_select}", flush=True)
        statuses = {}
        for k, i in enumerate(selected, start=1):
            i = int(i)
            scenario = apply_scenario(policy, rb, forecast, errors[i])
            original, success = run_pf_scenario(scenario)
            diagnostic = copy.deepcopy(scenario)
            # Restore original physical generator and bus bounds. Realized
            # net injections remain fixed; OPF can redispatch all generators.
            for key in ("f", "success", "et", "order", "raw", "mu"):
                diagnostic.pop(key, None)
            for col in (8, 9):  # PMAX, PMIN: explicit original physical limits
                diagnostic["gen"][:, col] = physical["gen"][:, col]
            for col in (3, 4):  # QMAX, QMIN
                diagnostic["gen"][:, col] = physical["gen"][:, col]
            for col in (11, 12):  # VMAX, VMIN
                diagnostic["bus"][:, col] = physical["bus"][:, col]
            status = "solver_failed_not_proven_infeasible"
            corrected = None
            try:
                corrected = solve_ac_opf(diagnostic)
                status = ("solved_physical_feasible"
                          if evaluate_constraints(corrected, tol)["operational_feasible"]
                          else "solved_limits_failed")
            except (RuntimeError, ValueError) as exc:
                print(f"  corrective {k}/{n_select} scenario={i} solver issue: {exc}", flush=True)
            statuses[status] = statuses.get(status, 0) + 1
            row = {"family": family, "scenario_id": i, "status": status,
                   "original_pf_converged": bool(success),
                   "corrective_objective": float(corrected["f"]) if corrected is not None else np.nan}
            if corrected is not None and success:
                p_diff = corrected["gen"][active, PG] - original["gen"][active, PG]
                q_diff = corrected["gen"][active, QG] - original["gen"][active, QG]
                v_diff = corrected["gen"][active, VG] - policy["gen"][active, VG]
                row.update(max_abs_delta_pg_mw=float(np.max(np.abs(p_diff))),
                           max_abs_delta_qg_mvar=float(np.max(np.abs(q_diff))),
                           max_abs_delta_vg_pu=float(np.max(np.abs(v_diff))),
                           sum_abs_delta_pg_mw=float(np.sum(np.abs(p_diff))),
                           sum_abs_delta_vg_pu=float(np.sum(np.abs(v_diff))))
                for g in active:
                    generator_rows.append({
                        "family": family, "scenario_id": i,
                        "generator_row_zero_based": int(g),
                        "bus_id": int(physical["gen"][g, GEN_BUS]),
                        "original_pg_mw": float(original["gen"][g, PG]),
                        "corrected_pg_mw": float(corrected["gen"][g, PG]),
                        "original_qg_mvar": float(original["gen"][g, QG]),
                        "corrected_qg_mvar": float(corrected["gen"][g, QG]),
                        "policy_vg_pu": float(policy["gen"][g, VG]),
                        "corrected_vg_pu": float(corrected["gen"][g, VG]),
                    })
            rows.append(row)
            print(f"  corrective {k}/{n_select} scenario={i} status={status} "
                  f"max_abs_delta_vg={row.get('max_abs_delta_vg_pu', float('nan')):.5f}",
                  flush=True)
        family_rows.append({"family": family, "n": args.n_samples,
                            "violating_count": len(violating),
                            "pf_failure_count": pf_fail,
                            "random_corrective_count": n_select,
                            **statuses})
        pd.DataFrame(rows).to_csv(out / "corrective_scenarios.csv", index=False)
        pd.DataFrame(generator_rows).to_csv(out / "generator_controls.csv", index=False)
        pd.DataFrame(family_rows).fillna(0).to_csv(out / "summary.csv", index=False)
    print("E10 corrective OPF is a perfect-information feasibility diagnostic, "
          "not deployable recourse; failed solves are not infeasibility proofs.", flush=True)
    print(pd.DataFrame(family_rows).fillna(0).to_string(index=False), flush=True)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
