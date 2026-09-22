"""E11: matched E10 corrective-control ablation; offline diagnostic only."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, REF, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QMAX, QMIN, VG
from graphopf.experiments import apply_scenario, generate_errors, renewable_forecast, run_pf_scenario, select_renewable_buses
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, BETA_P, BETA_Q, DELTA_V
from run_distribution_shift import FAMILIES, FROZEN_OFFSETS
from run_corrective_feasibility import SEED

BAND_MW = 1e-4


def solve(ppc, tol):
    try:
        result = solve_ac_opf(ppc)
        status = ("feasible" if evaluate_constraints(result, tol)["operational_feasible"]
                  else "solved_limits_failed")
        return result, status
    except (RuntimeError, ValueError):
        return None, "solver_failed_not_proven_infeasible"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--e10-csv", default="results/E10_corrective_feasibility/corrective_scenarios.csv")
    ap.add_argument("--n-samples", type=int, default=1000)
    args = ap.parse_args()
    chosen = pd.read_csv(args.e10_csv)
    if args.n_samples < 1 or chosen.empty:
        raise ValueError("Positive sample count and nonempty E10 selections required")
    if chosen.duplicated(["family", "scenario_id"]).any():
        raise ValueError("Duplicate E10 scenario selection")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    physical = copy.deepcopy(raw)
    lookup = {int(b): i for i, b in enumerate(physical["bus"][:, BUS_I])}
    for b, p in zip(rb, forecast):
        physical["bus"][lookup[int(b)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    policy = adjusted(restore_physical_limits(tight, physical), physical, FROZEN_OFFSETS)
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    corr = exponential_correlation(topological_distance(raw, rb),
                                   float(cfg["uncertainty"]["correlation_length"]))
    df = float(cfg["uncertainty"]["student_df"])
    active = np.flatnonzero(physical["gen"][:, GEN_STATUS] > 0)
    refs = set(physical["bus"][physical["bus"][:, BUS_TYPE] == REF, BUS_I].astype(int))
    slack = [g for g in active if int(physical["gen"][g, GEN_BUS]) in refs]
    if len(slack) != 1:
        raise RuntimeError("Expected one active slack generator")
    slack = slack[0]
    other = np.array([g for g in active if g != slack], dtype=int)
    out = Path("results/E11_control_ablation")
    out.mkdir(parents=True, exist_ok=True)
    rows, controls = [], []
    for j, family in enumerate(FAMILIES):
        group = chosen[chosen["family"] == family]
        if group.empty:
            continue
        errors = generate_errors(family, np.random.default_rng(SEED+j),
                                 args.n_samples, std, corr, df)
        print(f"E11 {family}: {len(group)} E10 scenarios", flush=True)
        for k, rec in enumerate(group.itertuples(index=False), 1):
            i = int(rec.scenario_id)
            if i < 0 or i >= args.n_samples:
                raise ValueError("E10 scenario ID outside n-samples")
            scenario = apply_scenario(policy, rb, forecast, errors[i])
            original, ok = run_pf_scenario(scenario)
            if ok and evaluate_constraints(original, tol)["operational_feasible"]:
                raise RuntimeError(f"E10 violation not reproduced: {family} {i}")
            full = copy.deepcopy(scenario)
            for key in ("f", "success", "et", "order", "raw", "mu"):
                full.pop(key, None)
            for col in (PMAX, PMIN, QMAX, QMIN):
                full["gen"][:, col] = physical["gen"][:, col]
            for col in (VMAX, VMIN):
                full["bus"][:, col] = physical["bus"][:, col]
            fixed = copy.deepcopy(full)
            schedule = scenario["gen"][:, PG].copy()
            fixed["gen"][other, PMIN] = np.maximum(
                physical["gen"][other, PMIN], schedule[other] - BAND_MW)
            fixed["gen"][other, PMAX] = np.minimum(
                physical["gen"][other, PMAX], schedule[other] + BAND_MW)
            if np.any(fixed["gen"][other, PMIN] > fixed["gen"][other, PMAX]):
                restricted, restricted_status = None, "scheduled_pg_outside_limits"
            else:
                restricted, restricted_status = solve(fixed, tol)
            corrected, full_status = solve(full, tol)
            row = {"family": family, "scenario_id": i,
                   "voltage_slack_status": restricted_status,
                   "full_redispatch_status": full_status}
            for label, result in (("voltage_slack", restricted), ("full_redispatch", corrected)):
                if result is None:
                    continue
                row[label + "_max_abs_vg_change_pu"] = float(np.max(
                    np.abs(result["gen"][active, VG] - policy["gen"][active, VG])))
                row[label + "_max_abs_non_slack_pg_change_mw"] = float(np.max(
                    np.abs(result["gen"][other, PG] - schedule[other])))
                row[label + "_slack_pg_change_mw"] = (
                    float(result["gen"][slack, PG] - original["gen"][slack, PG])
                    if ok else np.nan)
                for g in active:
                    controls.append({"family": family, "scenario_id": i,
                                     "method": label, "generator_row": int(g),
                                     "bus_id": int(physical["gen"][g, GEN_BUS]),
                                     "is_slack": bool(g == slack),
                                     "scheduled_pg_mw": float(schedule[g]),
                                     "corrected_pg_mw": float(result["gen"][g, PG]),
                                     "policy_vg_pu": float(policy["gen"][g, VG]),
                                     "corrected_vg_pu": float(result["gen"][g, VG])})
            rows.append(row)
            print(f"  {k}/{len(group)} scenario={i} voltage+slack={restricted_status} "
                  f"full={full_status}", flush=True)
        pd.DataFrame(rows).to_csv(out / "matched_scenarios.csv", index=False)
        pd.DataFrame(controls).to_csv(out / "generator_controls.csv", index=False)
    frame = pd.DataFrame(rows)
    for family, group in frame.groupby("family", sort=False):
        a = int((group["voltage_slack_status"] == "feasible").sum())
        b = int((group["full_redispatch_status"] == "feasible").sum())
        print(f"SUMMARY {family}: voltage_with_slack={a}/{len(group)}, "
              f"full_redispatch={b}/{len(group)}", flush=True)
    print("Offline perfect-information diagnostics. Voltage+slack holds non-slack "
          "PG approximately fixed but allows slack loss balancing and reactive output "
          "changes. Solver failure does not establish infeasibility.", flush=True)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
