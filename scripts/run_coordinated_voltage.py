"""E07 coordinated fixed-voltage baseline: tune on new scenarios, evaluate on held-out.

Exploratory E06 motivated the predeclared 3x3 grid at buses 8 and 12.
No access to held-out scenarios until after selection. Not adaptive recourse.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD, VM, VMIN, VMAX
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from graphopf.experiments import apply_scenario, generate_errors, renewable_forecast, run_pf_scenario, select_renewable_buses
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf
from graphopf.metrics import wilson_interval
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits

TRAIN_SEED = 20261121
TEST_SEED = 20261221
OFFSETS = (0.0, 0.00125, 0.0025)
BETA_P, BETA_Q, DELTA_V = 0.025, 0.10, 0.0025


def adjusted(base, physical, offsets):
    candidate = copy.deepcopy(base)
    bus_lookup = {int(row[BUS_I]): i for i, row in enumerate(candidate["bus"])}
    for bus_id, delta in offsets.items():
        gen_rows = np.flatnonzero((candidate["gen"][:, GEN_STATUS] > 0) &
                                   (candidate["gen"][:, GEN_BUS].astype(int) == bus_id))
        if len(gen_rows) != 1:
            raise ValueError(f"Expected exactly one active generator at bus {bus_id}")
        g = int(gen_rows[0])
        b = bus_lookup[bus_id]
        new_v = float(base["gen"][g, VG]) + delta
        if not (physical["bus"][b, VMIN] <= new_v <= physical["bus"][b, VMAX]):
            raise ValueError(f"Bus {bus_id} setpoint outside physical voltage limits")
        candidate["gen"][g, VG] = new_v
        candidate["bus"][b, VM] = new_v
    return candidate


def evaluate(policy, rb, forecast, errors, tol):
    counts = dict(joint=0, voltage=0, pg=0, qg=0, thermal=0, pf_fail=0)
    for e in errors:
        result, success = run_pf_scenario(apply_scenario(policy, rb, forecast, e))
        if not success:
            counts["joint"] += 1
            counts["pf_fail"] += 1
            continue
        m = evaluate_constraints(result, tol)
        counts["joint"] += int(not m["operational_feasible"])
        counts["voltage"] += int(m["max_voltage_violation_pu"] > tol)
        counts["pg"] += int(m["max_pg_violation_mw"] > tol)
        counts["qg"] += int(m["max_qg_violation_mvar"] > tol)
        counts["thermal"] += int(m["max_thermal_overload_pu"] > tol)
    return {f"{k}_rate": v / len(errors) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()
    if args.n_train < 1 or args.n_test < 1:
        ap.error("Sample counts must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for bus_id, p in zip(rb, forecast):
        physical["bus"][lookup[int(bus_id)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    baseline = restore_physical_limits(tight, physical)
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    df = float(cfg["uncertainty"]["student_df"])
    train = generate_errors("iid_gaussian", np.random.default_rng(TRAIN_SEED), args.n_train, std, None, df)
    # Held-out draws are generated only after candidate selection.
    policies = {}
    records = []
    grid = list(itertools.product(OFFSETS, OFFSETS))
    print(f"E07 tuning: {len(grid)} predeclared candidates, n_train={args.n_train}, "
          f"seed={TRAIN_SEED}; test seed={TEST_SEED} held out", flush=True)
    for i, (d8, d12) in enumerate(grid, start=1):
        key = (d8, d12)
        try:
            policy = adjusted(baseline, physical, {8: d8, 12: d12})
            nominal, ok = run_pf_scenario(apply_scenario(policy, rb, forecast, np.zeros_like(forecast)))
            nominal_ok = bool(ok and evaluate_constraints(nominal, tol)["operational_feasible"])
        except (ValueError, RuntimeError) as exc:
            policy, nominal_ok = None, False
            print(f"candidate {i}/{len(grid)} offsets={key} rejected: {exc}", flush=True)
        if not nominal_ok:
            records.append({"delta_v8": d8, "delta_v12": d12,
                            "nominal_feasible": False, "train_joint_rate": np.nan})
            print(f"candidate {i}/{len(grid)} offsets={key} nominal infeasible", flush=True)
            continue
        policies[key] = policy
        rates = evaluate(policy, rb, forecast, train, tol)
        records.append({"delta_v8": d8, "delta_v12": d12,
                        "nominal_feasible": True,
                        **{f"train_{k}": v for k, v in rates.items()}})
        print(f"candidate {i}/{len(grid)} offsets={key} "
              f"train_joint={rates['joint_rate']:.4f} "
              f"train_q={rates['qg_rate']:.4f} "
              f"train_voltage={rates['voltage_rate']:.4f}", flush=True)
    out = Path("results/E07_coordinated_voltage")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "tuning.csv", index=False)
    eligible = frame[frame["nominal_feasible"]].sort_values(
        ["train_joint_rate", "delta_v8", "delta_v12"])
    if eligible.empty:
        print("No nominally feasible candidate. Held-out test not run.", flush=True)
        return
    chosen = eligible.iloc[0]
    key = (float(chosen["delta_v8"]), float(chosen["delta_v12"]))
    print(f"SELECTED ON TRAIN ONLY: offsets={key}, "
          f"train_joint={chosen['train_joint_rate']:.4f}", flush=True)
    test = generate_errors("iid_gaussian", np.random.default_rng(TEST_SEED),
                           args.n_test, std, None, df)
    test_rows = []
    for label, policy in (("baseline", baseline), ("selected", policies[key])):
        rates = evaluate(policy, rb, forecast, test, tol)
        violations = int(round(rates["joint_rate"] * args.n_test))
        low, high = wilson_interval(violations, args.n_test, confidence)
        test_rows.append({"policy": label, "delta_v8": 0.0 if label == "baseline" else key[0],
                          "delta_v12": 0.0 if label == "baseline" else key[1],
                          "test_seed": TEST_SEED, "n_test": args.n_test,
                          **rates, "joint_ci_low": low, "joint_ci_high": high,
                          "meets_5pct_upper_ci": high <= 0.05})
        print(f"HELD-OUT {label}: joint={rates['joint_rate']:.4f} "
              f"CI=[{low:.4f}, {high:.4f}] "
              f"q={rates['qg_rate']:.4f} voltage={rates['voltage_rate']:.4f} "
              f"pf_fail={rates['pf_fail_rate']:.4f}", flush=True)
    pd.DataFrame(test_rows).to_csv(out / "held_out_test.csv", index=False)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
