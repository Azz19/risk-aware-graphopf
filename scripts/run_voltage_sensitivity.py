"""E06 pilot: fixed generator-voltage setpoint sensitivity, common random scenarios.

A voltage offset is applied to the nominal E04 generator VG setpoint before
each scenario PF; the original active-power AGC and physical limits are unchanged.
This is a controlled open-loop sensitivity experiment, NOT learned/closed-loop
voltage recourse. Inspect pilot before designing a corrective policy.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, PD, VM, VMIN, VMAX
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG, QG, QMAX, QMIN
from graphopf.experiments import (
    apply_scenario, generate_errors, renewable_forecast,
    run_pf_scenario, select_renewable_buses,
)
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits, VALIDATION_SEED

BETA_P, BETA_Q, DELTA_V = 0.025, 0.10, 0.0025
OFFSETS = (-0.005, -0.0025, 0.0, 0.0025, 0.005)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error("--n-samples must be positive")
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
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    errors = generate_errors("iid_gaussian", np.random.default_rng(VALIDATION_SEED),
                             args.n_samples, std, None, float(cfg["uncertainty"]["student_df"]))
    active = np.flatnonzero(opf["gen"][:, GEN_STATUS] > 0)
    buses = opf["gen"][active, GEN_BUS].astype(int)
    if 9 not in buses:
        raise RuntimeError("Expected bus 9 generator is not active")
    row9 = int(active[np.flatnonzero(buses == 9)[0]])
    cases = [(None, 0.0)] + [(int(i), d) for i in active for d in OFFSETS if d != 0.0]
    rows = []
    out = Path("results/E06_voltage_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    print(f"E06 open-loop pilot n={args.n_samples}, seed={VALIDATION_SEED}, "
          f"policies={len(cases)}; offsets in pu; original physical limits", flush=True)

    for idx, (gen_idx, offset) in enumerate(cases, start=1):
        modified = copy.deepcopy(opf)
        label = "baseline" if gen_idx is None else f"bus_{int(modified['gen'][gen_idx, GEN_BUS])}"
        if gen_idx is not None:
            bus_id = int(modified["gen"][gen_idx, GEN_BUS])
            bidx = lookup[bus_id]
            new_v = float(modified["gen"][gen_idx, VG] + offset)
            if not (physical["bus"][bidx, VMIN] <= new_v <= physical["bus"][bidx, VMAX]):
                print(f"skip {label} offset={offset:+.4f}: setpoint outside physical voltage bounds", flush=True)
                continue
            # Both are set explicitly so PYPOWER initialization and PV control
            # start from the same voltage. Duplicate generators at a bus are
            # excluded from this single-generator sensitivity experiment.
            if np.count_nonzero((modified["gen"][:, GEN_STATUS] > 0) &
                                (modified["gen"][:, GEN_BUS].astype(int) == bus_id)) != 1:
                print(f"skip {label}: multiple active generators share this bus", flush=True)
                continue
            modified["gen"][gen_idx, VG] = new_v
            modified["bus"][bidx, VM] = new_v
        zero, ok = run_pf_scenario(apply_scenario(modified, rb, forecast, np.zeros_like(forecast)))
        if not ok:
            print(f"skip {label} offset={offset:+.4f}: zero-error PF failed", flush=True)
            continue
        zero_m = evaluate_constraints(zero, tol)
        counts = dict(joint=0, q=0, voltage=0, pg=0, thermal=0, pf_fail=0, bus9_upper=0, bus9_lower=0)
        upper_sum = 0.0
        for e in errors:
            result, success = run_pf_scenario(apply_scenario(modified, rb, forecast, e))
            if not success:
                counts["pf_fail"] += 1
                counts["joint"] += 1
                continue
            m = evaluate_constraints(result, tol)
            counts["joint"] += int(not m["operational_feasible"])
            counts["q"] += int(m["max_qg_violation_mvar"] > tol)
            counts["voltage"] += int(m["max_voltage_violation_pu"] > tol)
            counts["pg"] += int(m["max_pg_violation_mw"] > tol)
            counts["thermal"] += int(m["max_thermal_overload_pu"] > tol)
            q9 = float(result["gen"][row9, QG])
            excess = max(q9 - float(physical["gen"][row9, QMAX]), 0.0)
            upper_sum += excess
            counts["bus9_upper"] += int(excess > tol)
            counts["bus9_lower"] += int(q9 < float(physical["gen"][row9, QMIN]) - tol)
        row = {"generator": label, "offset_pu": offset,
               "zero_error_operational_feasible": zero_m["operational_feasible"],
               "zero_error_max_q_violation_mvar": zero_m["max_qg_violation_mvar"],
               **{f"{k}_rate": v / args.n_samples for k, v in counts.items()},
               "bus9_mean_upper_excess_mvar": upper_sum / args.n_samples}
        rows.append(row)
        print(f"{idx}/{len(cases)} {label} offset={offset:+.4f} "
              f"joint={row['joint_rate']:.4f} q={row['q_rate']:.4f} "
              f"bus9_upper={row['bus9_upper_rate']:.4f} "
              f"voltage={row['voltage_rate']:.4f} "
              f"zero_ok={row['zero_error_operational_feasible']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "iid_gaussian_open_loop_sensitivity.csv", index=False)
    print("\nE06 RESULTS (exploratory; no policy selected on this dataset)", flush=True)
    print(frame.to_string(index=False), flush=True)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
