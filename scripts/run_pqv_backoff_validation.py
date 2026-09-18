from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, VMAX, VMIN
from pypower.idx_gen import PMAX, PMIN, QMAX, QMIN

from graphopf.experiments import apply_scenario, generate_errors, renewable_forecast, run_pf_scenario, select_renewable_buses
from graphopf.metrics import wilson_interval
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf
from run_pqv_backoff_sweep import apply_pqv_backoff, BETA_P, BETA_Q_VALUES, DELTA_V_VALUES

VALIDATION_SEED = 20261019


def restore_physical_limits(result, physical):
    out = copy.deepcopy(result)
    out["gen"][:, PMIN] = physical["gen"][:, PMIN]
    out["gen"][:, PMAX] = physical["gen"][:, PMAX]
    out["gen"][:, QMIN] = physical["gen"][:, QMIN]
    out["gen"][:, QMAX] = physical["gen"][:, QMAX]
    out["bus"][:, VMIN] = physical["bus"][:, VMIN]
    out["bus"][:, VMAX] = physical["bus"][:, VMAX]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n = args.n_samples
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])

    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for b, p in zip(rb, forecast):
        physical["bus"][lookup[int(b)], 2] -= p

    base_cost = float(solve_ac_opf(physical)["f"])
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    errors = generate_errors("iid_gaussian", np.random.default_rng(VALIDATION_SEED), n, std, None, float(cfg["uncertainty"]["student_df"]))

    rows = []
    for bq, dv in itertools.product(BETA_Q_VALUES, DELTA_V_VALUES):
        try:
            tightened = apply_pqv_backoff(physical, BETA_P, bq, dv)
            tight_opf = solve_ac_opf(tightened)
        except (RuntimeError, ValueError):
            continue
        opf = restore_physical_limits(tight_opf, physical)

        z, ok = run_pf_scenario(apply_scenario(opf, rb, forecast, np.zeros_like(forecast), None))
        if not ok or not evaluate_constraints(z, tol)["operational_feasible"]:
            raise RuntimeError(f"zero-error invariant failed beta_q={bq}, delta_v={dv}")

        counts = dict(joint=0, voltage=0, pg=0, qg=0, thermal=0, nonconv=0)
        for e in errors:
            res, conv = run_pf_scenario(apply_scenario(opf, rb, forecast, e, None))
            if not conv:
                counts["nonconv"] += 1
                counts["joint"] += 1
                continue
            m = evaluate_constraints(res, tol)
            counts["joint"] += int(not m["operational_feasible"])
            counts["voltage"] += int(m["max_voltage_violation_pu"] > tol)
            counts["pg"] += int(m["max_pg_violation_mw"] > tol)
            counts["qg"] += int(m["max_qg_violation_mvar"] > tol)
            counts["thermal"] += int(m["max_thermal_overload_pu"] > tol)

        lo, hi = wilson_interval(counts["joint"], n, confidence)
        rows.append({
            "beta_p": BETA_P, "beta_q": bq, "delta_v_pu": dv,
            "cost_increase_pct": 100*(float(tight_opf["f"])-base_cost)/base_cost,
            "joint_violation_rate": counts["joint"]/n, "joint_ci_low": lo, "joint_ci_high": hi,
            "meets_5pct_upper_ci": hi <= .05,
            "voltage_violation_rate": counts["voltage"]/n,
            "pg_violation_rate": counts["pg"]/n,
            "qg_violation_rate": counts["qg"]/n,
            "thermal_violation_rate": counts["thermal"]/n,
            "pf_nonconvergence_rate": counts["nonconv"]/n,
        })

    frame = pd.DataFrame(rows).sort_values(["cost_increase_pct","beta_q","delta_v_pu"])
    out = Path("results/E04_pqv_validation"); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/"iid_gaussian_validation.csv", index=False)
    eligible = frame[frame["meets_5pct_upper_ci"]]
    print(f"validation_seed={VALIDATION_SEED}, n={n}, beta_p={BETA_P}")
    print(frame.to_string(index=False))
    print("\nSELECTION")
    if eligible.empty:
        print("No feasible tested P/Q/V backoff satisfies upper 95% CI <= 5%.")
    else:
        best = eligible.iloc[0]
        print("Least-cost candidate satisfying upper 95% CI <= 5%:")
        print(best.to_string())


if __name__ == "__main__":
    main()
