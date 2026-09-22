"""E12: exploratory linear voltage controller trained on E11 corrective labels.

Inputs are REALIZED renewable forecast errors, so this is a post-realization
oracle-input diagnostic, not day-ahead control. E11 labels were selected from
violations; performance must be judged only on fresh held-out scenarios.
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
from graphopf.metrics import wilson_interval
from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf, topological_distance
from graphopf.uncertainty import exponential_correlation
from run_pqv_backoff_sweep import apply_pqv_backoff
from run_pqv_backoff_validation import restore_physical_limits
from run_coordinated_voltage import adjusted, BETA_P, BETA_Q, DELTA_V
from run_distribution_shift import FAMILIES, FROZEN_OFFSETS
from run_corrective_feasibility import SEED as E10_SEED

TEST_SEED = 20270521


def effective_error(error, forecast):
    return np.maximum(forecast + error, 0.0) - forecast


def fit_labels(path, forecast, std, corr, df, active):
    labels = pd.read_csv(path)
    labels = labels[labels["method"] == "voltage_slack"]
    if labels.empty:
        raise ValueError("E11 generator_controls.csv contains no voltage_slack labels")
    expected = set(int(g) for g in active)
    x, y, keys = [], [], []
    for j, family in enumerate(FAMILIES):
        sub = labels[labels["family"] == family]
        if sub.empty:
            continue
        ids = sorted(int(i) for i in sub["scenario_id"].unique())
        draws = generate_errors(family, np.random.default_rng(E10_SEED+j),
                                max(ids)+1, std, corr, df)
        for i in ids:
            group = sub[sub["scenario_id"] == i]
            if len(group) != len(active) or set(group["generator_row"].astype(int)) != expected:
                raise ValueError(f"Incomplete or duplicated E11 generator labels: {family} {i}")
            group = group.set_index("generator_row").loc[active]
            x.append(effective_error(draws[i], forecast) / std)
            y.append(group["corrected_vg_pu"].to_numpy(dtype=float) -
                     group["policy_vg_pu"].to_numpy(dtype=float))
            keys.append((family, i))
    if not x:
        raise ValueError("No usable E11 labels")
    return np.asarray(x), np.asarray(y), keys


def fit_ridge(x, y, alpha):
    # Fixed regularization, no held-out model selection. Intercept unpenalized.
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag([0.0] + [alpha] * x.shape[1])
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_policy(base, physical, active, coeff, normalized_error):
    candidate = copy.deepcopy(base)
    changes = np.r_[1.0, normalized_error] @ coeff
    lookup = {int(row[BUS_I]): i for i, row in enumerate(candidate["bus"])}
    for g, delta in zip(active, changes):
        bus_id = int(candidate["gen"][g, GEN_BUS])
        b = lookup[bus_id]
        new_v = float(np.clip(base["gen"][g, VG] + delta,
                              physical["bus"][b, VMIN], physical["bus"][b, VMAX]))
        candidate["gen"][g, VG] = new_v
        candidate["bus"][b, VM] = new_v
    return candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/E01_uncertainty.yaml")
    ap.add_argument("--labels", default="results/E11_control_ablation/generator_controls.csv")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    if args.n_test < 1 or args.alpha <= 0:
        ap.error("n-test and alpha must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tol = float(cfg["evaluation"]["feasibility_tolerance"])
    confidence = float(cfg["evaluation"]["confidence_level"])
    raw = load_case(cfg["case"]["path"])
    rb = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, rb, float(cfg["renewables"]["penetration"]))
    physical = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(physical["bus"])}
    for b, p in zip(rb, forecast):
        physical["bus"][lookup[int(b)], PD] -= p
    tight = solve_ac_opf(apply_pqv_backoff(physical, BETA_P, BETA_Q, DELTA_V))
    baseline = restore_physical_limits(tight, physical)
    frozen = adjusted(baseline, physical, FROZEN_OFFSETS)
    std = float(cfg["uncertainty"]["relative_sigma"]) * forecast
    corr = exponential_correlation(topological_distance(raw, rb),
                                   float(cfg["uncertainty"]["correlation_length"]))
    df = float(cfg["uncertainty"]["student_df"])
    active = np.flatnonzero(physical["gen"][:, GEN_STATUS] > 0)
    x, y, keys = fit_labels(args.labels, forecast, std, corr, df, active)
    coeff = fit_ridge(x, y, args.alpha)
    print(f"E12: trained linear controller on {len(x)} E11 violating-scenario "
          f"labels; alpha={args.alpha}; no held-out tuning", flush=True)
    print("Inputs use realized renewable errors (oracle-input diagnostic), "
          "not an implementable day-ahead forecast.", flush=True)
    out = Path("results/E12_linear_voltage")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coeff, columns=[f"gen_row_{g}" for g in active],
                 index=["intercept"]+[f"normalized_error_{int(b)}" for b in rb]).to_csv(
                     out / "coefficients.csv")
    rows = []
    for j, family in enumerate(FAMILIES):
        errors = generate_errors(family, np.random.default_rng(TEST_SEED+j),
                                 args.n_test, std, corr, df)
        counts = {label: dict(joint=0, q=0, v=0, pg=0, thermal=0, pf_fail=0)
                  for label in ("frozen_e07", "linear_e12")}
        for k, error in enumerate(errors):
            normalized = effective_error(error, forecast) / std
            learned = predict_policy(frozen, physical, active, coeff, normalized)
            for label, policy in (("frozen_e07", frozen), ("linear_e12", learned)):
                result, ok = run_pf_scenario(apply_scenario(policy, rb, forecast, error))
                c = counts[label]
                if not ok:
                    c["joint"] += 1
                    c["pf_fail"] += 1
                    continue
                m = evaluate_constraints(result, tol)
                c["joint"] += int(not m["operational_feasible"])
                for key, metric in (("q", "max_qg_violation_mvar"),
                                    ("v", "max_voltage_violation_pu"),
                                    ("pg", "max_pg_violation_mw"),
                                    ("thermal", "max_thermal_overload_pu")):
                    c[key] += int(m[metric] > tol)
            if (k+1) % 500 == 0:
                print(f"  {family}: {k+1}/{args.n_test}", flush=True)
        for label, c in counts.items():
            low, high = wilson_interval(c["joint"], args.n_test, confidence)
            row = dict(family=family, policy=label, seed=TEST_SEED+j,
                       n=args.n_test, **{key+"_count": val for key, val in c.items()},
                       **{key+"_rate": val/args.n_test for key, val in c.items()},
                       joint_ci_low=low, joint_ci_high=high,
                       meets_5pct_upper_ci=bool(high <= .05))
            rows.append(row)
            print(f"HELD-OUT {family} {label}: joint={c['joint']/args.n_test:.4f} "
                  f"CI=[{low:.4f}, {high:.4f}] q={c['q']/args.n_test:.4f} "
                  f"v={c['v']/args.n_test:.4f} pg={c['pg']/args.n_test:.4f} "
                  f"pf_fail={c['pf_fail']/args.n_test:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "held_out.csv", index=False)
    print(f"CSV output: {out}", flush=True)


if __name__ == "__main__":
    main()
