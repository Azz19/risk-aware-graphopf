"""M12: explicit AC-Jacobian boundary-sensitivity audit.

M11 showed that hand-crafted primal proxies rank the rare bus-31 regime well but
fail to transfer a validation threshold to frozen active cases.  M12 therefore
removes classifier fitting entirely and asks a sharper physical question:

    Does the local Newton AC power-flow Jacobian, evaluated at each solved OPF
    point, predict the direction and distance to the bus-31 Vmin boundary?

The experiment uses only the frozen G14 continuation in the observed lambda
range [0, 1].  Each continuation point is re-solved with PYPOWER AC-OPF, then
PYPOWER's standard Newton Jacobian blocks are assembled from dSbus_dV.  The
load direction for a path is known from its frozen control/active scenario pair.
Solving J dx = dS_spec/dlambda gives the local dV/dlambda for PQ buses.  From
that derivative and the current voltage margin we form a no-fit predicted
boundary location.

No test/G14 LMP, dual, active label, or future point is used to construct the
Jacobian score.  Solver duals are read only after prediction to locate the true
activation for evaluation.  This is a mechanism/falsification audit, not a
trained model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from pypower.api import ppoption, runopf
from graphopf.powerflow.case import load_case
from pypower.bustypes import bustypes
from pypower.dSbus_dV import dSbus_dV
from pypower.ext2int import ext2int
from pypower.idx_bus import BUS_I, PD, QD, VM, VA, VMIN
from pypower.makeYbus import makeYbus


def ff(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_config_case(config: Path) -> Path:
    # Avoid adding a YAML dependency to the script logic; the project already
    # depends on PyYAML, but this keeps the required field explicit.
    import yaml

    cfg = yaml.safe_load(config.read_text())
    p = Path(cfg["case"]["path"])
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"MATPOWER/PGLib case not found: {p}")
    return p


def continuation_rows(data: Path, test_x: np.ndarray):
    src = data / "g14_active_set_continuation.csv"
    rows = []
    with src.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("solved", "")).lower() not in {"true", "1", "yes"}:
                continue
            la = ff(r.get("lambda"))
            if not (0 <= la <= 1):
                continue
            s = int(r["active_scenario"])
            c = int(r["control_scenario"])
            direction = test_x[s].astype(float) - test_x[c].astype(float)
            x = test_x[c].astype(float) + la * direction
            rows.append(
                {
                    "path": s,
                    "control": c,
                    "lambda": la,
                    "x": x,
                    "direction": direction,
                    "true_lmp": ff(r.get("true_lmp")),
                }
            )
    return rows


def set_loads(ppc, x, bus_ids):
    q = {int(b): k for k, b in enumerate(bus_ids)}
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    bus = out["bus"].copy()
    for i in range(bus.shape[0]):
        b = int(round(bus[i, BUS_I]))
        if b in q:
            k = q[b]
            bus[i, PD] = float(x[k, 0])
            bus[i, QD] = float(x[k, 1])
    out["bus"] = bus
    return out


def jacobian_direction(opf_result, direction, bus_ids, trigger_bus):
    """Return local Newton-PF dV_trigger/dlambda and diagnostics.

    Uses the same reduced Jacobian structure as PYPOWER newtonpf: P equations
    for PV+PQ buses and Q equations for PQ buses; state variables are their
    voltage angles and PQ voltage magnitudes.
    """
    r = ext2int(opf_result)
    base = float(r["baseMVA"])
    bus = r["bus"]
    gen = r["gen"]
    branch = r["branch"]
    Ybus, _, _ = makeYbus(base, bus, branch)
    V = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
    ref, pv, pq = bustypes(bus, gen)
    pvpq = np.r_[pv, pq].astype(int)
    pq = np.asarray(pq, dtype=int)

    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
    J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
    J12 = dS_dVm[np.array([pvpq]).T, pq].real
    J21 = dS_dVa[np.array([pq]).T, pvpq].imag
    J22 = dS_dVm[np.array([pq]).T, pq].imag
    J = vstack([hstack([J11, J12]), hstack([J21, J22])], format="csr")

    # ext2int renumbers buses consecutively but preserves original bus numbers
    # in r['order']['bus']['i2e'].
    i2e = np.asarray(r["order"]["bus"]["i2e"]).astype(int)
    ext_to_int = {int(b): i for i, b in enumerate(i2e)}
    data_pos = {int(b): k for k, b in enumerate(bus_ids)}
    dP = np.zeros(len(bus), float)
    dQ = np.zeros(len(bus), float)
    for b, ii in ext_to_int.items():
        if b in data_pos:
            k = data_pos[b]
            dP[ii] = -float(direction[k, 0]) / base
            dQ[ii] = -float(direction[k, 1]) / base

    rhs = np.r_[dP[pvpq], dQ[pq]]
    dx = spsolve(J, rhs)
    dvm = dx[len(pvpq) :]
    pq_pos = {int(b): k for k, b in enumerate(pq)}
    ti = ext_to_int[int(trigger_bus)]
    dv = float(dvm[pq_pos[ti]]) if ti in pq_pos else np.nan

    dense = J.toarray()
    cond = float(np.linalg.cond(dense))
    return dv, cond, int(len(ref)), int(len(pv)), int(len(pq))


def trigger_values(opf_result, trigger_bus):
    bus = opf_result["bus"]
    hit = np.where(np.rint(bus[:, BUS_I]).astype(int) == int(trigger_bus))[0]
    if len(hit) != 1:
        raise RuntimeError(f"trigger bus {trigger_bus} not uniquely found")
    i = int(hit[0])
    return float(bus[i, VM]), float(bus[i, VMIN])


def first_crossing(rr, key="mu"):
    for r in sorted(rr, key=lambda z: z["lambda"]):
        if r[key] > 0:
            return float(r["lambda"])
    return np.nan


def bootstrap_path(values, rng, B):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [np.nan, np.nan]
    z = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(B)])
    return [float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--config", default="configs/case57.yaml")
    p.add_argument("--trigger-bus", type=int, default=31)
    p.add_argument("--dual-tol", type=float, default=1e-6)
    p.add_argument("--approach-eps", type=float, default=1e-10)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20271006)
    a = p.parse_args()

    data = Path(a.data)
    te = load_npz(data / "m01_constraint_dual" / "test_constraint_dual.npz")
    bus_ids = te["bus_ids"].astype(int)
    cont = continuation_rows(data, te["x"].astype(np.float32))
    paths = sorted(set(r["path"] for r in cont))
    case_path = load_config_case(Path(a.config))
    base_case = load_case(str(case_path))
        if not isinstance(base_case, dict):
    raise RuntimeError(
        f"Case loader returned {type(base_case).__name__}, expected dict: {case_path}"
    )
    opt = ppoption(VERBOSE=0, OUT_ALL=0)

    print("M12 AC-JACOBIAN / KKT BOUNDARY-SENSITIVITY AUDIT")
    print(f"source={data/'g14_active_set_continuation.csv'} paths={paths} observed_lambda=[0,1] trigger=vmin@{a.trigger_bus}")
    print("No classifier is fit. Test/G14 LMPs, duals and future points are evaluation-only.")
    print("Local score = Newton AC-Jacobian directional dV/dlambda + current physical Vmin margin.")

    out = []
    failures = 0
    for k, q in enumerate(cont, 1):
        case = set_loads(base_case, q["x"], bus_ids)
        res = runopf(case, opt)
        if not bool(res.get("success", False)):
            failures += 1
            continue
        vm, vmin = trigger_values(res, a.trigger_bus)
        margin = vm - vmin
        try:
            dv, cond, nref, npv, npq = jacobian_direction(
                res, q["direction"], bus_ids, a.trigger_bus
            )
        except Exception as e:
            failures += 1
            print(f" jacobian failure path={q['path']} lambda={q['lambda']:.6f}: {e}", flush=True)
            continue

        # Predicted crossing under first-order local continuation.  Only a
        # negative dV/dlambda means the current direction approaches Vmin.
        if np.isfinite(dv) and dv < -a.approach_eps:
            distance = margin / (-dv)
            hit = q["lambda"] + max(distance, 0.0)
        else:
            distance = np.inf
            hit = np.inf

        # Evaluation-only dual: PYPOWER OPF appends MU_VMIN after the standard
        # 13 bus columns.  idx_bus MU_VMIN is imported lazily for compatibility.
        from pypower.idx_bus import MU_VMIN
        bhit = np.where(np.rint(res["bus"][:, BUS_I]).astype(int) == a.trigger_bus)[0][0]
        mu = float(res["bus"][bhit, MU_VMIN]) if res["bus"].shape[1] > MU_VMIN else 0.0

        out.append(
            {
                "path": q["path"],
                "control": q["control"],
                "lambda": q["lambda"],
                "vm31": vm,
                "vmin31": vmin,
                "margin31": margin,
                "dV31_dlambda": dv,
                "distance_to_vmin_lambda": distance,
                "predicted_lambda_hit": hit,
                "jacobian_condition": cond,
                "mu31_eval_only": mu,
                "active_eval_only": int(mu > a.dual_tol),
                "true_lmp_eval_only": q["true_lmp"],
                "nref": nref,
                "npv": npv,
                "npq": npq,
            }
        )
        if k % 25 == 0:
            print(f" solved {k}/{len(cont)} failures={failures}", flush=True)

    if not out:
        raise RuntimeError("M12 produced no solved Jacobian points")

    print("\nPATHWISE BOUNDARY ANTICIPATION")
    summary = {
        "configuration": vars(a),
        "case": str(case_path),
        "n_requested": len(cont),
        "n_solved": len(out),
        "failures": failures,
        "paths": {},
    }
    crossing_errors = []
    preactive_hit_errors = []
    for path in paths:
        rr = sorted([r for r in out if r["path"] == path], key=lambda z: z["lambda"])
        if not rr:
            continue
        true_hit = first_crossing(rr, "mu31_eval_only")
        pre = [r for r in rr if not np.isfinite(true_hit) or r["lambda"] < true_hit]
        finite = [r for r in pre if np.isfinite(r["predicted_lambda_hit"])]
        # Last pre-activation prediction is the strictest prospective estimate:
        # it uses only current state + known load direction, not future labels.
        last = finite[-1] if finite else None
        pred_hit = float(last["predicted_lambda_hit"]) if last else np.nan
        err = abs(pred_hit - true_hit) if np.isfinite(pred_hit) and np.isfinite(true_hit) else np.nan
        crossing_errors.append(err)
        all_pre_err = [abs(r["predicted_lambda_hit"] - true_hit) for r in finite if np.isfinite(true_hit)]
        preactive_hit_errors.extend(all_pre_err)
        approaching = sum(r["dV31_dlambda"] < -a.approach_eps for r in pre)
        summary["paths"][str(path)] = {
            "n": len(rr),
            "true_activation_lambda": true_hit,
            "last_preactive_predicted_lambda": pred_hit,
            "last_preactive_abs_error": err,
            "preactive_n": len(pre),
            "preactive_approaching_fraction": approaching / max(len(pre), 1),
            "median_preactive_predicted_hit_abs_error": float(np.median(all_pre_err)) if all_pre_err else None,
            "max_jacobian_condition": float(np.max([r["jacobian_condition"] for r in rr])),
            "min_margin": float(np.min([r["margin31"] for r in rr])),
            "min_dV31_dlambda": float(np.min([r["dV31_dlambda"] for r in rr])),
        }
        print(
            f"path={path} true_hit={true_hit:.6f} last_pre_pred={pred_hit:.6f} "
            f"abs_err={err:.6f} approaching={approaching}/{len(pre)} "
            f"median_pre_err={np.median(all_pre_err) if all_pre_err else np.nan:.6f}"
        )

    # One-step voltage-margin prediction is an additional local linearization
    # check and never uses a future point to construct the prediction itself.
    one_step = []
    for path in paths:
        rr = sorted([r for r in out if r["path"] == path], key=lambda z: z["lambda"])
        for left, right in zip(rr[:-1], rr[1:]):
            dl = right["lambda"] - left["lambda"]
            pred = left["margin31"] + left["dV31_dlambda"] * dl
            one_step.append(
                {
                    "path": path,
                    "lambda_left": left["lambda"],
                    "lambda_right": right["lambda"],
                    "pred_margin_right": pred,
                    "true_margin_right": right["margin31"],
                    "abs_error": abs(pred - right["margin31"]),
                }
            )
    ose = np.array([r["abs_error"] for r in one_step], float)
    rng = np.random.default_rng(a.seed)
    summary["aggregate"] = {
        "mean_last_preactive_activation_abs_error": float(np.nanmean(crossing_errors)),
        "median_all_preactive_activation_abs_error": float(np.median(preactive_hit_errors)) if preactive_hit_errors else None,
        "one_step_margin_mae_pu": float(ose.mean()) if len(ose) else None,
        "one_step_margin_max_abs_pu": float(ose.max()) if len(ose) else None,
        "path_block_bootstrap_last_preactive_error_ci95": bootstrap_path(crossing_errors, rng, a.bootstrap),
        "bootstrap_unit": "continuation path",
    }

    csvout = data / "m12_ac_jacobian_boundary_sensitivity.csv"
    jsout = data / "m12_ac_jacobian_boundary_sensitivity_summary.json"
    stepout = data / "m12_ac_jacobian_one_step_margin.csv"
    with csvout.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    if one_step:
        with stepout.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(one_step[0].keys()))
            w.writeheader()
            w.writerows(one_step)
    jsout.write_text(json.dumps(summary, indent=2) + "\n")

    ag = summary["aggregate"]
    print("\nAGGREGATE")
    print(f"mean last-preactivation lambda error={ag['mean_last_preactive_activation_abs_error']:.6f}")
    print(f"median all-preactivation lambda error={ag['median_all_preactive_activation_abs_error']}")
    print(f"one-step voltage-margin MAE={ag['one_step_margin_mae_pu']:.8f} pu")
    print(f"path-block bootstrap 95% CI={ag['path_block_bootstrap_last_preactive_error_ci95']}")
    print(f"CSV output: {csvout}")
    print(f"One-step CSV: {stepout}")
    print(f"Summary: {jsout}")
    print(
        "Decision: small prospective boundary-location error across all frozen paths supports "
        "an explicit Jacobian/KKT transition feature for M13. Large/path-specific error means "
        "a single-point Newton Jacobian is insufficient and the next model must differentiate "
        "the full OPF KKT system or use continuation-aware state information. Do not interpret "
        "this audit as causal evidence."
    )


if __name__ == "__main__":
    main()
