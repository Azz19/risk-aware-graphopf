"""G06: diagnose physical active-set changes behind rare extreme LMP labels.

Re-solves selected G03 scenarios from their stored pre-OPF loads, then compares
extreme-price cases with load-matched regular cases. Reports generator P/Q
margins and duals, voltage margins and duals, branch loading and flow duals,
and the buses carrying the largest LMPs.

This is a diagnostic only. It does not remove labels or change GNN training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower.idx_bus import BUS_I, PD, QD, VM, VMIN, VMAX, LAM_P, MU_VMAX, MU_VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, PMIN, PMAX, QMIN, QMAX, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN

from graphopf.powerflow import load_case, solve_ac_opf


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float64), z["y"].astype(np.float64)


def reconstruct_case(base: dict, x: np.ndarray) -> dict:
    case = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    case["bus"][:, PD] = x[:, 0]
    case["bus"][:, QD] = x[:, 1]
    return case


def scenario_metrics(solved: dict) -> dict:
    bus, gen, branch = solved["bus"], solved["gen"], solved["branch"]
    active = gen[:, GEN_STATUS] > 0
    g = gen[active]

    p_margin = np.minimum(g[:, PG] - g[:, PMIN], g[:, PMAX] - g[:, PG])
    q_margin = np.minimum(g[:, QG] - g[:, QMIN], g[:, QMAX] - g[:, QG])
    v_margin = np.minimum(bus[:, VM] - bus[:, VMIN], bus[:, VMAX] - bus[:, VM])

    rate = branch[:, RATE_A]
    rated = rate > 0
    sf = np.hypot(branch[:, PF], branch[:, QF])
    st = np.hypot(branch[:, PT], branch[:, QT])
    loading = np.zeros(len(branch))
    loading[rated] = np.maximum(sf[rated], st[rated]) / rate[rated]

    return {
        "objective": float(solved["f"]),
        "min_p_margin_mw": float(np.min(p_margin)),
        "min_q_margin_mvar": float(np.min(q_margin)),
        "min_v_margin_pu": float(np.min(v_margin)),
        "max_branch_loading_pu": float(np.max(loading)) if rated.any() else 0.0,
        "max_lmp": float(np.max(bus[:, LAM_P])),
        "min_lmp": float(np.min(bus[:, LAM_P])),
        "active_p_dual_count": int(np.sum((g[:, MU_PMAX] > 1e-6) | (g[:, MU_PMIN] > 1e-6))),
        "active_q_dual_count": int(np.sum((g[:, MU_QMAX] > 1e-6) | (g[:, MU_QMIN] > 1e-6))),
        "active_v_dual_count": int(np.sum((bus[:, MU_VMAX] > 1e-6) | (bus[:, MU_VMIN] > 1e-6))),
        "active_flow_dual_count": int(np.sum((branch[:, MU_SF] > 1e-6) | (branch[:, MU_ST] > 1e-6))),
    }


def print_details(solved: dict, top: int = 8):
    bus, gen, branch = solved["bus"], solved["gen"], solved["branch"]
    active_idx = np.where(gen[:, GEN_STATUS] > 0)[0]

    print("  generators nearest limits:")
    entries = []
    for i in active_idx:
        pmar = min(gen[i, PG] - gen[i, PMIN], gen[i, PMAX] - gen[i, PG])
        qmar = min(gen[i, QG] - gen[i, QMIN], gen[i, QMAX] - gen[i, QG])
        dual = max(gen[i, MU_PMAX], gen[i, MU_PMIN], gen[i, MU_QMAX], gen[i, MU_QMIN])
        entries.append((min(pmar, qmar), i, pmar, qmar, dual))
    for _, i, pmar, qmar, dual in sorted(entries)[:top]:
        print(f"    gen={i:2d} bus={int(gen[i,GEN_BUS]):2d} PG={gen[i,PG]:9.3f} QG={gen[i,QG]:9.3f} "
              f"Pmargin={pmar:9.4f} Qmargin={qmar:9.4f} max_dual={dual:10.4g}")

    vmarg = np.minimum(bus[:, VM] - bus[:, VMIN], bus[:, VMAX] - bus[:, VM])
    print("  buses nearest voltage limits:")
    for i in np.argsort(vmarg)[:top]:
        dual = max(bus[i, MU_VMAX], bus[i, MU_VMIN])
        print(f"    bus={int(bus[i,BUS_I]):2d} VM={bus[i,VM]:.6f} margin={vmarg[i]:.6g} "
              f"LMP={bus[i,LAM_P]:10.4f} v_dual={dual:10.4g}")

    rate = branch[:, RATE_A]
    sf = np.hypot(branch[:, PF], branch[:, QF])
    st = np.hypot(branch[:, PT], branch[:, QT])
    loading = np.where(rate > 0, np.maximum(sf, st) / np.maximum(rate, 1e-12), 0.0)
    print("  most-loaded branches:")
    for i in np.argsort(loading)[::-1][:top]:
        dual = max(branch[i, MU_SF], branch[i, MU_ST])
        print(f"    branch={i:2d} {int(branch[i,F_BUS]):2d}->{int(branch[i,T_BUS]):2d} "
              f"loading={loading[i]:.5f} flow_dual={dual:10.4g}")

    print("  largest LMP buses:")
    for i in np.argsort(bus[:, LAM_P])[::-1][:top]:
        print(f"    bus={int(bus[i,BUS_I]):2d} LMP={bus[i,LAM_P]:12.4f} VM={bus[i,VM]:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--config", default="configs/case57.yaml")
    p.add_argument("--z-threshold", type=float, default=10.0)
    p.add_argument("--matched-per-extreme", type=int, default=1)
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    data = Path(args.data)
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_case(cfg["case"]["path"])
    splits = {s: load_split(data / f"{s}.npz") for s in ("train", "val", "test")}

    train_y = splits["train"][1]
    mu = float(train_y[..., 0].mean())
    sd = max(float(train_y[..., 0].std()), 1e-12)

    print("G06 ACTIVE-CONSTRAINT AUDIT")
    print(f"train LMP normalization mean={mu:.8g} std={sd:.8g}; extreme threshold={args.z_threshold:g} sigma")
    print("Matched controls are regular scenarios nearest in (total PD,total QD) within the same split.\n")

    rows = []
    for split, (x, y) in splits.items():
        scen_z = np.max(np.abs((y[..., 0] - mu) / sd), axis=1)
        extreme_idx = np.where(scen_z >= args.z_threshold)[0]
        regular_idx = np.where(scen_z < args.z_threshold)[0]
        pd = x[..., 0].sum(axis=1)
        qd = x[..., 1].sum(axis=1)
        pd_scale = max(float(np.std(pd)), 1e-9)
        qd_scale = max(float(np.std(qd)), 1e-9)

        selected = []
        used_controls = set()
        for e in extreme_idx:
            selected.append((int(e), "extreme", int(e)))
            if len(regular_idx):
                dist = ((pd[regular_idx] - pd[e]) / pd_scale) ** 2 + ((qd[regular_idx] - qd[e]) / qd_scale) ** 2
                for pos in np.argsort(dist):
                    c = int(regular_idx[pos])
                    if c not in used_controls:
                        selected.append((c, "matched_regular", int(e)))
                        used_controls.add(c)
                        if sum(1 for a,b,ref in selected if b == "matched_regular" and ref == int(e)) >= args.matched_per_extreme:
                            break

        print(f"{split.upper()}: extremes={len(extreme_idx)} selected_with_controls={len(selected)}")
        for idx, role, matched_to in selected:
            solved = solve_ac_opf(reconstruct_case(base, x[idx]))
            m = scenario_metrics(solved)
            row = {"split": split, "scenario": idx, "role": role, "matched_to_extreme": matched_to,
                   "stored_max_abs_z_lmp": float(scen_z[idx]), "total_pd_mw": float(pd[idx]),
                   "total_qd_mvar": float(qd[idx]), **m}
            rows.append(row)
            print(f"\n{split} scenario={idx} role={role} matched_to={matched_to} z={scen_z[idx]:.3f} "
                  f"PD={pd[idx]:.3f} QD={qd[idx]:.3f}")
            print(f"  LMP=[{m['min_lmp']:.4f},{m['max_lmp']:.4f}] objective={m['objective']:.6g} "
                  f"minP={m['min_p_margin_mw']:.4g} minQ={m['min_q_margin_mvar']:.4g} "
                  f"minV={m['min_v_margin_pu']:.4g} maxLoading={m['max_branch_loading_pu']:.4g}")
            print(f"  active dual counts: P={m['active_p_dual_count']} Q={m['active_q_dual_count']} "
                  f"V={m['active_v_dual_count']} flow={m['active_flow_dual_count']}")
            print_details(solved, args.top)
        print()

    out_csv = data / "g06_active_constraint_audit.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    summary = {
        "train_lmp_mean": mu, "train_lmp_std": sd, "z_threshold": args.z_threshold,
        "matched_per_extreme": args.matched_per_extreme,
        "n_audited": len(rows),
        "extreme_counts": {s: int(np.sum(np.max(np.abs((splits[s][1][...,0]-mu)/sd), axis=1) >= args.z_threshold)) for s in splits},
        "note": "Diagnostic only; matched controls are same-split load-matched regular scenarios."
    }
    out_json = data / "g06_active_constraint_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"CSV output: {out_csv}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
