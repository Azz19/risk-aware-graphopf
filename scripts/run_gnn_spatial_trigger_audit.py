"""G07: spatial trigger audit for rare voltage-constrained LMP regimes.

Uses the same extreme definition and same-split aggregate-load matching as G06,
then compares nodal PD/QD between each extreme and its matched regular control.
Reports the buses with the largest spatial load changes and their unweighted
network distance to the primary voltage-transition bus (default: bus 31).

Diagnostic only: does not alter labels or training data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pypower.idx_bus import BUS_I, PD, QD, VM, LAM_P, MU_VMAX, MU_VMIN

from graphopf.powerflow import load_case, solve_ac_opf


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float64), z["y"].astype(np.float64)


def reconstruct_case(base: dict, x: np.ndarray) -> dict:
    case = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    case["bus"][:, PD] = x[:, 0]
    case["bus"][:, QD] = x[:, 1]
    return case


def graph_distances(base: dict, source_bus: int) -> dict[int, int]:
    buses = [int(v) for v in base["bus"][:, BUS_I]]
    adj = {b: set() for b in buses}
    for row in base["branch"]:
        if row[BR_STATUS] <= 0:
            continue
        a, b = int(row[F_BUS]), int(row[T_BUS])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    dist = {source_bus: 0}
    q = deque([source_bus])
    while q:
        a = q.popleft()
        for b in adj.get(a, ()):
            if b not in dist:
                dist[b] = dist[a] + 1
                q.append(b)
    return dist


def solve_summary(base: dict, x: np.ndarray, trigger_bus: int) -> dict:
    solved = solve_ac_opf(reconstruct_case(base, x))
    bus = solved["bus"]
    ids = bus[:, BUS_I].astype(int)
    where = np.where(ids == trigger_bus)[0]
    if not len(where):
        raise ValueError(f"trigger bus {trigger_bus} not found")
    i = int(where[0])
    return {
        "objective": float(solved["f"]),
        "trigger_vm_pu": float(bus[i, VM]),
        "trigger_lmp": float(bus[i, LAM_P]),
        "trigger_v_dual": float(max(bus[i, MU_VMAX], bus[i, MU_VMIN])),
        "max_lmp": float(np.max(bus[:, LAM_P])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--config", default="configs/case57.yaml")
    p.add_argument("--z-threshold", type=float, default=10.0)
    p.add_argument("--trigger-bus", type=int, default=31)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    data = Path(args.data)
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_case(cfg["case"]["path"])
    bus_ids = base["bus"][:, BUS_I].astype(int)
    distances = graph_distances(base, args.trigger_bus)
    splits = {s: load_split(data / f"{s}.npz") for s in ("train", "val", "test")}

    train_y = splits["train"][1]
    mu = float(train_y[..., 0].mean())
    sd = max(float(train_y[..., 0].std()), 1e-12)

    print("G07 SPATIAL TRIGGER AUDIT")
    print(f"train LMP normalization mean={mu:.8g} std={sd:.8g}; extreme threshold={args.z_threshold:g} sigma")
    print(f"primary voltage-transition bus={args.trigger_bus}; distances are unweighted branch-hop distances")
    print("Controls are regular scenarios nearest in (total PD,total QD), matching G06.\n")

    rows = []
    pair_summaries = []
    aggregate_abs_dp = np.zeros(len(bus_ids))
    aggregate_abs_dq = np.zeros(len(bus_ids))
    n_pairs = 0

    for split, (x, y) in splits.items():
        scen_z = np.max(np.abs((y[..., 0] - mu) / sd), axis=1)
        extreme_idx = np.where(scen_z >= args.z_threshold)[0]
        regular_idx = np.where(scen_z < args.z_threshold)[0]
        total_pd = x[..., 0].sum(axis=1)
        total_qd = x[..., 1].sum(axis=1)
        pd_scale = max(float(np.std(total_pd)), 1e-9)
        qd_scale = max(float(np.std(total_qd)), 1e-9)
        used_controls = set()

        print(f"{split.upper()}: extremes={len(extreme_idx)}")
        for e in extreme_idx:
            if not len(regular_idx):
                continue
            d = ((total_pd[regular_idx] - total_pd[e]) / pd_scale) ** 2 + ((total_qd[regular_idx] - total_qd[e]) / qd_scale) ** 2
            control = None
            for pos in np.argsort(d):
                c = int(regular_idx[pos])
                if c not in used_controls:
                    control = c
                    used_controls.add(c)
                    break
            if control is None:
                continue

            e = int(e)
            dp = x[e, :, 0] - x[control, :, 0]
            dq = x[e, :, 1] - x[control, :, 1]
            # Scale-free joint magnitude used only for ranking within the pair.
            dp_sd = max(float(np.std(x[:, :, 0])), 1e-12)
            dq_sd = max(float(np.std(x[:, :, 1])), 1e-12)
            score = np.sqrt((dp / dp_sd) ** 2 + (dq / dq_sd) ** 2)
            order = np.argsort(score)[::-1]

            ext_sol = solve_summary(base, x[e], args.trigger_bus)
            ctl_sol = solve_summary(base, x[control], args.trigger_bus)
            n_pairs += 1
            aggregate_abs_dp += np.abs(dp)
            aggregate_abs_dq += np.abs(dq)

            pair = {
                "split": split, "extreme_scenario": e, "control_scenario": control,
                "extreme_z": float(scen_z[e]),
                "extreme_total_pd_mw": float(total_pd[e]), "control_total_pd_mw": float(total_pd[control]),
                "extreme_total_qd_mvar": float(total_qd[e]), "control_total_qd_mvar": float(total_qd[control]),
                "delta_total_pd_mw": float(total_pd[e] - total_pd[control]),
                "delta_total_qd_mvar": float(total_qd[e] - total_qd[control]),
                "extreme_trigger_vm_pu": ext_sol["trigger_vm_pu"], "control_trigger_vm_pu": ctl_sol["trigger_vm_pu"],
                "extreme_trigger_lmp": ext_sol["trigger_lmp"], "control_trigger_lmp": ctl_sol["trigger_lmp"],
                "extreme_trigger_v_dual": ext_sol["trigger_v_dual"], "control_trigger_v_dual": ctl_sol["trigger_v_dual"],
            }
            pair_summaries.append(pair)

            print(f"\n{split} extreme={e} control={control} z={scen_z[e]:.3f}")
            print(f"  totals: dPD={pair['delta_total_pd_mw']:+.4f} MW dQD={pair['delta_total_qd_mvar']:+.4f} MVAr")
            print(f"  bus {args.trigger_bus}: VM {ctl_sol['trigger_vm_pu']:.6f}->{ext_sol['trigger_vm_pu']:.6f}, "
                  f"LMP {ctl_sol['trigger_lmp']:.4f}->{ext_sol['trigger_lmp']:.4f}, "
                  f"Vdual {ctl_sol['trigger_v_dual']:.4g}->{ext_sol['trigger_v_dual']:.4g}")
            print("  largest nodal load changes:")
            for rank, i in enumerate(order[:args.top], 1):
                bid = int(bus_ids[i])
                hops = distances.get(bid, -1)
                print(f"   {rank:2d} bus={bid:2d} hops_to_{args.trigger_bus}={hops:2d} "
                      f"dPD={dp[i]:+9.4f} dQD={dq[i]:+9.4f} score={score[i]:.5f}")
                rows.append({
                    **pair, "rank": rank, "bus_id": bid, "hops_to_trigger": hops,
                    "extreme_pd_mw": float(x[e, i, 0]), "control_pd_mw": float(x[control, i, 0]),
                    "delta_pd_mw": float(dp[i]), "extreme_qd_mvar": float(x[e, i, 1]),
                    "control_qd_mvar": float(x[control, i, 1]), "delta_qd_mvar": float(dq[i]),
                    "joint_change_score": float(score[i]),
                })
        print()

    if n_pairs:
        mean_abs_dp = aggregate_abs_dp / n_pairs
        mean_abs_dq = aggregate_abs_dq / n_pairs
        global_score = np.sqrt((mean_abs_dp / max(float(np.std(splits['train'][0][:,:,0])), 1e-12)) ** 2 +
                               (mean_abs_dq / max(float(np.std(splits['train'][0][:,:,1])), 1e-12)) ** 2)
        print("CROSS-PAIR REPEATED SPATIAL CHANGES")
        for rank, i in enumerate(np.argsort(global_score)[::-1][:args.top], 1):
            bid = int(bus_ids[i])
            print(f" {rank:2d} bus={bid:2d} hops_to_{args.trigger_bus}={distances.get(bid,-1):2d} "
                  f"mean|dPD|={mean_abs_dp[i]:.5f} mean|dQD|={mean_abs_dq[i]:.5f} score={global_score[i]:.6f}")
    else:
        mean_abs_dp = np.zeros(len(bus_ids)); mean_abs_dq = np.zeros(len(bus_ids)); global_score = np.zeros(len(bus_ids))

    out_csv = data / "g07_spatial_trigger_audit.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    summary = {
        "train_lmp_mean": mu, "train_lmp_std": sd, "z_threshold": args.z_threshold,
        "trigger_bus": args.trigger_bus, "n_pairs": n_pairs, "pairs": pair_summaries,
        "cross_pair_bus_ranking": [
            {"rank": r + 1, "bus_id": int(bus_ids[i]), "hops_to_trigger": int(distances.get(int(bus_ids[i]), -1)),
             "mean_abs_delta_pd_mw": float(mean_abs_dp[i]), "mean_abs_delta_qd_mvar": float(mean_abs_dq[i]),
             "score": float(global_score[i])}
            for r, i in enumerate(np.argsort(global_score)[::-1][:args.top])
        ],
        "note": "Diagnostic only. Hop distance is topological, not impedance-weighted electrical distance."
    }
    out_json = data / "g07_spatial_trigger_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nCSV output: {out_csv}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
