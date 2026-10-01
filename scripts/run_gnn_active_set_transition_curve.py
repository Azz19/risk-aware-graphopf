"""G09: active-set transition curve audit for the already-trained G02 GNN.

No retraining. Reuses the held-out G03 test set and G02 checkpoint. Each test
scenario is re-solved with AC-OPF and ordered by the physical voltage margin at
a designated trigger bus (default bus 31). The audit asks whether GNN price
error remains small within the inactive regime and rises abruptly when the
lower-voltage constraint activates.

Outputs scenario-level data plus margin-binned error summaries suitable for a
transition-curve figure/table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from pypower.idx_bus import BUS_I, PD, QD, VM, VMIN, MU_VMIN

from graphopf.gnn_baseline import SupervisedGraphOPF
from graphopf.powerflow import load_case, solve_ac_opf


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float32), z["y"].astype(np.float32)


def reconstruct_case(base: dict, x: np.ndarray) -> dict:
    case = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    case["bus"][:, PD] = x[:, 0]
    case["bus"][:, QD] = x[:, 1]
    return case


def predict(model, x, edge_index, edge_attr, x_mean, x_std, y_mean, y_std, device, batch_size):
    chunks = []
    for start in range(0, len(x), batch_size):
        xb = (x[start:start + batch_size] - x_mean) / x_std
        with torch.no_grad():
            z = model(torch.tensor(xb, dtype=torch.float32, device=device), edge_index, edge_attr)
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0) * y_std + y_mean


def bin_stats(rows, lo, hi, include_hi=False):
    if include_hi:
        rr = [r for r in rows if lo <= r["trigger_margin_pu"] <= hi]
    else:
        rr = [r for r in rows if lo <= r["trigger_margin_pu"] < hi]
    if not rr:
        return None
    vals = lambda k: np.asarray([r[k] for r in rr], dtype=float)
    return {
        "margin_lo_pu": lo, "margin_hi_pu": hi, "n": len(rr),
        "mean_margin_pu": float(vals("trigger_margin_pu").mean()),
        "mean_true_lmp": float(vals("trigger_true_lmp").mean()),
        "mean_pred_lmp": float(vals("trigger_pred_lmp").mean()),
        "trigger_lmp_mae": float(vals("trigger_abs_lmp_error").mean()),
        "trigger_lmp_rmse": float(np.sqrt(np.mean(vals("trigger_lmp_error") ** 2))),
        "mean_true_vm_pu": float(vals("trigger_true_vm_pu").mean()),
        "mean_pred_vm_pu": float(vals("trigger_pred_vm_pu").mean()),
        "trigger_vm_mae_pu": float(vals("trigger_abs_vm_error_pu").mean()),
        "mean_lower_v_dual": float(vals("trigger_lower_v_dual").mean()),
        "max_lower_v_dual": float(vals("trigger_lower_v_dual").max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--config", default="configs/case57.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--trigger-bus", type=int, default=31)
    p.add_argument("--active-tol", type=float, default=1e-5)
    p.add_argument("--dual-tol", type=float, default=1e-6)
    p.add_argument("--z-threshold", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    data = Path(args.data)
    model_path = Path(args.model) if args.model else data / "g02_model.pt"
    train_x, train_y = load_split(data / "train.npz")
    test_x, test_y = load_split(data / "test.npz")
    graph = np.load(data / "graph.npz")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedGraphOPF(hidden_dim=int(ckpt["hidden"]), layers=int(ckpt["layers"])).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    edge_mean = np.asarray(ckpt["edge_mean"], dtype=np.float32)
    edge_std = np.asarray(ckpt["edge_std"], dtype=np.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.tensor((graph["edge_features"].astype(np.float32)-edge_mean)/edge_std,
                             dtype=torch.float32, device=device)
    pred = predict(model, test_x, edge_index, edge_attr, x_mean, x_std, y_mean, y_std,
                   device, args.batch_size)

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_case(cfg["case"]["path"])
    ids = base["bus"][:, BUS_I].astype(int)
    loc = np.where(ids == args.trigger_bus)[0]
    if not len(loc):
        raise ValueError(f"trigger bus {args.trigger_bus} not found")
    trigger_idx = int(loc[0])

    lmp_mu = float(train_y[..., 0].mean())
    lmp_sd = max(float(train_y[..., 0].std()), 1e-12)
    max_abs_z = np.max(np.abs((test_y[..., 0]-lmp_mu)/lmp_sd), axis=1)

    print("G09 ACTIVE-SET TRANSITION CURVE AUDIT")
    print(f"model={model_path} device={device} test={len(test_x)} trigger_bus={args.trigger_bus}")
    print("Re-solving held-out test scenarios and measuring distance to the lower-voltage boundary...", flush=True)

    rows = []
    for i in range(len(test_x)):
        solved = solve_ac_opf(reconstruct_case(base, test_x[i]))
        bus = solved["bus"]; bid = bus[:, BUS_I].astype(int)
        j = int(np.where(bid == args.trigger_bus)[0][0])
        vm = float(bus[j, VM]); vmin = float(bus[j, VMIN]); dual = float(bus[j, MU_VMIN])
        margin = vm-vmin
        active = bool(margin <= args.active_tol or dual > args.dual_tol)
        true_lmp = float(test_y[i, trigger_idx, 0]); pred_lmp = float(pred[i, trigger_idx, 0])
        true_vm = float(test_y[i, trigger_idx, 1]); pred_vm = float(pred[i, trigger_idx, 1])
        rows.append({
            "scenario": i, "active": active, "extreme_lmp": bool(max_abs_z[i] >= args.z_threshold),
            "max_abs_z_lmp": float(max_abs_z[i]), "trigger_margin_pu": margin,
            "trigger_lower_v_dual": dual, "trigger_true_lmp": true_lmp,
            "trigger_pred_lmp": pred_lmp, "trigger_lmp_error": pred_lmp-true_lmp,
            "trigger_abs_lmp_error": abs(pred_lmp-true_lmp), "trigger_true_vm_pu": true_vm,
            "trigger_pred_vm_pu": pred_vm, "trigger_vm_error_pu": pred_vm-true_vm,
            "trigger_abs_vm_error_pu": abs(pred_vm-true_vm),
            "total_pd_mw": float(test_x[i, :, 0].sum()), "total_qd_mvar": float(test_x[i, :, 1].sum()),
        })
        if (i+1) % 100 == 0 or i+1 == len(test_x):
            print(f"  solved {i+1}/{len(test_x)}", flush=True)

    # Fine bins near the transition, then progressively wider bins farther away.
    edges = [-1e-6, args.active_tol, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 1.0]
    bins = []
    for k in range(len(edges)-1):
        b = bin_stats(rows, edges[k], edges[k+1], include_hi=(k == len(edges)-2))
        if b is not None:
            bins.append(b)

    active_rows = [r for r in rows if r["active"]]
    inactive_rows = [r for r in rows if not r["active"]]
    nearest_inactive = sorted(inactive_rows, key=lambda r: r["trigger_margin_pu"])[:15]
    active_sorted = sorted(active_rows, key=lambda r: r["trigger_abs_lmp_error"], reverse=True)

    print("\nTRANSITION BINS")
    print(" margin interval (pu)       n   true_LMP  pred_LMP  LMP_MAE   VM_MAE     mean_dual")
    for b in bins:
        print(f"[{b['margin_lo_pu']:8.5f},{b['margin_hi_pu']:8.5f}) {b['n']:4d} "
              f"{b['mean_true_lmp']:9.3f} {b['mean_pred_lmp']:9.3f} {b['trigger_lmp_mae']:9.3f} "
              f"{b['trigger_vm_mae_pu']:9.6f} {b['mean_lower_v_dual']:11.3g}")

    print("\nACTIVE SCENARIOS")
    for r in active_sorted:
        print(f"scenario={r['scenario']:4d} margin={r['trigger_margin_pu']:.3e} dual={r['trigger_lower_v_dual']:.4g} "
              f"z={r['max_abs_z_lmp']:.3f} LMP true/pred={r['trigger_true_lmp']:.4f}/{r['trigger_pred_lmp']:.4f} "
              f"abs_err={r['trigger_abs_lmp_error']:.4f} VM true/pred={r['trigger_true_vm_pu']:.6f}/{r['trigger_pred_vm_pu']:.6f}")

    print("\n15 NEAREST INACTIVE SCENARIOS")
    for r in nearest_inactive:
        print(f"scenario={r['scenario']:4d} margin={r['trigger_margin_pu']:.6f} dual={r['trigger_lower_v_dual']:.3g} "
              f"LMP true/pred={r['trigger_true_lmp']:.4f}/{r['trigger_pred_lmp']:.4f} "
              f"abs_err={r['trigger_abs_lmp_error']:.4f}")

    # Correlations are diagnostic only; the transition bins are the primary test.
    margins = np.asarray([r["trigger_margin_pu"] for r in inactive_rows])
    errs = np.asarray([r["trigger_abs_lmp_error"] for r in inactive_rows])
    corr = float(np.corrcoef(margins, errs)[0, 1]) if len(margins) > 1 else None
    summary = {
        "model": str(model_path), "device": str(device), "n_test": len(rows),
        "trigger_bus": args.trigger_bus, "active_tol_pu": args.active_tol, "dual_tol": args.dual_tol,
        "z_threshold": args.z_threshold, "n_active": len(active_rows), "n_inactive": len(inactive_rows),
        "inactive_margin_vs_abs_lmp_error_pearson": corr, "bins": bins,
        "active_scenarios": active_sorted, "nearest_inactive_scenarios": nearest_inactive,
    }

    out_csv = data / "g09_active_set_transition_curve.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out_json = data / "g09_active_set_transition_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\ninactive margin-vs-|LMP error| Pearson r={corr:.6f}" if corr is not None else "")
    print(f"CSV output: {out_csv}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
