"""G08: regime-aware held-out evaluation of the already-trained G02 GNN.

No retraining. Loads g02_model.pt and the G03 test split, reconstructs each
held-out AC-OPF case, and stratifies prediction errors by the physical voltage
regime at a designated trigger bus (default bus 31):
  regular       : VM > VMIN + near_margin
  near_boundary : VMIN + active_tol < VM <= VMIN + near_margin
  active        : VM <= VMIN + active_tol or lower-voltage dual > dual_tol

Also reports errors on rare extreme-LMP scenarios using the G05/G06 definition:
max bus |z_LMP| >= z_threshold under train-only LMP normalization.

Diagnostic only. Does not modify training data or model weights.
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


def error_metrics(pred: np.ndarray, target: np.ndarray, trigger_idx: int) -> dict:
    if len(pred) == 0:
        return {k: None for k in (
            "lmp_mae", "lmp_rmse", "vm_mae_pu", "vm_rmse_pu", "vm_max_abs_error_pu",
            "trigger_lmp_mae", "trigger_lmp_rmse", "trigger_vm_mae_pu", "trigger_vm_rmse_pu")}
    e = pred - target
    return {
        "lmp_mae": float(np.mean(np.abs(e[..., 0]))),
        "lmp_rmse": float(np.sqrt(np.mean(e[..., 0] ** 2))),
        "vm_mae_pu": float(np.mean(np.abs(e[..., 1]))),
        "vm_rmse_pu": float(np.sqrt(np.mean(e[..., 1] ** 2))),
        "vm_max_abs_error_pu": float(np.max(np.abs(e[..., 1]))),
        "trigger_lmp_mae": float(np.mean(np.abs(e[:, trigger_idx, 0]))),
        "trigger_lmp_rmse": float(np.sqrt(np.mean(e[:, trigger_idx, 0] ** 2))),
        "trigger_vm_mae_pu": float(np.mean(np.abs(e[:, trigger_idx, 1]))),
        "trigger_vm_rmse_pu": float(np.sqrt(np.mean(e[:, trigger_idx, 1] ** 2))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--config", default="configs/case57.yaml")
    p.add_argument("--model", default=None, help="default: <data>/g02_model.pt")
    p.add_argument("--trigger-bus", type=int, default=31)
    p.add_argument("--near-margin", type=float, default=0.01,
                   help="pu above VMIN counted as near-boundary")
    p.add_argument("--active-tol", type=float, default=1e-5,
                   help="pu tolerance for treating lower voltage limit as active")
    p.add_argument("--dual-tol", type=float, default=1e-6)
    p.add_argument("--z-threshold", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    data = Path(args.data)
    model_path = Path(args.model) if args.model else data / "g02_model.pt"
    train_x, train_y = load_split(data / "train.npz")
    test_x, test_y = load_split(data / "test.npz")
    graph = np.load(data / "graph.npz")

    # Load with CPU mapping first so the script is portable across machines.
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedGraphOPF(hidden_dim=int(ckpt["hidden"]), layers=int(ckpt["layers"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    edge_mean = np.asarray(ckpt["edge_mean"], dtype=np.float32)
    edge_std = np.asarray(ckpt["edge_std"], dtype=np.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.tensor((graph["edge_features"].astype(np.float32) - edge_mean) / edge_std,
                             dtype=torch.float32, device=device)

    pred_chunks = []
    for start in range(0, len(test_x), args.batch_size):
        xb = (test_x[start:start + args.batch_size] - x_mean) / x_std
        with torch.no_grad():
            z = model(torch.tensor(xb, dtype=torch.float32, device=device), edge_index, edge_attr)
        pred_chunks.append(z.cpu().numpy())
    pred_z = np.concatenate(pred_chunks, axis=0)
    pred = pred_z * y_std + y_mean

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_case(cfg["case"]["path"])
    base_ids = base["bus"][:, BUS_I].astype(int)
    loc = np.where(base_ids == args.trigger_bus)[0]
    if not len(loc):
        raise ValueError(f"trigger bus {args.trigger_bus} not found")
    trigger_idx = int(loc[0])

    # Extreme-price definition exactly follows G05/G06: train-only global LMP mean/std.
    lmp_mu = float(train_y[..., 0].mean())
    lmp_sd = max(float(train_y[..., 0].std()), 1e-12)
    max_abs_z = np.max(np.abs((test_y[..., 0] - lmp_mu) / lmp_sd), axis=1)
    extreme_lmp = max_abs_z >= args.z_threshold

    regimes = []
    trigger_vm = np.zeros(len(test_x), dtype=float)
    trigger_vmin = np.zeros(len(test_x), dtype=float)
    trigger_dual = np.zeros(len(test_x), dtype=float)

    print("G08 REGIME-AWARE GNN EVALUATION")
    print(f"model={model_path} device={device} test={len(test_x)}")
    print(f"trigger bus={args.trigger_bus}; near_margin={args.near_margin:g} pu; active_tol={args.active_tol:g} pu")
    print(f"extreme LMP threshold={args.z_threshold:g} train sigma")
    print("Re-solving held-out test scenarios to classify the physical voltage regime...", flush=True)

    for i in range(len(test_x)):
        solved = solve_ac_opf(reconstruct_case(base, test_x[i]))
        bus = solved["bus"]
        ids = bus[:, BUS_I].astype(int)
        j = int(np.where(ids == args.trigger_bus)[0][0])
        vm = float(bus[j, VM]); vmin = float(bus[j, VMIN]); dual = float(bus[j, MU_VMIN])
        trigger_vm[i] = vm; trigger_vmin[i] = vmin; trigger_dual[i] = dual
        margin = vm - vmin
        if margin <= args.active_tol or dual > args.dual_tol:
            regime = "active"
        elif margin <= args.near_margin:
            regime = "near_boundary"
        else:
            regime = "regular"
        regimes.append(regime)
        if (i + 1) % 100 == 0 or i + 1 == len(test_x):
            print(f"  classified {i+1}/{len(test_x)}", flush=True)

    regimes = np.asarray(regimes)
    groups = [
        ("all", np.ones(len(test_x), dtype=bool)),
        ("regular", regimes == "regular"),
        ("near_boundary", regimes == "near_boundary"),
        ("active", regimes == "active"),
        ("extreme_lmp", extreme_lmp),
        ("non_extreme_lmp", ~extreme_lmp),
    ]

    summary = {
        "model": str(model_path), "device": str(device), "n_test": int(len(test_x)),
        "trigger_bus": args.trigger_bus, "near_margin_pu": args.near_margin,
        "active_tol_pu": args.active_tol, "dual_tol": args.dual_tol,
        "z_threshold": args.z_threshold, "train_lmp_mean": lmp_mu, "train_lmp_std": lmp_sd,
        "groups": {},
    }

    print("\nREGIME RESULTS")
    for name, mask in groups:
        m = error_metrics(pred[mask], test_y[mask], trigger_idx)
        entry = {"n": int(mask.sum()), "fraction": float(mask.mean()), **m}
        summary["groups"][name] = entry
        if entry["n"]:
            print(f"{name:16s} n={entry['n']:4d} frac={entry['fraction']:.4f} "
                  f"LMP_MAE={m['lmp_mae']:.6f} LMP_RMSE={m['lmp_rmse']:.6f} "
                  f"VM_MAE={m['vm_mae_pu']:.7f} trigger_LMP_MAE={m['trigger_lmp_mae']:.6f} "
                  f"trigger_VM_MAE={m['trigger_vm_mae_pu']:.7f}")
        else:
            print(f"{name:16s} n=0")

    # Scenario-level table makes the rare failures auditable rather than hiding them in averages.
    rows = []
    for i in range(len(test_x)):
        err = pred[i] - test_y[i]
        rows.append({
            "scenario": i, "regime": str(regimes[i]), "max_abs_z_lmp": float(max_abs_z[i]),
            "extreme_lmp": bool(extreme_lmp[i]), "trigger_vm_pu": float(trigger_vm[i]),
            "trigger_vmin_pu": float(trigger_vmin[i]), "trigger_margin_pu": float(trigger_vm[i] - trigger_vmin[i]),
            "trigger_lower_v_dual": float(trigger_dual[i]),
            "lmp_mae": float(np.mean(np.abs(err[:, 0]))),
            "lmp_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "vm_mae_pu": float(np.mean(np.abs(err[:, 1]))),
            "vm_max_abs_error_pu": float(np.max(np.abs(err[:, 1]))),
            "trigger_true_lmp": float(test_y[i, trigger_idx, 0]),
            "trigger_pred_lmp": float(pred[i, trigger_idx, 0]),
            "trigger_abs_lmp_error": float(abs(err[trigger_idx, 0])),
            "trigger_true_vm_pu": float(test_y[i, trigger_idx, 1]),
            "trigger_pred_vm_pu": float(pred[i, trigger_idx, 1]),
            "trigger_abs_vm_error_pu": float(abs(err[trigger_idx, 1])),
        })

    out_csv = data / "g08_regime_evaluation.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    out_json = data / "g08_regime_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nTOP TEST SCENARIOS BY LMP ERROR")
    for r, row in enumerate(sorted(rows, key=lambda q: q["lmp_rmse"], reverse=True)[:10], 1):
        print(f"{r:2d} scenario={row['scenario']:4d} regime={row['regime']:13s} "
              f"z={row['max_abs_z_lmp']:8.3f} LMP_RMSE={row['lmp_rmse']:.6f} "
              f"bus{args.trigger_bus}_true/pred={row['trigger_true_lmp']:.4f}/{row['trigger_pred_lmp']:.4f} "
              f"VM={row['trigger_vm_pu']:.6f} dual={row['trigger_lower_v_dual']:.4g}")

    print(f"\nCSV output: {out_csv}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
