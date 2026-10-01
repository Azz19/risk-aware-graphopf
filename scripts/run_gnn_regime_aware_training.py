"""G10: compare MSE, Huber, and tail-weighted training on the fixed G03 split.

The experiment keeps architecture, normalization, optimizer settings, seed, and
train/validation/test splits fixed. Tail weighting is defined only from TRAIN
targets: a scenario receives extra weight when its maximum bus |z_LMP| exceeds
a configurable threshold. Validation uses the corresponding training objective
for early stopping. Evaluation reports global and rare-tail metrics; if G08's
scenario table exists, its physical regime labels are reused without re-solving
OPF cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphopf.gnn_baseline import SupervisedGraphOPF


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float32), z["y"].astype(np.float32)


def metrics(pred, target, trigger_idx=30):
    if len(pred) == 0:
        return {k: None for k in ("lmp_mae", "lmp_rmse", "vm_mae_pu", "vm_rmse_pu",
                                  "trigger_lmp_mae", "trigger_vm_mae_pu")}
    e = pred - target
    return {
        "lmp_mae": float(np.mean(np.abs(e[..., 0]))),
        "lmp_rmse": float(np.sqrt(np.mean(e[..., 0] ** 2))),
        "vm_mae_pu": float(np.mean(np.abs(e[..., 1]))),
        "vm_rmse_pu": float(np.sqrt(np.mean(e[..., 1] ** 2))),
        "trigger_lmp_mae": float(np.mean(np.abs(e[:, trigger_idx, 0]))),
        "trigger_vm_mae_pu": float(np.mean(np.abs(e[:, trigger_idx, 1]))),
    }


def load_g08_masks(data: Path, n_test: int):
    path = data / "g08_regime_evaluation.csv"
    regimes = np.array(["unknown"] * n_test, dtype=object)
    if not path.exists():
        return regimes
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["scenario"])
            if 0 <= i < n_test:
                regimes[i] = row["regime"]
    return regimes


def objective(pred, target, kind, weights=None, huber_delta=1.0):
    if kind == "mse":
        per = (pred - target) ** 2
    elif kind == "huber":
        per = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    elif kind == "tail_weighted":
        per = (pred - target) ** 2
    else:
        raise ValueError(kind)
    per_scenario = per.mean(dim=(1, 2))
    if kind == "tail_weighted":
        return (per_scenario * weights).sum() / weights.sum()
    return per_scenario.mean()


def train_one(kind, train_x, train_y, val_x, val_y, test_x, graph, args, stats, train_w, val_w, device):
    x_mean, x_std, y_mean, y_std, ea_mean, ea_std = stats
    tx = torch.tensor((train_x - x_mean) / x_std, dtype=torch.float32, device=device)
    ty = torch.tensor((train_y - y_mean) / y_std, dtype=torch.float32, device=device)
    vx = torch.tensor((val_x - x_mean) / x_std, dtype=torch.float32, device=device)
    vy = torch.tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
    ex = torch.tensor((test_x - x_mean) / x_std, dtype=torch.float32, device=device)
    tw = torch.tensor(train_w, dtype=torch.float32, device=device)
    vw = torch.tensor(val_w, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.tensor((graph["edge_features"].astype(np.float32) - ea_mean) / ea_std,
                             dtype=torch.float32, device=device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model = SupervisedGraphOPF(hidden_dim=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss = objective(model(tx, edge_index, edge_attr), ty, kind, tw, args.huber_delta)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = objective(model(vx, edge_index, edge_attr), vy, kind, vw, args.huber_delta).item()
        if vl < best - 1e-8:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"{kind:13s} epoch={epoch:4d} train={loss.item():.6f} val={vl:.6f}", flush=True)
        if stale >= args.patience:
            print(f"{kind:13s} early_stop epoch={epoch} best_val={best:.6f}", flush=True)
            break
    model.load_state_dict(best_state); model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(ex), args.batch_size):
            chunks.append(model(ex[s:s + args.batch_size], edge_index, edge_attr).cpu().numpy())
    pred_z = np.concatenate(chunks, axis=0)
    pred = pred_z * y_std + y_mean
    return pred, best, best_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=20271001)
    p.add_argument("--z-threshold", type=float, default=10.0)
    p.add_argument("--tail-weight", type=float, default=25.0)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--trigger-bus", type=int, default=31)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    data = Path(args.data)
    train_x, train_y = load_split(data / "train.npz")
    val_x, val_y = load_split(data / "val.npz")
    test_x, test_y = load_split(data / "test.npz")
    graph = np.load(data / "graph.npz")
    trigger_idx = args.trigger_bus - 1
    if not 0 <= trigger_idx < train_y.shape[1]:
        raise ValueError("trigger-bus assumes contiguous 1-based bus numbering and is out of range")

    x_mean = train_x.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(train_x.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = train_y.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(train_y.std(axis=(0, 1), keepdims=True), 1e-6)
    ef = graph["edge_features"].astype(np.float32)
    ea_mean = ef.mean(axis=0, keepdims=True); ea_std = np.maximum(ef.std(axis=0, keepdims=True), 1e-6)
    stats = (x_mean, x_std, y_mean, y_std, ea_mean, ea_std)

    lmp_mu = float(train_y[..., 0].mean()); lmp_sd = max(float(train_y[..., 0].std()), 1e-12)
    train_z = np.max(np.abs((train_y[..., 0] - lmp_mu) / lmp_sd), axis=1)
    val_z = np.max(np.abs((val_y[..., 0] - lmp_mu) / lmp_sd), axis=1)
    test_z = np.max(np.abs((test_y[..., 0] - lmp_mu) / lmp_sd), axis=1)
    train_tail = train_z >= args.z_threshold; val_tail = val_z >= args.z_threshold; test_tail = test_z >= args.z_threshold
    train_w = np.where(train_tail, args.tail_weight, 1.0).astype(np.float32)
    val_w = np.where(val_tail, args.tail_weight, 1.0).astype(np.float32)
    regimes = load_g08_masks(data, len(test_x))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_models = data / "g10_models"; out_models.mkdir(exist_ok=True)

    print("G10 REGIME-AWARE GNN TRAINING")
    print(f"device={device} split={len(train_x)}/{len(val_x)}/{len(test_x)} seed={args.seed}")
    print(f"train-tail threshold={args.z_threshold:g} sigma count={train_tail.sum()} weight={args.tail_weight:g}")
    print(f"val-tail count={val_tail.sum()} test-tail count={test_tail.sum()}")
    if np.all(regimes == "unknown"):
        print("WARNING: g08_regime_evaluation.csv absent; physical active/near-boundary metrics unavailable.")

    summary = {"configuration": vars(args), "device": str(device), "train_tail_count": int(train_tail.sum()),
               "val_tail_count": int(val_tail.sum()), "test_tail_count": int(test_tail.sum()), "models": {}}
    rows = []
    for kind in ("mse", "huber", "tail_weighted"):
        print(f"\nTRAINING {kind.upper()}", flush=True)
        pred, best, state = train_one(kind, train_x, train_y, val_x, val_y, test_x, graph,
                                      args, stats, train_w, val_w, device)
        masks = {"all": np.ones(len(test_x), bool), "extreme_lmp": test_tail,
                 "non_extreme_lmp": ~test_tail}
        if not np.all(regimes == "unknown"):
            masks.update({"regular": regimes == "regular", "near_boundary": regimes == "near_boundary",
                          "active": regimes == "active"})
        group_results = {name: {"n": int(mask.sum()), **metrics(pred[mask], test_y[mask], trigger_idx)}
                         for name, mask in masks.items()}
        summary["models"][kind] = {"best_validation_objective": best, "groups": group_results}
        torch.save({"model_state": state, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean,
                    "y_std": y_std, "edge_mean": ea_mean, "edge_std": ea_std, "hidden": args.hidden,
                    "layers": args.layers, "seed": args.seed, "loss": kind}, out_models / f"{kind}.pt")
        print(f"{kind.upper()} RESULTS")
        for name, r in group_results.items():
            if r["n"]:
                print(f"{name:16s} n={r['n']:4d} LMP_MAE={r['lmp_mae']:.6f} LMP_RMSE={r['lmp_rmse']:.6f} "
                      f"VM_MAE={r['vm_mae_pu']:.7f} trigger_LMP_MAE={r['trigger_lmp_mae']:.6f}")
        for i in range(len(test_x)):
            rows.append({"model": kind, "scenario": i, "regime": str(regimes[i]),
                         "max_abs_z_lmp": float(test_z[i]), "extreme_lmp": bool(test_tail[i]),
                         "trigger_true_lmp": float(test_y[i, trigger_idx, 0]),
                         "trigger_pred_lmp": float(pred[i, trigger_idx, 0]),
                         "trigger_abs_lmp_error": float(abs(pred[i, trigger_idx, 0] - test_y[i, trigger_idx, 0])),
                         "lmp_mae": float(np.mean(np.abs(pred[i, :, 0] - test_y[i, :, 0]))),
                         "lmp_rmse": float(np.sqrt(np.mean((pred[i, :, 0] - test_y[i, :, 0]) ** 2)))})
        print("stress cases:")
        for i in (575, 889, 984):
            if i < len(test_x):
                print(f"  scenario={i:4d} true/pred={test_y[i, trigger_idx, 0]:.4f}/{pred[i, trigger_idx, 0]:.4f} "
                      f"abs_err={abs(pred[i, trigger_idx, 0]-test_y[i, trigger_idx, 0]):.4f}")

    out_csv = data / "g10_regime_aware_training.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out_json = data / "g10_regime_aware_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nCSV output: {out_csv}")
    print(f"Summary: {out_json}")
    print(f"Models: {out_models}")


if __name__ == "__main__":
    main()
