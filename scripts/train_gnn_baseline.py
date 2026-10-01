"""Train/evaluate the G02 supervised edge-aware GNN baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from graphopf.gnn_baseline import SupervisedGraphOPF


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float32), z["y"].astype(np.float32)


def standardize(a, mean, std):
    return (a - mean) / std


def metrics(pred, target):
    err = pred - target
    return {
        "lmp_mae": float(np.mean(np.abs(err[..., 0]))),
        "lmp_rmse": float(np.sqrt(np.mean(err[..., 0] ** 2))),
        "vm_mae_pu": float(np.mean(np.abs(err[..., 1]))),
        "vm_rmse_pu": float(np.sqrt(np.mean(err[..., 1] ** 2))),
        "vm_max_abs_error_pu": float(np.max(np.abs(err[..., 1]))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G01_supervised_dataset")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=20271001)
    args = p.parse_args()
    if min(args.epochs, args.hidden, args.layers, args.patience) < 1:
        p.error("epochs, hidden, layers, patience must be positive")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = Path(args.data)
    train_x, train_y = load_split(data / "train.npz")
    val_x, val_y = load_split(data / "val.npz")
    test_x, test_y = load_split(data / "test.npz")
    graph = np.load(data / "graph.npz")

    # Statistics are computed on TRAIN ONLY. This is a leakage guardrail.
    x_mean = train_x.mean(axis=(0, 1), keepdims=True)
    x_std = train_x.std(axis=(0, 1), keepdims=True)
    y_mean = train_y.mean(axis=(0, 1), keepdims=True)
    y_std = train_y.std(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x_std, 1e-6); y_std = np.maximum(y_std, 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx = torch.tensor(standardize(train_x, x_mean, x_std), device=device)
    ty = torch.tensor(standardize(train_y, y_mean, y_std), device=device)
    vx = torch.tensor(standardize(val_x, x_mean, x_std), device=device)
    vy = torch.tensor(standardize(val_y, y_mean, y_std), device=device)
    ex = torch.tensor(standardize(test_x, x_mean, x_std), device=device)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    edge_attr_np = graph["edge_features"].astype(np.float32)
    ea_mean = edge_attr_np.mean(axis=0, keepdims=True)
    ea_std = np.maximum(edge_attr_np.std(axis=0, keepdims=True), 1e-6)
    edge_attr = torch.tensor((edge_attr_np - ea_mean) / ea_std, device=device)

    model = SupervisedGraphOPF(hidden_dim=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(tx, edge_index, edge_attr), ty)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vx, edge_index, edge_attr), vy).item()
        if val_loss < best - 1e-8:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:4d} train_loss={loss.item():.6f} val_loss={val_loss:.6f}", flush=True)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} best_val_loss={best:.6f}", flush=True)
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_z = model(ex, edge_index, edge_attr).cpu().numpy()
    pred = pred_z * y_std + y_mean
    result = metrics(pred, test_y)
    result.update({"device": str(device), "best_val_standardized_mse": best,
                   "n_train": len(train_x), "n_val": len(val_x), "n_test": len(test_x)})
    out = data / "g02_metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"model_state": best_state, "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std, "edge_mean": ea_mean,
                "edge_std": ea_std, "hidden": args.hidden, "layers": args.layers,
                "seed": args.seed}, data / "g02_model.pt")
    print("G02 HELD-OUT TEST")
    for k, v in result.items(): print(f"{k}={v}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
