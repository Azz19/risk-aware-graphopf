"""G04 audit for GNN train/validation/test distributions and normalization."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_split(path: Path):
    z = np.load(path)
    return z["x"].astype(np.float64), z["y"].astype(np.float64)


def stats(a):
    flat = a.reshape(-1)
    return {
        "mean": np.mean(flat), "std": np.std(flat), "min": np.min(flat),
        "p01": np.quantile(flat, .01), "p50": np.quantile(flat, .50),
        "p99": np.quantile(flat, .99), "max": np.max(flat),
    }


def show_stats(label, a):
    s = stats(a)
    print(f"{label:16s} " + " ".join(f"{k}={v:.8g}" for k, v in s.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    data = Path(args.data)

    splits = {}
    for name in ("train", "val", "test"):
        x, y = load_split(data / f"{name}.npz")
        splits[name] = (x, y)

    train_x, train_y = splits["train"]
    x_mean = train_x.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(train_x.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = train_y.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(train_y.std(axis=(0, 1), keepdims=True), 1e-6)

    print("G04 DATASET / NORMALIZATION AUDIT")
    print("\nRAW TARGET STATISTICS")
    for name, (_, y) in splits.items():
        show_stats(f"{name} LMP", y[..., 0])
        show_stats(f"{name} VM", y[..., 1])

    print("\nTRAIN-ONLY NORMALIZATION")
    print("y_mean=", y_mean.reshape(-1).tolist())
    print("y_std =", y_std.reshape(-1).tolist())

    print("\nSTANDARDIZED TARGET STATISTICS / ZERO-PREDICTOR MSE")
    component = {}
    for name, (_, y) in splits.items():
        yz = (y - y_mean) / y_std
        show_stats(f"{name} zLMP", yz[..., 0])
        show_stats(f"{name} zVM", yz[..., 1])
        lmp_mse = float(np.mean(yz[..., 0] ** 2))
        vm_mse = float(np.mean(yz[..., 1] ** 2))
        combined = float(np.mean(yz ** 2))
        component[name] = (lmp_mse, vm_mse, combined)
        print(f"{name:5s}: zLMP_MSE={lmp_mse:.6f} zVM_MSE={vm_mse:.6f} combined={combined:.6f}")

    print("\nINPUT FEATURE STATISTICS (raw, flattened over scenarios/buses)")
    for j in range(train_x.shape[-1]):
        vals = []
        for name, (x, _) in splits.items():
            vals.append(f"{name}:mean={x[...,j].mean():.6g},std={x[...,j].std():.6g}")
        print(f"feature_{j}: " + " | ".join(vals))

    print("\nINTEGRITY")
    for name, (x, y) in splits.items():
        print(f"{name}: x_nan={np.isnan(x).sum()} x_inf={np.isinf(x).sum()} "
              f"y_nan={np.isnan(y).sum()} y_inf={np.isinf(y).sum()}")

    # Exact scenario overlap based on x bytes.
    keys = {name: {np.ascontiguousarray(row).tobytes() for row in x} for name, (x, _) in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        print(f"exact_overlap_{a}_{b}={len(keys[a] & keys[b])}")
    for name, (x, _) in splits.items():
        print(f"duplicate_scenarios_{name}={len(x)-len(keys[name])}")

    print("\nTOP EXTREME STANDARDIZED LMP TARGETS")
    for name in ("val", "test"):
        x, y = splits[name]
        z = np.abs((y[..., 0] - y_mean[..., 0]) / y_std[..., 0])
        flat_idx = np.argsort(z.reshape(-1))[-args.top:][::-1]
        print(f"-- {name} --")
        for rank, idx in enumerate(flat_idx, 1):
            scen, bus = np.unravel_index(idx, z.shape)
            raw = y[scen, bus, 0]
            zz = (raw - y_mean.reshape(-1)[0]) / y_std.reshape(-1)[0]
            feat_mean = x[scen].mean(axis=0)
            print(f"{rank:2d} scenario={scen:4d} bus_index={bus:2d} raw_lmp={raw:.8g} "
                  f"z_lmp={zz:.4f} scenario_feature_mean={np.array2string(feat_mean, precision=5)}")

    train_lmp_var = component["train"][0]
    train_vm_var = component["train"][1]
    val_lmp_var = component["val"][0]
    val_vm_var = component["val"][1]
    print("\nDIAGNOSIS")
    print(f"val/train standardized LMP second-moment ratio={val_lmp_var/train_lmp_var:.6f}")
    print(f"val/train standardized VM second-moment ratio={val_vm_var/train_vm_var:.6f}")
    dominant = "LMP" if val_lmp_var > val_vm_var else "VM"
    print(f"dominant validation zero-predictor loss component={dominant}")
    if val_lmp_var > 10 * val_vm_var:
        print("FLAG: validation standardized loss is strongly LMP-dominated; inspect extreme LMP targets/OPF regimes.")
    elif val_vm_var > 10 * val_lmp_var:
        print("FLAG: validation standardized loss is strongly voltage-dominated.")
    else:
        print("No >10x single-target dominance in the raw standardized validation targets.")


if __name__ == "__main__":
    main()
