"""G05: identify OPF regimes behind extreme LMP labels in G03.

This is a dataset diagnostic, not a model-training step.  It ranks scenarios by
maximum absolute train-standardized LMP and reports load/objective/voltage
statistics so isolated price spikes can be distinguished from broad dataset
shift.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_split(path: Path):
    z = np.load(path)
    return (z["x"].astype(np.float64), z["y"].astype(np.float64),
            z["objective"].astype(np.float64) if "objective" in z.files else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/G03_gnn_batch")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--z-threshold", type=float, default=10.0)
    args = p.parse_args()
    data = Path(args.data)

    splits = {name: load_split(data / f"{name}.npz")
              for name in ("train", "val", "test")}
    train_y = splits["train"][1]
    mu = float(train_y[..., 0].mean())
    sd = max(float(train_y[..., 0].std()), 1e-12)

    print("G05 OPF REGIME AUDIT")
    print(f"train LMP normalization: mean={mu:.8g} std={sd:.8g}")
    print(f"extreme threshold: max_bus_abs_zLMP >= {args.z_threshold:g}\n")

    rows = []
    for name, (x, y, obj) in splits.items():
        zlmp = (y[..., 0] - mu) / sd
        scen_max_z = np.max(np.abs(zlmp), axis=1)
        scen_max_lmp = np.max(y[..., 0], axis=1)
        scen_min_lmp = np.min(y[..., 0], axis=1)
        extreme = scen_max_z >= args.z_threshold
        pd = x[..., 0].sum(axis=1)
        qd = x[..., 1].sum(axis=1)
        vm_min = y[..., 1].min(axis=1)
        vm_max = y[..., 1].max(axis=1)

        print(f"{name.upper()}: n={len(y)} extreme={int(extreme.sum())} "
              f"fraction={extreme.mean():.6f} max_abs_z={scen_max_z.max():.3f}")
        if extreme.any():
            print(f"  extreme total_PD mean/range={pd[extreme].mean():.4f}/"
                  f"[{pd[extreme].min():.4f},{pd[extreme].max():.4f}]")
            print(f"  regular total_PD mean/range={pd[~extreme].mean():.4f}/"
                  f"[{pd[~extreme].min():.4f},{pd[~extreme].max():.4f}]" if (~extreme).any() else "  no regular scenarios")
        order = np.argsort(scen_max_z)[::-1][:args.top]
        print("  top scenarios:")
        for rank, i in enumerate(order, 1):
            objective = float(np.asarray(obj[i]).reshape(-1)[0]) if obj is not None else float("nan")
            print(f"  {rank:2d} scenario={i:4d} max_abs_z={scen_max_z[i]:9.3f} "
                  f"LMP=[{scen_min_lmp[i]:.4f},{scen_max_lmp[i]:.4f}] "
                  f"PD={pd[i]:.4f} QD={qd[i]:.4f} VM=[{vm_min[i]:.5f},{vm_max[i]:.5f}] "
                  f"objective={objective:.6g}")
            rows.append(dict(split=name, rank=rank, scenario=int(i),
                             max_abs_z_lmp=float(scen_max_z[i]),
                             min_lmp=float(scen_min_lmp[i]), max_lmp=float(scen_max_lmp[i]),
                             total_pd_mw=float(pd[i]), total_qd_mvar=float(qd[i]),
                             min_vm_pu=float(vm_min[i]), max_vm_pu=float(vm_max[i]),
                             objective=objective, extreme=bool(extreme[i])))
        print()

    out_csv = data / "g05_regime_audit.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for name, (_, y, _) in splits.items():
        z = np.max(np.abs((y[..., 0] - mu) / sd), axis=1)
        summary[name] = {"n": int(len(z)), "extreme_count": int((z >= args.z_threshold).sum()),
                         "extreme_fraction": float((z >= args.z_threshold).mean()),
                         "max_abs_z_lmp": float(z.max())}
    (data / "g05_regime_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"CSV output: {out_csv}")
    print(f"Summary: {data / 'g05_regime_summary.json'}")


if __name__ == "__main__":
    main()
