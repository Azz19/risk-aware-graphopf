"""Run the complete current GNN experiment batch (G01-G03).

Stages
------
1. Optional pytest regression suite.
2. Build a disjoint OPF-labelled dataset.
3. Train/evaluate the supervised edge-aware GNN.
4. Audit dataset/model outputs and write one machine-readable summary.

Example:
  python -u scripts/run_gnn_experiments.py

Fast smoke test:
  python -u scripts/run_gnn_experiments.py --train 100 --val 20 --test 20 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val", type=int, default=400)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--load-sigma", type=float, default=0.05)
    p.add_argument("--dataset-seed", type=int, default=20270901)
    p.add_argument("--model-seed", type=int, default=20271001)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--output", default="results/G03_gnn_batch")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()
    if min(args.train, args.val, args.test, args.epochs, args.hidden, args.layers, args.patience) < 1:
        p.error("sample counts and training integer arguments must be positive")
    if args.load_sigma < 0 or args.lr <= 0:
        p.error("load-sigma must be nonnegative and lr positive")

    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"])

    run([py, "-u", "scripts/build_gnn_dataset.py",
         "--train", str(args.train), "--val", str(args.val), "--test", str(args.test),
         "--seed", str(args.dataset_seed), "--load-sigma", str(args.load_sigma),
         "--output", args.output])

    run([py, "-u", "scripts/train_gnn_baseline.py",
         "--data", args.output, "--epochs", str(args.epochs),
         "--hidden", str(args.hidden), "--layers", str(args.layers),
         "--lr", str(args.lr), "--patience", str(args.patience),
         "--seed", str(args.model_seed)])

    metadata = json.loads((out / "metadata.json").read_text())
    metrics = json.loads((out / "g02_metrics.json").read_text())
    graph = np.load(out / "graph.npz")
    train = np.load(out / "train.npz")
    val = np.load(out / "val.npz")
    test = np.load(out / "test.npz")

    checks = {
        "all_finite": bool(all(np.isfinite(a).all() for a in
            (train["x"], train["y"], val["x"], val["y"], test["x"], test["y"], graph["edge_features"]))),
        "node_count_consistent": bool(train["x"].shape[1] == val["x"].shape[1] == test["x"].shape[1] == metadata["n_bus"]),
        "target_dim_is_2": bool(train["y"].shape[-1] == val["y"].shape[-1] == test["y"].shape[-1] == 2),
        "directed_edge_count_consistent": bool(graph["edge_index"].shape[1] == metadata["n_directed_edges"]),
        "expected_split_sizes": bool(len(train["x"]) == args.train and len(val["x"]) == args.val and len(test["x"]) == args.test),
    }
    if not all(checks.values()):
        raise RuntimeError(f"G03 audit failed: {checks}")

    summary = {
        "experiment": "G03_gnn_batch",
        "purpose": "scaled supervised GNN baseline before physical/risk-aware evaluation",
        "configuration": vars(args),
        "dataset": metadata,
        "held_out_regression_metrics": metrics,
        "audit": checks,
        "interpretation_guardrail": (
            "Regression accuracy alone does not establish AC feasibility, calibration, "
            "robustness under renewable uncertainty, or superiority of risk-aware training."
        ),
    }
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nG03 BATCH COMPLETE", flush=True)
    print(f"dataset={args.train}/{args.val}/{args.test}", flush=True)
    print(f"lmp_mae={metrics['lmp_mae']:.8f}", flush=True)
    print(f"vm_mae_pu={metrics['vm_mae_pu']:.8f}", flush=True)
    print(f"vm_max_abs_error_pu={metrics['vm_max_abs_error_pu']:.8f}", flush=True)
    print(f"Summary: {out / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
