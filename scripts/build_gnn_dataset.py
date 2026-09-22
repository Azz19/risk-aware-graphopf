"""Generate disjoint supervised OPF splits for the first GNN baseline.

Usage: python -u scripts/build_gnn_dataset.py --train 100 --val 20 --test 20
Outputs contain OPF labels; never use held-out labels as model features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from graphopf.powerflow import load_case
from graphopf.supervised_data import electrical_graph, generate_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/case57.yaml")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20270901)
    parser.add_argument("--load-sigma", type=float, default=0.05)
    parser.add_argument("--output", default="results/G01_supervised_dataset")
    args = parser.parse_args()
    if min(args.train, args.val, args.test) < 1 or args.load_sigma < 0:
        parser.error("train, val, test must be positive; load-sigma nonnegative")
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = load_case(cfg["case"]["path"])
    edges, edge_features = electrical_graph(base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "graph.npz", edge_index=edges,
                        edge_features=edge_features,
                        bus_ids=base["bus"][:, 0].astype(np.int64))
    metadata = dict(case=cfg["case"]["name"], case_path=cfg["case"]["path"],
                    seed=args.seed, load_sigma=args.load_sigma,
                    node_features=["PD_MW", "QD_MVAr", "generator_count",
                                   "sum_PMIN_MW", "sum_PMAX_MW"],
                    targets=["LAM_P_currency_per_MWh", "VM_pu"],
                    edge_features=["R_pu", "X_pu", "B_pu", "RATE_A_MVA",
                                   "tap", "shift_deg"],
                    n_bus=len(base["bus"]), n_directed_edges=edges.shape[1],
                    splits={})
    for i, (name, n) in enumerate((("train", args.train),
                                    ("val", args.val), ("test", args.test))):
        split = generate_split(base, n, args.seed + i, args.load_sigma)
        np.savez_compressed(out / f"{name}.npz", x=split["x"],
                            y=split["y"], objective=split["objective"])
        metadata["splits"][name] = dict(n=n, seed=split["seed"],
                                         attempts=split["attempts"])
        print(f"{name}: {n} feasible OPF labels / {split['attempts']} attempts; "
              f"x={split['x'].shape} y={split['y'].shape}", flush=True)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset output: {out}", flush=True)


if __name__ == "__main__":
    main()
