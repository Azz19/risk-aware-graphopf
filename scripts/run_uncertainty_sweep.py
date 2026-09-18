from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    config_path = Path(args.config)
    root = Path("results/E02_uncertainty_sweep")
    root.mkdir(parents=True, exist_ok=True)
    frames = []

    for level in LEVELS:
        cfg = yaml.safe_load(config_path.read_text())
        cfg["uncertainty"]["relative_sigma"] = level
        cfg["experiment"]["n_samples"] = args.n_samples
        cfg["experiment"]["output_dir"] = str(root / f"sigma_{level:g}")
        temp = root / f"config_sigma_{level:g}.yaml"
        temp.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"\n=== relative_sigma={level:g}, n={args.n_samples} per family ===")
        subprocess.run(
            [sys.executable, "scripts/run_monte_carlo.py", "--config", str(temp)],
            check=True,
        )
        summary = pd.read_csv(Path(cfg["experiment"]["output_dir"]) / "summary.csv")
        summary.insert(0, "relative_sigma", level)
        frames.append(summary)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "sweep_summary.csv", index=False)
    cols = [
        "relative_sigma", "family", "joint_violation_rate",
        "voltage_violation_rate", "pg_violation_rate",
        "qg_violation_rate", "thermal_violation_rate",
    ]
    print("\n=== Combined sweep ===")
    print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
