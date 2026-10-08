"""Recover R1 diagnostic validation from the saved post-training checkpoint.

Does not train, choose a model, or use frozen R1 calibration/test seeds.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r1_risk_aware_calibration import setup, zero_error_audit, authoritative_eval
from graphopf.risk_aware_model import RiskAwareGraphOPF

def encode_np(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/R1_risk_aware_calibration.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force-eval", action="store_true",
                        help="Re-run the same independent 1000 diagnostic scenarios")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["experiment"]["output_dir"])
    ck_path = Path(args.checkpoint) if args.checkpoint else out / "r1_pretrain_risk_continuation.pt"
    if not ck_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}. Do not retrain until checked.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, case, _, rb, fc, corr, _, t = setup(cfg, device)
    checkpoint = torch.load(ck_path, map_location=device, weights_only=False)
    model = RiskAwareGraphOPF(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["message_passing_layers"])
    ).to(device).double()
    model.load_state_dict(checkpoint["state"])
    model.eval()
    z = zero_error_audit(cfg, model, t, case, rb, fc)
    seed = int(cfg["uncertainty"]["train_seed"]) + 1991
    scenario_file = out / "r1_pretrain_risk_validation_scenarios.csv"
    if args.force_eval or not scenario_file.is_file():
        val = authoritative_eval(
            cfg, model, t, case, rb, fc, corr, seed, 1000,
            cfg["uncertainty"]["train_family"], scenario_file
        )
        mode = "authoritative power flow rerun; no training"
    else:
        import pandas as pd
        from graphopf.metrics import wilson_interval
        df = pd.read_csv(scenario_file)
        if len(df) != 1000 or "joint_violation" not in df:
            raise ValueError("Scenario CSV incomplete; run again with --force-eval")
        def rate(col):
            if col not in df:
                raise ValueError(f"Missing column {col} in scenario CSV; run with --force-eval")
            return float((df[col].fillna(0).astype(float) > float(cfg["experiment"]["feasibility_tolerance"])).sum() / len(df))
        joint = int(df["joint_violation"].astype(str).str.lower().eq("true").sum())
        nonconv = int(df["pf_converged"].astype(str).str.lower().eq("false").sum())
        lo, hi = wilson_interval(joint, len(df), float(cfg["experiment"]["confidence_level"]))
        eps = float(cfg["experiment"]["target_epsilon"])
        val = {
            "n": len(df), "joint_violation_rate": joint / len(df),
            "joint_ci_low": lo, "joint_ci_high": hi,
            "meets_5pct_point_estimate": joint / len(df) <= eps,
            "meets_5pct_upper_ci": hi <= eps,
            "voltage_violation_rate": rate("max_voltage_violation_pu"),
            "pg_violation_rate": rate("max_pg_violation_mw"),
            "qg_violation_rate": rate("max_qg_violation_mvar"),
            "thermal_violation_rate": rate("max_thermal_overload_pu"),
            "pf_nonconvergence_rate": nonconv / len(df)
        }
        mode = "existing scenario CSV; no training or PF rerun"
    summary = {
        "zero_error_audit": z, "diagnostic_validation": val,
        "diagnostic_seed": seed, "recovery": mode,
        "checkpoint": str(ck_path),
        "note": "Diagnostic only; frozen calibration/test remain untouched."
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "r1_pretrain_risk_summary.json").write_text(
        json.dumps(summary, indent=2, default=encode_np) + "\n"
    )
    print("\nR1 PRETRAIN -> RISK FINAL (RECOVERED)\n" +
          json.dumps(summary, indent=2, default=encode_np))

if __name__ == "__main__":
    main()
