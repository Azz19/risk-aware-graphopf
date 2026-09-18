from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from graphopf.powerflow import evaluate_constraints, load_case, solve_ac_opf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ppc = load_case(cfg["case"]["path"])
    result = solve_ac_opf(ppc, cfg["solver"].get("opf_verbose", False))
    metrics = evaluate_constraints(result, cfg["solver"]["feasibility_tolerance"])

    print(f'case={cfg["case"]["name"]}')
    print(f'objective={result["f"]:.8f}')
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
