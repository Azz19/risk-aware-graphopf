from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pypower.idx_bus import BUS_I, BUS_TYPE, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN

from graphopf.experiments import renewable_forecast, select_renewable_buses
from graphopf.powerflow import load_case, solve_ac_opf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E01_uncertainty.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = load_case(cfg["case"]["path"])
    buses = select_renewable_buses(raw, int(cfg["renewables"]["n_sites"]))
    forecast = renewable_forecast(raw, buses, float(cfg["renewables"]["penetration"]))
    case = copy.deepcopy(raw)
    lookup = {int(row[BUS_I]): i for i, row in enumerate(case["bus"])}
    for b, p in zip(buses, forecast):
        case["bus"][lookup[int(b)], 2] -= p

    opf = solve_ac_opf(case)
    active = opf["gen"][:, GEN_STATUS] > 0

    gen_rows = []
    for i, g in enumerate(opf["gen"]):
        if not active[i]:
            continue
        gen_rows.append({
            "gen_index": i,
            "bus": int(g[GEN_BUS]),
            "pg_mw": g[PG],
            "pmin_mw": g[PMIN],
            "pmax_mw": g[PMAX],
            "down_p_margin_mw": g[PG] - g[PMIN],
            "up_p_margin_mw": g[PMAX] - g[PG],
            "qg_mvar": g[QG],
            "qmin_mvar": g[QMIN],
            "qmax_mvar": g[QMAX],
            "down_q_margin_mvar": g[QG] - g[QMIN],
            "up_q_margin_mvar": g[QMAX] - g[QG],
            "q_range_mvar": g[QMAX] - g[QMIN],
        })

    bus_rows = []
    gen_bus_ids = set(opf["gen"][active, GEN_BUS].astype(int))
    for row in opf["bus"]:
        bus_id = int(row[BUS_I])
        bus_rows.append({
            "bus": bus_id,
            "bus_type": int(row[BUS_TYPE]),
            "is_generator_bus": bus_id in gen_bus_ids,
            "vm_pu": row[VM],
            "vmin_pu": row[VMIN],
            "vmax_pu": row[VMAX],
            "lower_v_margin_pu": row[VM] - row[VMIN],
            "upper_v_margin_pu": row[VMAX] - row[VM],
        })

    gf = pd.DataFrame(gen_rows)
    bf = pd.DataFrame(bus_rows)
    out = Path("results/E03_vq_margin_audit")
    out.mkdir(parents=True, exist_ok=True)
    gf.to_csv(out / "generator_margins.csv", index=False)
    bf.to_csv(out / "bus_voltage_margins.csv", index=False)

    qtol = 1e-4
    vnear = 1e-3
    print(f"objective={float(opf['f']):.6f}")
    print("\nACTIVE GENERATORS -- sorted by nearest Q limit")
    showg = gf.assign(nearest_q_margin=gf[["down_q_margin_mvar","up_q_margin_mvar"]].min(axis=1))
    print(showg.sort_values("nearest_q_margin").to_string(index=False))
    print("\nBUSES -- 15 smallest voltage margins")
    showb = bf.assign(nearest_v_margin=bf[["lower_v_margin_pu","upper_v_margin_pu"]].min(axis=1))
    print(showb.sort_values("nearest_v_margin").head(15).to_string(index=False))
    print("\nAUDIT SUMMARY")
    print(f"active_generators={len(gf)}")
    print(f"generators_at_qmin={(gf['down_q_margin_mvar'] <= qtol).sum()}")
    print(f"generators_at_qmax={(gf['up_q_margin_mvar'] <= qtol).sum()}")
    print(f"generator_buses_within_0.001pu_voltage_limit={((bf['is_generator_bus']) & (showb['nearest_v_margin'] <= vnear)).sum()}")
    print(f"all_buses_within_0.001pu_voltage_limit={(showb['nearest_v_margin'] <= vnear).sum()}")
    print(f"minimum_voltage_margin_pu={showb['nearest_v_margin'].min():.9f}")
    print(f"minimum_q_margin_mvar={showg['nearest_q_margin'].min():.9f}")


if __name__ == "__main__":
    main()
