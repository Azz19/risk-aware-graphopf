from __future__ import annotations

import networkx as nx
import numpy as np
from pypower.api import ppoption, runopf
from pypower.loadcase import loadcase
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I


def load_case(path: str):
    ppc = loadcase(path)
    if not isinstance(ppc, dict):
        raise RuntimeError(f"Could not load MATPOWER/PGLib case: {path}")
    return ppc


def solve_ac_opf(ppc: dict, verbose: bool = False):
    opt = ppoption(VERBOSE=1 if verbose else 0, OUT_ALL=0)
    result = runopf(ppc, opt)
    if not result.get("success", False):
        raise RuntimeError("AC-OPF did not converge")
    return result


def topological_distance(ppc: dict, bus_ids: np.ndarray) -> np.ndarray:
    graph = nx.Graph()
    all_ids = ppc["bus"][:, BUS_I].astype(int)
    graph.add_nodes_from(all_ids.tolist())
    for row in ppc["branch"]:
        graph.add_edge(int(row[F_BUS]), int(row[T_BUS]))
    bus_ids = np.asarray(bus_ids, dtype=int)
    distance = np.zeros((bus_ids.size, bus_ids.size), dtype=float)
    for i, source in enumerate(bus_ids):
        lengths = nx.single_source_shortest_path_length(graph, int(source))
        for j, target in enumerate(bus_ids):
            if int(target) not in lengths:
                raise ValueError("Renewable buses are not in one connected component")
            distance[i, j] = lengths[int(target)]
    return distance
