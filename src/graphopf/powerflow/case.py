from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
from pypower.api import ppoption, runopf
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I


def _parse_matpower_matrix(text: str, name: str) -> np.ndarray:
    pattern = rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find mpc.{name} in MATPOWER case")

    rows = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(";").strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])

    if not rows:
        raise ValueError(f"mpc.{name} is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent row width in mpc.{name}")
    return np.asarray(rows, dtype=float)


def _parse_matpower_m_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    version_match = re.search(r"mpc\.version\s*=\s*['\"]([^'\"]+)['\"]\s*;", text)
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE+\-.]+)\s*;", text)
    if not base_match:
        raise ValueError("Could not find mpc.baseMVA in MATPOWER case")

    ppc = {
        "version": version_match.group(1) if version_match else "2",
        "baseMVA": float(base_match.group(1)),
        "bus": _parse_matpower_matrix(text, "bus"),
        "gen": _parse_matpower_matrix(text, "gen"),
        "branch": _parse_matpower_matrix(text, "branch"),
        "gencost": _parse_matpower_matrix(text, "gencost"),
    }
    return ppc


def load_case(path: str):
    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise FileNotFoundError(
            f"MATPOWER/PGLib case not found: {case_path}\n"
            "Check the YAML path and your working directory."
        )

    if case_path.suffix.lower() == ".m":
        return _parse_matpower_m_file(case_path)

    from pypower.loadcase import loadcase

    ppc = loadcase(str(case_path))
    if not isinstance(ppc, dict):
        raise RuntimeError(f"Could not load PYPOWER case: {case_path}")
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
