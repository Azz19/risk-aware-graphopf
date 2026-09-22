import numpy as np
import pytest
from pypower.case9 import case9
from pypower.idx_bus import PD, QD

from graphopf.supervised_data import (
    build_example, electrical_graph, sample_operating_case,
)


def test_graph_has_bidirectional_edges_and_no_inactive_branches():
    case = case9()
    case["branch"][0, 10] = 0  # BR_STATUS
    edge_index, features = electrical_graph(case)
    assert edge_index.shape == (2, 2 * (len(case["branch"]) - 1))
    assert features.shape == (edge_index.shape[1], 6)
    directed = {tuple(edge) for edge in edge_index.T}
    assert all((v, u) in directed for u, v in directed)


def test_load_sample_is_reproducible_and_does_not_mutate_case():
    base = case9()
    original = base["bus"].copy()
    a = sample_operating_case(base, np.random.default_rng(42), 0.05)
    b = sample_operating_case(base, np.random.default_rng(42), 0.05)
    np.testing.assert_allclose(a["bus"], b["bus"])
    np.testing.assert_array_equal(base["bus"], original)
    assert np.all(a["bus"][:, PD] >= 0)
    assert np.all(a["bus"][:, QD] >= 0)


def test_example_rejects_unsolved_opf():
    case = case9()
    with pytest.raises(ValueError, match="did not solve"):
        build_example(case, {"success": False})
