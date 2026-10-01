import numpy as np

from graphopf.gnn_baseline import SupervisedGraphOPF
from graphopf.supervised_data import electrical_graph


def test_edge_model_rejects_zero_layers():
    try:
        SupervisedGraphOPF(layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-layer model should be rejected")


def test_electrical_graph_has_symmetric_directed_edges():
    # Minimal PYPOWER-style arrays with one in-service branch. Only columns used
    # by electrical_graph need meaningful values.
    bus = np.zeros((2, 13), dtype=float)
    bus[:, 0] = [1, 2]
    branch = np.zeros((1, 13), dtype=float)
    branch[0, 0:2] = [1, 2]
    branch[0, 2:6] = [0.01, 0.1, 0.02, 100.0]
    branch[0, 8] = 0.0
    branch[0, 9] = 0.0
    branch[0, 10] = 1.0
    edge_index, edge_attr = electrical_graph({"bus": bus, "branch": branch})
    assert edge_index.shape == (2, 2)
    assert edge_attr.shape == (2, 6)
    assert tuple(edge_index[:, 0]) == (0, 1)
    assert tuple(edge_index[:, 1]) == (1, 0)
    np.testing.assert_allclose(edge_attr[0], edge_attr[1])
    assert edge_attr[0, 4] == 1.0
