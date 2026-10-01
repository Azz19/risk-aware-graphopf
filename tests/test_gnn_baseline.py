import torch

from graphopf.gnn_baseline import SupervisedGraphOPF


def test_gnn_output_shape_and_gradient():
    torch.manual_seed(1)
    model = SupervisedGraphOPF(node_dim=5, edge_dim=6, hidden_dim=16, layers=2, out_dim=2)
    x = torch.randn(4, 5, 5)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]])
    edge_attr = torch.randn(edge_index.shape[1], 6)
    y = model(x, edge_index, edge_attr)
    assert y.shape == (4, 5, 2)
    y.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
