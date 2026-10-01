"""Small edge-aware GNN used as a supervised pipeline baseline.

This model is intentionally simple. It establishes a reproducible graph-learning
baseline before physics-informed and risk-aware components are introduced.
"""
from __future__ import annotations

import torch
from torch import nn


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        hs = h[:, src, :]
        hd = h[:, dst, :]
        ea = edge_attr.unsqueeze(0).expand(h.shape[0], -1, -1)
        msg = self.message(torch.cat((hs, hd, ea), dim=-1))
        agg = torch.zeros_like(h)
        idx = dst.view(1, -1, 1).expand(h.shape[0], -1, h.shape[-1])
        agg.scatter_add_(1, idx, msg)
        degree = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / degree.clamp_min(1).view(1, -1, 1)
        return h + self.update(torch.cat((h, agg), dim=-1))


class SupervisedGraphOPF(nn.Module):
    def __init__(self, node_dim: int = 5, edge_dim: int = 6,
                 hidden_dim: int = 64, layers: int = 3, out_dim: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            EdgeMessageLayer(hidden_dim, edge_dim) for _ in range(layers)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.decoder(h)
