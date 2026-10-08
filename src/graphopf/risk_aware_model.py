"""Risk-aware heterogeneous graph policy for unsupervised AC-OPF.

The policy consumes only exogenous/static bus information. It predicts active
generation, generator-voltage setpoints, and nonnegative AGC participation.
Fixed-range generators are excluded from AGC participation by construction:
a unit with PMAX == PMIN has zero physical active-power recourse capability.
No OPF solution labels are used by this module.
"""
from __future__ import annotations
import torch
from torch import nn
from .gnn_baseline import EdgeMessageLayer


class RiskAwareGraphOPF(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden_dim=128, layers=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList(EdgeMessageLayer(hidden_dim, edge_dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def forward(self, x, edge_index, edge_attr, gen_mask, pmin, pmax, vmin, vmax):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        raw = self.head(h)
        mask = gen_mask.to(dtype=x.dtype).unsqueeze(0)
        pg = pmin + (pmax - pmin) * torch.sigmoid(raw[..., 0])
        vg = vmin + (vmax - vmin) * torch.sigmoid(raw[..., 1])

        # Only generators with nonzero active-power range can provide AGC
        # recourse. This structural mask prevents zero-range units from being
        # moved away from their mandatory PG setpoint under any nonzero mismatch.
        flexible = gen_mask & ((pmax - pmin) > 1e-9)
        flex_mask = flexible.to(dtype=x.dtype).unsqueeze(0)
        logits = raw[..., 2].masked_fill(~flexible.unsqueeze(0), -1e9)
        alpha = torch.softmax(logits, dim=1) * flex_mask
        return {"pg_bus": pg * mask, "vg_bus": vg, "alpha_bus": alpha}
