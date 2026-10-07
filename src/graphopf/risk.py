"""CVaR utilities used by R1 risk-aware training."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def smooth_positive(x: torch.Tensor, temperature: float = 40.0) -> torch.Tensor:
    return F.softplus(temperature * x) / temperature


def empirical_cvar(loss: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Differentiable empirical upper-tail CVaR.

    Uses the Rockafellar-Uryasev representation with the empirical alpha
    quantile detached as the optimizing threshold. Gradients flow through tail
    excesses, not through the order statistic.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    flat = loss.reshape(-1)
    eta = torch.quantile(flat.detach(), alpha)
    return eta + torch.relu(flat - eta).mean() / (1.0 - alpha)


def risk_objective(cost, violations, alpha, multiplier, weights):
    risk = cost.mean()
    pieces = {}
    for name, value in violations.items():
        term = empirical_cvar(value, alpha)
        pieces[name] = term
        risk = risk + multiplier * float(weights.get(name, 1.0)) * term
    return risk, pieces
