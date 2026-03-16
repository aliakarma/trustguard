"""
trustguard/marl/centralized_critic.py
========================================
Centralised critic V_ψ(s) for TrustGuard CTDE training.

During training, the critic receives the full global state sₜ and outputs a
scalar value estimate used to compute GAE-λ advantages for all three agents.
At deployment, the critic is discarded and each agent executes its actor
using only local observations.

Global state sₜ composition (see §4.2 of the paper):
  sₜ = (Uᵗ, ρᵗ, Eᵗ)
  where
    Uᵗ ∈ {0,1}^(N × |𝒫|)  — stacked permission usage profiles
    ρᵗ ∈ [0,1]^N           — latent risk levels
    Eᵗ ∈ ℝ^(N × d)         — application semantic embeddings

The state is flattened to a single vector before being fed to the critic.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class CentralizedCritic(nn.Module):
    """
    Centralised value network V_ψ(s).

    Parameters
    ----------
    state_dim : int
        Dimension of the flattened global state vector.
    hidden_dims : tuple[int, ...]
        Widths of the hidden layers.
    dropout : float

    Example
    -------
    >>> critic = CentralizedCritic(state_dim=2048)
    >>> s = torch.randn(8, 2048)
    >>> v = critic(s)
    >>> v.shape
    torch.Size([8])
    """

    def __init__(
        self,
        state_dim:   int = 2048,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Value head: orthogonal init with gain=1
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, global_state: Tensor) -> Tensor:
        """
        Parameters
        ----------
        global_state : Tensor  shape (B, state_dim)

        Returns
        -------
        value : Tensor  shape (B,)
        """
        return self.net(global_state).squeeze(-1)

    # ------------------------------------------------------------------
    @staticmethod
    def build_global_state(
        usage_matrix:  Tensor,   # (B, N, |𝒫|)
        risk_vector:   Tensor,   # (B, N)
        embeddings:    Tensor,   # (B, N, d)
    ) -> Tensor:
        """
        Flatten and concatenate the three global state components.

        Parameters
        ----------
        usage_matrix : Tensor  shape (B, N, |𝒫|)
        risk_vector  : Tensor  shape (B, N)
        embeddings   : Tensor  shape (B, N, d)

        Returns
        -------
        state : Tensor  shape (B, N*|𝒫| + N + N*d)
        """
        B = usage_matrix.shape[0]
        u = usage_matrix.flatten(1)      # (B, N*|𝒫|)
        r = risk_vector                  # (B, N)
        e = embeddings.flatten(1)        # (B, N*d)
        return torch.cat([u, r, e], dim=-1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"state_dim={self.state_dim}"
