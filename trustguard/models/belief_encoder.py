"""
trustguard/models/belief_encoder.py
=====================================
Shared Belief State Encoder for TrustGuard's Dec-POMDP.

Implements the GRU-based belief update (Eq. 6 of the paper):

    bₜ = f_ψ(bₜ₋₁, o¹ₜ, o²ₜ, o³ₜ)

where o^k_t is the local observation of agent k at time t. The belief state
bₜ serves as a compressed sufficient statistic over the partially observable
global state and is used by the Enforcement Agent to condition its policy.

Key design choices
------------------
- The raw observations from all three agents are first projected to a common
  embedding space before being concatenated and fed to the GRU.
- A separate ``belief_projection`` head maps the GRU hidden state to the
  final belief vector consumed by the Enforcement Agent.
- The encoder is shared across all agents but each agent receives its own
  private projection of bₜ.

Reference: §5.4 (Eq. 6) of the TrustGuard paper.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class AgentObservationProjector(nn.Module):
    """
    Lightweight MLP that projects a single agent's local observation into a
    shared embedding space.

    Parameters
    ----------
    obs_dim : int
        Dimension of the agent's raw observation vector.
    embed_dim : int
        Projected embedding dimension.
    """

    def __init__(self, obs_dim: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs : Tensor  shape (..., obs_dim)

        Returns
        -------
        Tensor  shape (..., embed_dim)
        """
        return self.projector(obs)


# ─────────────────────────────────────────────────────────────────────────────
class BeliefEncoder(nn.Module):
    """
    Recurrent belief state encoder that aggregates partial observations from
    all three TrustGuard agents into a shared belief bₜ.

    Architecture
    ------------
    1. Each agent's observation oᵏₜ is projected independently to embed_dim.
    2. The three projections are concatenated: x_t ∈ ℝ^(3 × embed_dim).
    3. A GRU cell updates the hidden state: hₜ = GRU(xₜ, hₜ₋₁).
    4. A linear projection produces the final belief: bₜ = W hₜ + b.

    Parameters
    ----------
    obs_dim_monitor : int
        Observation dimension of the Monitoring Agent (agent 1).
    obs_dim_risk : int
        Observation dimension of the Risk-Analysis Agent (agent 2).
    obs_dim_enforce : int
        Observation dimension of the Enforcement Agent (agent 3).
    embed_dim : int
        Common projection dimension for each agent's observation.
    gru_hidden_dim : int
        Hidden state dimension of the GRU.
    belief_dim : int
        Output belief vector dimension (fed to policy networks).
    num_gru_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout applied between GRU layers.

    Example
    -------
    >>> encoder = BeliefEncoder(obs_dim_monitor=64,
    ...                         obs_dim_risk=128,
    ...                         obs_dim_enforce=96)
    >>> o1 = torch.randn(2, 4, 64)   # (B, T, obs_dim)
    >>> o2 = torch.randn(2, 4, 128)
    >>> o3 = torch.randn(2, 4, 96)
    >>> belief, h_n = encoder(o1, o2, o3)
    >>> belief.shape
    torch.Size([2, 4, 256])
    """

    def __init__(
        self,
        obs_dim_monitor: int = 64,
        obs_dim_risk: int = 128,
        obs_dim_enforce: int = 96,
        embed_dim: int = 128,
        gru_hidden_dim: int = 512,
        belief_dim: int = 256,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim     = embed_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.belief_dim    = belief_dim
        self.num_gru_layers = num_gru_layers

        # ── Per-agent observation projectors ──────────────────────────
        self.proj_monitor = AgentObservationProjector(obs_dim_monitor, embed_dim)
        self.proj_risk    = AgentObservationProjector(obs_dim_risk,    embed_dim)
        self.proj_enforce = AgentObservationProjector(obs_dim_enforce, embed_dim)

        # ── Recurrent core ────────────────────────────────────────────
        gru_input_dim = 3 * embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # ── Belief projection head ────────────────────────────────────
        self.belief_projection = nn.Sequential(
            nn.Linear(gru_hidden_dim, belief_dim),
            nn.LayerNorm(belief_dim),
            nn.Tanh(),   # bound belief to [-1, 1] for numerical stability
        )

        self._init_gru_weights()

    # ------------------------------------------------------------------
    def _init_gru_weights(self) -> None:
        """Orthogonal initialisation for GRU weights (improves gradient flow)."""
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1.0 for GRU-like initialisation
                # (GRU does not have a forget gate but initialising biases
                #  to ones can help gradient flow in early training)
                n = param.shape[0]
                param.data[n // 3: 2 * n // 3].fill_(1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        obs_monitor: Tensor,
        obs_risk: Tensor,
        obs_enforce: Tensor,
        h_prev: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Update the shared belief state.

        Parameters
        ----------
        obs_monitor : Tensor  shape (B, T, obs_dim_monitor)
            Sequential observations from the Monitoring Agent.
        obs_risk : Tensor  shape (B, T, obs_dim_risk)
            Sequential observations from the Risk-Analysis Agent.
        obs_enforce : Tensor  shape (B, T, obs_dim_enforce)
            Sequential observations from the Enforcement Agent.
        h_prev : Tensor, optional  shape (num_layers, B, gru_hidden_dim)
            Previous GRU hidden state. If None, initialised to zeros.

        Returns
        -------
        belief : Tensor  shape (B, T, belief_dim)
            Per-timestep belief vectors bₜ.
        h_n : Tensor  shape (num_layers, B, gru_hidden_dim)
            Final GRU hidden state (carry forward between episodes/chunks).
        """
        # ── Project each agent's observations ─────────────────────────
        e1 = self.proj_monitor(obs_monitor)    # (B, T, embed_dim)
        e2 = self.proj_risk(obs_risk)          # (B, T, embed_dim)
        e3 = self.proj_enforce(obs_enforce)    # (B, T, embed_dim)

        # ── Concatenate along feature dim ─────────────────────────────
        x = torch.cat([e1, e2, e3], dim=-1)   # (B, T, 3 * embed_dim)

        # ── GRU recurrence ────────────────────────────────────────────
        gru_out, h_n = self.gru(x, h_prev)    # (B, T, gru_hidden_dim)

        # ── Project to belief space ───────────────────────────────────
        belief = self.belief_projection(gru_out)  # (B, T, belief_dim)

        return belief, h_n

    # ------------------------------------------------------------------
    def step(
        self,
        obs_monitor: Tensor,
        obs_risk: Tensor,
        obs_enforce: Tensor,
        h_prev: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Single-step belief update for deployment (no sequence axis).

        Parameters
        ----------
        obs_monitor : Tensor  shape (B, obs_dim_monitor)
        obs_risk    : Tensor  shape (B, obs_dim_risk)
        obs_enforce : Tensor  shape (B, obs_dim_enforce)
        h_prev : Tensor, optional  shape (num_layers, B, gru_hidden_dim)

        Returns
        -------
        belief_t : Tensor  shape (B, belief_dim)
        h_n      : Tensor  shape (num_layers, B, gru_hidden_dim)
        """
        belief_seq, h_n = self.forward(
            obs_monitor.unsqueeze(1),
            obs_risk.unsqueeze(1),
            obs_enforce.unsqueeze(1),
            h_prev=h_prev,
        )
        return belief_seq.squeeze(1), h_n

    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Return a zero-initialised hidden state for a new episode.

        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        Tensor  shape (num_layers, batch_size, gru_hidden_dim)
        """
        return torch.zeros(
            self.num_gru_layers,
            batch_size,
            self.gru_hidden_dim,
            device=device,
        )

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"gru_hidden={self.gru_hidden_dim}, "
            f"belief_dim={self.belief_dim}"
        )
