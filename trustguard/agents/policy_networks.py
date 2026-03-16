"""
trustguard/agents/policy_networks.py
======================================
Shared policy network building blocks for TrustGuard's three Dec-POMDP agents.

Provides:
  - ``ActorNetwork``   : stochastic policy π^k(aᵏ | oᵏ; θᵏ)
  - ``CriticNetwork``  : centralised value function V_ψ(s)
  - ``EnforcementHead``: action-specific output head for the Enforcement Agent
    with support for discrete multi-categorical action selection (revoke,
    rate-limit, alert, no-op) and per-permission targeting.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Bernoulli

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def _build_mlp(
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    activation: type[nn.Module] = nn.Tanh,
    dropout: float = 0.0,
    layer_norm: bool = True,
) -> nn.Sequential:
    """Utility to build a fully-connected MLP."""
    layers: list[nn.Module] = []
    current = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(current, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = h
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """
    Stochastic actor for a discrete action space.

    Outputs a categorical distribution over ``action_dim`` actions.

    Parameters
    ----------
    obs_dim : int
        Dimension of the agent's local observation (or belief projection).
    action_dim : int
        Number of discrete actions.
    hidden_dims : tuple[int, ...]
        Widths of MLP hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            in_dim=obs_dim,
            hidden_dims=hidden_dims,
            out_dim=action_dim,
            dropout=dropout,
        )
        # Orthogonal init for the output layer (common in PPO)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, obs: Tensor) -> Categorical:
        """
        Parameters
        ----------
        obs : Tensor  shape (B, obs_dim)

        Returns
        -------
        dist : Categorical  over action_dim actions
        """
        logits = self.net(obs)
        return Categorical(logits=logits)

    # ------------------------------------------------------------------
    def get_action_and_log_prob(
        self, obs: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Sample an action and return its log-probability.

        Parameters
        ----------
        obs : Tensor  shape (B, obs_dim)
        deterministic : bool
            If True, return the greedy argmax action.

        Returns
        -------
        action   : Tensor  shape (B,)  — integer action indices
        log_prob : Tensor  shape (B,)
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    # ------------------------------------------------------------------
    def evaluate_actions(
        self, obs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate log-probabilities and entropy for recorded actions.

        Parameters
        ----------
        obs     : Tensor  shape (B, obs_dim)
        actions : Tensor  shape (B,)

        Returns
        -------
        log_probs : Tensor  shape (B,)
        entropy   : Tensor  scalar (mean entropy)
        """
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy().mean()


# ─────────────────────────────────────────────────────────────────────────────
class CriticNetwork(nn.Module):
    """
    Centralised critic V_ψ(s) with access to the full global state.

    Used during training only (CTDE). At deployment, each agent runs its
    actor with local observations only.

    Parameters
    ----------
    state_dim : int
        Dimension of the global state vector.
    hidden_dims : tuple[int, ...]
    dropout : float
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            in_dim=state_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            dropout=dropout,
        )
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


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement action types
# ─────────────────────────────────────────────────────────────────────────────
ENFORCEMENT_ACTIONS = ["no_op", "alert", "rate_limit", "revoke"]
NUM_ENFORCEMENT_ACTIONS = len(ENFORCEMENT_ACTIONS)
ACTION_NO_OP      = 0
ACTION_ALERT      = 1
ACTION_RATE_LIMIT = 2
ACTION_REVOKE     = 3


class EnforcementHead(nn.Module):
    """
    Two-head output for the Enforcement Agent:

      1. **Action type head**: selects from {no_op, alert, rate_limit, revoke}
         as a 4-way Categorical distribution.
      2. **Permission target head**: for actions that target specific permissions
         (rate_limit, revoke), selects which permission via a Bernoulli mask
         over all |𝒫| permissions.

    Parameters
    ----------
    belief_dim : int
        Input dimension (belief vector from BeliefEncoder).
    num_permissions : int
        |𝒫|, the total number of permissions.
    hidden_dims : tuple[int, ...]
    dropout : float
    """

    def __init__(
        self,
        belief_dim: int = 256,
        num_permissions: int = 42,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_permissions = num_permissions

        # Shared trunk
        self.trunk = _build_mlp(
            in_dim=belief_dim,
            hidden_dims=hidden_dims,
            out_dim=hidden_dims[-1],
            dropout=dropout,
        )

        # Action type head
        self.action_head = nn.Linear(hidden_dims[-1], NUM_ENFORCEMENT_ACTIONS)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

        # Permission target head (binary per permission)
        self.permission_head = nn.Linear(hidden_dims[-1], num_permissions)
        nn.init.orthogonal_(self.permission_head.weight, gain=0.01)

    # ------------------------------------------------------------------
    def forward(
        self, belief: Tensor
    ) -> tuple[Categorical, Bernoulli]:
        """
        Parameters
        ----------
        belief : Tensor  shape (B, belief_dim)

        Returns
        -------
        action_dist     : Categorical  over 4 action types
        permission_dist : Bernoulli    over num_permissions permission targets
        """
        trunk_out = self.trunk(belief)   # (B, hidden)

        action_logits     = self.action_head(trunk_out)       # (B, 4)
        permission_logits = self.permission_head(trunk_out)   # (B, |𝒫|)

        return Categorical(logits=action_logits), Bernoulli(logits=permission_logits)

    # ------------------------------------------------------------------
    def select_action(
        self,
        belief: Tensor,
        risk_vector: Tensor,
        risk_threshold: float = 0.5,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample enforcement action type and targeted permissions.

        A risk-gated override forces no_op when the EMA risk is below
        ``risk_threshold`` to avoid spurious enforcement.

        Parameters
        ----------
        belief         : Tensor  shape (B, belief_dim)
        risk_vector    : Tensor  shape (B,)  — per-app EMA risk
        risk_threshold : float
        deterministic  : bool

        Returns
        -------
        action_type    : Tensor  shape (B,)         — integer in [0, 3]
        perm_targets   : Tensor  shape (B, |𝒫|)     — binary permission mask
        action_log_prob: Tensor  shape (B,)
        perm_log_prob  : Tensor  shape (B,)
        """
        action_dist, perm_dist = self.forward(belief)

        if deterministic:
            action_type  = action_dist.probs.argmax(dim=-1)
            perm_targets = (perm_dist.probs > 0.5).float()
        else:
            action_type  = action_dist.sample()
            perm_targets = perm_dist.sample()

        # Risk-gate: force no_op for low-risk apps
        low_risk_mask = (risk_vector < risk_threshold)
        action_type = action_type.masked_fill(low_risk_mask, ACTION_NO_OP)

        action_log_prob = action_dist.log_prob(action_type)
        perm_log_prob   = perm_dist.log_prob(perm_targets).sum(dim=-1)

        return action_type, perm_targets, action_log_prob, perm_log_prob
