"""
trustguard/agents/enforcement_agent.py
========================================
Enforcement Agent (k = 3) in TrustGuard's Dec-POMDP.

Responsibility
--------------
Conditions its policy on the shared belief state bₜ (from BeliefEncoder) and
selects enforcement actions from 𝒜³ = {no_op, alert, rate_limit, revoke}.
For targeted actions (rate_limit, revoke), also selects which permissions
to act on via the per-permission binary head in ``EnforcementHead``.

Observation space
-----------------
o³ₜ ∈ ℝ^obs_dim (concatenation of):
  - EMA risk scores ρ̄: (N_apps,)
  - belief state bₜ: (belief_dim,)
  - enforcement history (rolling K-step): mean revoke/alert rates (2,)
  → obs_dim = N_apps + belief_dim + 2

Action space
------------
  - Action type:       {no_op (0), alert (1), rate_limit (2), revoke (3)}
  - Permission target: binary mask over |𝒫| permissions

Safety constraint integration
------------------------------
The Enforcement Agent receives the Lagrange multiplier μ as an auxiliary
scalar observation during training, allowing the policy to internalise the
false-revocation penalty without needing manual threshold engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from trustguard.agents.policy_networks import (
    EnforcementHead,
    ENFORCEMENT_ACTIONS,
    ACTION_NO_OP,
    ACTION_ALERT,
    ACTION_RATE_LIMIT,
    ACTION_REVOKE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnforcementObservation:
    """
    Typed observation bundle for the Enforcement Agent.

    Attributes
    ----------
    ema_risks : Tensor  shape (B, N_apps)
    belief    : Tensor  shape (B, belief_dim)
    revoke_rate_history : Tensor  shape (B,)
        Rolling mean revocation rate over last K steps.
    alert_rate_history  : Tensor  shape (B,)
        Rolling mean alert rate over last K steps.
    lagrange_mu : Tensor  shape (B,)
        Current Lagrange multiplier (passed at training time only).
    """

    ema_risks: Tensor
    belief: Tensor
    revoke_rate_history: Tensor
    alert_rate_history: Tensor
    lagrange_mu: Optional[Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnforcementRecord:
    """Record of a single enforcement decision for audit and metrics."""

    timestep: int
    app_id: str
    action_type: int
    permission_targets: list[int]
    ema_risk_at_decision: float
    was_false_revocation: Optional[bool] = None   # labelled post-hoc if known


# ─────────────────────────────────────────────────────────────────────────────
class EnforcementAgent(nn.Module):
    """
    Autonomous enforcement agent conditioned on the shared belief state.

    The agent wraps the ``EnforcementHead`` and adds:
      - observation normalisation
      - a rolling enforcement history tracker
      - a ``apply_actions`` interface that translates the agent's decisions
        into concrete permission-state mutations in the environment

    Parameters
    ----------
    num_apps : int
    num_permissions : int
    belief_dim : int
    hidden_dims : tuple[int, ...]
    dropout : float
    risk_threshold : float
        Minimum EMA risk for any non-no_op action.
    history_window : int
        Rolling window size for computing enforcement rate history.
    """

    def __init__(
        self,
        num_apps: int = 500,
        num_permissions: int = 42,
        belief_dim: int = 256,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        risk_threshold: float = 0.5,
        history_window: int = 20,
    ) -> None:
        super().__init__()
        self.num_apps        = num_apps
        self.num_permissions = num_permissions
        self.belief_dim      = belief_dim
        self.risk_threshold  = risk_threshold
        self.history_window  = history_window

        # ── Observation dimension ─────────────────────────────────────
        # [mean_ema_risk (1), belief (belief_dim), revoke_rate (1), alert_rate (1)]
        # + optional lagrange_mu (1) concatenated during training
        self.obs_dim = 1 + belief_dim + 2

        # ── Observation projector (maps obs_dim → belief_dim as actor input) ──
        self.obs_projector = nn.Sequential(
            nn.Linear(self.obs_dim, belief_dim),
            nn.LayerNorm(belief_dim),
            nn.Tanh(),
        )

        # ── Two-head enforcement output ───────────────────────────────
        self.enforcement_head = EnforcementHead(
            belief_dim=belief_dim,
            num_permissions=num_permissions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # ── Rolling enforcement history ───────────────────────────────
        self._revoke_history: list[float] = []
        self._alert_history:  list[float] = []

        # ── Audit log ─────────────────────────────────────────────────
        self.audit_log: list[EnforcementRecord] = []

        # ── Obs normalisation ─────────────────────────────────────────
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var",  torch.ones(self.obs_dim))
        self._obs_count = 0

    # ------------------------------------------------------------------
    def build_observation(self, raw_obs: EnforcementObservation) -> Tensor:
        """
        Flatten to (B, obs_dim).

        Parameters
        ----------
        raw_obs : EnforcementObservation

        Returns
        -------
        Tensor  shape (B, obs_dim)
        """
        mean_risk = raw_obs.ema_risks.mean(dim=-1, keepdim=True)  # (B, 1)
        obs = torch.cat(
            [
                mean_risk,
                raw_obs.belief,
                raw_obs.revoke_rate_history.unsqueeze(-1),
                raw_obs.alert_rate_history.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, obs_dim)
        return obs

    # ------------------------------------------------------------------
    def normalise_obs(self, obs: Tensor) -> Tensor:
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    # ------------------------------------------------------------------
    def update_obs_stats(self, obs: Tensor) -> None:
        batch_mean = obs.mean(dim=0)
        batch_var  = obs.var(dim=0, unbiased=False)
        n = self._obs_count + obs.shape[0]
        delta = batch_mean - self.obs_mean
        self.obs_mean = self.obs_mean + delta * (obs.shape[0] / n)
        self.obs_var  = (
            (self._obs_count * self.obs_var + obs.shape[0] * batch_var)
            / n
            + delta ** 2 * (self._obs_count * obs.shape[0] / n ** 2)
        )
        self._obs_count = n

    # ------------------------------------------------------------------
    def forward(
        self,
        raw_obs: EnforcementObservation,
        deterministic: bool = False,
        update_stats: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Select enforcement action type and permission targets.

        Parameters
        ----------
        raw_obs : EnforcementObservation
        deterministic : bool
        update_stats : bool

        Returns
        -------
        action_type    : Tensor  shape (B,)       — integer in [0,3]
        perm_targets   : Tensor  shape (B, |𝒫|)   — binary mask
        action_log_prob : Tensor shape (B,)
        perm_log_prob  : Tensor  shape (B,)
        obs_norm       : Tensor  shape (B, obs_dim)
        """
        obs_flat = self.build_observation(raw_obs)
        if update_stats:
            self.update_obs_stats(obs_flat)
        obs_norm = self.normalise_obs(obs_flat)

        # Project observation to belief-dim input for EnforcementHead
        projected = self.obs_projector(obs_norm)   # (B, belief_dim)

        # Use mean EMA risk for risk-gating
        mean_risk = raw_obs.ema_risks.mean(dim=-1)  # (B,)

        action_type, perm_targets, action_log_prob, perm_log_prob = (
            self.enforcement_head.select_action(
                belief=projected,
                risk_vector=mean_risk,
                risk_threshold=self.risk_threshold,
                deterministic=deterministic,
            )
        )

        # Update enforcement history
        self._update_history(action_type)

        return action_type, perm_targets, action_log_prob, perm_log_prob, obs_norm

    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_flat: Tensor,
        action_types: Tensor,
        perm_targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate log-probs and entropy for stored actions (called by MAPPO).

        Parameters
        ----------
        obs_flat     : Tensor  shape (B, obs_dim)
        action_types : Tensor  shape (B,)
        perm_targets : Tensor  shape (B, |𝒫|)

        Returns
        -------
        action_log_prob : Tensor  shape (B,)
        perm_log_prob   : Tensor  shape (B,)
        entropy         : Tensor  scalar
        """
        from torch.distributions import Categorical, Bernoulli

        obs_norm  = self.normalise_obs(obs_flat)
        projected = self.obs_projector(obs_norm)
        action_dist, perm_dist = self.enforcement_head(projected)

        action_log_prob = action_dist.log_prob(action_types)
        perm_log_prob   = perm_dist.log_prob(perm_targets).sum(dim=-1)
        entropy         = (action_dist.entropy() + perm_dist.entropy().sum(dim=-1)).mean()

        return action_log_prob, perm_log_prob, entropy

    # ------------------------------------------------------------------
    def _update_history(self, action_types: Tensor) -> None:
        """Update rolling enforcement rate history."""
        revoke_rate = (action_types == ACTION_REVOKE).float().mean().item()
        alert_rate  = (action_types == ACTION_ALERT).float().mean().item()

        self._revoke_history.append(revoke_rate)
        self._alert_history.append(alert_rate)

        if len(self._revoke_history) > self.history_window:
            self._revoke_history.pop(0)
            self._alert_history.pop(0)

    # ------------------------------------------------------------------
    def get_enforcement_rates(self) -> tuple[float, float]:
        """
        Return rolling mean revoke rate and alert rate.

        Returns
        -------
        (revoke_rate, alert_rate) : tuple[float, float]
        """
        if not self._revoke_history:
            return 0.0, 0.0
        return (
            sum(self._revoke_history) / len(self._revoke_history),
            sum(self._alert_history)  / len(self._alert_history),
        )

    # ------------------------------------------------------------------
    def log_enforcement(
        self,
        timestep: int,
        app_id: str,
        action_type: int,
        perm_indices: list[int],
        ema_risk: float,
        was_false: Optional[bool] = None,
    ) -> None:
        """Append an enforcement record to the audit log."""
        self.audit_log.append(
            EnforcementRecord(
                timestep=timestep,
                app_id=app_id,
                action_type=action_type,
                permission_targets=perm_indices,
                ema_risk_at_decision=ema_risk,
                was_false_revocation=was_false,
            )
        )

    # ------------------------------------------------------------------
    def false_revocation_rate(self) -> float:
        """
        Compute the fraction of revocations in the audit log that were
        labelled as false positives.

        Returns
        -------
        float in [0, 1]
        """
        revocations = [
            r for r in self.audit_log
            if r.action_type == ACTION_REVOKE and r.was_false_revocation is not None
        ]
        if not revocations:
            return 0.0
        return sum(r.was_false_revocation for r in revocations) / len(revocations)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, "
            f"belief_dim={self.belief_dim}, "
            f"risk_threshold={self.risk_threshold}"
        )
