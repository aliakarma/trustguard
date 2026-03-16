"""
trustguard/agents/risk_analysis_agent.py
==========================================
Risk-Analysis Agent (k = 2) in TrustGuard's Dec-POMDP.

Responsibility
--------------
Receives the permission usage delta δuᵢᵗ from the Monitoring Agent and
produces an updated risk estimate for each application, incorporating both
instantaneous deviation and semantic risk from the Permission Prediction Model.

Observation space
-----------------
o²ₜ ∈ ℝ^obs_dim:
  - δu (usage delta): (B, N_apps, |𝒫|) flattened to mean delta
  - predicted probabilities p̂: (B, |𝒫|) — forwarded from prediction model
  - current EMA risk scores ρ̄: (B, N_apps)
  → obs_dim = 2 × |𝒫| + N_apps

Action space
------------
Discrete: { DEFER, ANALYSE } — 2 actions.
ANALYSE triggers a full risk-score recomputation and forwards the updated
estimate to the Enforcement Agent.

Outputs written to shared belief
---------------------------------
Updated per-application risk vector ρᵢᵗ → forwarded to belief state and
Enforcement Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from trustguard.agents.policy_networks import ActorNetwork
from trustguard.models.runtime_risk_estimator import RuntimeRiskEstimator
from trustguard.models.permission_predictor import PermissionPredictionModel

logger = logging.getLogger(__name__)

DEFER   = 0
ANALYSE = 1


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskAnalysisObservation:
    """
    Typed observation bundle for the Risk-Analysis Agent.

    Attributes
    ----------
    usage_delta : Tensor  shape (B, N_apps, |𝒫|)
        Change in permission usage counts since last Monitoring Agent sample.
    predicted_probs : Tensor  shape (B, |𝒫|)
        Expected permission probabilities from the prediction model
        (computed once per update of the semantic encoder).
    ema_risks : Tensor  shape (B, N_apps)
        Current per-application EMA risk scores from the RuntimeRiskEstimator.
    """

    usage_delta: Tensor
    predicted_probs: Tensor
    ema_risks: Tensor


# ─────────────────────────────────────────────────────────────────────────────
class RiskAnalysisAgent(nn.Module):
    """
    Learning-based risk analysis agent.

    Takes the Monitoring Agent's sampled usage delta and current semantic
    predictions as observations; decides whether to trigger a full risk
    recomputation (ANALYSE) or defer to the next timestep (DEFER).

    The agent also wraps the RuntimeRiskEstimator so that when ANALYSE is
    triggered it directly returns updated risk scores to the caller.

    Parameters
    ----------
    num_apps : int
        Maximum tracked applications.
    num_permissions : int
        |𝒫|
    hidden_dims : tuple[int, ...]
    dropout : float
    ema_alpha : float
        EMA smoothing coefficient forwarded to RuntimeRiskEstimator.
    """

    def __init__(
        self,
        num_apps: int = 500,
        num_permissions: int = 42,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        ema_alpha: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_apps        = num_apps
        self.num_permissions = num_permissions

        # ── Observation: [mean_delta(|𝒫|), mean_pred(|𝒫|), mean_ema_risk(1)]
        self.obs_dim = 2 * num_permissions + 1

        # ── Actor policy ──────────────────────────────────────────────
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=2,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # ── Risk computation backend ──────────────────────────────────
        self.risk_estimator = RuntimeRiskEstimator(
            num_permissions=num_permissions,
            ema_alpha=ema_alpha,
        )

        # ── Obs normalisation ─────────────────────────────────────────
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var",  torch.ones(self.obs_dim))
        self._obs_count = 0

    # ------------------------------------------------------------------
    def build_observation(self, raw_obs: RiskAnalysisObservation) -> Tensor:
        """
        Flatten a ``RiskAnalysisObservation`` to (B, obs_dim).

        Parameters
        ----------
        raw_obs : RiskAnalysisObservation

        Returns
        -------
        Tensor  shape (B, obs_dim)
        """
        # Mean over apps and permissions where appropriate
        mean_delta  = raw_obs.usage_delta.mean(dim=1)       # (B, |𝒫|)
        mean_pred   = raw_obs.predicted_probs                # (B, |𝒫|)
        mean_risk   = raw_obs.ema_risks.mean(dim=1, keepdim=True)  # (B, 1)

        obs = torch.cat([mean_delta, mean_pred, mean_risk], dim=-1)
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
        raw_obs: RiskAnalysisObservation,
        app_ids: Optional[list[str]] = None,
        deterministic: bool = False,
        update_stats: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Decide whether to run a full risk analysis.

        Parameters
        ----------
        raw_obs : RiskAnalysisObservation
        app_ids : list[str], optional
            Application identifiers for persistent EMA state update.
        deterministic : bool
        update_stats : bool

        Returns
        -------
        action      : Tensor  shape (B,)   — 0=DEFER, 1=ANALYSE
        log_prob    : Tensor  shape (B,)
        obs_norm    : Tensor  shape (B, obs_dim)
        updated_risk : Tensor shape (B,) or None
            Per-batch-element mean risk (only computed if any action == ANALYSE).
        """
        obs_flat = self.build_observation(raw_obs)
        if update_stats:
            self.update_obs_stats(obs_flat)
        obs_norm = self.normalise_obs(obs_flat)

        action, log_prob = self.actor.get_action_and_log_prob(
            obs_norm, deterministic=deterministic
        )

        updated_risk: Optional[Tensor] = None
        analyse_mask = action == ANALYSE

        if analyse_mask.any():
            # Run risk recomputation for apps flagged by this agent
            usage_batch  = raw_obs.usage_delta[analyse_mask].mean(dim=1)  # reduce apps
            pred_batch   = raw_obs.predicted_probs[analyse_mask]
            ids_batch    = (
                [aid for aid, flag in zip(app_ids, analyse_mask.tolist()) if flag]
                if app_ids else None
            )

            risk_scores = self.risk_estimator.compute_risk(
                usage_vectors=usage_batch,
                predicted_probs=pred_batch,
                app_ids=ids_batch,
            )

            # Scatter back into a full-batch tensor
            updated_risk = torch.zeros(
                action.shape[0], device=action.device, dtype=torch.float32
            )
            updated_risk[analyse_mask] = risk_scores

        return action, log_prob, obs_norm, updated_risk

    # ------------------------------------------------------------------
    def evaluate_actions(
        self, obs_flat: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        obs_norm = self.normalise_obs(obs_flat)
        return self.actor.evaluate_actions(obs_norm, actions)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"obs_dim={self.obs_dim}"
