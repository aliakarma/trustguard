"""
trustguard/environment/observation_builder.py
===============================================
Constructs typed observation bundles for each TrustGuard agent from the raw
environment state tensors (usage matrix, EMA risks, etc.).

Each agent observes only a partial view of the global state, enforcing the
Dec-POMDP partial observability constraint:
  - Agent 1 (Monitoring):   raw usage counts + system load
  - Agent 2 (Risk):         usage delta + predicted probs + EMA risks
  - Agent 3 (Enforcement):  EMA risks + belief state + enforcement history
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor

from trustguard.agents.monitoring_agent import MonitoringObservation
from trustguard.agents.risk_analysis_agent import RiskAnalysisObservation
from trustguard.agents.enforcement_agent import EnforcementObservation


# ─────────────────────────────────────────────────────────────────────────────
class ObservationBuilder:
    """
    Translates raw environment state into typed agent observations.

    Parameters
    ----------
    num_apps : int
    num_permissions : int
    device : torch.device
    """

    def __init__(
        self,
        num_apps:        int,
        num_permissions: int,
        device:          torch.device,
    ) -> None:
        self.num_apps        = num_apps
        self.num_permissions = num_permissions
        self.device          = device

        # Cached predicted probs (updated when permission predictor runs)
        self._cached_pred_probs: Optional[Tensor] = None
        # Cached belief state (updated by BeliefEncoder)
        self._cached_belief: Optional[Tensor] = None

        # Rolling enforcement rate history (for Agent 3)
        self._revoke_rate: float = 0.0
        self._alert_rate:  float = 0.0

    # ------------------------------------------------------------------
    def update_predicted_probs(self, pred_probs: Tensor) -> None:
        """
        Cache the latest predicted permission probabilities from the
        PermissionPredictionModel (called after each supervised forward pass).

        Parameters
        ----------
        pred_probs : Tensor  shape (N, |𝒫|) or (|𝒫|,)
        """
        if pred_probs.dim() == 1:
            pred_probs = pred_probs.unsqueeze(0).expand(self.num_apps, -1)
        self._cached_pred_probs = pred_probs.to(self.device)

    # ------------------------------------------------------------------
    def update_belief(self, belief: Tensor) -> None:
        """
        Cache the latest belief state from the BeliefEncoder.

        Parameters
        ----------
        belief : Tensor  shape (belief_dim,) or (1, belief_dim)
        """
        if belief.dim() == 2:
            belief = belief.squeeze(0)
        self._cached_belief = belief.to(self.device)

    # ------------------------------------------------------------------
    def update_enforcement_rates(self, revoke_rate: float, alert_rate: float) -> None:
        """Update the rolling enforcement rate scalars for Agent 3."""
        self._revoke_rate = revoke_rate
        self._alert_rate  = alert_rate

    # ------------------------------------------------------------------
    def build(
        self,
        usage_matrix: Tensor,   # (N, |𝒫|)
        prev_usage:   Tensor,   # (N, |𝒫|)
        ema_risk:     Tensor,   # (N,)
        step:         int,
    ) -> dict:
        """
        Build all three agents' observation bundles.

        Parameters
        ----------
        usage_matrix : Tensor  shape (N, |𝒫|)
        prev_usage   : Tensor  shape (N, |𝒫|)
        ema_risk     : Tensor  shape (N,)
        step         : int

        Returns
        -------
        dict with keys "monitor", "risk", "enforce"
        """
        usage_delta = usage_matrix - prev_usage   # (N, |𝒫|)

        # Predicted probs: use cached or fall back to zeros
        pred_probs = (
            self._cached_pred_probs
            if self._cached_pred_probs is not None
            else torch.zeros(self.num_apps, self.num_permissions, device=self.device)
        )

        # Belief: use cached or zeros
        belief_dim = (
            self._cached_belief.shape[-1]
            if self._cached_belief is not None
            else 256
        )
        belief = (
            self._cached_belief
            if self._cached_belief is not None
            else torch.zeros(belief_dim, device=self.device)
        )

        # Unsqueeze to add a batch dimension of 1
        usage_3d   = usage_matrix.unsqueeze(0)   # (1, N, |𝒫|)
        delta_3d   = usage_delta.unsqueeze(0)    # (1, N, |𝒫|)
        ema_2d     = ema_risk.unsqueeze(0)       # (1, N)
        pred_2d    = pred_probs.mean(0).unsqueeze(0)  # (1, |𝒫|) — mean over apps
        belief_2d  = belief.unsqueeze(0)         # (1, belief_dim)

        # ── Monitoring Agent observation ──────────────────────────────
        system_load = torch.tensor([min(step / 1000.0, 1.0)], device=self.device)
        time_since  = torch.tensor([1.0], device=self.device)
        obs_monitor = MonitoringObservation(
            usage_counts=usage_3d,
            time_since_sample=time_since,
            system_load=system_load,
        )

        # ── Risk-Analysis Agent observation ───────────────────────────
        obs_risk = RiskAnalysisObservation(
            usage_delta=delta_3d,
            predicted_probs=pred_2d,
            ema_risks=ema_2d,
        )

        # ── Enforcement Agent observation ─────────────────────────────
        revoke_rate_t = torch.tensor([self._revoke_rate], device=self.device)
        alert_rate_t  = torch.tensor([self._alert_rate],  device=self.device)
        obs_enforce = EnforcementObservation(
            ema_risks=ema_2d,
            belief=belief_2d,
            revoke_rate_history=revoke_rate_t,
            alert_rate_history=alert_rate_t,
        )

        return {
            "monitor": obs_monitor,
            "risk":    obs_risk,
            "enforce": obs_enforce,
        }

    # ------------------------------------------------------------------
    def observation_dims(self) -> dict[str, int]:
        """
        Return flat observation dimensions for each agent.
        Useful for initialising policy networks.
        """
        return {
            "monitor": self.num_permissions + 2,
            "risk":    2 * self.num_permissions + 1,
            "enforce": 1 + 256 + 2,   # mean_risk + belief_dim + rates
        }
