"""
trustguard/models/runtime_risk_estimator.py
============================================
Layer 3 of TrustGuard: Runtime Risk Estimator.

Implements the aggregate risk score from the paper (Eq. 3):

    ρᵢᵗ = (1 / |𝒫ᵢᵗ|) Σ_{p ∈ 𝒫ᵢᵗ} |uᵢ,ₚᵗ − p̂ᵢ,ₚ|

where:
  - uᵢ,ₚᵗ  : binary runtime usage indicator (was p invoked in [t-1, t)?)
  - p̂ᵢ,ₚ   : predicted probability from the permission prediction model
  - 𝒫ᵢᵗ   : set of permissions actively used at time t

Also maintains an exponential moving average (EMA) of the risk signal:

    ρ̄ᵢᵗ = α · ρᵢᵗ + (1 − α) · ρ̄ᵢᵗ⁻¹

to suppress transient noise from single-use permission invocations.

Reference: §5.3 of the TrustGuard paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from trustguard.models.permission_predictor import NUM_PERMISSIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AppRiskState:
    """
    Per-application risk tracking state maintained by the estimator.

    Attributes
    ----------
    app_id : str
        Unique application identifier.
    ema_risk : float
        Exponential moving average of the instantaneous risk score.
    instantaneous_risk : float
        Most recently computed ρᵢᵗ.
    num_updates : int
        Number of timestep updates applied.
    risk_history : list[float]
        Full chronological risk score history (optional, capped at
        ``history_len`` entries).
    """

    app_id: str
    ema_risk: float = 0.0
    instantaneous_risk: float = 0.0
    num_updates: int = 0
    risk_history: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
class RuntimeRiskEstimator(nn.Module):
    """
    Stateful runtime risk estimator.

    Maintains per-application EMA risk state across governance timesteps
    and exposes both instantaneous and smoothed risk tensors to the agents.

    Parameters
    ----------
    num_permissions : int
        Number of permissions |𝒫| (must match PermissionPredictionModel).
    ema_alpha : float
        EMA smoothing coefficient α ∈ (0, 1].
        Higher → faster response; lower → smoother signal.
    history_len : int
        Maximum number of risk history entries stored per application.
        Set to 0 to disable history storage.

    Usage
    -----
    Typical usage inside the environment step:

    >>> estimator = RuntimeRiskEstimator()
    >>> risk = estimator.compute_risk(
    ...     usage_vectors=u_t,        # (B, |𝒫|) float in [0,1]
    ...     predicted_probs=p_hat,    # (B, |𝒫|) float in [0,1]
    ...     app_ids=["com.foo", ...],
    ... )
    >>> ema_risk = estimator.get_ema_risk(app_ids=["com.foo", ...])
    """

    def __init__(
        self,
        num_permissions: int = NUM_PERMISSIONS,
        ema_alpha: float = 0.3,
        history_len: int = 1000,
    ) -> None:
        super().__init__()
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}")

        self.num_permissions = num_permissions
        self.ema_alpha       = ema_alpha
        self.history_len     = history_len

        # ── App-level state registry ──────────────────────────────────
        self._states: dict[str, AppRiskState] = {}

    # ------------------------------------------------------------------
    # Core risk computation
    # ------------------------------------------------------------------

    def compute_risk(
        self,
        usage_vectors: Tensor,
        predicted_probs: Tensor,
        app_ids: Optional[list[str]] = None,
        active_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute per-application instantaneous risk ρᵢᵗ and update EMA states.

        Parameters
        ----------
        usage_vectors : Tensor  shape (B, |𝒫|)
            Binary (or soft) runtime permission-usage indicators at time t.
            uᵢ,ₚᵗ = 1 if permission p was invoked during [t-1, t).
        predicted_probs : Tensor  shape (B, |𝒫|)
            Expected permission probabilities p̂ᵢ,ₚ from the predictor model.
        app_ids : list[str], optional
            Application identifiers for state tracking (length B).
            If None, no persistent state is updated.
        active_mask : Tensor, optional  shape (B, |𝒫|)
            Boolean mask indicating which permissions are in 𝒫ᵢᵗ (actively
            used). If None, all non-zero usage entries are treated as active.

        Returns
        -------
        instantaneous_risk : Tensor  shape (B,)
            Raw risk score ρᵢᵗ for each application in the batch.
        """
        # Validate shapes
        if usage_vectors.shape != predicted_probs.shape:
            raise ValueError(
                f"Shape mismatch: usage_vectors {usage_vectors.shape} vs "
                f"predicted_probs {predicted_probs.shape}"
            )

        batch_size = usage_vectors.shape[0]

        # Determine active permission mask 𝒫ᵢᵗ
        if active_mask is None:
            active_mask = usage_vectors > 0.0   # (B, |𝒫|)

        active_count = active_mask.float().sum(dim=-1).clamp(min=1.0)   # (B,)

        # Per-permission absolute deviation, zeroed for inactive permissions
        deviation = (usage_vectors - predicted_probs).abs()             # (B, |𝒫|)
        deviation = deviation * active_mask.float()

        # Aggregate risk per app: mean over active permissions
        inst_risk = deviation.sum(dim=-1) / active_count                # (B,)

        # ── Update persistent EMA states ──────────────────────────────
        if app_ids is not None:
            if len(app_ids) != batch_size:
                raise ValueError(
                    f"app_ids length {len(app_ids)} != batch_size {batch_size}"
                )
            inst_np = inst_risk.detach().cpu().tolist()
            for app_id, risk_val in zip(app_ids, inst_np):
                self._update_state(app_id, risk_val)

        return inst_risk

    # ------------------------------------------------------------------
    def _update_state(self, app_id: str, instantaneous_risk: float) -> None:
        """Update EMA state for a single application."""
        if app_id not in self._states:
            self._states[app_id] = AppRiskState(
                app_id=app_id,
                ema_risk=instantaneous_risk,
                instantaneous_risk=instantaneous_risk,
            )
        else:
            state = self._states[app_id]
            state.instantaneous_risk = instantaneous_risk
            state.ema_risk = (
                self.ema_alpha * instantaneous_risk
                + (1.0 - self.ema_alpha) * state.ema_risk
            )
            state.num_updates += 1

        state = self._states[app_id]
        if self.history_len > 0:
            state.risk_history.append(state.ema_risk)
            if len(state.risk_history) > self.history_len:
                state.risk_history.pop(0)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_ema_risk(self, app_ids: list[str]) -> Tensor:
        """
        Retrieve the current EMA risk score for each requested application.

        Parameters
        ----------
        app_ids : list[str]

        Returns
        -------
        Tensor  shape (len(app_ids),)
        """
        risks = [
            self._states[a].ema_risk if a in self._states else 0.0
            for a in app_ids
        ]
        return torch.tensor(risks, dtype=torch.float32)

    # ------------------------------------------------------------------
    def get_risk_state(self, app_id: str) -> Optional[AppRiskState]:
        """Return the full AppRiskState for a given application, or None."""
        return self._states.get(app_id)

    # ------------------------------------------------------------------
    def batch_ema_risk_tensor(
        self,
        app_ids: list[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Convenience: return EMA risks on the specified device.

        Parameters
        ----------
        app_ids : list[str]
        device : torch.device, optional

        Returns
        -------
        Tensor  shape (B,)
        """
        t = self.get_ema_risk(app_ids)
        return t.to(device) if device is not None else t

    # ------------------------------------------------------------------
    def reset_app(self, app_id: str) -> None:
        """Reset the risk state of a single application (e.g. after update)."""
        if app_id in self._states:
            del self._states[app_id]
            logger.debug("Risk state reset for app: %s", app_id)

    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        """Clear all tracked application states."""
        self._states.clear()
        logger.info("All RuntimeRiskEstimator states cleared.")

    # ------------------------------------------------------------------
    @staticmethod
    def compute_per_permission_risk(
        usage_vector: Tensor,
        predicted_probs: Tensor,
    ) -> Tensor:
        """
        Return the per-permission risk vector ϱ(p, fᵢ) for a single app.

        Combines the instantaneous deviation with the static semantic risk
        (1 − p̂ᵢ,ₚ) via their geometric mean to produce a composite score:

            ϱ_composite(p, fᵢ) = |uᵢ,ₚᵗ − p̂ᵢ,ₚ| · (1 − p̂ᵢ,ₚ)

        Parameters
        ----------
        usage_vector : Tensor  shape (|𝒫|,)
        predicted_probs : Tensor  shape (|𝒫|,)

        Returns
        -------
        risk_vector : Tensor  shape (|𝒫|,)  values in [0, 1]
        """
        deviation     = (usage_vector - predicted_probs).abs()
        semantic_risk = 1.0 - predicted_probs
        return deviation * semantic_risk

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"num_permissions={self.num_permissions}, "
            f"ema_alpha={self.ema_alpha}"
        )
