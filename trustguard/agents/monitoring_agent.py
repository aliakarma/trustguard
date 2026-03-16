"""
trustguard/agents/monitoring_agent.py
=======================================
Monitoring Agent (k = 1) in TrustGuard's Dec-POMDP.

Responsibility
--------------
Adaptively samples permission-usage observations from the runtime environment.
Balances monitoring coverage (detect anomalies early) against device overhead
(battery, CPU) by learning a sampling-interval policy.

Observation space
-----------------
o¹ₜ ∈ ℝ^obs_dim:
  - raw permission invocation counts per application (|𝒫| × N_apps features)
  - time-since-last-sample scalar
  - battery / CPU utilisation (optional)

Action space
------------
Discrete: { SAMPLE_NOW, IDLE } — 2 actions.
The agent learns when to trigger a full permission-usage snapshot.

Outputs written to shared belief
---------------------------------
δuᵢᵗ = Δ permission usage vector (difference from last sample) per app,
forwarded to the Risk-Analysis Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from trustguard.agents.policy_networks import ActorNetwork

logger = logging.getLogger(__name__)

# Action indices
IDLE       = 0
SAMPLE_NOW = 1


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MonitoringObservation:
    """
    Typed observation bundle for the Monitoring Agent.

    Attributes
    ----------
    usage_counts : Tensor  shape (B, N_apps, |𝒫|)
        Raw permission invocation counts since last sample.
    time_since_sample : Tensor  shape (B,)
        Seconds elapsed since the last SAMPLE_NOW action.
    system_load : Tensor  shape (B,)
        Normalised system CPU/battery load in [0, 1].
    """

    usage_counts: Tensor
    time_since_sample: Tensor
    system_load: Tensor


# ─────────────────────────────────────────────────────────────────────────────
class MonitoringAgent(nn.Module):
    """
    Adaptive permission-usage monitor with a learned sampling policy.

    The agent encodes its local observation and outputs a two-way decision:
    trigger a full environment sample (SAMPLE_NOW) or remain idle (IDLE).
    By learning an adaptive schedule, the agent can front-load monitoring
    effort toward anomalous applications while idling on stable ones.

    Parameters
    ----------
    num_apps : int
        Maximum number of concurrently tracked applications.
    num_permissions : int
        |𝒫|
    hidden_dims : tuple[int, ...]
        MLP hidden widths in the actor network.
    dropout : float

    Observation construction
    ------------------------
    The flat observation vector is built internally from a
    ``MonitoringObservation`` instance:
      [mean_usage_per_app (|𝒫|), time_since_sample (1), system_load (1)]
    → obs_dim = |𝒫| + 2
    """

    def __init__(
        self,
        num_apps: int = 500,
        num_permissions: int = 42,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_apps        = num_apps
        self.num_permissions = num_permissions

        # ── Observation dimension ─────────────────────────────────────
        # [mean per-permission usage (|𝒫|), time_since_sample, system_load]
        self.obs_dim = num_permissions + 2

        # ── Actor network ─────────────────────────────────────────────
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=2,             # {IDLE, SAMPLE_NOW}
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # ── Observation normalisation (running statistics) ────────────
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var",  torch.ones(self.obs_dim))
        self._obs_count = 0

    # ------------------------------------------------------------------
    def build_observation(self, raw_obs: MonitoringObservation) -> Tensor:
        """
        Flatten a ``MonitoringObservation`` into a (B, obs_dim) tensor.

        Parameters
        ----------
        raw_obs : MonitoringObservation

        Returns
        -------
        Tensor  shape (B, obs_dim)
        """
        B = raw_obs.usage_counts.shape[0]
        device = raw_obs.usage_counts.device

        # Aggregate: mean permission usage across all apps
        mean_usage = raw_obs.usage_counts.mean(dim=1)   # (B, |𝒫|)

        obs = torch.cat(
            [
                mean_usage,
                raw_obs.time_since_sample.unsqueeze(-1),
                raw_obs.system_load.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, obs_dim)
        return obs

    # ------------------------------------------------------------------
    def normalise_obs(self, obs: Tensor) -> Tensor:
        """
        Apply running-mean normalisation to the observation.

        Parameters
        ----------
        obs : Tensor  shape (B, obs_dim)

        Returns
        -------
        Tensor  shape (B, obs_dim)
        """
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    # ------------------------------------------------------------------
    def update_obs_stats(self, obs: Tensor) -> None:
        """Welford online update of running observation statistics."""
        batch_mean = obs.mean(dim=0)
        batch_var  = obs.var(dim=0, unbiased=False)
        batch_size = obs.shape[0]
        n = self._obs_count + batch_size
        delta = batch_mean - self.obs_mean
        self.obs_mean = self.obs_mean + delta * (batch_size / n)
        self.obs_var  = (
            (self._obs_count * self.obs_var + batch_size * batch_var)
            / n
            + delta ** 2 * (self._obs_count * batch_size / n ** 2)
        )
        self._obs_count = n

    # ------------------------------------------------------------------
    def forward(
        self,
        raw_obs: MonitoringObservation,
        deterministic: bool = False,
        update_stats: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decide whether to sample the runtime environment at this timestep.

        Parameters
        ----------
        raw_obs : MonitoringObservation
        deterministic : bool
            Greedy action selection (for evaluation).
        update_stats : bool
            Update running normalisation statistics (enable during training).

        Returns
        -------
        action   : Tensor  shape (B,)  — 0 = IDLE, 1 = SAMPLE_NOW
        log_prob : Tensor  shape (B,)
        obs_flat : Tensor  shape (B, obs_dim)  — normalised observation
        """
        obs_flat = self.build_observation(raw_obs)
        if update_stats:
            self.update_obs_stats(obs_flat)
        obs_norm = self.normalise_obs(obs_flat)

        action, log_prob = self.actor.get_action_and_log_prob(
            obs_norm, deterministic=deterministic
        )
        return action, log_prob, obs_norm

    # ------------------------------------------------------------------
    def evaluate_actions(
        self, obs_flat: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute log-probs and entropy for stored (obs, action) pairs.
        Called by MAPPO trainer during policy update.

        Parameters
        ----------
        obs_flat : Tensor  shape (B, obs_dim)
        actions  : Tensor  shape (B,)

        Returns
        -------
        log_probs : Tensor  shape (B,)
        entropy   : Tensor  scalar
        """
        obs_norm = self.normalise_obs(obs_flat)
        return self.actor.evaluate_actions(obs_norm, actions)

    # ------------------------------------------------------------------
    def should_sample(self, action: Tensor) -> Tensor:
        """
        Return a boolean mask indicating which batch entries should trigger
        a full environment sample.

        Parameters
        ----------
        action : Tensor  shape (B,)

        Returns
        -------
        Tensor  shape (B,)  dtype=bool
        """
        return action == SAMPLE_NOW

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"obs_dim={self.obs_dim}, num_permissions={self.num_permissions}"
