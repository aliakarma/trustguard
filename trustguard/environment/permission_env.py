"""
trustguard/environment/permission_env.py
==========================================
TrustGuard Permission Governance Environment.

Implements the Dec-POMDP simulation environment for training and evaluating
the three TrustGuard agents. Simulates a mobile device with N installed
applications whose permission usage evolves over discrete timesteps.

Environment dynamics
--------------------
At each timestep t:
  1. Each application stochastically invokes permissions according to its
     usage model (benign profile or malicious escalation pattern).
  2. The Monitoring Agent decides whether to sample (SAMPLE_NOW / IDLE).
  3. The Risk-Analysis Agent receives the usage delta and decides whether
     to run a full risk computation (ANALYSE / DEFER).
  4. The Enforcement Agent selects an enforcement action conditioned on the
     shared belief state.
  5. Enforcement actions modify the permission state of affected applications.
  6. The joint reward is computed from risk reduction minus false-revocation
     penalty (Eq. 1 of the paper).

Reward function (Eq. 1):
    rₜ = Σᵢ (ρ̄ᵢᵗ⁻¹ − ρ̄ᵢᵗ)
         − λ₁ · Σᵢ 𝟙[false_revoke_i]
         − λ₂ · C_enforcement

Episode termination
-------------------
Episodes terminate after ``max_steps`` timesteps (simulating a 72-hour window
as reported in §6.2 of the paper, discretised to 5-minute governance steps).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from trustguard.environment.app_simulator import AppSimulator, AppProfile
from trustguard.environment.observation_builder import ObservationBuilder
from trustguard.models.permission_predictor import NUM_PERMISSIONS
from trustguard.agents.policy_networks import (
    ACTION_NO_OP, ACTION_ALERT, ACTION_RATE_LIMIT, ACTION_REVOKE
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    """Environment configuration."""
    num_benign_apps:    int   = 50
    num_malicious_apps: int   = 10
    num_permissions:    int   = NUM_PERMISSIONS
    max_steps:          int   = 1000      # ~72 h at 5-min steps
    risk_reduction_weight: float = 1.0    # λ₀ — reward scale
    false_revoc_penalty:   float = 5.0    # λ₁
    enforce_cost:          float = 0.01   # λ₂
    seed: int = 42


@dataclass
class StepInfo:
    """Structured info dict returned alongside each env step."""
    risk_reduction:    float
    false_revocations: int
    total_revocations: int
    enforcement_cost:  float
    privacy_risk:      float
    episode_step:      int
    done:              bool


# ─────────────────────────────────────────────────────────────────────────────
class PermissionEnv:
    """
    Vectorisable permission governance environment.

    Follows a gym-like interface but returns typed observation bundles
    instead of raw numpy arrays, to integrate cleanly with the agent
    dataclass interfaces.

    Parameters
    ----------
    config : EnvConfig
    device : torch.device
    """

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg    = config
        self.device = device or torch.device("cpu")

        self._rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # ── Simulators ────────────────────────────────────────────────
        self.app_simulator = AppSimulator(
            num_benign=config.num_benign_apps,
            num_malicious=config.num_malicious_apps,
            num_permissions=config.num_permissions,
            seed=config.seed,
        )
        self.obs_builder = ObservationBuilder(
            num_apps=config.num_benign_apps + config.num_malicious_apps,
            num_permissions=config.num_permissions,
            device=self.device,
        )

        # ── State ─────────────────────────────────────────────────────
        self.N = config.num_benign_apps + config.num_malicious_apps
        self.P = config.num_permissions

        # Usage matrix U ∈ {0,1}^(N × |𝒫|)
        self.usage_matrix: Tensor = torch.zeros(self.N, self.P, device=self.device)
        # Previous usage matrix (for delta computation)
        self.prev_usage:   Tensor = torch.zeros(self.N, self.P, device=self.device)
        # EMA risk vector ρ̄ ∈ [0,1]^N
        self.ema_risk:     Tensor = torch.zeros(self.N, device=self.device)
        # Permission revocation state (True = revoked)
        self.revoked:      Tensor = torch.zeros(self.N, self.P, dtype=torch.bool,
                                                device=self.device)
        # Ground-truth malicious mask
        self.is_malicious: Tensor = self.app_simulator.malicious_mask.to(self.device)

        self._step_count: int = 0
        self._episode_false_revocs: int = 0
        self._episode_total_revocs: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """
        Reset the environment to an initial state.

        Returns
        -------
        obs : dict
            Initial observations for all three agents.
        """
        self.app_simulator.reset()
        self.usage_matrix = torch.zeros(self.N, self.P, device=self.device)
        self.prev_usage   = torch.zeros(self.N, self.P, device=self.device)
        self.ema_risk     = torch.zeros(self.N, device=self.device)
        self.revoked      = torch.zeros(self.N, self.P, dtype=torch.bool,
                                        device=self.device)
        self._step_count = 0
        self._episode_false_revocs = 0
        self._episode_total_revocs = 0

        # Simulate first usage snapshot
        self._simulate_usage()
        return self._build_observations()

    # ------------------------------------------------------------------
    def step(
        self,
        action_monitor:   int,
        action_risk:      int,
        action_enforce:   int,
        perm_targets:     Tensor,   # (N, |𝒫|) binary
        risk_threshold:   float = 0.5,
    ) -> tuple[dict, float, bool, StepInfo]:
        """
        Advance the environment by one governance timestep.

        Parameters
        ----------
        action_monitor   : int  — 0=IDLE, 1=SAMPLE_NOW
        action_risk      : int  — 0=DEFER, 1=ANALYSE
        action_enforce   : int  — 0=no_op, 1=alert, 2=rate_limit, 3=revoke
        perm_targets     : Tensor shape (N, |𝒫|) — binary enforcement targets
        risk_threshold   : float

        Returns
        -------
        obs     : dict   — next observations
        reward  : float
        done    : bool
        info    : StepInfo
        """
        self._step_count += 1

        # ── 1. Simulate app usage ─────────────────────────────────────
        self._simulate_usage()

        # ── 2. Update EMA risks ───────────────────────────────────────
        prev_ema = self.ema_risk.clone()
        self._update_ema_risks()

        # ── 3. Apply enforcement actions ──────────────────────────────
        false_revocs, total_revocs, enforce_cost = self._apply_enforcement(
            action_enforce, perm_targets, risk_threshold
        )

        # ── 4. Compute reward ─────────────────────────────────────────
        risk_reduction = (prev_ema - self.ema_risk).sum().item()
        risk_reduction = max(risk_reduction, 0.0)   # clip negative

        reward = (
            self.cfg.risk_reduction_weight * risk_reduction
            - self.cfg.false_revoc_penalty  * false_revocs
            - self.cfg.enforce_cost         * enforce_cost
        )

        # ── 5. Accumulate episode stats ───────────────────────────────
        self._episode_false_revocs += false_revocs
        self._episode_total_revocs += total_revocs

        done = self._step_count >= self.cfg.max_steps

        obs  = self._build_observations()
        info = StepInfo(
            risk_reduction=risk_reduction,
            false_revocations=false_revocs,
            total_revocations=total_revocs,
            enforcement_cost=enforce_cost,
            privacy_risk=self.ema_risk.mean().item(),
            episode_step=self._step_count,
            done=done,
        )

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _simulate_usage(self) -> None:
        """
        Ask each application simulator to emit permission usage for this step.
        Respects the revocation state (revoked permissions cannot be invoked).
        """
        self.prev_usage = self.usage_matrix.clone()
        new_usage = self.app_simulator.step_usage(self._step_count)  # (N, |𝒫|)
        new_usage_t = torch.from_numpy(new_usage).float().to(self.device)

        # Mask out revoked permissions
        new_usage_t = new_usage_t * (~self.revoked).float()
        self.usage_matrix = new_usage_t

    # ------------------------------------------------------------------
    def _update_ema_risks(self, alpha: float = 0.3) -> None:
        """
        Recompute EMA risk from current usage matrix using a simple heuristic
        (full risk estimator with learned predictions is used during training
        via the agents; this lightweight version drives the reward signal).
        """
        # Instantaneous risk proxy: fraction of non-zero permissions per app
        inst_risk = (self.usage_matrix > 0).float().mean(dim=-1)   # (N,)

        # Upscale risk for malicious apps that are actively escalating
        malicious_factor = torch.where(
            self.is_malicious,
            torch.ones(self.N, device=self.device) * 2.0,
            torch.ones(self.N, device=self.device),
        )
        inst_risk = (inst_risk * malicious_factor).clamp(0.0, 1.0)

        self.ema_risk = alpha * inst_risk + (1.0 - alpha) * self.ema_risk

    # ------------------------------------------------------------------
    def _apply_enforcement(
        self,
        action_type:   int,
        perm_targets:  Tensor,   # (N, |𝒫|)
        risk_threshold: float,
    ) -> tuple[int, int, float]:
        """
        Mutate the revocation state based on the enforcement action.

        Returns
        -------
        (false_revocations, total_revocations, enforcement_cost)
        """
        false_revocs = 0
        total_revocs = 0
        cost         = 0.0

        if action_type == ACTION_NO_OP:
            return 0, 0, 0.0

        if action_type == ACTION_ALERT:
            cost = 0.1 * (perm_targets.sum().item() > 0)
            return 0, 0, cost

        if action_type in (ACTION_RATE_LIMIT, ACTION_REVOKE):
            # Only act on apps exceeding risk threshold
            high_risk_mask = self.ema_risk > risk_threshold   # (N,)

            for app_idx in range(self.N):
                if not high_risk_mask[app_idx]:
                    continue
                target_perms = perm_targets[app_idx]   # (|𝒫|,)
                if not target_perms.any():
                    continue

                if action_type == ACTION_REVOKE:
                    self.revoked[app_idx] |= target_perms.bool()
                    n_revoked = target_perms.sum().item()
                    total_revocs += n_revoked
                    cost += 0.5 * n_revoked

                    # False revocation: benign app targeted
                    if not self.is_malicious[app_idx]:
                        false_revocs += n_revoked

                elif action_type == ACTION_RATE_LIMIT:
                    # Rate limiting halves the usage probability (in simulator)
                    self.app_simulator.apply_rate_limit(app_idx, target_perms.cpu())
                    cost += 0.2 * target_perms.sum().item()

        return int(false_revocs), int(total_revocs), cost

    # ------------------------------------------------------------------
    def _build_observations(self) -> dict:
        """Build typed observation bundles for all three agents."""
        return self.obs_builder.build(
            usage_matrix=self.usage_matrix,
            prev_usage=self.prev_usage,
            ema_risk=self.ema_risk,
            step=self._step_count,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_global_state(self, embeddings: Tensor) -> Tensor:
        """
        Construct the flattened global state for the centralised critic.

        Parameters
        ----------
        embeddings : Tensor  shape (N, d)

        Returns
        -------
        Tensor  shape (N*|𝒫| + N + N*d)
        """
        u = self.usage_matrix.flatten()
        r = self.ema_risk
        e = embeddings.flatten()
        return torch.cat([u, r, e])

    # ------------------------------------------------------------------
    @property
    def privacy_risk_reduction(self) -> float:
        """
        Overall privacy risk reduction relative to the no-intervention baseline.
        Computed as 1 − (current mean EMA risk / initial mean EMA risk proxy).
        """
        initial_proxy = 0.5   # expected mean risk without any enforcement
        current       = self.ema_risk.mean().item()
        return max(0.0, (initial_proxy - current) / initial_proxy)

    # ------------------------------------------------------------------
    @property
    def false_revocation_rate(self) -> float:
        """Cumulative false-revocation rate for the current episode."""
        if self._episode_total_revocs == 0:
            return 0.0
        return self._episode_false_revocs / self._episode_total_revocs

    # ------------------------------------------------------------------
    @property
    def app_ids(self) -> list[str]:
        return self.app_simulator.app_ids

    # ------------------------------------------------------------------
    @property
    def observation_spaces(self) -> dict[str, int]:
        """Return observation dimensions for each agent (for network init)."""
        return self.obs_builder.observation_dims()
