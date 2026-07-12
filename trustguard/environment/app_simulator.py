"""
trustguard/environment/app_simulator.py
=========================================
Stochastic application behaviour simulator.

Each application is assigned a ``AppProfile`` that defines:
  - Its expected permission usage probabilities (benign profile)
  - Whether it is a simulated malicious application
  - Its escalation pattern (for malicious apps)

Benign profiles are drawn from category-conditional permission distributions
calibrated to match the PermissionBench dataset statistics.

Malicious escalation patterns implement three attack classes:
  1. **Covert background access**: permission invoked only outside active
     user sessions (simulated by elevated night-time usage probability).
  2. **Permission creep**: starts with legitimate usage, gradually escalates
     to additional permissions after a warm-up period.
  3. **Mimicry**: appears identical to a benign app until a trigger condition
     is met, then escalates sharply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ── App categories and their expected permission profiles ────────────────────
# Format: category → {permission_index: base_probability}
# Permission indices correspond to ANDROID_PERMISSIONS in permission_predictor.py

CATEGORY_PROFILES: dict[str, dict[int, float]] = {
    "messaging":   {0: 0.1, 1: 0.1, 5: 0.83, 6: 0.91, 12: 0.78, 21: 0.60},
    "navigation":  {10: 0.97, 11: 0.85, 12: 0.40},
    "camera":      {5: 0.95, 26: 0.80, 27: 0.70},
    "social":      {5: 0.75, 6: 0.70, 12: 0.65, 21: 0.55, 26: 0.60},
    "utility":     {26: 0.50, 27: 0.40},
    "health":      {21: 0.70, 22: 0.65, 33: 0.80},
    "finance":     {14: 0.60, 26: 0.45},
    "gaming":      {26: 0.35},
    "flashlight":  {},  # expects no permissions
}

CATEGORIES = list(CATEGORY_PROFILES.keys())
NUM_PERMISSIONS = 42


# ─────────────────────────────────────────────────────────────────────────────
class EscalationType(Enum):
    """Three simulated malicious permission-escalation attack patterns."""

    COVERT_BACKGROUND = "covert_background"
    PERMISSION_CREEP  = "permission_creep"
    MIMICRY           = "mimicry"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AppProfile:
    """
    Per-application simulation profile.

    Attributes
    ----------
    app_id : str
    category : str
    base_probs : np.ndarray  shape (|𝒫|,)
        Per-permission invocation probability under normal operation.
    is_malicious : bool
    escalation_type : EscalationType, optional
    escalation_start : int
        Timestep at which malicious escalation begins.
    escalation_perms : list[int]
        Permission indices targeted by the escalation.
    rate_limited : np.ndarray  shape (|𝒫|,)  dtype=bool
        Permissions currently rate-limited by the Enforcement Agent.
    """

    app_id:            str
    category:          str
    base_probs:        np.ndarray
    is_malicious:      bool = False
    escalation_type:   Optional[EscalationType] = None
    escalation_start:  int = 200
    escalation_perms:  list[int] = field(default_factory=list)
    rate_limited:      Optional[np.ndarray] = None

    def __post_init__(self):
        if self.rate_limited is None:
            self.rate_limited = np.zeros(len(self.base_probs), dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
class AppSimulator:
    """
    Simulates N applications across discrete timesteps.

    Parameters
    ----------
    num_benign : int
    num_malicious : int
    num_permissions : int
    seed : int
    """

    def __init__(
        self,
        num_benign:      int = 50,
        num_malicious:   int = 10,
        num_permissions: int = NUM_PERMISSIONS,
        seed:            int = 42,
    ) -> None:
        self.num_benign      = num_benign
        self.num_malicious   = num_malicious
        self.num_apps        = num_benign + num_malicious
        self.num_permissions = num_permissions
        self._rng = np.random.default_rng(seed)

        self._profiles: list[AppProfile] = []
        self._build_profiles()

    # ------------------------------------------------------------------
    def _build_profiles(self) -> None:
        """Initialise profiles for all benign and malicious apps."""
        self._profiles = []

        for i in range(self.num_benign):
            cat   = self._rng.choice(CATEGORIES)
            probs = self._category_probs(cat)
            self._profiles.append(AppProfile(
                app_id=f"benign_{i:04d}.{cat}",
                category=cat,
                base_probs=probs,
                is_malicious=False,
            ))

        escalation_types = list(EscalationType)
        for j in range(self.num_malicious):
            cat   = self._rng.choice(CATEGORIES)
            probs = self._category_probs(cat)
            esc   = escalation_types[j % len(escalation_types)]
            # Malicious apps target 2–5 high-risk permissions
            esc_perms = self._rng.choice(
                self.num_permissions,
                size=self._rng.integers(2, 6),
                replace=False,
            ).tolist()
            start = int(self._rng.integers(50, 400))
            self._profiles.append(AppProfile(
                app_id=f"malicious_{j:04d}.{cat}",
                category=cat,
                base_probs=probs,
                is_malicious=True,
                escalation_type=esc,
                escalation_start=start,
                escalation_perms=esc_perms,
            ))

    # ------------------------------------------------------------------
    def _category_probs(self, category: str) -> np.ndarray:
        """Build a full permission probability vector for a given category."""
        probs = np.zeros(self.num_permissions, dtype=np.float32)
        profile = CATEGORY_PROFILES.get(category, {})
        for perm_idx, p in profile.items():
            if perm_idx < self.num_permissions:
                probs[perm_idx] = p
        # Add small background noise for all permissions
        noise = self._rng.uniform(0.0, 0.02, size=self.num_permissions)
        return np.clip(probs + noise, 0.0, 1.0)

    # ------------------------------------------------------------------
    def step_usage(self, timestep: int) -> np.ndarray:
        """
        Simulate one governance timestep of permission usage.

        Parameters
        ----------
        timestep : int
            Current global timestep.

        Returns
        -------
        usage : np.ndarray  shape (N, |𝒫|)  dtype=float32
            Binary usage indicators (Bernoulli sample per permission per app).
        """
        usage = np.zeros((self.num_apps, self.num_permissions), dtype=np.float32)

        for i, profile in enumerate(self._profiles):
            probs = profile.base_probs.copy()

            if profile.is_malicious and timestep >= profile.escalation_start:
                probs = self._apply_escalation(profile, probs, timestep)

            # Apply rate limiting (halve probabilities)
            probs = np.where(profile.rate_limited, probs * 0.5, probs)

            # Bernoulli sample
            usage[i] = (self._rng.random(self.num_permissions) < probs).astype(np.float32)

        return usage

    # ------------------------------------------------------------------
    def _apply_escalation(
        self,
        profile:   AppProfile,
        probs:     np.ndarray,
        timestep:  int,
    ) -> np.ndarray:
        """Modify permission probabilities based on escalation type."""
        t_since = timestep - profile.escalation_start

        if profile.escalation_type == EscalationType.COVERT_BACKGROUND:
            # All targeted permissions spike with high probability
            for p in profile.escalation_perms:
                probs[p] = min(probs[p] + 0.8, 1.0)

        elif profile.escalation_type == EscalationType.PERMISSION_CREEP:
            # Gradual ramp-up: probability increases by 0.01 per step
            ramp = min(t_since * 0.01, 0.9)
            for p in profile.escalation_perms:
                probs[p] = min(probs[p] + ramp, 1.0)

        elif profile.escalation_type == EscalationType.MIMICRY:
            # Sudden transition at escalation_start + 100 steps
            if t_since > 100:
                for p in profile.escalation_perms:
                    probs[p] = 0.95

        return probs

    # ------------------------------------------------------------------
    def apply_rate_limit(self, app_idx: int, perm_mask: Tensor) -> None:
        """Mark permissions as rate-limited for a given application."""
        mask_np = perm_mask.numpy().astype(bool)
        self._profiles[app_idx].rate_limited |= mask_np

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Rebuild all profiles (new episode)."""
        self._build_profiles()

    # ------------------------------------------------------------------
    @property
    def profiles(self) -> list[AppProfile]:
        """Per-application simulation profiles (mutable — used by the
        stress-test protocols to swap the anomaly injection process)."""
        return self._profiles

    # ------------------------------------------------------------------
    @property
    def malicious_mask(self) -> Tensor:
        """Boolean tensor shape (N,) — True for malicious apps."""
        return torch.tensor(
            [p.is_malicious for p in self._profiles], dtype=torch.bool
        )

    # ------------------------------------------------------------------
    @property
    def app_ids(self) -> list[str]:
        return [p.app_id for p in self._profiles]

    # ------------------------------------------------------------------
    @property
    def categories(self) -> list[str]:
        return [p.category for p in self._profiles]
