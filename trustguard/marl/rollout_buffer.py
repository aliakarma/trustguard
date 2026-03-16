"""
trustguard/marl/rollout_buffer.py
===================================
On-policy rollout buffer for TrustGuard MAPPO.

Stores transitions for all three agents during a rollout, then exposes
mini-batch iterators for the PPO update step.

Each ``Transition`` captures the full agent interaction at a single timestep:
observations, actions, log-probabilities, reward, done flag, global state,
and auxiliary tensors (EMA risks, permission targets).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Transition:
    """
    Single timestep transition for all three agents.

    Shapes assume B = number of parallel environments.

    Attributes
    ----------
    obs_monitor     : Tensor  (B, obs_dim_monitor)
    obs_risk        : Tensor  (B, obs_dim_risk)
    obs_enforce     : Tensor  (B, obs_dim_enforce)
    global_state    : Tensor  (B, state_dim)
    action_monitor  : Tensor  (B,)  — int
    action_risk     : Tensor  (B,)  — int
    action_enforce  : Tensor  (B,)  — int
    perm_targets    : Tensor  (B, |𝒫|)  — binary
    logp_monitor    : Tensor  (B,)
    logp_risk       : Tensor  (B,)
    logp_enforce    : Tensor  (B,)
    logp_perm       : Tensor  (B,)
    value           : Tensor  (B,)   — centralised critic output
    reward          : Tensor  (B,)
    done            : Tensor  (B,)   — float 0/1
    ema_risk        : Tensor  (B, N_apps)   — per-app EMA risk at this step
    """

    obs_monitor:    Tensor
    obs_risk:       Tensor
    obs_enforce:    Tensor
    global_state:   Tensor
    action_monitor: Tensor
    action_risk:    Tensor
    action_enforce: Tensor
    perm_targets:   Tensor
    logp_monitor:   Tensor
    logp_risk:      Tensor
    logp_enforce:   Tensor
    logp_perm:      Tensor
    value:          Tensor
    reward:         Tensor
    done:           Tensor
    ema_risk:       Optional[Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores a variable-length sequence of ``Transition`` objects and exposes
    tensor views for the MAPPO update.

    Usage
    -----
    >>> buf = RolloutBuffer()
    >>> buf.add(transition)
    >>> batch = buf.get_all(device)
    >>> buf.clear()
    """

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    # ------------------------------------------------------------------
    def add(self, t: Transition) -> None:
        """Append a transition."""
        self._transitions.append(t)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._transitions)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all stored transitions."""
        self._transitions.clear()

    # ------------------------------------------------------------------
    def get_all(self, device: torch.device) -> dict[str, Tensor]:
        """
        Stack all stored transitions into batched tensors.

        Returns
        -------
        dict mapping field names → Tensor of shape (T*B, ...)
        where T = rollout length, B = number of parallel envs.
        """
        if not self._transitions:
            raise RuntimeError("RolloutBuffer is empty.")

        def _stack(attr: str) -> Optional[Tensor]:
            vals = [getattr(t, attr) for t in self._transitions]
            if any(v is None for v in vals):
                return None
            return torch.cat(vals, dim=0).to(device)

        return {
            "obs_monitor":    _stack("obs_monitor"),
            "obs_risk":       _stack("obs_risk"),
            "obs_enforce":    _stack("obs_enforce"),
            "global_state":   _stack("global_state"),
            "action_monitor": _stack("action_monitor"),
            "action_risk":    _stack("action_risk"),
            "action_enforce": _stack("action_enforce"),
            "perm_targets":   _stack("perm_targets"),
            "logp_monitor":   _stack("logp_monitor"),
            "logp_risk":      _stack("logp_risk"),
            "logp_enforce":   _stack("logp_enforce"),
            "logp_perm":      _stack("logp_perm"),
            "value":          _stack("value"),
            "reward":         _stack("reward"),
            "done":           _stack("done"),
            "ema_risk_at_step": _stack("ema_risk"),
        }

    # ------------------------------------------------------------------
    def get_value_inputs(
        self, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return tensors needed for GAE computation.

        Returns
        -------
        global_states : Tensor  (T, B, state_dim)
        rewards       : Tensor  (T, B)
        dones         : Tensor  (T, B)
        values        : Tensor  (T, B)
        """
        T = len(self._transitions)
        B = self._transitions[0].reward.shape[0]

        global_states = torch.stack(
            [t.global_state for t in self._transitions], dim=0
        ).to(device)
        rewards = torch.stack(
            [t.reward for t in self._transitions], dim=0
        ).to(device)
        dones   = torch.stack(
            [t.done for t in self._transitions], dim=0
        ).to(device)
        values  = torch.stack(
            [t.value for t in self._transitions], dim=0
        ).to(device)

        return global_states, rewards, dones, values

    # ------------------------------------------------------------------
    def compute_rewards_to_go(
        self, gamma: float = 0.99, device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Compute discounted rewards-to-go (no bootstrapping).

        Used for quick sanity checks; GAE is preferred during training.

        Returns
        -------
        Tensor  shape (T, B)
        """
        T = len(self._transitions)
        B = self._transitions[0].reward.shape[0]
        rtg = torch.zeros(T, B)
        running = torch.zeros(B)
        for t in reversed(range(T)):
            running = self._transitions[t].reward + gamma * running * (
                1.0 - self._transitions[t].done
            )
            rtg[t] = running
        return rtg.to(device) if device else rtg
