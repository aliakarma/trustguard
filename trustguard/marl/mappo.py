"""
trustguard/marl/mappo.py
=========================
Multi-Agent PPO (MAPPO) trainer with Lagrangian safety constraint.

Implements Algorithm 1 from the TrustGuard paper:

    Constrained MAPPO for three cooperative Dec-POMDP agents with:
      - Centralised training / decentralised execution (CTDE)
      - PPO-clip policy update for each agent
      - GAE-λ advantage estimation from the centralised critic
      - Lagrangian dual ascent for false-revocation safety constraint

Lagrangian objective (Eq. 5 of the paper):
    ℒ(θ, μ) = 𝔼[Σ γᵗ rₜ] − μ(𝔼[false_revocations] − ε_safe)

Multiplier update:
    μ ← max(0,  μ + η_μ · (c̄ − ε_safe))
    where c̄ = empirical false-revocation rate over the current rollout.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from trustguard.marl.rollout_buffer import RolloutBuffer, Transition
from trustguard.marl.centralized_critic import CentralizedCritic
from trustguard.agents.monitoring_agent import MonitoringAgent
from trustguard.agents.risk_analysis_agent import RiskAnalysisAgent
from trustguard.agents.enforcement_agent import EnforcementAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class MAPPOTrainer:
    """
    Centralised MAPPO trainer for TrustGuard's three-agent system.

    Parameters
    ----------
    monitoring_agent  : MonitoringAgent
    risk_agent        : RiskAnalysisAgent
    enforcement_agent : EnforcementAgent
    critic            : CentralizedCritic
    device            : torch.device
    lr_actor          : float   — actor learning rate
    lr_critic         : float   — critic learning rate
    lr_lagrange       : float   — Lagrange multiplier update step η_μ
    gamma             : float   — discount factor γ
    gae_lambda        : float   — GAE λ
    eps_clip          : float   — PPO clip ε
    value_loss_coef   : float
    entropy_coef      : float   — entropy bonus coefficient
    max_grad_norm     : float
    ppo_epochs        : int     — number of PPO update epochs per rollout
    mini_batch_size   : int
    eps_safe          : float   — false-revocation safety budget ε_safe
    lagrange_init     : float   — initial μ
    lagrange_max      : float   — μ ceiling
    """

    def __init__(
        self,
        monitoring_agent:  MonitoringAgent,
        risk_agent:        RiskAnalysisAgent,
        enforcement_agent: EnforcementAgent,
        critic:            CentralizedCritic,
        device:            torch.device,
        lr_actor:     float = 3e-4,
        lr_critic:    float = 1e-3,
        lr_lagrange:  float = 1e-3,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        eps_clip:     float = 0.2,
        value_loss_coef:  float = 0.5,
        entropy_coef:     float = 0.01,
        max_grad_norm:    float = 10.0,
        ppo_epochs:       int   = 10,
        mini_batch_size:  int   = 256,
        eps_safe:         float = 0.025,
        lagrange_init:    float = 0.0,
        lagrange_max:     float = 10.0,
    ) -> None:
        self.device = device

        # ── Agents and critic ─────────────────────────────────────────
        self.monitoring_agent  = monitoring_agent.to(device)
        self.risk_agent        = risk_agent.to(device)
        self.enforcement_agent = enforcement_agent.to(device)
        self.critic            = critic.to(device)

        # ── Hyper-parameters ──────────────────────────────────────────
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.eps_clip        = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.ppo_epochs      = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.eps_safe        = eps_safe
        self.lagrange_max    = lagrange_max

        # ── Lagrange multiplier μ (scalar, non-negative) ──────────────
        self.log_lagrange_mu = nn.Parameter(
            torch.tensor(max(lagrange_init, 1e-8)).log().to(device)
        )
        self.lr_lagrange = lr_lagrange

        # ── Optimisers ────────────────────────────────────────────────
        actor_params = (
            list(monitoring_agent.parameters())
            + list(risk_agent.parameters())
            + list(enforcement_agent.parameters())
        )
        self.actor_optimizer  = optim.Adam(actor_params, lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic, eps=1e-5)
        self.lagrange_optimizer = optim.Adam(
            [self.log_lagrange_mu], lr=lr_lagrange
        )

        # ── Rollout buffer ────────────────────────────────────────────
        self.buffer = RolloutBuffer()

        # ── Training metrics ──────────────────────────────────────────
        self._train_steps = 0
        self._total_env_steps = 0

    # ------------------------------------------------------------------
    @property
    def lagrange_mu(self) -> Tensor:
        """Current μ ≥ 0 (derived from log-parameterisation)."""
        return self.log_lagrange_mu.exp().clamp(max=self.lagrange_max)

    # ------------------------------------------------------------------
    def store_transition(self, transition: Transition) -> None:
        """Add a single transition to the rollout buffer."""
        self.buffer.add(transition)
        self._total_env_steps += transition.obs_monitor.shape[0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_gae(
        self,
        rewards:    Tensor,
        values:     Tensor,
        dones:      Tensor,
        last_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Generalised Advantage Estimation (GAE-λ).

        Parameters
        ----------
        rewards    : Tensor  shape (T, B)
        values     : Tensor  shape (T, B)
        dones      : Tensor  shape (T, B)  — episode-end indicators
        last_value : Tensor  shape (B,)    — bootstrap value

        Returns
        -------
        advantages : Tensor  shape (T, B)
        returns    : Tensor  shape (T, B)
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae   = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def _ppo_loss(
        self,
        log_probs_new: Tensor,
        log_probs_old: Tensor,
        advantages:    Tensor,
        entropy:       Tensor,
        false_revoc_rate: float,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the constrained PPO-clip objective.

        ℒ_PPO = -𝔼[min(r·A, clip(r, 1-ε, 1+ε)·A)] - c_ent·H
               - μ · max(0, c̄ - ε_safe)

        Parameters
        ----------
        log_probs_new : Tensor  shape (B,)
        log_probs_old : Tensor  shape (B,)
        advantages    : Tensor  shape (B,)
        entropy       : Tensor  scalar
        false_revoc_rate : float — empirical constraint violation

        Returns
        -------
        loss  : Tensor  scalar
        info  : dict[str, float]
        """
        ratio = (log_probs_new - log_probs_old).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy_loss = -self.entropy_coef * entropy

        # Lagrangian penalty term: μ · max(0, c̄ − ε_safe)
        constraint_violation = max(0.0, false_revoc_rate - self.eps_safe)
        lagrange_penalty = self.lagrange_mu.detach() * constraint_violation

        total_loss = policy_loss + entropy_loss + lagrange_penalty

        info = {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "lagrange_penalty": lagrange_penalty if isinstance(lagrange_penalty, float)
                                  else lagrange_penalty.item(),
            "constraint_violation": constraint_violation,
        }
        return total_loss, info

    # ------------------------------------------------------------------
    def update(self, last_global_state: Tensor) -> dict[str, float]:
        """
        Run PPO update epochs over the collected rollout buffer.

        Parameters
        ----------
        last_global_state : Tensor  shape (B, state_dim)
            Global state at the end of the rollout, for value bootstrapping.

        Returns
        -------
        metrics : dict[str, float]
        """
        if len(self.buffer) == 0:
            logger.warning("update() called with empty buffer — skipping.")
            return {}

        # ── Compute returns and advantages ────────────────────────────
        with torch.no_grad():
            last_value = self.critic(last_global_state)

        (
            all_global_states,
            all_rewards,
            all_dones,
            all_values_old,
        ) = self.buffer.get_value_inputs(self.device)

        advantages, returns = self.compute_gae(
            rewards=all_rewards,
            values=all_values_old,
            dones=all_dones,
            last_value=last_value,
        )

        # Normalise advantages per-batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten time × batch → single batch axis
        advantages_flat  = advantages.flatten(0, 1)
        returns_flat     = returns.flatten(0, 1)
        states_flat      = all_global_states.flatten(0, 1)

        # Retrieve stored observations and actions
        batch = self.buffer.get_all(self.device)
        false_revoc_rate = self._estimate_false_revoc_rate(batch)

        metric_accum: dict[str, list[float]] = {
            "policy_loss_monitor":    [],
            "policy_loss_risk":       [],
            "policy_loss_enforce":    [],
            "value_loss":             [],
            "entropy":                [],
            "lagrange_mu":            [],
            "constraint_violation":   [],
        }

        T_total = advantages_flat.shape[0]

        for _ in range(self.ppo_epochs):
            # Mini-batch iteration
            indices = torch.randperm(T_total, device=self.device)
            for start in range(0, T_total, self.mini_batch_size):
                idx = indices[start: start + self.mini_batch_size]

                adv_mb    = advantages_flat[idx]
                ret_mb    = returns_flat[idx]
                state_mb  = states_flat[idx]

                # ── Critic update ─────────────────────────────────────
                value_pred = self.critic(state_mb)
                value_loss = self.value_loss_coef * nn.functional.mse_loss(
                    value_pred, ret_mb
                )
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # ── Monitoring Agent update ───────────────────────────
                lp_new_mon, ent_mon = self.monitoring_agent.evaluate_actions(
                    batch["obs_monitor"][idx], batch["action_monitor"][idx]
                )
                loss_mon, info_mon = self._ppo_loss(
                    lp_new_mon,
                    batch["logp_monitor"][idx],
                    adv_mb,
                    ent_mon,
                    false_revoc_rate,
                )

                # ── Risk Agent update ─────────────────────────────────
                lp_new_risk, ent_risk = self.risk_agent.evaluate_actions(
                    batch["obs_risk"][idx], batch["action_risk"][idx]
                )
                loss_risk, info_risk = self._ppo_loss(
                    lp_new_risk,
                    batch["logp_risk"][idx],
                    adv_mb,
                    ent_risk,
                    false_revoc_rate,
                )

                # ── Enforcement Agent update ──────────────────────────
                lp_new_enf, lp_perm_enf, ent_enf = self.enforcement_agent.evaluate_actions(
                    batch["obs_enforce"][idx],
                    batch["action_enforce"][idx],
                    batch["perm_targets"][idx],
                )
                combined_lp_enf = lp_new_enf + lp_perm_enf
                combined_lp_old = batch["logp_enforce"][idx] + batch["logp_perm"][idx]
                loss_enf, info_enf = self._ppo_loss(
                    combined_lp_enf,
                    combined_lp_old,
                    adv_mb,
                    ent_enf,
                    false_revoc_rate,
                )

                # ── Joint actor update ────────────────────────────────
                total_actor_loss = loss_mon + loss_risk + loss_enf
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.monitoring_agent.parameters())
                    + list(self.risk_agent.parameters())
                    + list(self.enforcement_agent.parameters()),
                    self.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Accumulate metrics
                metric_accum["policy_loss_monitor"].append(info_mon["policy_loss"])
                metric_accum["policy_loss_risk"].append(info_risk["policy_loss"])
                metric_accum["policy_loss_enforce"].append(info_enf["policy_loss"])
                metric_accum["value_loss"].append(value_loss.item())
                metric_accum["entropy"].append(ent_enf.item())

        # ── Lagrange multiplier dual ascent ───────────────────────────
        lagrange_loss = -self.lagrange_mu * (false_revoc_rate - self.eps_safe)
        self.lagrange_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optimizer.step()

        self.buffer.clear()
        self._train_steps += 1

        metrics = {k: sum(v) / max(len(v), 1) for k, v in metric_accum.items()}
        metrics["lagrange_mu"] = self.lagrange_mu.item()
        metrics["constraint_violation"] = max(0.0, false_revoc_rate - self.eps_safe)
        metrics["false_revoc_rate"] = false_revoc_rate
        metrics["train_step"] = self._train_steps

        logger.info(
            "Step %d | VLoss %.4f | μ %.4f | FRR %.4f",
            self._train_steps,
            metrics["value_loss"],
            metrics["lagrange_mu"],
            metrics["false_revoc_rate"],
        )
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_false_revoc_rate(batch: dict) -> float:
        """
        Estimate false-revocation rate from the rollout batch.

        A revocation is counted as false if the ground-truth risk of the
        revoked app was below the safety threshold at the time of revocation.
        This is approximated using the stored risk vectors.
        """
        from trustguard.agents.policy_networks import ACTION_REVOKE
        revoke_mask = batch["action_enforce"] == ACTION_REVOKE
        if not revoke_mask.any():
            return 0.0
        # App whose risk was below 0.3 at revocation time = false revocation
        risk_at_revoke = batch.get("ema_risk_at_step")
        if risk_at_revoke is None:
            return 0.0
        mean_risk = risk_at_revoke.mean(dim=-1)   # mean over apps
        false_mask = (mean_risk < 0.3) & revoke_mask
        return false_mask.float().sum().item() / max(revoke_mask.float().sum().item(), 1)
