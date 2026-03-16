"""
tests/test_agents.py
======================
Unit tests for TrustGuard agent components.

Covers:
  - ActorNetwork, CriticNetwork, EnforcementHead
  - MonitoringAgent, RiskAnalysisAgent, EnforcementAgent
"""

from __future__ import annotations

import pytest
import torch

from trustguard.agents.policy_networks import (
    ActorNetwork, CriticNetwork, EnforcementHead,
    ACTION_REVOKE, ACTION_NO_OP,
)
from trustguard.agents.monitoring_agent import MonitoringAgent, MonitoringObservation
from trustguard.agents.risk_analysis_agent import RiskAnalysisAgent, RiskAnalysisObservation
from trustguard.agents.enforcement_agent import EnforcementAgent, EnforcementObservation


# ── Fixtures ──────────────────────────────────────────────────────────────────
B  = 4   # batch size
NP = 10  # num_permissions (small for test speed)
BD = 32  # belief_dim


@pytest.fixture
def actor():
    return ActorNetwork(obs_dim=16, action_dim=4, hidden_dims=(32, 32))


@pytest.fixture
def critic():
    return CriticNetwork(state_dim=64, hidden_dims=(32,))


@pytest.fixture
def enf_head():
    return EnforcementHead(belief_dim=BD, num_permissions=NP, hidden_dims=(32, 32))


@pytest.fixture
def mon_agent():
    return MonitoringAgent(num_apps=5, num_permissions=NP, hidden_dims=(32, 32))


@pytest.fixture
def risk_agent():
    return RiskAnalysisAgent(num_apps=5, num_permissions=NP, hidden_dims=(32, 32))


@pytest.fixture
def enf_agent():
    return EnforcementAgent(
        num_apps=5, num_permissions=NP, belief_dim=BD, hidden_dims=(32, 32)
    )


@pytest.fixture
def mon_obs():
    return MonitoringObservation(
        usage_counts=torch.rand(B, 5, NP),
        time_since_sample=torch.rand(B),
        system_load=torch.rand(B),
    )


@pytest.fixture
def risk_obs():
    return RiskAnalysisObservation(
        usage_delta=torch.rand(B, 5, NP),
        predicted_probs=torch.rand(B, NP),
        ema_risks=torch.rand(B, 5),
    )


@pytest.fixture
def enf_obs():
    return EnforcementObservation(
        ema_risks=torch.rand(B, 5),
        belief=torch.rand(B, BD),
        revoke_rate_history=torch.rand(B),
        alert_rate_history=torch.rand(B),
    )


# ─────────────────────────────────────────────────────────────────────────────
class TestActorNetwork:
    def test_output_shapes(self, actor):
        obs  = torch.randn(B, 16)
        dist = actor(obs)
        act, lp = actor.get_action_and_log_prob(obs)
        assert act.shape == (B,)
        assert lp.shape  == (B,)

    def test_deterministic_action(self, actor):
        obs  = torch.randn(B, 16)
        a1, _ = actor.get_action_and_log_prob(obs, deterministic=True)
        a2, _ = actor.get_action_and_log_prob(obs, deterministic=True)
        assert torch.equal(a1, a2)

    def test_evaluate_actions_shapes(self, actor):
        obs     = torch.randn(B, 16)
        actions = torch.randint(0, 4, (B,))
        lp, ent = actor.evaluate_actions(obs, actions)
        assert lp.shape == (B,)
        assert ent.ndim == 0   # scalar entropy


class TestCriticNetwork:
    def test_output_shape(self, critic):
        state = torch.randn(B, 64)
        v     = critic(state)
        assert v.shape == (B,)


class TestEnforcementHead:
    def test_forward_distributions(self, enf_head):
        belief = torch.randn(B, BD)
        act_dist, perm_dist = enf_head(belief)
        assert act_dist.probs.shape == (B, 4)
        assert perm_dist.probs.shape == (B, NP)

    def test_select_action_shapes(self, enf_head):
        belief     = torch.randn(B, BD)
        risk_vector = torch.rand(B)
        at, pt, lp_a, lp_p = enf_head.select_action(belief, risk_vector)
        assert at.shape == (B,)
        assert pt.shape == (B, NP)
        assert lp_a.shape == (B,)
        assert lp_p.shape == (B,)

    def test_risk_gate_forces_noop(self, enf_head):
        """All apps below threshold → all actions should be no_op."""
        belief      = torch.randn(B, BD)
        risk_vector = torch.zeros(B)   # all zero risk
        at, _, _, _ = enf_head.select_action(
            belief, risk_vector, risk_threshold=0.5, deterministic=False
        )
        assert (at == ACTION_NO_OP).all()


# ─────────────────────────────────────────────────────────────────────────────
class TestMonitoringAgent:
    def test_forward_shapes(self, mon_agent, mon_obs):
        act, lp, obs_flat = mon_agent.forward(mon_obs)
        assert act.shape     == (B,)
        assert lp.shape      == (B,)
        assert obs_flat.shape == (B, mon_agent.obs_dim)

    def test_action_binary(self, mon_agent, mon_obs):
        act, _, _ = mon_agent.forward(mon_obs)
        assert set(act.tolist()).issubset({0, 1})

    def test_should_sample_mask(self, mon_agent):
        acts = torch.tensor([0, 1, 0, 1])
        mask = mon_agent.should_sample(acts)
        assert mask.tolist() == [False, True, False, True]


class TestRiskAnalysisAgent:
    def test_forward_shapes(self, risk_agent, risk_obs):
        act, lp, obs_flat, updated = risk_agent.forward(risk_obs)
        assert act.shape     == (B,)
        assert lp.shape      == (B,)
        assert obs_flat.shape == (B, risk_agent.obs_dim)

    def test_action_binary(self, risk_agent, risk_obs):
        act, _, _, _ = risk_agent.forward(risk_obs)
        assert set(act.tolist()).issubset({0, 1})


class TestEnforcementAgent:
    def test_forward_shapes(self, enf_agent, enf_obs):
        at, pt, lp_a, lp_p, obs_flat = enf_agent.forward(enf_obs)
        assert at.shape       == (B,)
        assert pt.shape       == (B, NP)
        assert lp_a.shape     == (B,)
        assert lp_p.shape     == (B,)
        assert obs_flat.shape == (B, enf_agent.obs_dim)

    def test_action_in_valid_range(self, enf_agent, enf_obs):
        at, _, _, _, _ = enf_agent.forward(enf_obs, deterministic=True)
        assert (at >= 0).all() and (at <= 3).all()

    def test_false_revocation_rate_empty(self, enf_agent):
        assert enf_agent.false_revocation_rate() == 0.0

    def test_audit_log(self, enf_agent):
        enf_agent.log_enforcement(
            timestep=10, app_id="com.test", action_type=ACTION_REVOKE,
            perm_indices=[0, 3], ema_risk=0.8, was_false=True
        )
        enf_agent.log_enforcement(
            timestep=11, app_id="com.other", action_type=ACTION_REVOKE,
            perm_indices=[1], ema_risk=0.9, was_false=False
        )
        frr = enf_agent.false_revocation_rate()
        assert abs(frr - 0.5) < 1e-6
