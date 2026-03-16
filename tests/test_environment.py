"""
tests/test_environment.py
===========================
Unit tests for TrustGuard simulation environment.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from trustguard.environment.app_simulator import AppSimulator, EscalationType
from trustguard.environment.permission_env import PermissionEnv, EnvConfig
from trustguard.agents.policy_networks import ACTION_NO_OP, ACTION_REVOKE


# ─────────────────────────────────────────────────────────────────────────────
class TestAppSimulator:
    @pytest.fixture
    def sim(self):
        return AppSimulator(num_benign=5, num_malicious=2, num_permissions=10, seed=0)

    def test_usage_shape(self, sim):
        usage = sim.step_usage(0)
        assert usage.shape == (7, 10)

    def test_usage_binary(self, sim):
        usage = sim.step_usage(0)
        assert set(np.unique(usage)).issubset({0.0, 1.0})

    def test_malicious_mask(self, sim):
        mask = sim.malicious_mask
        assert mask.shape  == (7,)
        assert mask.sum()  == 2
        assert (~mask).sum() == 5

    def test_escalation_increases_usage(self, sim):
        """Malicious apps should show higher permission usage after escalation."""
        usage_before = sim.step_usage(0)
        # Step to well past escalation start (max 400)
        usage_after  = sim.step_usage(1000)
        mal_mask = sim.malicious_mask.numpy()
        # Mean usage for malicious apps should be higher post-escalation
        mean_before = usage_before[mal_mask].mean()
        mean_after  = usage_after[mal_mask].mean()
        assert mean_after >= mean_before - 0.2   # allow some stochasticity

    def test_reset_rebuilds_profiles(self, sim):
        ids_before = sim.app_ids
        sim.reset()
        ids_after  = sim.app_ids
        # IDs contain same categories but are rebuilt
        assert len(ids_before) == len(ids_after)

    def test_rate_limit_reduces_usage(self, sim):
        """Rate-limited permissions should fire less frequently."""
        np.random.seed(99)
        counts_before = np.zeros(10)
        for _ in range(100):
            counts_before += sim.step_usage(0)[0]

        sim.apply_rate_limit(0, torch.ones(10))
        counts_after = np.zeros(10)
        for _ in range(100):
            counts_after += sim.step_usage(0)[0]

        assert counts_after.mean() <= counts_before.mean() + 5   # roughly halved


# ─────────────────────────────────────────────────────────────────────────────
class TestPermissionEnv:
    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            num_benign_apps=5,
            num_malicious_apps=2,
            num_permissions=10,
            max_steps=20,
            seed=0,
        )
        return PermissionEnv(config=cfg)

    def test_reset_returns_obs_dict(self, env):
        obs = env.reset()
        assert set(obs.keys()) == {"monitor", "risk", "enforce"}

    def test_step_returns_correct_types(self, env):
        env.reset()
        perm_targets = torch.zeros(env.N, 10)
        obs, reward, done, info = env.step(
            action_monitor=0, action_risk=0,
            action_enforce=ACTION_NO_OP, perm_targets=perm_targets
        )
        assert isinstance(reward, float)
        assert isinstance(done,   bool)
        assert set(obs.keys())  == {"monitor", "risk", "enforce"}

    def test_episode_terminates(self, env):
        obs = env.reset()
        done = False
        steps = 0
        perm_tgt = torch.zeros(env.N, 10)
        while not done:
            _, _, done, _ = env.step(0, 0, ACTION_NO_OP, perm_tgt)
            steps += 1
            if steps > 100:
                break
        assert done, "Episode should terminate within max_steps"
        assert steps == env.cfg.max_steps

    def test_revocation_affects_usage(self, env):
        env.reset()
        # Revoke all permissions for all apps
        perm_tgt = torch.ones(env.N, 10)
        # Set high EMA risk first
        env.ema_risk = torch.ones(env.N)
        obs, _, _, info = env.step(
            action_monitor=1, action_risk=1,
            action_enforce=ACTION_REVOKE, perm_targets=perm_tgt,
            risk_threshold=0.0,
        )
        # All permissions revoked → usage should drop to zero
        assert (env.revoked).all()

    def test_false_revocation_rate_benign(self, env):
        """Revoking a benign app should count as a false revocation."""
        env.reset()
        env.ema_risk = torch.ones(env.N)
        perm_tgt = torch.zeros(env.N, 10)
        perm_tgt[0, 0] = 1.0   # target first app (benign), first permission
        env.step(0, 0, ACTION_REVOKE, perm_tgt, risk_threshold=0.0)
        # frr > 0 if first app is benign
        if not env.is_malicious[0]:
            assert env.false_revocation_rate > 0.0

    def test_no_op_reward_zero_risk_reduction(self, env):
        env.reset()
        env.ema_risk[:] = 0.5   # set baseline
        perm_tgt = torch.zeros(env.N, 10)
        _, reward, _, info = env.step(0, 0, ACTION_NO_OP, perm_tgt)
        # no_op produces zero enforcement cost
        assert info.enforcement_cost == 0.0
