"""
tests/conftest.py
==================
Shared pytest fixtures for the TrustGuard test suite.

Provides lightweight model and environment fixtures that avoid heavy
dependencies (no HuggingFace downloads, no real data) so the full
test suite runs fast on any machine.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from trustguard.models.permission_predictor import PermissionPredictionModel
from trustguard.models.runtime_risk_estimator import RuntimeRiskEstimator
from trustguard.models.belief_encoder import BeliefEncoder
from trustguard.agents.monitoring_agent import MonitoringAgent
from trustguard.agents.risk_analysis_agent import RiskAnalysisAgent
from trustguard.agents.enforcement_agent import EnforcementAgent
from trustguard.marl.centralized_critic import CentralizedCritic
from trustguard.environment.permission_env import PermissionEnv, EnvConfig

# ── Fixed test dimensions (small for speed) ───────────────────────────────────
N_APPS  = 5
N_PERMS = 10
EMBED   = 32
BELIEF  = 32


# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def small_perm_pred() -> PermissionPredictionModel:
    return PermissionPredictionModel(
        embedding_dim=EMBED, num_permissions=N_PERMS, hidden_dims=(64, 32)
    ).eval()


@pytest.fixture(scope="session")
def small_risk_estimator() -> RuntimeRiskEstimator:
    return RuntimeRiskEstimator(num_permissions=N_PERMS, ema_alpha=0.3)


@pytest.fixture(scope="session")
def small_belief_encoder() -> BeliefEncoder:
    return BeliefEncoder(
        obs_dim_monitor=N_PERMS + 2,
        obs_dim_risk=2 * N_PERMS + 1,
        obs_dim_enforce=1 + BELIEF + 2,
        embed_dim=16,
        gru_hidden_dim=32,
        belief_dim=BELIEF,
    ).eval()


@pytest.fixture(scope="session")
def small_monitoring_agent() -> MonitoringAgent:
    return MonitoringAgent(
        num_apps=N_APPS, num_permissions=N_PERMS, hidden_dims=(32, 32)
    ).eval()


@pytest.fixture(scope="session")
def small_risk_agent() -> RiskAnalysisAgent:
    return RiskAnalysisAgent(
        num_apps=N_APPS, num_permissions=N_PERMS, hidden_dims=(32, 32)
    ).eval()


@pytest.fixture(scope="session")
def small_enforcement_agent() -> EnforcementAgent:
    return EnforcementAgent(
        num_apps=N_APPS, num_permissions=N_PERMS,
        belief_dim=BELIEF, hidden_dims=(32, 32)
    ).eval()


@pytest.fixture(scope="session")
def small_critic() -> CentralizedCritic:
    # state_dim = N_APPS * N_PERMS + N_APPS + N_APPS * EMBED
    state_dim = N_APPS * N_PERMS + N_APPS + N_APPS * EMBED
    return CentralizedCritic(state_dim=state_dim, hidden_dims=(32,)).eval()


@pytest.fixture
def small_env() -> PermissionEnv:
    cfg = EnvConfig(
        num_benign_apps=N_APPS - 1,
        num_malicious_apps=1,
        num_permissions=N_PERMS,
        max_steps=10,
        seed=0,
    )
    return PermissionEnv(config=cfg)


@pytest.fixture
def random_phi(device) -> torch.Tensor:
    """Random batch of application embeddings."""
    return torch.randn(4, EMBED, device=device)


@pytest.fixture
def random_perm_vector() -> torch.Tensor:
    """Random batch of binary permission vectors."""
    return torch.randint(0, 2, (4, N_PERMS)).float()
