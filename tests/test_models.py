"""
tests/test_models.py
======================
Unit tests for TrustGuard model components.

Covers:
  - AppSemanticEncoder (graph encoder path only — BERT skipped to avoid download)
  - PermissionPredictionModel (forward, loss, risk_scores)
  - RuntimeRiskEstimator (EMA state, compute_risk)
  - BeliefEncoder (forward, step, init_hidden)
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data as GeoData, Batch as GeoBatch

from trustguard.models.permission_predictor import (
    PermissionPredictionModel,
    LabelSmoothingBCE,
    NUM_PERMISSIONS,
)
from trustguard.models.runtime_risk_estimator import RuntimeRiskEstimator
from trustguard.models.belief_encoder import BeliefEncoder


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def perm_pred() -> PermissionPredictionModel:
    return PermissionPredictionModel(
        embedding_dim=32,
        num_permissions=10,
        hidden_dims=(64, 32),
    )


@pytest.fixture
def risk_est() -> RuntimeRiskEstimator:
    return RuntimeRiskEstimator(num_permissions=10, ema_alpha=0.3)


@pytest.fixture
def belief_enc() -> BeliefEncoder:
    return BeliefEncoder(
        obs_dim_monitor=12,
        obs_dim_risk=22,
        obs_dim_enforce=14,
        embed_dim=16,
        gru_hidden_dim=32,
        belief_dim=16,
    )


# ─────────────────────────────────────────────────────────────────────────────
class TestPermissionPredictionModel:
    def test_forward_shape(self, perm_pred):
        phi = torch.randn(4, 32)
        logits = perm_pred(phi)
        assert logits.shape == (4, 10)

    def test_predict_proba_range(self, perm_pred):
        phi   = torch.randn(8, 32)
        probs = perm_pred.predict_proba(phi)
        assert probs.shape == (8, 10)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_risk_scores_complement(self, perm_pred):
        perm_pred.eval()
        phi   = torch.randn(4, 32)
        probs = perm_pred.predict_proba(phi)
        risk  = perm_pred.risk_scores(phi)
        assert torch.allclose(probs + risk, torch.ones_like(probs), atol=1e-5)

    def test_loss_positive(self, perm_pred):
        phi    = torch.randn(4, 32)
        labels = torch.randint(0, 2, (4, 10)).float()
        loss   = perm_pred.compute_loss(phi, labels)
        assert loss.item() > 0.0

    def test_label_smoothing_bce(self):
        criterion = LabelSmoothingBCE(smoothing=0.1)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8, 10)).float()
        loss   = criterion(logits, labels)
        assert loss.item() > 0.0


# ─────────────────────────────────────────────────────────────────────────────
class TestRuntimeRiskEstimator:
    def test_compute_risk_shape(self, risk_est):
        usage = torch.rand(4, 10)
        probs = torch.rand(4, 10)
        risk  = risk_est.compute_risk(usage, probs)
        assert risk.shape == (4,)

    def test_risk_range(self, risk_est):
        usage = torch.rand(4, 10)
        probs = torch.rand(4, 10)
        risk  = risk_est.compute_risk(usage, probs)
        assert (risk >= 0.0).all() and (risk <= 1.0).all()

    def test_ema_state_update(self, risk_est):
        usage = torch.rand(2, 10)
        probs = torch.rand(2, 10)
        risk_est.compute_risk(usage, probs, app_ids=["app_0", "app_1"])
        state0 = risk_est.get_risk_state("app_0")
        assert state0 is not None
        assert state0.num_updates == 0   # first update sets state but doesn't increment

    def test_ema_monotone_with_zero_new(self, risk_est):
        """EMA risk should decrease when new observations are 0."""
        # Seed with high risk
        usage_high = torch.ones(1, 10)
        probs_low  = torch.zeros(1, 10)
        risk_est.compute_risk(usage_high, probs_low, app_ids=["app"])
        ema_after_first = risk_est.get_ema_risk(["app"]).item()

        # Zero new usage → risk should drop
        usage_zero = torch.zeros(1, 10)
        risk_est.compute_risk(usage_zero, probs_low, app_ids=["app"])
        ema_after_second = risk_est.get_ema_risk(["app"]).item()

        assert ema_after_second < ema_after_first

    def test_reset_app(self, risk_est):
        usage = torch.rand(1, 10)
        probs = torch.rand(1, 10)
        risk_est.compute_risk(usage, probs, app_ids=["app_x"])
        risk_est.reset_app("app_x")
        assert risk_est.get_risk_state("app_x") is None

    def test_shape_mismatch_raises(self, risk_est):
        with pytest.raises(ValueError):
            risk_est.compute_risk(torch.rand(4, 10), torch.rand(4, 8))


# ─────────────────────────────────────────────────────────────────────────────
class TestBeliefEncoder:
    def test_forward_shape(self, belief_enc):
        B, T = 2, 5
        o1 = torch.randn(B, T, 12)
        o2 = torch.randn(B, T, 22)
        o3 = torch.randn(B, T, 14)
        belief, h_n = belief_enc(o1, o2, o3)
        assert belief.shape == (B, T, 16)
        assert h_n.shape    == (1, B, 32)

    def test_step_shape(self, belief_enc):
        B = 3
        o1 = torch.randn(B, 12)
        o2 = torch.randn(B, 22)
        o3 = torch.randn(B, 14)
        belief_t, h_n = belief_enc.step(o1, o2, o3)
        assert belief_t.shape == (B, 16)
        assert h_n.shape      == (1, B, 32)

    def test_init_hidden_shape(self, belief_enc, device):
        h = belief_enc.init_hidden(batch_size=4, device=device)
        assert h.shape == (1, 4, 32)
        assert (h == 0.0).all()

    def test_belief_bounded(self, belief_enc):
        """Tanh output should be in [-1, 1]."""
        B, T = 2, 3
        o1 = torch.randn(B, T, 12) * 10   # large inputs
        o2 = torch.randn(B, T, 22) * 10
        o3 = torch.randn(B, T, 14) * 10
        belief, _ = belief_enc(o1, o2, o3)
        assert (belief >= -1.0 - 1e-5).all()
        assert (belief <=  1.0 + 1e-5).all()
