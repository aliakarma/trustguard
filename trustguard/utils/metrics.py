"""
trustguard/utils/metrics.py
=============================
Evaluation metrics for all three experimental tasks in the TrustGuard paper.

Task 1 — Permission Risk Prediction:
    Accuracy, Macro-F1, AUROC

Task 2 — Autonomous Enforcement:
    Privacy Risk Reduction (PRR), False Revocation Rate (FRR),
    Enforcement Latency

Task 3 — Adversarial Robustness:
    AUROC under mimicry attack, degradation delta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PredictionMetrics:
    """Task 1 metric bundle."""
    accuracy:  float
    macro_f1:  float
    auroc:     float
    ap:        float   # Average Precision (summarises precision-recall curve)

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.4f} | "
            f"Macro-F1={self.macro_f1:.4f} | "
            f"AUROC={self.auroc:.4f} | "
            f"AP={self.ap:.4f}"
        )


@dataclass
class EnforcementMetrics:
    """Task 2 metric bundle."""
    privacy_risk_reduction: float    # PRR (%)
    false_revocation_rate:  float    # FRR
    enforcement_latency_s:  float    # mean seconds from anomaly onset to action
    total_revocations:      int
    total_false_revocations: int

    def __str__(self) -> str:
        return (
            f"PRR={self.privacy_risk_reduction:.1f}% | "
            f"FRR={self.false_revocation_rate:.4f} | "
            f"Latency={self.enforcement_latency_s:.2f}s"
        )


@dataclass
class AdversarialMetrics:
    """Task 3 metric bundle."""
    auroc_clean:    float
    auroc_attack:   float
    auroc_delta:    float   # degradation

    def __str__(self) -> str:
        return (
            f"AUROC (clean)={self.auroc_clean:.4f} | "
            f"AUROC (attack)={self.auroc_attack:.4f} | "
            f"Δ={self.auroc_delta:+.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
def compute_prediction_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> PredictionMetrics:
    """
    Compute Task 1 metrics.

    Parameters
    ----------
    y_true   : np.ndarray shape (N,)   — binary ground-truth labels
    y_pred   : np.ndarray shape (N,)   — binary predictions (after threshold)
    y_score  : np.ndarray shape (N,)   — continuous risk scores / probabilities
    threshold : float

    Returns
    -------
    PredictionMetrics
    """
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUROC requires at least two classes
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        ap    = float("nan")
        logger.warning("Only one class present; AUROC is undefined.")
    else:
        auroc = roc_auc_score(y_true, y_score)
        ap    = average_precision_score(y_true, y_score)

    return PredictionMetrics(accuracy=acc, macro_f1=f1, auroc=auroc, ap=ap)


# ─────────────────────────────────────────────────────────────────────────────
def compute_per_permission_metrics(
    perm_labels_true: np.ndarray,
    perm_scores:      np.ndarray,
    threshold:        float = 0.5,
) -> dict[str, float]:
    """
    Compute per-permission prediction metrics averaged across permissions.

    Parameters
    ----------
    perm_labels_true : np.ndarray  shape (N, NUM_PERMISSIONS)
    perm_scores      : np.ndarray  shape (N, NUM_PERMISSIONS)

    Returns
    -------
    dict with macro-averaged metrics per permission
    """
    from trustguard.models.permission_predictor import ANDROID_PERMISSIONS

    per_perm: dict[str, dict] = {}
    for i, perm in enumerate(ANDROID_PERMISSIONS):
        yt = perm_labels_true[:, i]
        ys = perm_scores[:, i]
        yp = (ys >= threshold).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        per_perm[perm] = {
            "f1":    f1_score(yt, yp, zero_division=0),
            "auroc": roc_auc_score(yt, ys),
        }

    if not per_perm:
        return {}

    mean_f1    = np.mean([v["f1"]    for v in per_perm.values()])
    mean_auroc = np.mean([v["auroc"] for v in per_perm.values()])

    return {
        "mean_per_perm_f1":    float(mean_f1),
        "mean_per_perm_auroc": float(mean_auroc),
        "per_permission":      per_perm,
    }


# ─────────────────────────────────────────────────────────────────────────────
def compute_enforcement_metrics(
    initial_risk:           float,
    final_risk:             float,
    total_revocations:      int,
    false_revocations:      int,
    anomaly_onset_steps:    list[int],
    enforcement_steps:      list[int],
    step_duration_s:        float = 300.0,   # 5-minute governance steps
) -> EnforcementMetrics:
    """
    Compute Task 2 enforcement quality metrics.

    Parameters
    ----------
    initial_risk        : float — mean EMA risk at episode start
    final_risk          : float — mean EMA risk at episode end
    total_revocations   : int
    false_revocations   : int
    anomaly_onset_steps : list[int]  — steps at which anomalies first appeared
    enforcement_steps   : list[int]  — steps at which enforcement first triggered
    step_duration_s     : float — real-world duration per step (seconds)

    Returns
    -------
    EnforcementMetrics
    """
    # Privacy Risk Reduction (PRR)
    if initial_risk > 0:
        prr = 100.0 * (initial_risk - final_risk) / initial_risk
    else:
        prr = 0.0
    prr = max(prr, 0.0)

    # False Revocation Rate (FRR)
    frr = false_revocations / max(total_revocations, 1)

    # Enforcement Latency: mean (enforcement_step − anomaly_onset) × step_duration
    latency_s = 0.0
    paired = list(zip(anomaly_onset_steps, enforcement_steps))
    if paired:
        latency_steps = [max(0, enf - onset) for onset, enf in paired]
        latency_s = np.mean(latency_steps) * step_duration_s

    return EnforcementMetrics(
        privacy_risk_reduction=prr,
        false_revocation_rate=frr,
        enforcement_latency_s=latency_s,
        total_revocations=total_revocations,
        total_false_revocations=false_revocations,
    )


# ─────────────────────────────────────────────────────────────────────────────
def compute_adversarial_metrics(
    y_true_clean:   np.ndarray,
    scores_clean:   np.ndarray,
    y_true_attack:  np.ndarray,
    scores_attack:  np.ndarray,
) -> AdversarialMetrics:
    """
    Compute Task 3 adversarial robustness metrics.

    Parameters
    ----------
    y_true_clean  : np.ndarray shape (N,)
    scores_clean  : np.ndarray shape (N,)
    y_true_attack : np.ndarray shape (N,)
    scores_attack : np.ndarray shape (N,)

    Returns
    -------
    AdversarialMetrics
    """
    auroc_clean  = roc_auc_score(y_true_clean,  scores_clean)
    auroc_attack = roc_auc_score(y_true_attack, scores_attack)
    return AdversarialMetrics(
        auroc_clean=auroc_clean,
        auroc_attack=auroc_attack,
        auroc_delta=auroc_attack - auroc_clean,
    )


# ─────────────────────────────────────────────────────────────────────────────
class MetricTracker:
    """
    Lightweight running-average tracker for training metrics.

    Usage
    -----
    >>> tracker = MetricTracker()
    >>> tracker.update({"loss": 0.4, "auroc": 0.91})
    >>> tracker.averages()
    {'loss': 0.4, 'auroc': 0.91}
    """

    def __init__(self) -> None:
        self._sums:   dict[str, float] = {}
        self._counts: dict[str, int]   = {}

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0)  + v * n
            self._counts[k] = self._counts.get(k, 0)   + n

    def averages(self) -> dict[str, float]:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
