"""
experiments/evaluate_prediction.py
=====================================
Task 1: Permission Risk Prediction Evaluation (§6.1 of the TrustGuard paper).

Evaluates the PermissionPredictionModel on the PermissionBench test split and
reports Accuracy, Macro-F1, and AUROC — the three primary metrics from Table 2.

Also computes per-permission AUROC to identify which permission categories the
model finds most/least difficult to classify.

Usage
-----
    python experiments/evaluate_prediction.py \
        --checkpoint outputs/run_001/checkpoint_best.pt \
        --data-dir   data/permissionbench \
        --config-dir configs/ \
        --output-dir outputs/eval_task1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from trustguard.models.permission_predictor import (
    PermissionPredictionModel, NUM_PERMISSIONS, ANDROID_PERMISSIONS,
)
from trustguard.dataset.permissionbench_loader import (
    PermissionBenchLoader, PermissionBenchDataset,
)
from trustguard.utils.config_utils import load_all_configs, seed_everything, get_device
from trustguard.utils.logging_utils import setup_logger
from trustguard.utils.metrics import (
    compute_prediction_metrics,
    compute_per_permission_metrics,
    PredictionMetrics,
)

logger = logging.getLogger("trustguard.eval_prediction")


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model:     PermissionPredictionModel,
    dataset:   PermissionBenchDataset,
    device:    torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference over a dataset split.

    Returns
    -------
    risk_labels   : (N,)           — app-level binary labels
    app_scores    : (N,)           — app-level aggregated risk score
    perm_labels   : (N, |𝒫|)      — per-permission binary labels
    perm_scores   : (N, |𝒫|)      — per-permission risk scores
    """
    model.eval()

    all_risk_labels: list[np.ndarray] = []
    all_app_scores:  list[np.ndarray] = []
    all_perm_labels: list[np.ndarray] = []
    all_perm_scores: list[np.ndarray] = []

    loader = PermissionBenchLoader.get_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    for batch in tqdm(loader, desc="Inference", leave=False):
        perm_vec   = batch["perm_vector"].to(device)    # (B, |𝒫|)
        risk_label = batch["risk_label"].numpy()         # (B,)
        perm_label = batch["perm_labels"].numpy()        # (B, |𝒫|)

        # Forward pass — using permission vector as proxy embedding
        # (identical to supervised pre-training setup)
        probs      = model.predict_proba(perm_vec).cpu().numpy()  # (B, |𝒫|)

        # Aggregate to app-level score: max over all per-permission risk scores
        perm_risk  = 1.0 - probs                           # (B, |𝒫|)
        app_score  = perm_risk.max(axis=-1)                # (B,)

        all_risk_labels.append(risk_label)
        all_app_scores.append(app_score)
        all_perm_labels.append(perm_label)
        all_perm_scores.append(perm_risk)

    risk_labels = np.concatenate(all_risk_labels)
    app_scores  = np.concatenate(all_app_scores)
    perm_labels = np.concatenate(all_perm_labels)
    perm_scores = np.concatenate(all_perm_scores)

    return risk_labels, app_scores, perm_labels, perm_scores


# ─────────────────────────────────────────────────────────────────────────────
def _baseline_results() -> dict:
    """
    Return paper-reported baseline metrics for comparison table.
    Values from Table 2 of the TrustGuard paper.
    """
    return {
        "Android Static Policy": {"accuracy": 0.614, "macro_f1": 0.501, "auroc": None},
        "DREBIN":                 {"accuracy": 0.881, "macro_f1": 0.864, "auroc": 0.907},
        "MaMaDroid":              {"accuracy": 0.893, "macro_f1": 0.877, "auroc": 0.921},
        "Rule-Based Threshold":   {"accuracy": 0.902, "macro_f1": 0.888, "auroc": 0.913},
        "Single-Agent RL":        {"accuracy": 0.931, "macro_f1": 0.918, "auroc": 0.947},
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(
    trustguard_metrics: PredictionMetrics,
    baselines:          dict,
) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Method':<25} {'Accuracy':>10} {'Macro-F1':>10} {'AUROC':>10}"
    sep    = "-" * len(header)
    logger.info("\n%s\n%s", header, sep)

    for name, vals in baselines.items():
        auroc_str = f"{vals['auroc']:.3f}" if vals["auroc"] is not None else "  —   "
        logger.info(
            "%-25s %10.3f %10.3f %10s",
            name, vals["accuracy"], vals["macro_f1"], auroc_str,
        )

    logger.info(sep)
    logger.info(
        "%-25s %10.3f %10.3f %10.3f  ◄ TrustGuard",
        "TrustGuard (ours)",
        trustguard_metrics.accuracy,
        trustguard_metrics.macro_f1,
        trustguard_metrics.auroc,
    )
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Permission risk prediction eval")
    parser.add_argument("--checkpoint", required=True,             type=str)
    parser.add_argument("--data-dir",   required=True,             type=str)
    parser.add_argument("--config-dir", default="configs/",        type=str)
    parser.add_argument("--output-dir", default="outputs/eval_task1", type=str)
    parser.add_argument("--batch-size", default=512,               type=int)
    parser.add_argument("--seed",       default=42,                type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("trustguard.eval_prediction", log_file=output_dir / "eval_prediction.log")
    seed_everything(args.seed)
    device = get_device()

    cfg = load_all_configs(args.config_dir)
    mc  = cfg.get("model", cfg)

    # ── Load model ────────────────────────────────────────────────────
    perm_pred = PermissionPredictionModel(
        embedding_dim=mc["permission_predictor"]["embedding_dim"],
        hidden_dims=tuple(mc["permission_predictor"]["hidden_dims"]),
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    perm_pred.load_state_dict(ckpt["permission_predictor"])
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── Load test split ───────────────────────────────────────────────
    try:
        loader             = PermissionBenchLoader(args.data_dir)
        _, _, test_dataset = loader.get_datasets()
        logger.info("Test split: %d samples", len(test_dataset))
    except FileNotFoundError:
        logger.warning("Dataset not found — generating synthetic test data.")
        test_dataset = _make_synthetic_dataset(n=2000)

    # ── Inference ─────────────────────────────────────────────────────
    risk_labels, app_scores, perm_labels, perm_scores = run_inference(
        perm_pred, test_dataset, device, batch_size=args.batch_size
    )

    # ── Task 1 metrics ────────────────────────────────────────────────
    app_preds   = (app_scores >= 0.5).astype(int)
    app_metrics = compute_prediction_metrics(risk_labels, app_preds, app_scores)

    logger.info("=== Task 1: Permission Risk Prediction ===")
    logger.info("App-level: %s", app_metrics)

    per_perm = compute_per_permission_metrics(perm_labels, perm_scores)
    if per_perm:
        logger.info(
            "Per-permission — mean AUROC: %.4f | mean F1: %.4f",
            per_perm["mean_per_perm_auroc"],
            per_perm["mean_per_perm_f1"],
        )

        # Top-5 hardest and easiest permissions by AUROC
        perm_aurocs = {
            p: v["auroc"]
            for p, v in per_perm["per_permission"].items()
        }
        sorted_perms = sorted(perm_aurocs.items(), key=lambda x: x[1])
        logger.info("Hardest permissions (lowest AUROC):")
        for p, a in sorted_perms[:5]:
            logger.info("  %-35s  AUROC=%.3f", p, a)
        logger.info("Easiest permissions (highest AUROC):")
        for p, a in sorted_perms[-5:][::-1]:
            logger.info("  %-35s  AUROC=%.3f", p, a)

    # ── Comparison table ──────────────────────────────────────────────
    print_comparison_table(app_metrics, _baseline_results())

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "app_level": {
            "accuracy": app_metrics.accuracy,
            "macro_f1": app_metrics.macro_f1,
            "auroc":    app_metrics.auroc,
            "ap":       app_metrics.ap,
        },
        "per_permission": {
            "mean_auroc": per_perm.get("mean_per_perm_auroc", None),
            "mean_f1":    per_perm.get("mean_per_perm_f1", None),
        } if per_perm else {},
    }

    out_path = output_dir / "prediction_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # ── Save score arrays for further analysis ────────────────────────
    np.save(output_dir / "risk_labels.npy",  risk_labels)
    np.save(output_dir / "app_scores.npy",   app_scores)
    np.save(output_dir / "perm_labels.npy",  perm_labels)
    np.save(output_dir / "perm_scores.npy",  perm_scores)
    logger.info("Score arrays saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
def _make_synthetic_dataset(n: int = 2000):
    """
    Create a synthetic PermissionBenchDataset for offline testing when the
    real dataset is unavailable.
    """
    import pandas as pd
    from trustguard.dataset.permissionbench_loader import PermissionBenchDataset
    import json

    np.random.seed(42)
    records = []
    for i in range(n):
        is_mal = int(i < n // 4)
        perms  = np.random.randint(0, 2, NUM_PERMISSIONS).tolist()
        records.append({
            "app_id":      f"synthetic_{i}",
            "category":    "utility",
            "description": f"Synthetic app {i}",
            "api_features": "open read write",
            "permissions": json.dumps(
                [ANDROID_PERMISSIONS[j] for j, v in enumerate(perms) if v]
            ),
            "risk_label":  is_mal,
        })
    return PermissionBenchDataset(pd.DataFrame(records), augment=False)


if __name__ == "__main__":
    main()
