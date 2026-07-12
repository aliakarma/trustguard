"""
experiments/evaluate_temporal.py
==================================
Temporal robustness evaluation (§"Temporal Robustness" of the paper).

Follows the temporally consistent protocol of TESSERACT
(Pendlebury et al., 2019): models train on 2012–2018 records (~53,000) and
evaluate on 2019–2021 records (~23,350) with no post-2018 fine-tuning.

The PermissionBench release carries a ``year`` column (first-seen year from
AndroZoo/Drebin metadata); this script re-splits the corpus by year, runs
inference with the supplied checkpoint on the post-gap split, and prints the
final paper table (tab:temporal; also stored in
results/temporal_holdout.json) alongside the measured values.

Usage
-----
    python experiments/evaluate_temporal.py \
        --checkpoint outputs/run_001/checkpoint_best.pt \
        --data-dir   data/permissionbench \
        --output-dir outputs/eval_temporal \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from trustguard.models.permission_predictor import PermissionPredictionModel
from trustguard.dataset.permissionbench_loader import (
    PermissionBenchLoader, PermissionBenchDataset,
)
from trustguard.utils.config_utils import load_all_configs, seed_everything, get_device
from trustguard.utils.logging_utils import setup_logger
from trustguard.utils.metrics import compute_prediction_metrics

logger = logging.getLogger("trustguard.eval_temporal")

TRAIN_YEARS = (2012, 2018)
TEST_YEARS  = (2019, 2021)

# Final values as reported in the paper (tab:temporal); mean ± std, 5 seeds.
PAPER_REFERENCE = {
    "DREBIN":            {"auroc": "0.846 ± 0.005", "aipr": "—",          "frr": "—"},
    "MaMaDroid":         {"auroc": "0.867 ± 0.005", "aipr": "—",          "frr": "—"},
    "DexRay":            {"auroc": "0.905 ± 0.004", "aipr": "—",          "frr": "—"},
    "Rule Threshold":    {"auroc": "0.882 ± 0.004", "aipr": "32.8 ± 2.3", "frr": "13.9 ± 1.0"},
    "Single-Agent RL":   {"auroc": "0.918 ± 0.004", "aipr": "43.1 ± 2.0", "frr": "8.1 ± 0.6"},
    "SA PPO-Lagr.":      {"auroc": "0.916 ± 0.004", "aipr": "41.9 ± 1.9", "frr": "2.9 ± 0.4"},
    "MAPPO-Lagr.":       {"auroc": "0.931 ± 0.004", "aipr": "50.6 ± 1.9", "frr": "2.8 ± 0.4"},
    "TrustGuard (ours)": {"auroc": "0.941 ± 0.004", "aipr": "55.2 ± 1.8", "frr": "2.6 ± 0.4"},
}


def print_paper_table() -> None:
    header = f"{'Method':<22} {'AUROC':<16} {'AIPR (%)':<14} {'FRR (%)':<12}"
    logger.info("\n%s\n%s", header, "-" * len(header))
    for name, row in PAPER_REFERENCE.items():
        marker = "  ◄" if "TrustGuard" in name else ""
        logger.info("%-22s %-16s %-14s %-12s%s",
                    name, row["auroc"], row["aipr"], row["frr"], marker)
    logger.info("-" * len(header))


@torch.no_grad()
def evaluate_split(model, dataset, device, batch_size: int = 512):
    """Inference over one temporal split; returns (labels, scores)."""
    model.eval()
    labels, scores = [], []
    loader = PermissionBenchLoader.get_dataloader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    for batch in loader:
        perm_vec = batch["perm_vector"].to(device)
        probs    = model.predict_proba(perm_vec).cpu().numpy()
        scores.append((1.0 - probs).max(axis=-1))
        labels.append(batch["risk_label"].numpy())
    return np.concatenate(labels), np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal hold-out evaluation")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-dir",   required=True, type=str)
    parser.add_argument("--config-dir", default="configs/", type=str)
    parser.add_argument("--output-dir", default="outputs/eval_temporal", type=str)
    parser.add_argument("--seed",       default=42, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("trustguard.eval_temporal", log_file=output_dir / "eval_temporal.log")
    seed_everything(args.seed)
    device = get_device()

    cfg = load_all_configs(args.config_dir)
    mc  = cfg.get("model", cfg)

    perm_pred = PermissionPredictionModel(
        embedding_dim=mc["permission_predictor"]["embedding_dim"],
        hidden_dims=tuple(mc["permission_predictor"]["hidden_dims"]),
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    perm_pred.load_state_dict(ckpt["permission_predictor"])

    measured = None
    try:
        loader = PermissionBenchLoader(args.data_dir)
        df = loader.load_dataframe() if hasattr(loader, "load_dataframe") else None
        if df is None:
            # fall back to the standard splits and their concatenation
            train_ds, val_ds, test_ds = loader.get_datasets()
            import pandas as pd
            df = pd.concat([train_ds.df, val_ds.df, test_ds.df], ignore_index=True)

        if "year" not in df.columns:
            logger.warning(
                "Dataset has no 'year' column — cannot form the temporal split. "
                "Rebuild PermissionBench with scripts/build_dataset.py (which "
                "records first-seen year) to run this protocol."
            )
        else:
            test_df = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1])]
            logger.info(
                "Temporal split: %d train-era (%d–%d) / %d test-era (%d–%d) records",
                int(((df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])).sum()),
                *TRAIN_YEARS, len(test_df), *TEST_YEARS,
            )
            test_ds = PermissionBenchDataset(test_df, augment=False)
            y, s = evaluate_split(perm_pred, test_ds, device)
            m = compute_prediction_metrics(y, (s >= 0.5).astype(int), s)
            measured = {"auroc": m.auroc, "macro_f1": m.macro_f1}
            logger.info("Measured on %d–%d split: %s", *TEST_YEARS, m)
    except FileNotFoundError:
        logger.warning("PermissionBench not found at %s — printing paper "
                       "reference table only.", args.data_dir)

    print_paper_table()

    results = {
        "_provenance": "Final values as reported in the paper (tab:temporal); "
                       "see results/temporal_holdout.json.",
        "protocol": {"train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
                     "fine_tuning": "none post-2018"},
        "paper_reference": PAPER_REFERENCE,
        "measured_this_run": measured,
    }
    out = output_dir / "temporal_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
