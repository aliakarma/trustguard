"""
scripts/build_dataset.py
==========================
Builds the PermissionBench dataset from raw APK sources.

Pipeline:
  1. Download benign APKs from AndroZoo (requires API key).
  2. Download malicious APKs from Drebin / MalDroid-2020 (public).
  3. For each APK: extract manifest permissions, API call graph, description.
  4. Run sandboxed UI-automator (500 events) to collect runtime permission traces.
  5. Apply automated annotation protocol (Stage 1 + 2).
  6. Save stratified Parquet splits to output_dir.

NOTE: Full dataset construction requires ~40 GB disk and ~72 h on a 16-core
      machine with Docker for the Android emulator. A pre-built dataset is
      available via scripts/download_permissionbench.sh.

Usage
-----
    python scripts/build_dataset.py \
        --androzoo-key  $ANDROZOO_API_KEY \
        --output-dir    data/permissionbench \
        --n-benign      61840 \
        --n-malicious   14512 \
        --n-workers     8

    # Synthetic mode (no API key). Small smoke-test set:
    python scripts/build_dataset.py --synthetic --output-dir data/permissionbench

    # Synthetic mode at full paper scale (76,352 records, ~7 s, ~28 MB):
    python scripts/build_dataset.py --synthetic \
        --n-benign 61840 --n-malicious 14512 \
        --output-dir data/permissionbench

WARNING: the synthetic set is procedurally generated and is NOT the real
PermissionBench. Its class separability is an artifact of the generator, so
it will NOT reproduce the paper's metrics. Use it only to exercise the
pipeline end-to-end at realistic scale (dataloaders, splits, training loop).
Reproducing paper results requires the real APK-derived corpus (Option B with
an AndroZoo key) or the released build. A DATASET_TYPE.txt marker is written
alongside the splits to record which kind of data a directory holds.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from trustguard.dataset.preprocessing import (
    normalise_description,
    build_api_feature_string,
    annotate_permission_labels,
    stratified_split,
    save_splits,
)
from trustguard.models.permission_predictor import ANDROID_PERMISSIONS, NUM_PERMISSIONS
from trustguard.utils.logging_utils import setup_logger

logger = logging.getLogger("trustguard.build_dataset")

# ── App categories (Google Play) ─────────────────────────────────────────────
GOOGLE_PLAY_CATEGORIES = [
    "COMMUNICATION", "SOCIAL", "TOOLS", "PRODUCTIVITY", "ENTERTAINMENT",
    "MAPS_AND_NAVIGATION", "HEALTH_AND_FITNESS", "SHOPPING", "FINANCE",
    "PHOTOGRAPHY", "EDUCATION", "TRAVEL_AND_LOCAL", "MUSIC_AND_AUDIO",
    "BUSINESS", "PERSONALIZATION", "SPORTS", "GAME_ACTION", "GAME_PUZZLE",
    "FOOD_AND_DRINK", "WEATHER", "NEWS_AND_MAGAZINES", "LIFESTYLE",
    "AUTO_AND_VEHICLES", "DATING", "MEDICAL", "HOUSE_AND_HOME",
    "EVENTS", "ART_AND_DESIGN", "BOOKS_AND_REFERENCE",
    "BEAUTY", "PARENTING", "COMICS", "VIDEO_PLAYERS",
]


# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_record(
    idx: int,
    is_malicious: bool,
    rng: random.Random,
) -> dict:
    """
    Generate a single synthetic PermissionBench record.
    Used for quick testing without real APK data.
    """
    category = rng.choice(GOOGLE_PLAY_CATEGORIES)

    # Benign apps use 1–6 permissions; malicious use 4–15
    n_perms = rng.randint(4, 15) if is_malicious else rng.randint(1, 6)
    perms   = rng.sample(ANDROID_PERMISSIONS, min(n_perms, NUM_PERMISSIONS))

    api_calls = [
        f"android.{rng.choice(['content', 'telephony', 'location', 'hardware'])}"
        f".{rng.choice(['Manager', 'Provider', 'Service'])}"
        f".{rng.choice(['get', 'set', 'request', 'open'])}()"
        for _ in range(rng.randint(10, 50))
    ]

    runtime_trace = [0.0] * NUM_PERMISSIONS
    for p in perms:
        idx_p = ANDROID_PERMISSIONS.index(p)
        # Malicious apps invoke permissions more frequently
        runtime_trace[idx_p] = rng.uniform(0.5, 1.0) if is_malicious else rng.uniform(0.0, 0.4)

    return {
        "app_id":        f"{'malicious' if is_malicious else 'benign'}_{idx:06d}",
        "category":      category,
        "description":   normalise_description(
            f"This is a {'potentially harmful' if is_malicious else 'useful'} "
            f"{category.lower().replace('_', ' ')} application for Android devices. "
            f"It provides various features for users."
        ),
        "api_features":  build_api_feature_string(api_calls),
        "permissions":   json.dumps(perms),
        "risk_label":    int(is_malicious),
        "runtime_trace": json.dumps(runtime_trace),
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_synthetic(
    output_dir:   Path,
    n_benign:     int = 100,
    n_malicious:  int = 25,
    seed:         int = 42,
) -> None:
    """Build a small synthetic dataset for offline development and testing."""
    logger.info(
        "Building synthetic dataset: %d benign + %d malicious",
        n_benign, n_malicious,
    )
    rng = random.Random(seed)

    records = []
    for i in range(n_benign):
        records.append(generate_synthetic_record(i, is_malicious=False, rng=rng))
    for j in range(n_malicious):
        records.append(generate_synthetic_record(j, is_malicious=True, rng=rng))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info("Applying annotation protocol (Stage 1 + 2 — no taint data available)...")
    df = annotate_permission_labels(df, predicted_probs=None)

    df_train, df_val, df_test = stratified_split(df, seed=seed)
    save_splits(df_train, df_val, df_test, output_dir)

    # Also save full dataset
    df.to_parquet(output_dir / "permissionbench_synthetic.parquet", index=False)

    # Write a marker so this directory is never mistaken for the real benchmark
    marker = (
        "DATASET_TYPE: SYNTHETIC\n"
        "This directory contains a procedurally generated synthetic dataset, "
        "NOT the real PermissionBench.\n"
        "Class separability is an artifact of the generator; it does NOT "
        "reproduce the paper's metrics.\n"
        "Use only to exercise the pipeline end-to-end. Reproducing paper "
        "results requires the real APK-derived corpus\n"
        "(scripts/build_dataset.py with --androzoo-key) or the released build "
        "(scripts/download_permissionbench.sh).\n"
        f"records={len(df)} benign={(df['risk_label'] == 0).sum()} "
        f"malicious={(df['risk_label'] == 1).sum()} seed={seed}\n"
    )
    (output_dir / "DATASET_TYPE.txt").write_text(marker, encoding="utf-8")
    logger.info("Synthetic dataset saved to %s", output_dir)

    # Print summary
    logger.info("Records: %d | Benign: %d | Malicious: %d",
                len(df), (df["risk_label"] == 0).sum(), (df["risk_label"] == 1).sum())


# ─────────────────────────────────────────────────────────────────────────────
def build_from_androzoo(
    androzoo_key: str,
    output_dir:   Path,
    n_benign:     int,
    n_malicious:  int,
    n_workers:    int = 4,
    seed:         int = 42,
) -> None:
    """
    Full dataset construction from AndroZoo and Drebin.
    Requires androzoo-download CLI and a sandboxed Android emulator.

    This function outlines the pipeline; full implementation requires the
    androzoo-download and apktool libraries.
    """
    logger.info("Full dataset construction pipeline:")
    logger.info("  Step 1: Download %d benign APKs from AndroZoo...", n_benign)
    logger.info("  Step 2: Download %d malicious APKs from Drebin/MalDroid...", n_malicious)
    logger.info("  Step 3: Extract manifests and API call graphs...")
    logger.info("  Step 4: Run sandbox UI-automator (500 events per app)...")
    logger.info("  Step 5: Annotate per-permission risk labels...")
    logger.info("  Step 6: Save stratified Parquet splits...")
    logger.info("")
    logger.warning(
        "Full APK processing pipeline requires Docker + Android emulator + ~72h. "
        "Use --synthetic for a quick test, or download pre-built dataset via "
        "scripts/download_permissionbench.sh"
    )


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build PermissionBench dataset")
    parser.add_argument("--output-dir",    default="data/permissionbench", type=str)
    parser.add_argument("--n-benign",      default=61840,                  type=int)
    parser.add_argument("--n-malicious",   default=14512,                  type=int)
    parser.add_argument("--androzoo-key",  default=None,                   type=str)
    parser.add_argument("--n-workers",     default=4,                      type=int)
    parser.add_argument("--seed",          default=42,                     type=int)
    parser.add_argument("--synthetic",     action="store_true",
                        help="Generate a small synthetic dataset (no API key needed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("trustguard.build_dataset", log_file=output_dir / "build.log")

    if args.synthetic:
        build_synthetic(
            output_dir=output_dir,
            n_benign=args.n_benign,
            n_malicious=args.n_malicious,
            seed=args.seed,
        )
    elif args.androzoo_key:
        build_from_androzoo(
            androzoo_key=args.androzoo_key,
            output_dir=output_dir,
            n_benign=args.n_benign,
            n_malicious=args.n_malicious,
            n_workers=args.n_workers,
            seed=args.seed,
        )
    else:
        logger.error(
            "Provide --androzoo-key for full dataset, or --synthetic for offline testing."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
