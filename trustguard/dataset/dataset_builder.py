"""
trustguard/dataset/dataset_builder.py
========================================
Programmatic dataset construction API used by scripts/build_dataset.py.

Provides ``DatasetBuilder``, a high-level coordinator that orchestrates:
  - APK feature extraction (manifest permissions, API call graph, description)
  - Runtime trace collection
  - Annotation (Stage 1–2 automated + Stage 3 human-review flagging)
  - Train / val / test split and serialisation

This module is designed to be called either from the build script or from
a Jupyter notebook for interactive dataset assembly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from trustguard.dataset.preprocessing import (
    normalise_description,
    build_api_feature_string,
    annotate_permission_labels,
    stratified_split,
    save_splits,
)
from trustguard.models.permission_predictor import ANDROID_PERMISSIONS, NUM_PERMISSIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AppRecord:
    """
    Raw application record before conversion to a DataFrame row.

    Attributes
    ----------
    app_id          : str
    category        : str
    description     : str
    permissions     : list[str]   — declared permission names
    api_calls       : list[str]   — API method signatures
    call_edges      : list[tuple] — (src, dst) call graph edges
    risk_label      : int         — 0=benign, 1=malicious
    runtime_counts  : list[float] — per-permission invocation counts (optional)
    taint_flags     : list[float] — per-permission taint flags (optional)
    source          : str         — "androzoo", "drebin", "maldroid"
    sha256          : str         — APK SHA-256 hash
    """
    app_id:         str
    category:       str
    description:    str
    permissions:    list[str]
    api_calls:      list[str]
    call_edges:     list[tuple]         = field(default_factory=list)
    risk_label:     int                 = 0
    runtime_counts: Optional[list[float]] = None
    taint_flags:    Optional[list[float]] = None
    source:         str                 = "unknown"
    sha256:         str                 = ""

    def to_dict(self) -> dict:
        return {
            "app_id":        self.app_id,
            "category":      self.category,
            "description":   normalise_description(self.description),
            "permissions":   json.dumps(self.permissions),
            "api_features":  build_api_feature_string(self.api_calls),
            "risk_label":    self.risk_label,
            "runtime_trace": json.dumps(
                self.runtime_counts if self.runtime_counts
                else [0.0] * NUM_PERMISSIONS
            ),
            "taint_flags":   json.dumps(
                self.taint_flags if self.taint_flags
                else [0.0] * NUM_PERMISSIONS
            ),
            "source":        self.source,
            "sha256":        self.sha256,
        }


# ─────────────────────────────────────────────────────────────────────────────
class DatasetBuilder:
    """
    Incremental dataset assembler.

    Usage
    -----
    >>> builder = DatasetBuilder(output_dir="data/permissionbench")
    >>> builder.add_record(app_record)
    >>> builder.add_records(list_of_records)
    >>> builder.finalise(predicted_probs=pred_probs_array)
    """

    def __init__(
        self,
        output_dir:    Union[str, Path],
        split_ratios:  tuple[float, float, float] = (0.70, 0.10, 0.20),
        seed:          int = 42,
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_ratios = split_ratios
        self.seed         = seed
        self._records:    list[dict] = []

    # ------------------------------------------------------------------
    def add_record(self, record: AppRecord) -> None:
        """Append a single AppRecord to the internal buffer."""
        self._records.append(record.to_dict())

    # ------------------------------------------------------------------
    def add_records(self, records: list[AppRecord]) -> None:
        """Append multiple AppRecord objects."""
        for r in records:
            self._records.append(r.to_dict())
        logger.info("Buffer now holds %d records.", len(self._records))

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the internal buffer to a pandas DataFrame."""
        if not self._records:
            raise ValueError("No records in buffer. Call add_record() first.")
        return pd.DataFrame(self._records)

    # ------------------------------------------------------------------
    def finalise(
        self,
        predicted_probs: Optional[np.ndarray] = None,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply annotation protocol, split, and (optionally) serialise.

        Parameters
        ----------
        predicted_probs : np.ndarray, optional  shape (N, NUM_PERMISSIONS)
            Output of PermissionPredictionModel; enables Stage 2 annotation.
        save : bool
            If True, write Parquet splits to output_dir.

        Returns
        -------
        (df_train, df_val, df_test)
        """
        df = self.to_dataframe()
        logger.info(
            "Finalising dataset: %d records | %d benign | %d malicious",
            len(df),
            (df["risk_label"] == 0).sum(),
            (df["risk_label"] == 1).sum(),
        )

        df = annotate_permission_labels(df, predicted_probs)
        df_train, df_val, df_test = stratified_split(
            df, ratios=self.split_ratios, seed=self.seed
        )

        if save:
            save_splits(df_train, df_val, df_test, self.output_dir)
            # Also save full dataset
            full_path = self.output_dir / "permissionbench_full.parquet"
            df.to_parquet(full_path, index=False)
            logger.info("Full dataset saved to %s", full_path)

            # Save dataset card
            self._write_dataset_card(df)

        return df_train, df_val, df_test

    # ------------------------------------------------------------------
    def _write_dataset_card(self, df: pd.DataFrame) -> None:
        """Write a JSON dataset card with summary statistics."""
        card = {
            "name": "PermissionBench",
            "version": "1.0",
            "total_records": len(df),
            "benign":    int((df["risk_label"] == 0).sum()),
            "malicious": int((df["risk_label"] == 1).sum()),
            "num_permissions": NUM_PERMISSIONS,
            "categories": df["category"].nunique(),
            "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "needs_human_review": int(df["needs_human_review"].sum())
                if "needs_human_review" in df.columns else 0,
        }
        card_path = self.output_dir / "dataset_card.json"
        import json
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        logger.info("Dataset card saved to %s", card_path)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"DatasetBuilder(records={len(self._records)}, output='{self.output_dir}')"
