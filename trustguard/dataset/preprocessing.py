"""
trustguard/dataset/preprocessing.py
======================================
Feature extraction and preprocessing utilities for PermissionBench.

Handles:
  - APK manifest parsing (permissions, API declarations)
  - API call graph construction (PyG Data objects)
  - Text normalisation for BERT / CodeBERT tokenisation
  - Per-permission risk label annotation (automated + rule-based stage)
  - Train / val / test split stratification
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as GeoData

from trustguard.models.permission_predictor import (
    ANDROID_PERMISSIONS, NUM_PERMISSIONS, PERMISSION_TO_IDX
)

logger = logging.getLogger(__name__)

# ── Permission annotation thresholds (§5.5 of the paper) ────────────────────
TAU_LOW  = 0.05   # below this predicted prob → anomalous
TAU_HIGH = 0.70   # above this → legitimate


# ─────────────────────────────────────────────────────────────────────────────
def normalise_description(text: str, max_chars: int = 2000) -> str:
    """
    Clean app-store description text for BERT tokenisation.

    - Strips HTML tags
    - Normalises whitespace
    - Truncates to max_chars
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
def build_api_feature_string(api_calls: list[str], max_calls: int = 200) -> str:
    """
    Concatenate a list of API call strings into a single CodeBERT-ready string.

    Parameters
    ----------
    api_calls : list[str]
        List of Android API method signatures.
    max_calls : int
        Truncate to this many calls.

    Returns
    -------
    str — space-separated API signatures
    """
    truncated = api_calls[:max_calls]
    # Keep only the method name (strip package prefix for brevity)
    short_names = [call.split(".")[-1] for call in truncated]
    return " ".join(short_names)


# ─────────────────────────────────────────────────────────────────────────────
def build_api_call_graph(
    api_calls:  list[str],
    call_edges: list[tuple[int, int]],
    vocab_size: int = 10_000,
) -> GeoData:
    """
    Construct a PyTorch Geometric graph from an API call list and edge list.

    Nodes are API call indices (from a simple integer vocabulary).
    Node features are one-hot encoded API call indices projected to 128-dim.

    Parameters
    ----------
    api_calls   : list[str]  — ordered list of API calls (nodes)
    call_edges  : list[(src, dst)]  — caller→callee edges
    vocab_size  : int  — vocabulary size for node embedding lookup

    Returns
    -------
    torch_geometric.data.Data
    """
    n_nodes = len(api_calls)
    if n_nodes == 0:
        # Degenerate graph: single dummy node, no edges
        x = torch.zeros(1, 128)
        return GeoData(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

    # Node features: clipped integer IDs (hash-based for reproducibility)
    node_ids = torch.tensor(
        [hash(call) % vocab_size for call in api_calls],
        dtype=torch.long,
    )
    # Simple embedding: scatter to 128-dim via modular indexing
    x = torch.zeros(n_nodes, 128)
    for i, nid in enumerate(node_ids):
        x[i, nid % 128] = 1.0   # sparse one-hot, 128-dim

    if call_edges:
        src = torch.tensor([e[0] for e in call_edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in call_edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        # Clamp to valid node range
        mask       = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
        edge_index = edge_index[:, mask]
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return GeoData(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────────────────────────────────────
def annotate_permission_labels(
    df:              pd.DataFrame,
    predicted_probs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Assign per-permission risk labels using the three-step annotation
    protocol described in §5.5 of the TrustGuard paper.

    Stage 1 (automated):   Taint-tracking flags (requires ``taint_flags`` col)
    Stage 2 (rule-based):  Threshold on predicted probability
    Stage 3 (human):       Ambiguous cases marked for manual review

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``permissions`` (list) and ``risk_label`` (int) columns.
    predicted_probs : np.ndarray, optional  shape (N, NUM_PERMISSIONS)
        Output of PermissionPredictionModel.predict_proba. Required for Stage 2.

    Returns
    -------
    pd.DataFrame with added column ``perm_labels`` (list of NUM_PERMISSIONS floats).
    """
    from trustguard.dataset.permissionbench_loader import parse_permission_vector

    perm_vecs = np.stack(df["permissions"].apply(parse_permission_vector).values)
    N = len(df)
    perm_labels = perm_vecs.copy()   # default: declared = legitimate

    # ── Stage 1: taint-tracking flags ────────────────────────────────
    if "taint_flags" in df.columns:
        taint_flags = np.stack(
            df["taint_flags"].apply(
                lambda x: np.array(json.loads(x) if isinstance(x, str) else x,
                                   dtype=np.float32)
            ).values
        )
        perm_labels = np.maximum(perm_labels, taint_flags)
        logger.info("Stage 1: taint flags applied to %d records.", N)

    # ── Stage 2: prediction-based threshold ──────────────────────────
    if predicted_probs is not None:
        if predicted_probs.shape != (N, NUM_PERMISSIONS):
            logger.warning(
                "predicted_probs shape %s does not match (%d, %d). Skipping Stage 2.",
                predicted_probs.shape, N, NUM_PERMISSIONS,
            )
        else:
            # Declared permissions with low predicted prob → anomalous
            low_prob_mask  = (predicted_probs < TAU_LOW) & (perm_vecs > 0)
            # Declared permissions with high predicted prob → legitimate
            high_prob_mask = (predicted_probs > TAU_HIGH) & (perm_vecs > 0)

            perm_labels = np.where(low_prob_mask,  1.0, perm_labels)
            perm_labels = np.where(high_prob_mask, 0.0, perm_labels)
            logger.info("Stage 2: threshold labels applied.")

    # ── Stage 3: identify ambiguous cases ────────────────────────────
    if predicted_probs is not None:
        ambiguous = (
            (predicted_probs >= TAU_LOW) &
            (predicted_probs <= TAU_HIGH) &
            (perm_vecs > 0)
        )
        df["needs_human_review"] = ambiguous.any(axis=1)
        n_ambig = df["needs_human_review"].sum()
        logger.info(
            "Stage 3: %d records (%.1f%%) flagged for human review.",
            n_ambig, 100.0 * n_ambig / N,
        )

    df["perm_labels"] = list(perm_labels.astype(np.float32))
    return df


# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(
    df:           pd.DataFrame,
    ratios:       tuple[float, float, float] = (0.70, 0.10, 0.20),
    label_col:    str = "risk_label",
    seed:         int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split preserving class balance.

    Parameters
    ----------
    df        : pd.DataFrame
    ratios    : (train, val, test)
    label_col : str
    seed      : int

    Returns
    -------
    (df_train, df_val, df_test)
    """
    from sklearn.model_selection import train_test_split

    tr_r, val_r, te_r = ratios
    assert abs(tr_r + val_r + te_r - 1.0) < 1e-6

    df_train, df_temp = train_test_split(
        df, test_size=(val_r + te_r), stratify=df[label_col], random_state=seed
    )
    val_frac = val_r / (val_r + te_r)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1.0 - val_frac),
        stratify=df_temp[label_col], random_state=seed,
    )

    logger.info(
        "Stratified split — train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
def save_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    output_dir: Union[str, Path],
) -> None:
    """Save dataframe splits as Parquet files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out / "train.parquet", index=False)
    df_val.to_parquet(out / "val.parquet",     index=False)
    df_test.to_parquet(out / "test.parquet",   index=False)
    logger.info("Splits saved to %s", out)
