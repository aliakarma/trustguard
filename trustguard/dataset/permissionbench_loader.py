"""
trustguard/dataset/permissionbench_loader.py
=============================================
PermissionBench dataset loader and PyTorch Dataset wrapper.

PermissionBench comprises 76,352 annotated application records:
  - 61,840 benign (from AndroZoo, Google Play)
  - 14,512 malicious (from Drebin + MalDroid-2020)

Each record contains:
  - app_id:          unique identifier
  - category:        Google Play category string
  - description:     app-store text description
  - permissions:     list of declared permission strings
  - api_features:    concatenated API call names
  - risk_label:      0 = benign, 1 = malicious
  - perm_labels:     per-permission binary risk label (42-dim vector)
  - runtime_trace:   (optional) per-permission usage counts from sandbox

File format: HDF5 or Parquet (auto-detected from extension).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from trustguard.models.permission_predictor import ANDROID_PERMISSIONS, NUM_PERMISSIONS

logger = logging.getLogger(__name__)

# ── Expected schema ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "app_id", "category", "description",
    "permissions", "api_features", "risk_label",
]


# ─────────────────────────────────────────────────────────────────────────────
def parse_permission_vector(perm_list: Union[str, list]) -> np.ndarray:
    """
    Convert a list of permission name strings to a binary indicator vector.

    Parameters
    ----------
    perm_list : list[str] or str (JSON-encoded list)

    Returns
    -------
    np.ndarray  shape (NUM_PERMISSIONS,)  dtype=float32
    """
    if isinstance(perm_list, str):
        import json
        try:
            perm_list = json.loads(perm_list)
        except json.JSONDecodeError:
            perm_list = perm_list.split(",")

    vec = np.zeros(NUM_PERMISSIONS, dtype=np.float32)
    for p in perm_list:
        p_clean = p.strip().upper()
        # Strip common prefixes
        for prefix in ("ANDROID.PERMISSION.", "android.permission."):
            p_clean = p_clean.replace(prefix, "")
        if p_clean in ANDROID_PERMISSIONS:
            vec[ANDROID_PERMISSIONS.index(p_clean)] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
class PermissionBenchDataset(Dataset):
    """
    PyTorch Dataset for PermissionBench.

    Each ``__getitem__`` returns a dict with keys:
      - ``app_id``       : str
      - ``description``  : str
      - ``api_features`` : str
      - ``perm_vector``  : Tensor (NUM_PERMISSIONS,)  — declared permissions
      - ``risk_label``   : Tensor ()                  — binary 0/1
      - ``perm_labels``  : Tensor (NUM_PERMISSIONS,)  — per-permission labels
      - ``runtime_trace``: Tensor (NUM_PERMISSIONS,)  — usage counts (zeros if absent)

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pre-loaded and validated dataframe.
    augment : bool
        If True, apply light data augmentation (description token shuffling,
        permission dropout with p=0.05) during training.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        augment: bool = False,
    ) -> None:
        self.df      = dataframe.reset_index(drop=True)
        self.augment = augment
        self._validate()

        # Pre-compute permission vectors to avoid repeated parsing
        logger.info("Pre-computing permission vectors for %d records...", len(self.df))
        self._perm_vectors = np.stack(
            self.df["permissions"].apply(parse_permission_vector).values
        )  # (N, NUM_PERMISSIONS)

        # Per-permission labels (may be absent — fall back to risk_label broadcast)
        if "perm_labels" in self.df.columns:
            self._perm_labels = np.stack(
                self.df["perm_labels"].apply(
                    lambda x: np.frombuffer(x, dtype=np.float32)
                    if isinstance(x, bytes)
                    else parse_permission_vector(x)
                ).values
            )
        else:
            # Broadcast app-level risk label to all permissions
            self._perm_labels = (
                self.df["risk_label"].values[:, None]
                * self._perm_vectors
            ).astype(np.float32)

        # Runtime traces (optional)
        if "runtime_trace" in self.df.columns:
            self._runtime_traces = np.stack(
                self.df["runtime_trace"].apply(
                    lambda x: np.frombuffer(x, dtype=np.float32)
                    if isinstance(x, bytes)
                    else np.zeros(NUM_PERMISSIONS, dtype=np.float32)
                ).values
            )
        else:
            self._runtime_traces = np.zeros(
                (len(self.df), NUM_PERMISSIONS), dtype=np.float32
            )

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"PermissionBench missing required columns: {missing}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        perm_vec     = self._perm_vectors[idx].copy()
        perm_labels  = self._perm_labels[idx].copy()
        runtime_tr   = self._runtime_traces[idx].copy()

        # ── Light augmentation (training only) ───────────────────────
        if self.augment:
            # Random permission dropout: zero out observed permissions with p=0.05
            dropout_mask = np.random.rand(NUM_PERMISSIONS) > 0.05
            perm_vec    *= dropout_mask.astype(np.float32)

        return {
            "app_id":        row["app_id"],
            "description":   str(row["description"]),
            "api_features":  str(row.get("api_features", "")),
            "category":      str(row.get("category", "unknown")),
            "perm_vector":   torch.from_numpy(perm_vec),
            "risk_label":    torch.tensor(float(row["risk_label"]), dtype=torch.float32),
            "perm_labels":   torch.from_numpy(perm_labels),
            "runtime_trace": torch.from_numpy(runtime_tr),
        }

    # ------------------------------------------------------------------
    @property
    def class_weights(self) -> Tensor:
        """
        Per-sample weights for WeightedRandomSampler (handles class imbalance).
        Weight = 1 / class_frequency.
        """
        labels  = self.df["risk_label"].values.astype(np.float32)
        n_pos   = labels.sum()
        n_neg   = len(labels) - n_pos
        w_pos   = len(labels) / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg   = len(labels) / (2.0 * n_neg) if n_neg > 0 else 1.0
        weights = np.where(labels == 1, w_pos, w_neg)
        return torch.from_numpy(weights.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
class PermissionBenchLoader:
    """
    Factory that loads PermissionBench from disk and returns train/val/test
    Dataset objects with correct splits.

    Parameters
    ----------
    data_root : str or Path
        Directory containing the PermissionBench data files.
    split_ratios : tuple[float, float, float]
        Train / validation / test split ratios (must sum to 1.0).
    seed : int
        Random seed for reproducible splits.

    Example
    -------
    >>> loader = PermissionBenchLoader("data/permissionbench")
    >>> train_ds, val_ds, test_ds = loader.get_datasets()
    >>> train_loader = loader.get_dataloader(train_ds, batch_size=256)
    """

    SUPPORTED_FORMATS = (".parquet", ".csv", ".h5", ".hdf5")

    def __init__(
        self,
        data_root:    Union[str, Path],
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed:         int = 42,
    ) -> None:
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")
        self.data_root    = Path(data_root)
        self.split_ratios = split_ratios
        self.seed         = seed

    # ------------------------------------------------------------------
    def _find_data_file(self) -> Path:
        """Locate the main dataset file in data_root."""
        for ext in self.SUPPORTED_FORMATS:
            candidates = list(self.data_root.glob(f"*permissionbench*{ext}"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"No PermissionBench data file found in {self.data_root}. "
            f"Run scripts/build_dataset.py first."
        )

    # ------------------------------------------------------------------
    def load_dataframe(self) -> pd.DataFrame:
        """Load and validate the raw dataset."""
        path = self._find_data_file()
        logger.info("Loading PermissionBench from %s", path)

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in (".h5", ".hdf5"):
            df = pd.read_hdf(path, key="permissionbench")
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        logger.info("Loaded %d records (%d columns)", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    def get_datasets(
        self,
    ) -> tuple[PermissionBenchDataset, PermissionBenchDataset, PermissionBenchDataset]:
        """
        Load data and return (train, val, test) Dataset objects.

        Returns
        -------
        train_dataset, val_dataset, test_dataset
        """
        df = self.load_dataframe()
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n      = len(df)
        n_tr   = int(n * self.split_ratios[0])
        n_val  = int(n * self.split_ratios[1])

        df_train = df.iloc[:n_tr]
        df_val   = df.iloc[n_tr: n_tr + n_val]
        df_test  = df.iloc[n_tr + n_val:]

        logger.info(
            "Splits — train: %d | val: %d | test: %d",
            len(df_train), len(df_val), len(df_test),
        )

        return (
            PermissionBenchDataset(df_train, augment=True),
            PermissionBenchDataset(df_val,   augment=False),
            PermissionBenchDataset(df_test,  augment=False),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def get_dataloader(
        dataset:         PermissionBenchDataset,
        batch_size:      int  = 256,
        num_workers:     int  = 4,
        balance_classes: bool = False,
        shuffle:         bool = True,
    ) -> DataLoader:
        """
        Construct a DataLoader for a PermissionBench split.

        Parameters
        ----------
        dataset         : PermissionBenchDataset
        batch_size      : int
        num_workers     : int
        balance_classes : bool
            Use WeightedRandomSampler to balance benign/malicious ratio.
        shuffle         : bool

        Returns
        -------
        DataLoader
        """
        sampler = None
        if balance_classes:
            weights = dataset.class_weights
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False   # sampler and shuffle are mutually exclusive

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def load_from_splits(
        data_root: Union[str, Path],
    ) -> tuple[PermissionBenchDataset, PermissionBenchDataset, PermissionBenchDataset]:
        """
        Load from pre-existing train/val/test Parquet files (if available).

        Looks for files named train.parquet, val.parquet, test.parquet in
        data_root.
        """
        root = Path(data_root)
        splits = {}
        for split in ("train", "val", "test"):
            p = root / f"{split}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Expected split file not found: {p}")
            splits[split] = pd.read_parquet(p)
            logger.info("Loaded %s split: %d records", split, len(splits[split]))

        return (
            PermissionBenchDataset(splits["train"], augment=True),
            PermissionBenchDataset(splits["val"],   augment=False),
            PermissionBenchDataset(splits["test"],  augment=False),
        )
