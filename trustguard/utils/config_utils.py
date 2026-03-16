"""
trustguard/utils/config_utils.py
==================================
YAML configuration loading and experiment reproducibility utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str | Path) -> dict:
    """
    Load and merge all YAML configs from a directory.
    Later files override earlier ones on key collision.
    """
    config_dir = Path(config_dir)
    merged: dict = {}
    for yml in sorted(config_dir.glob("*.yaml")):
        merged.update(load_config(yml))
    return merged


def save_config(config: dict, path: str | Path) -> None:
    """Serialise config dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)


# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy,
    PyTorch (CPU and CUDA), and CUDA determinism flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
def config_hash(config: dict) -> str:
    """Return a short deterministic hash of a config dict (for run naming)."""
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialised.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available compute device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
