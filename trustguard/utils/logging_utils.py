"""
trustguard/utils/logging_utils.py
===================================
Structured logging setup for TrustGuard experiments.

Provides:
  - ``setup_logger``     : configure root logger with rich console + file handler
  - ``ExperimentLogger`` : thin W&B / TensorBoard wrapper for metric logging
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(
    name:       str  = "trustguard",
    level:      str  = "INFO",
    log_file:   Optional[Path] = None,
    rich:       bool = True,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Parameters
    ----------
    name     : str
    level    : str    — "DEBUG", "INFO", "WARNING", "ERROR"
    log_file : Path   — if provided, also write to this file
    rich     : bool   — use RichHandler for pretty console output

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ── Console handler ───────────────────────────────────────────
        if rich:
            ch = RichHandler(console=_console, rich_tracebacks=True, markup=True)
            ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        else:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
        logger.addHandler(ch)

        # ── File handler ──────────────────────────────────────────────
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
class ExperimentLogger:
    """
    Unified metric logger supporting W&B and/or TensorBoard.

    Parameters
    ----------
    project   : str   — W&B project name
    run_name  : str   — experiment run identifier
    config    : dict  — hyperparameters to log
    use_wandb : bool
    use_tb    : bool
    tb_dir    : Path  — TensorBoard log directory
    """

    def __init__(
        self,
        project:   str  = "trustguard",
        run_name:  str  = "run",
        config:    Optional[dict] = None,
        use_wandb: bool = False,
        use_tb:    bool = True,
        tb_dir:    Path = Path("runs"),
    ) -> None:
        self._use_wandb = use_wandb
        self._use_tb    = use_tb
        self._step      = 0

        # ── W&B initialisation ────────────────────────────────────────
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config or {})
                self._wandb = wandb
            except ImportError:
                logging.getLogger(__name__).warning(
                    "wandb not installed — W&B logging disabled."
                )
                self._use_wandb = False

        # ── TensorBoard initialisation ────────────────────────────────
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(tb_dir) / run_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                logging.getLogger(__name__).warning(
                    "TensorBoard not installed — TB logging disabled."
                )
                self._use_tb = False

    # ------------------------------------------------------------------
    def log(
        self,
        metrics: dict[str, Any],
        step:    Optional[int] = None,
    ) -> None:
        """
        Log a dict of scalar metrics.

        Parameters
        ----------
        metrics : dict[str, float]
        step    : int, optional — if None, uses internal counter
        """
        if step is None:
            step = self._step
            self._step += 1

        if self._use_wandb:
            self._wandb.log(metrics, step=step)

        if self._use_tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, global_step=step)

    # ------------------------------------------------------------------
    def log_histogram(self, tag: str, values: Any, step: Optional[int] = None) -> None:
        if self._use_tb:
            self._writer.add_histogram(tag, values, global_step=step or self._step)

    # ------------------------------------------------------------------
    def finish(self) -> None:
        if self._use_wandb:
            self._wandb.finish()
        if self._use_tb:
            self._writer.close()
