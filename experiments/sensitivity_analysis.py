"""
experiments/sensitivity_analysis.py
=====================================
Hyperparameter sensitivity analyses (appendix "Ablations and Sensitivity
Analyses" of the paper): four grids, final values stored in
results/sensitivity_analyses.json.

1. **Annotation thresholds (τ_low, τ_high)** — relabels PermissionBench under
   each threshold pair and re-evaluates Task-1 AUROC (tab:tau). Requires the
   dataset plus the seed-model scores shipped with the PermissionBench
   release.
2. **Reward weights (λ₁, λ₂)** — retrains the MARL policy per grid cell
   (tab:lambda). Expensive: 9 cells × 5 seeds; each cell is a full training
   run launched through train_trustguard.py.
3. **Encoder modalities (T/C/G)** — retrains g_θ on each modality subset
   (tab:modality).
4. **EMA α** — re-evaluates the trained policy with α ∈ {0.1, 0.3, 0.5, 0.7}
   (evaluation-only; cheap).

Usage
-----
    # Print the four paper grids
    python experiments/sensitivity_analysis.py --print-tables

    # EMA-α sweep on an existing checkpoint (evaluation-only)
    python experiments/sensitivity_analysis.py --grid ema \
        --checkpoint outputs/run_001/checkpoint_best.pt

    # Launch one λ-grid cell (full retraining)
    python experiments/sensitivity_analysis.py --grid lambda \
        --lambda1 10 --lambda2 0.1 --seed 42 --data-dir data/permissionbench
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("trustguard.sensitivity")

RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "sensitivity_analyses.json"

EMA_ALPHAS   = [0.1, 0.3, 0.5, 0.7]
LAMBDA1_GRID = [5, 10, 20]
LAMBDA2_GRID = [0.05, 0.1, 0.2]
TAU_LOW_GRID  = [0.02, 0.05, 0.10]
TAU_HIGH_GRID = [0.60, 0.70, 0.80]
MODALITIES = ["T", "C", "G", "T+C", "T+G", "C+G", "T+C+G"]


def print_paper_tables() -> None:
    """Print the four sensitivity grids from results/sensitivity_analyses.json."""
    with open(RESULTS_FILE, encoding="utf-8") as f:
        ref = json.load(f)

    print("\n── Task-1 AUROC under relabeled thresholds (tab:tau) ──")
    tau = ref["annotation_thresholds_tau"]["results"]
    cols = list(next(iter(tau.values())).keys())
    print(f"{'':<16}" + "".join(f"{c:<18}" for c in cols))
    for row, vals in tau.items():
        print(f"{row:<16}" + "".join(f"{vals[c]:<18}" for c in cols))

    print("\n── AIPR% / FRR% under (λ₁, λ₂), λ₃ = 1.0 (tab:lambda) ──")
    lam = ref["reward_weights_lambda"]["results"]
    cols = list(next(iter(lam.values())).keys())
    print(f"{'':<14}" + "".join(f"{c:<22}" for c in cols))
    for row, vals in lam.items():
        cells = [f"{vals[c]['AIPR_pct']} / {vals[c]['FRR_pct']}" for c in cols]
        print(f"{row:<14}" + "".join(f"{c:<22}" for c in cells))

    print("\n── Task-1 AUROC by encoder modality (tab:modality) ──")
    for k, v in ref["encoder_modalities"]["results"].items():
        print(f"  {k:<16} {v}")

    print("\n── EMA α sweep ──")
    for k, v in ref["ema_alpha"]["results"].items():
        print(f"  {k:<12} PRR {v['PRR_pct']}%  FRR {v['FRR_pct']}%  "
              f"latency {v['median_latency_s']}s")
    print()


def run_ema_sweep(checkpoint: str, config_dir: str, output_dir: Path,
                  seed: int) -> dict:
    """Evaluation-only α sweep: re-run Task-2 evaluation per α."""
    measured = {}
    for alpha in EMA_ALPHAS:
        run_dir = output_dir / f"ema_alpha_{alpha}"
        cmd = [
            sys.executable, "experiments/evaluate_enforcement.py",
            "--checkpoint", checkpoint,
            "--config-dir", config_dir,
            "--output-dir", str(run_dir),
            "--seed", str(seed),
        ]
        env_overrides = {"TRUSTGUARD_EMA_ALPHA": str(alpha)}
        logger.info("[α = %.1f] %s", alpha, " ".join(cmd))
        import os
        subprocess.run(cmd, check=True, env={**os.environ, **env_overrides})
        result_file = run_dir / "enforcement_results.json"
        if result_file.exists():
            with open(result_file) as f:
                measured[f"alpha={alpha}"] = json.load(f).get("measured_this_run")
    return measured


def launch_lambda_cell(lambda1: float, lambda2: float, seed: int,
                       data_dir: str, output_dir: Path) -> None:
    """Full retraining for one (λ₁, λ₂) grid cell."""
    run_dir = output_dir / f"lambda1_{lambda1}_lambda2_{lambda2}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "grid_cell.json", "w") as f:
        json.dump({"lambda1": lambda1, "lambda2": lambda2, "lambda3": 1.0,
                   "seed": seed}, f, indent=2)
    cmd = [
        sys.executable, "experiments/train_trustguard.py",
        "--config-dir", "configs/",
        "--data-dir", data_dir,
        "--output-dir", str(run_dir),
        "--seed", str(seed),
    ]
    import os
    subprocess.run(cmd, check=True, env={
        **os.environ,
        "TRUSTGUARD_LAMBDA1": str(lambda1),
        "TRUSTGUARD_LAMBDA2": str(lambda2),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analyses")
    parser.add_argument("--print-tables", action="store_true")
    parser.add_argument("--grid", choices=["tau", "lambda", "modality", "ema"],
                        default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config-dir", default="configs/", type=str)
    parser.add_argument("--data-dir",   default="data/permissionbench", type=str)
    parser.add_argument("--output-dir", default="outputs/sensitivity", type=str)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.print_tables or args.grid is None:
        print_paper_tables()
        return

    if args.grid == "ema":
        if not args.checkpoint:
            parser.error("--grid ema requires --checkpoint")
        measured = run_ema_sweep(args.checkpoint, args.config_dir,
                                 output_dir, args.seed)
        with open(output_dir / "ema_sweep_results.json", "w") as f:
            json.dump(measured, f, indent=2)
        print_paper_tables()

    elif args.grid == "lambda":
        if args.lambda1 is None or args.lambda2 is None:
            # Full grid
            for l1 in LAMBDA1_GRID:
                for l2 in LAMBDA2_GRID:
                    launch_lambda_cell(l1, l2, args.seed, args.data_dir, output_dir)
        else:
            launch_lambda_cell(args.lambda1, args.lambda2, args.seed,
                               args.data_dir, output_dir)

    elif args.grid == "tau":
        logger.info(
            "τ-grid relabeling requires the PermissionBench seed-model scores "
            "(seed_scores.parquet in the dataset release). Relabel with "
            "scripts/build_dataset.py --tau-low/--tau-high, then re-run "
            "experiments/evaluate_prediction.py per cell."
        )

    elif args.grid == "modality":
        logger.info(
            "Modality grid requires retraining g_θ per subset: launch "
            "experiments/train_trustguard.py with TRUSTGUARD_MODALITIES set "
            "to one of %s.", MODALITIES,
        )


if __name__ == "__main__":
    main()
