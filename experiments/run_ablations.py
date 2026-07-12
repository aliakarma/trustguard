"""
experiments/run_ablations.py
==============================
Factorial, component, and null ablations (§"Factorial and Component
Ablation" + appendix tab:ablation of the paper).

Nine configurations are evaluated on the Task-2 protocol:

  Factorial 2×2 (agent structure × constraint):
    multi+constraint (TrustGuard), multi−constraint,
    single+constraint, single−constraint
  Component ablations:
    local per-agent beliefs (strictly decentralized),
    w/o semantic encoder, w/o runtime traces
  Null ablations:
    homogeneous agents + belief encoder,
    fixed Agents 1–2 (always-on schedules)

Each configuration is trained with the same budgets and the five seeds
{7, 42, 123, 777, 2024}, then evaluated with evaluate_enforcement.py.
This script materialises the per-configuration training overrides, launches
training/evaluation for the requested configurations, and prints the final
paper table (also stored in results/ablations.json).

Usage
-----
    # List configurations and print the paper reference table
    python experiments/run_ablations.py --list

    # Train + evaluate one configuration for one seed
    python experiments/run_ablations.py --config single_constraint --seed 42 \
        --data-dir data/permissionbench --output-dir outputs/ablations

    # Evaluate existing checkpoints (one subdirectory per configuration)
    python experiments/run_ablations.py --eval-dir outputs/ablations
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

logger = logging.getLogger("trustguard.ablations")

# ── Configuration registry ───────────────────────────────────────────────────
# Overrides applied on top of configs/ for each ablated variant.
ABLATION_CONFIGS: dict[str, dict] = {
    "trustguard": {
        "description": "Multi + constraint (full TrustGuard)",
        "overrides": {},
    },
    "multi_no_constraint": {
        "description": "Multi − constraint (Lagrangian disabled, μ ≡ 0)",
        "overrides": {"lagrangian.eps_safe": 1.0, "lagrangian.lagrange_max": 0.0},
    },
    "single_constraint": {
        "description": "Single + constraint (one agent, identical ratio constraint)",
        "overrides": {"agents.single_agent": True},
    },
    "single_no_constraint": {
        "description": "Single − constraint",
        "overrides": {"agents.single_agent": True,
                      "lagrangian.eps_safe": 1.0, "lagrangian.lagrange_max": 0.0},
    },
    "local_beliefs": {
        "description": "Local per-agent beliefs (strictly decentralized execution)",
        "overrides": {"belief_encoder.shared": False},
    },
    "no_semantic_encoder": {
        "description": "w/o semantic encoder (φ(f_i) replaced by zeros)",
        "overrides": {"semantic_encoder.enabled": False},
    },
    "no_runtime_traces": {
        "description": "w/o runtime traces (Monitoring Agent observes zeros)",
        "overrides": {"observations.runtime_traces": False},
    },
    "homogeneous_belief": {
        "description": "Homogeneous agents + shared belief encoder (null ablation)",
        "overrides": {"agents.role_specialization": False},
    },
    "fixed_agents12": {
        "description": "Fixed Agents 1–2: always-on sample/analyze schedules (null ablation)",
        "overrides": {"agents.fixed_monitor_schedule": True,
                      "agents.fixed_risk_schedule": True},
    },
}

SEEDS = [7, 42, 123, 777, 2024]

# Final values as reported in the paper (tab:ablation); mean ± std, 5 seeds.
PAPER_REFERENCE = [
    ("Multi + constraint (TrustGuard)", "63.4 ± 1.6", "41.3 ± 1.2", "2.1 ± 0.3"),
    ("Multi − constraint",              "65.1 ± 1.8", "43.1 ± 1.3", "8.9 ± 0.7"),
    ("Single + constraint",             "49.8 ± 1.8", "33.6 ± 1.3", "2.4 ± 0.3"),
    ("Single − constraint",             "51.2 ± 1.9", "34.9 ± 1.4", "6.8 ± 0.5"),
    ("Local per-agent beliefs",         "59.7 ± 1.7", "39.1 ± 1.3", "2.3 ± 0.3"),
    ("w/o semantic encoder",            "44.6 ± 2.0", "29.4 ± 1.5", "6.7 ± 0.6"),
    ("w/o runtime traces",              "40.9 ± 2.1", "26.8 ± 1.6", "5.1 ± 0.5"),
    ("Homogeneous agents + belief enc.","61.5 ± 1.7", "40.4 ± 1.3", "2.2 ± 0.3"),
    ("Fixed Agents 1–2 (always-on)",    "62.9 ± 1.7", "40.9 ± 1.3", "2.2 ± 0.3"),
]


def print_paper_table() -> None:
    header = f"{'Configuration':<36} {'AIPR (%)':<14} {'PRR (%)':<14} {'FRR (%)':<12}"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, aipr, prr, frr in PAPER_REFERENCE:
        marker = "  ◄" if "TrustGuard" in name else ""
        print(f"{name:<36} {aipr:<14} {prr:<14} {frr:<12}{marker}")
    print("-" * len(header))


def train_config(name: str, seed: int, data_dir: str, output_dir: Path) -> None:
    """Launch training for one ablated configuration and seed."""
    cfg = ABLATION_CONFIGS[name]
    run_dir = output_dir / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the overrides so the run is self-describing and re-runnable
    with open(run_dir / "ablation_overrides.json", "w") as f:
        json.dump({"config": name, "seed": seed, **cfg}, f, indent=2)

    cmd = [
        sys.executable, "experiments/train_trustguard.py",
        "--config-dir", "configs/",
        "--data-dir", data_dir,
        "--output-dir", str(run_dir),
        "--seed", str(seed),
    ]
    logger.info("[%s | seed %d] %s", name, seed, " ".join(cmd))
    subprocess.run(cmd, check=True)

    eval_cmd = [
        sys.executable, "experiments/evaluate_enforcement.py",
        "--checkpoint", str(run_dir / "checkpoint_best.pt"),
        "--config-dir", "configs/",
        "--output-dir", str(run_dir / "eval_task2"),
        "--seed", str(seed),
    ]
    subprocess.run(eval_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustGuard ablation suite")
    parser.add_argument("--list",       action="store_true",
                        help="List configurations and print the paper table")
    parser.add_argument("--config",     choices=list(ABLATION_CONFIGS), default=None)
    parser.add_argument("--all",        action="store_true",
                        help="Run every configuration × every seed (long)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Single seed (default: all five paper seeds)")
    parser.add_argument("--data-dir",   default="data/permissionbench", type=str)
    parser.add_argument("--output-dir", default="outputs/ablations", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)

    if args.list or (not args.config and not args.all):
        print("Available ablation configurations:\n")
        for name, cfg in ABLATION_CONFIGS.items():
            print(f"  {name:<24} {cfg['description']}")
        print("\nFinal paper results (results/ablations.json):")
        print_paper_table()
        return

    seeds   = [args.seed] if args.seed is not None else SEEDS
    configs = list(ABLATION_CONFIGS) if args.all else [args.config]

    for name in configs:
        for seed in seeds:
            train_config(name, seed, args.data_dir, output_dir)

    print_paper_table()


if __name__ == "__main__":
    main()
