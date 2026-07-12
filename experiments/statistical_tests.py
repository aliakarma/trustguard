"""
experiments/statistical_tests.py
==================================
Statistical significance tests (appendix "Statistical Tests" of the paper):

- **DeLong test** for paired AUROC comparison between TrustGuard and each
  baseline on the Task-1 test split.
- **Bootstrap 95% CIs** (1,000 resamples of the 72-hour window) for the
  Task-2 AIPR and FRR of the constrained methods.

Operates on the score arrays saved by evaluate_prediction.py
(risk_labels.npy, app_scores.npy) plus a second method's scores; the final
paper values are stored in results/statistical_tests.json and printed for
reference.

Usage
-----
    # DeLong: TrustGuard vs a baseline, from saved score arrays
    python experiments/statistical_tests.py delong \
        --labels  outputs/eval_task1/risk_labels.npy \
        --scores-a outputs/eval_task1/app_scores.npy \
        --scores-b outputs/eval_task1_dexray/app_scores.npy

    # Bootstrap CI over per-window AIPR values
    python experiments/statistical_tests.py bootstrap \
        --values outputs/eval_task2/aipr_windows.npy

    # Print the paper's reported test results
    python experiments/statistical_tests.py --print-paper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from trustguard.utils.metrics import delong_test, bootstrap_ci

logger = logging.getLogger("trustguard.stats")

RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "statistical_tests.json"


def print_paper_reference() -> None:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        ref = json.load(f)

    print("\n── DeLong tests (full-set Task-1 AUROC) ──")
    for pair, r in ref["delong_tests"].items():
        print(f"  {pair:<28} Δ = {r['delta_auroc']:+.3f}   p = {r['p_value']}")

    print("\n── Bootstrap 95% CIs (1,000 resamples of the 72-h window) ──")
    for method, r in ref["bootstrap_ci_95"].items():
        if method == "method":
            continue
        print(f"  {method:<20} AIPR {r['AIPR_pct']}   FRR {r['FRR_pct']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical tests")
    parser.add_argument("mode", nargs="?", choices=["delong", "bootstrap"],
                        default=None)
    parser.add_argument("--print-paper", action="store_true")
    parser.add_argument("--labels",   type=str, default=None)
    parser.add_argument("--scores-a", type=str, default=None)
    parser.add_argument("--scores-b", type=str, default=None)
    parser.add_argument("--values",   type=str, default=None)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.print_paper or args.mode is None:
        print_paper_reference()
        return

    if args.mode == "delong":
        if not (args.labels and args.scores_a and args.scores_b):
            parser.error("delong requires --labels, --scores-a, --scores-b")
        y  = np.load(args.labels)
        sa = np.load(args.scores_a)
        sb = np.load(args.scores_b)
        delta, p = delong_test(y, sa, sb)
        print(f"DeLong: ΔAUROC = {delta:+.4f}, p = {p:.4f}")
        print_paper_reference()

    elif args.mode == "bootstrap":
        if not args.values:
            parser.error("bootstrap requires --values")
        vals = np.load(args.values)
        lo, hi = bootstrap_ci(vals, n_resamples=args.n_resamples, seed=args.seed)
        print(f"Bootstrap 95% CI ({args.n_resamples} resamples): "
              f"[{lo:.2f}, {hi:.2f}]  (point est. {np.mean(vals):.2f})")
        print_paper_reference()


if __name__ == "__main__":
    main()
