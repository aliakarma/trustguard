#!/usr/bin/env bash
# scripts/run_all_seeds.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs the full TrustGuard pipeline for every training seed used in the
# paper ({7, 42, 123, 777, 2024}) so that per-seed means and standard
# deviations can be recomputed. Every table in the paper reports
# mean ± std over these five seeds.
#
# Usage:
#   bash scripts/run_all_seeds.sh [OUTPUT_BASE] [DATA_DIR]
#
# Example:
#   bash scripts/run_all_seeds.sh outputs/paper data/permissionbench

set -euo pipefail

OUTPUT_BASE=${1:-"outputs/paper"}
DATA_DIR=${2:-"data/permissionbench"}
SEEDS=(7 42 123 777 2024)

for SEED in "${SEEDS[@]}"; do
    echo "================================================================"
    echo "  Seed $SEED  →  $OUTPUT_BASE/seed_$SEED"
    echo "================================================================"
    bash scripts/run_full_experiment.sh "$OUTPUT_BASE/seed_$SEED" "$SEED" "$DATA_DIR"
done

echo ""
echo "All ${#SEEDS[@]} seeds complete. Aggregate with:"
echo "  python - <<'EOF'"
echo "  import json, glob, numpy as np"
echo "  runs = [json.load(open(p)) for p in glob.glob('$OUTPUT_BASE/seed_*/results_summary.json')]"
echo "  # per-seed summaries collected — compute mean/std per metric here"
echo "  EOF"
