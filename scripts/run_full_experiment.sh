#!/usr/bin/env bash
# scripts/run_full_experiment.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs the complete TrustGuard experimental pipeline:
#   1. Supervised pre-training of the permission prediction model
#   2. MARL training via Constrained MAPPO
#   3. Task 1: Permission risk prediction evaluation
#   4. Task 2: Autonomous enforcement evaluation (72h simulation)
#   5. Task 3: Adversarial robustness evaluation (mimicry attack)
#
# Usage:
#   bash scripts/run_full_experiment.sh [OUTPUT_DIR] [SEED] [DATA_DIR]
#
# Example:
#   bash scripts/run_full_experiment.sh outputs/run_001 42 data/permissionbench

set -euo pipefail

OUTPUT_DIR=${1:-"outputs/run_$(date +%Y%m%d_%H%M%S)"}
SEED=${2:-42}
DATA_DIR=${3:-"data/permissionbench"}
CONFIG_DIR="configs/"

echo "========================================================"
echo "  TrustGuard Full Experiment Pipeline"
echo "  Output:  $OUTPUT_DIR"
echo "  Seed:    $SEED"
echo "  Data:    $DATA_DIR"
echo "========================================================"

mkdir -p "$OUTPUT_DIR"

# ── Phase 1 + 2: Training ─────────────────────────────────────────────────────
echo ""
echo ">>> [1/5] Training TrustGuard (supervised pre-train + MARL) ..."
python experiments/train_trustguard.py \
    --config-dir "$CONFIG_DIR" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --seed       "$SEED"

CKPT="$OUTPUT_DIR/checkpoint_best.pt"

# ── Task 1: Permission risk prediction ───────────────────────────────────────
echo ""
echo ">>> [2/5] Task 1: Permission risk prediction evaluation ..."
python experiments/evaluate_prediction.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR/eval_task1" \
    --seed       "$SEED"

# ── Task 2: Autonomous enforcement ───────────────────────────────────────────
echo ""
echo ">>> [3/5] Task 2: Autonomous enforcement evaluation ..."
python experiments/evaluate_enforcement.py \
    --checkpoint "$CKPT" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR/eval_task2" \
    --n-episodes 10 \
    --seed       "$SEED"

# ── Task 3: Adversarial robustness ───────────────────────────────────────────
echo ""
echo ">>> [4/5] Task 3: Adversarial robustness (mimicry attack) ..."
python experiments/adversarial_evaluation.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR/eval_task3" \
    --seed       "$SEED"

# ── Collect results ───────────────────────────────────────────────────────────
echo ""
echo ">>> [5/5] Aggregating results ..."
python - <<'PYEOF'
import json, pathlib, sys

base = pathlib.Path("$OUTPUT_DIR")
results = {}

for task_dir in ["eval_task1", "eval_task2", "eval_task3"]:
    p = base / task_dir
    for jf in p.glob("*.json"):
        with open(jf) as f:
            results[task_dir] = json.load(f)
        break

out = base / "results_summary.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results summary saved to {out}")
for k, v in results.items():
    print(f"  {k}: {v}")
PYEOF

echo ""
echo "========================================================"
echo "  Pipeline complete. Results in $OUTPUT_DIR"
echo "========================================================"
