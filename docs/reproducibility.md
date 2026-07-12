# Reproducibility Guide

This document maps **every experiment and table in the TrustGuard paper** to
the script that reproduces it and the file holding the final reported
numbers. The paper's results are final; the reference values in `results/`
are the ground truth a reproduction run should be compared against.

## Global protocol

- **Seeds**: all learned methods use the five training seeds
  `{7, 42, 123, 777, 2024}`; every table cell is mean ± sample std across
  seeds. Use `scripts/run_all_seeds.sh` to run all five.
- **Task-2 simulation**: 72 hours, Δ = 60 s governance windows (4,320
  steps), N = 700 applications (500 benign / 200 malicious), identical
  budgets/episodes across methods.
- **Safety budget**: ε_safe = 0.025 on the ratio-form FRR (Eq. 3);
  configured in `configs/marl.yaml` (`lagrangian.eps_safe`).
- **Hardware used for the paper**: 4× A100 80 GB, ~96 h MARL training +
  ~6 h pre-training per seed. Evaluation-only reproduction is far cheaper;
  scripts accept scaled-down flags for smoke tests.

## Table-by-table mapping

| Paper table / section | Reference results | Reproduction command |
|---|---|---|
| Task 1 risk prediction (`tab:task1`) | `results/task1_prediction.json` | `python experiments/evaluate_prediction.py --checkpoint <ckpt> --data-dir data/permissionbench` |
| Task 2 enforcement (`tab:task2`, incl. FIR) | `results/task2_enforcement.json` | `python experiments/evaluate_enforcement.py --checkpoint <ckpt> --n-benign 500 --n-malicious 200 --max-steps 4320` |
| Stress tests (§5.3, AASE-B / 2% prevalence / recalibration) | `results/task2_stress_tests.json` | `python experiments/stress_tests.py --checkpoint <ckpt> --protocol all` |
| Factorial + component + null ablations (`tab:ablation`) | `results/ablations.json` | `python experiments/run_ablations.py --all` (9 configs × 5 seeds; long) |
| Adversarial robustness (`tab:adversarial`, MM/RTMA/composed) | `results/adversarial_robustness.json` | `python experiments/adversarial_evaluation.py --checkpoint <ckpt> --data-dir data/permissionbench` |
| Temporal hold-out (`tab:temporal`) | `results/temporal_holdout.json` | `python experiments/evaluate_temporal.py --checkpoint <ckpt> --data-dir data/permissionbench` |
| Threshold / λ / modality / EMA-α sensitivity (`tab:tau`, `tab:lambda`, `tab:modality`) | `results/sensitivity_analyses.json` | `python experiments/sensitivity_analysis.py --print-tables` (see script header for per-grid runs) |
| Constraint dynamics (`tab:curves`) + per-category FRR | `results/constraint_dynamics.json` | logged during `experiments/train_trustguard.py` (rolling FRR and μ per iteration) |
| DeLong tests + bootstrap CIs (appendix "Statistical Tests") | `results/statistical_tests.json` | `python experiments/statistical_tests.py delong/bootstrap ...` |
| Real-device pilot (§"Real-Device Pilot") | `results/pilot_summary.json` | **not re-runnable** — 14-day IRB-approved field study; traces were pseudonymized and deleted at study end |

## Baselines (B1–B9)

| ID | Method | Where |
|----|--------|-------|
| B1 | Android Static Policy | `experiments/evaluate_enforcement.py` (`StaticAndroidPolicy`) |
| B2–B4, B9 | DREBIN / MaMaDroid / DexRay / MaskDroid | retrained on PermissionBench from their released code; app-level scores propagated to pairs via `s(f_i)·(1−π₀(p\|cat))` (see `results/task1_prediction.json` protocol block) |
| B5 | Rule-Based Threshold (ϱ > 0.8) | `experiments/evaluate_enforcement.py` (`RuleBasedThreshold`) |
| B6 | Single-Agent RL | `experiments/run_ablations.py --config single_no_constraint` |
| B7 | Single-Agent PPO-Lagrangian | `experiments/run_ablations.py --config single_constraint` |
| B8 | MAPPO-Lagrangian (homogeneous) | `experiments/run_ablations.py --config homogeneous_belief` (without belief encoder: see script registry) |

Install-time methods (B1–B4, B9) cannot act at runtime; their Task-2 cells
are N/A by construction.

## Dataset (PermissionBench)

- **Download the processed release**: `bash scripts/download_permissionbench.sh`
  (feature vectors, per-permission labels, SHA-256 hashes, datasheet with
  per-split seed-corpus overlap counts).
- **Rebuild from source**: `python scripts/build_dataset.py --androzoo-key
  $ANDROZOO_API_KEY ...` — requires your own AndroZoo access agreement; raw
  APKs and Play metadata are not redistributed, so the text modality is
  reconstructible only with store access.
- Splits: 70/10/20 → train 43,288/10,158, val 6,184/1,451,
  test 12,368/2,903 (benign/malicious); 76,352 records total, content
  2012–2021.
- The externally grounded Task-1 subset (11,704 human-annotated or
  taint-verified pairs) is flagged by the `label_provenance` field in the
  release; `evaluate_prediction.py` reports "Ext. AUROC" on it.

## What a reviewer cannot reproduce and why

1. **Real-device pilot** — IRB-scoped human-subjects study; only the
   aggregate outcomes in `results/pilot_summary.json` are released.
2. **Raw APK/text retrieval** — gated by the AndroZoo agreement and Google
   Play terms; retrieval scripts are provided, redistribution is not.
3. **TaintDroid labeling** — requires Android 4.3-compatible emulator
   images; label outputs (including the taint-verified event pool used for
   EPR) ship with the dataset release.

## Known scaled-down defaults

For fast smoke tests, some defaults are smaller than the paper protocol
(e.g. `evaluate_enforcement.py` defaults to 60 apps / 1,000 steps;
`configs/training.yaml` defaults to 500 MARL iterations vs. the paper's
5,000). The paper-scale flags are documented in each script header and
above. Reported reference numbers always correspond to the paper protocol.
