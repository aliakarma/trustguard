# TrustGuard 🛡️

**A Multi-Agent Reinforcement Learning Framework for Autonomous Permission Governance in Mobile Ecosystems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Official implementation of the TrustGuard paper.  
> *Akarma, A., Jan, S., & Syed, T. A. (2026). TrustGuard: A Multi-Agent Reinforcement Learning Framework for Autonomous Permission Governance in Mobile Ecosystems.*

---

## Overview

Mobile permission systems rely on **static policies and uninformed user prompts** that cannot reason about application behaviour at runtime. TrustGuard replaces this with a **continuous, learning-based governance loop** formalised as a Decentralised Partially Observable Markov Decision Process (Dec-POMDP).

Three cooperative agents — **Monitoring**, **Risk-Analysis**, and **Enforcement** — are trained via Centralised Training / Decentralised Execution (CTDE) using MAPPO with a **Lagrangian safety constraint** that bounds the false-revocation rate.

### Key Results

| Metric | TrustGuard | Best Baseline |
|--------|-----------|---------------|
| Permission Risk AUROC | **0.963** | 0.921 (MaMaDroid) |
| Privacy Risk Reduction | **41.3%** | 34.9% (Single-Agent RL) |
| False Revocation Rate | **2.1%** | 6.8% (Single-Agent RL) |
| Enforcement Latency | **1.9 s** | 2.8 s |
| AUROC under Mimicry Attack | **0.891** | 0.739 (MaMaDroid) |

---

## Architecture

```
App Metadata ──► App Semantic Encoder (BERT + GATv2 + CodeBERT) ──► ϕ(fᵢ) ∈ ℝ²⁵⁶
                                                                          │
                                              Permission Prediction Model ◄─┘
                                                  gθ: ℝ²⁵⁶ → [0,1]^|𝒫|
                                                          │
Runtime Traces ──────────────────────────────► Runtime Risk Estimator
                                                  ρᵢᵗ (EMA-smoothed)
                                                          │
                            ┌─────────────────────────────┤
                            ▼         ▼         ▼
                      Monitoring   Risk       Enforcement
                       Agent(k=1) Agent(k=2) Agent(k=3)
                            └─────────────────────────────┘
                                          │
                                  Shared Belief bₜ
                                  (GRU Encoder f_ψ)
                                          │
                              Enforcement Action ∈ {no_op,
                               alert, rate_limit, revoke}
```

The system is trained end-to-end via **Constrained MAPPO**:

```
ℒ(θ, μ) = 𝔼[Σ γᵗ rₜ] − μ · (𝔼[false_revocations] − ε_safe)
```

---

## Repository Structure

```
trustguard/
├── trustguard/                  # Main package
│   ├── agents/                  # Three Dec-POMDP agents + policy networks
│   │   ├── monitoring_agent.py
│   │   ├── risk_analysis_agent.py
│   │   ├── enforcement_agent.py
│   │   └── policy_networks.py
│   ├── models/                  # Four-layer model stack
│   │   ├── semantic_encoder.py      # Layer 1: BERT + GATv2 + CodeBERT
│   │   ├── permission_predictor.py  # Layer 2: multi-label MLP
│   │   ├── runtime_risk_estimator.py # Layer 3: EMA risk tracker
│   │   └── belief_encoder.py        # GRU-based shared belief state
│   ├── marl/                    # MAPPO training infrastructure
│   │   ├── mappo.py             # Constrained MAPPO trainer
│   │   ├── rollout_buffer.py    # On-policy experience buffer
│   │   └── centralized_critic.py
│   ├── environment/             # Simulation environment
│   │   ├── permission_env.py    # Dec-POMDP environment
│   │   ├── app_simulator.py     # Benign + malicious app behaviour
│   │   └── observation_builder.py
│   ├── dataset/                 # PermissionBench utilities
│   │   ├── permissionbench_loader.py
│   │   ├── dataset_builder.py
│   │   └── preprocessing.py
│   └── utils/
│       ├── metrics.py           # PRR, FRR, AUROC, F1, ...
│       ├── logging_utils.py     # W&B + TensorBoard
│       └── config_utils.py
├── experiments/                 # Runnable experiment scripts
│   ├── train_trustguard.py      # Main training script
│   ├── evaluate_prediction.py   # Task 1: permission risk prediction
│   ├── evaluate_enforcement.py  # Task 2: autonomous enforcement
│   └── adversarial_evaluation.py # Task 3: mimicry attack
├── configs/                     # YAML configuration files
│   ├── model.yaml
│   ├── marl.yaml
│   ├── training.yaml
│   └── dataset.yaml
├── scripts/
│   ├── build_dataset.py
│   └── run_full_experiment.sh
├── tests/                       # pytest test suite
├── docs/                        # Extended documentation
└── notebooks/
    └── trustguard_demo.ipynb
```

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/aliakarma/trustguard.git
cd trustguard

conda create -n trustguard python=3.10 -y
conda activate trustguard
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### 4. Install TrustGuard

```bash
pip install -e ".[dev]"
```

---

## Dataset Preparation

### Option A: Download pre-built PermissionBench

```bash
bash scripts/download_permissionbench.sh
```

This downloads the pre-processed dataset (~2 GB) to `data/permissionbench/`.

### Option B: Build from source

```bash
# Requires AndroZoo API key — set ANDROZOO_API_KEY env variable
python scripts/build_dataset.py \
    --androzoo-key $ANDROZOO_API_KEY \
    --output-dir   data/permissionbench \
    --n-benign     61840 \
    --n-malicious  14512
```

---

## Training

### Full pipeline (Phase 1 + Phase 2)

```bash
python experiments/train_trustguard.py \
    --config-dir configs/ \
    --data-dir   data/permissionbench \
    --output-dir outputs/run_001 \
    --seed 42
```

### Phase 2 only (skip supervised pre-training)

```bash
python experiments/train_trustguard.py \
    --config-dir configs/ \
    --output-dir outputs/run_001 \
    --no-pretrain
```

### With Weights & Biases tracking

```bash
python experiments/train_trustguard.py ... --use-wandb
```

### Resume from checkpoint

```bash
python experiments/train_trustguard.py \
    --resume outputs/run_001/checkpoint_latest.pt ...
```

---

## Evaluation

### Task 1 — Permission Risk Prediction

```bash
python experiments/evaluate_prediction.py \
    --checkpoint outputs/run_001/checkpoint_best.pt \
    --data-dir   data/permissionbench \
    --output-dir outputs/eval_task1
```

Expected output:
```
Accuracy=0.951 | Macro-F1=0.939 | AUROC=0.963 | AP=0.941
```

### Task 2 — Autonomous Enforcement (72h simulation)

```bash
python experiments/evaluate_enforcement.py \
    --checkpoint outputs/run_001/checkpoint_best.pt \
    --output-dir outputs/eval_task2 \
    --n-episodes 10
```

Expected output:
```
PRR=41.3% | FRR=0.0210 | Latency=1.90s
```

### Task 3 — Adversarial Robustness (Mimicry Attack)

```bash
python experiments/adversarial_evaluation.py \
    --checkpoint outputs/run_001/checkpoint_best.pt \
    --data-dir   data/permissionbench \
    --output-dir outputs/eval_task3
```

Expected output:
```
AUROC (clean)=0.9630 | AUROC (attack)=0.8910 | Δ=-0.0720
```

### Run all experiments

```bash
bash scripts/run_full_experiment.sh outputs/run_001
```

---

## Running Tests

```bash
# Fast unit tests (no GPU, no data download required)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=trustguard --cov-report=html
```

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`.

Key parameters:

| File | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `marl.yaml` | `lagrangian.eps_safe` | `0.025` | Max false-revocation rate ε_safe |
| `marl.yaml` | `mappo.eps_clip` | `0.2` | PPO clip coefficient |
| `marl.yaml` | `mappo.gae_lambda` | `0.95` | GAE λ |
| `model.yaml` | `semantic_encoder.output_dim` | `256` | ϕ(fᵢ) dimension |
| `model.yaml` | `enforcement_agent.risk_threshold` | `0.5` | Minimum EMA risk for non-no_op |
| `training.yaml` | `training.marl_iterations` | `500` | Total MARL iterations |

---

## PermissionBench Dataset

PermissionBench is the first large-scale benchmark for mobile permission risk analysis with longitudinal runtime traces.

| Split | Benign | Malicious | Total |
|-------|--------|-----------|-------|
| Train (70%) | 43,288 | 10,158 | 53,446 |
| Val (10%) | 6,184 | 1,451 | 7,635 |
| Test (20%) | 12,368 | 2,903 | 15,271 |
| **Total** | **61,840** | **14,512** | **76,352** |

Each record contains: app ID, category, description, declared permissions, API call features, binary risk label, per-permission risk labels, and runtime permission traces.

**Download**: [github.com/aliakarma/PermissionBench](https://github.com/aliakarma/PermissionBench)  
**License**: CC-BY-4.0

---


## License

This project is released under the [MIT License](LICENSE).

---

## Contact

**Ali Akarma** — 443059463@stu.iu.edu.sa  
Islamic University of Madinah, Department of Information Technology
