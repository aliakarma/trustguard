# TrustGuard Training Guide

This guide covers the two-phase training procedure, hyperparameter tuning,
common failure modes, and checkpoint management.

---

## Two-Phase Training

### Phase 1 — Supervised Pre-training

**Goal**: Train the Permission Prediction Model `gθ` on PermissionBench binary
cross-entropy labels so it can produce meaningful `p̂ᵢ,ₚ` values before
reinforcement learning begins.

**What is trained**: `PermissionPredictionModel` only. The semantic encoder
(`AppSemanticEncoder`) is frozen if `freeze_text=true` in `configs/model.yaml`
to save GPU memory.

**When to stop**: Early stopping with patience=5 on validation loss.
Typical convergence: 15–25 epochs.

**Tip**: If PermissionBench is unavailable, skip pre-training with
`--no-pretrain`. The MARL agents will still learn, but convergence will be
slower.

---

### Phase 2 — MARL Training (Constrained MAPPO)

**Goal**: Train the three Dec-POMDP agents to govern permissions continuously,
subject to the false-revocation safety constraint.

**Algorithm 1 (paper)**:

```
1.  Pre-train gθ on 𝒟 (Phase 1)
2.  Initialise {θᵏ}, V_ψ, μ = 0
3.  For each training iteration:
4.    Collect H-step rollout under {πᵏ} in simulation
5.    Compute returns R̂ₜ and advantages Âₜ using V_ψ
6.    Update {θᵏ} via PPO-clip with Âₜ
7.    Compute c̄ = empirical false-revocation rate
8.    μ ← max(0, μ + η_μ(c̄ − ε_safe))
9.    Update V_ψ: minimise ‖V_ψ(sₜ) − R̂ₜ‖²
```

**Typical wall-clock time**: ~4h on 1× A100 for 500 iterations.

---

## Hyperparameter Tuning

### Most important parameters

| Parameter | Location | Effect | Tuning guidance |
|-----------|----------|--------|-----------------|
| `eps_safe` | `marl.yaml` | FRR ceiling | Lower → safer, slower learning |
| `eps_clip` | `marl.yaml` | PPO clip | 0.1–0.3; 0.2 is standard |
| `risk_threshold` | `model.yaml` | Minimum EMA risk to act | Raise if FRR too high |
| `ema_alpha` | `model.yaml` | Risk signal smoothing | Raise for faster response |
| `entropy_coef` | `marl.yaml` | Exploration bonus | Raise if agents converge early |
| `gae_lambda` | `marl.yaml` | GAE bias/variance tradeoff | 0.9–0.99 |

### Lagrange multiplier μ

The multiplier μ controls how aggressively the false-revocation penalty is
applied. If μ grows unboundedly, reduce `lagrange_max` (default 10.0) or
increase `lr_lagrange` (default 1e-3).

Monitor μ in TensorBoard: it should stabilise near a finite value once the
policy learns to respect the constraint.

### Signs of instability

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| FRR > `eps_safe` after 200+ iters | μ too small or `lr_lagrange` too low | Increase `lr_lagrange` |
| PRR near zero | Risk threshold too high | Lower `risk_threshold` |
| Value loss diverges | Critic LR too high | Reduce `lr_critic` |
| Lagrange penalty dominates | `eps_safe` too tight | Raise `eps_safe` slightly |

---

## Checkpoint Management

Checkpoints are saved at:
- `checkpoint_best.pt`  — highest PRR seen during training
- `checkpoint_latest.pt` — most recent checkpoint (every `save_interval` iters)

Each checkpoint contains:
```python
{
    "iteration":            int,
    "global_step":          int,
    "monitoring_agent":     state_dict,
    "risk_agent":           state_dict,
    "enforcement_agent":    state_dict,
    "critic":               state_dict,
    "permission_predictor": state_dict,
    "belief_encoder":       state_dict,
}
```

To resume training:
```bash
python experiments/train_trustguard.py \
    --resume outputs/run_001/checkpoint_latest.pt \
    --output-dir outputs/run_001 \
    ...
```

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/run_001/tb
```

Key scalars to watch:
- `prr_pct`            — privacy risk reduction (want ↑)
- `frr`                — false revocation rate  (want < `eps_safe`)
- `lagrange_mu`        — multiplier μ           (want stable, finite)
- `constraint_violation` — max(0, FRR − ε_safe) (want → 0)
- `value_loss`         — critic loss            (want ↓)

### Weights & Biases
```bash
python experiments/train_trustguard.py ... --use-wandb
```

### Log file
Full logs are written to `outputs/run_001/train.log`.

---

## Ablation Experiments

To reproduce the ablation results from Table 3 of the paper:

**No Lagrangian constraint** (ε_safe = ∞):
```yaml
# configs/marl.yaml
lagrangian:
  eps_safe: 999.0
```

**No shared belief state**:
Comment out the `belief_encoder` update in `experiments/train_trustguard.py`
and pass a zero belief tensor to the enforcement agent.

**No semantic encoder** (random ϕ):
```yaml
# configs/model.yaml
semantic_encoder:
  freeze_text: true
  output_dim: 256
# Then in training script, replace encoder forward with:
# phi = torch.randn(B, 256)
```

**Single-agent RL** (no MARL):
Use the `SingleAgentRL` baseline in `experiments/evaluate_enforcement.py`.
