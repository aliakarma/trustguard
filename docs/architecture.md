# TrustGuard Architecture

This document provides a detailed description of the TrustGuard system
architecture, expanding on §5 of the paper.

---

## Overview

TrustGuard is structured as four sequential layers:

```
Layer 1   App Semantic Encoder          ϕ(fᵢ) ∈ ℝ²⁵⁶
Layer 2   Permission Prediction Model   p̂ᵢ,ₚ = P(p ∈ 𝒫ᵢ* | ϕ(fᵢ))
Layer 3   Runtime Risk Estimator        ρᵢᵗ (EMA-smoothed)
Layer 4   Multi-Agent Governance        Dec-POMDP agents
```

Layers 1–2 are trained supervised on PermissionBench. Layers 3–4 are
deployed continuously and trained via MARL.

---

## Layer 1: App Semantic Encoder

### Purpose
Map an application's available metadata to a fixed-dimensional embedding
ϕ(fᵢ) ∈ ℝ²⁵⁶ that captures functional intent.

### Architecture

```
Description text ──► BERT (bert-base-uncased) ──► CLS ──► Linear(768→768)
                                                              │
API call names   ──► CodeBERT ──────────────────► CLS ──► Linear(768→768)
                                                              │
API call graph   ──► GATv2Conv (×2) ──► Global Pool ──► Linear(768→768)
                                                              │
              ─────────────────────────────────────────────────
                                Concat (2304-dim)
                                Linear(2304→512) → GELU → Dropout
                                Linear(512→256)  → LayerNorm → L2-Norm
                                          ϕ(fᵢ) ∈ ℝ²⁵⁶
```

**GATv2 details:**
- Layer 1: in_channels → 256, heads=4, concat=True  → 1024-dim
- Layer 2: 1024 → 256, heads=4, concat=False → 256-dim
- Pooling: global_mean_pool ‖ global_max_pool → 512-dim → Linear(512→768)

### Design choices
- L2 normalisation of ϕ enables cosine-similarity comparisons between apps.
- `freeze_text=True` during MARL training to prevent catastrophic forgetting.
- GATv2 outperforms GATv1 on heterogeneous API call graphs (Brody et al., 2022).

---

## Layer 2: Permission Prediction Model

### Purpose
For each application, predict the probability that each permission p is
legitimately warranted by the application's functionality.

### Formulation
```
gθ: ℝ²⁵⁶ → [0,1]^|𝒫|

p̂ᵢ,ₚ = σ(MLP(ϕ(fᵢ)))_p

ℒ_supervised = BCE_smoothed(ŷ, y, ε=0.1)
```

### Architecture
```
ϕ (256) → Linear(512) → LayerNorm → GELU → Dropout(0.1)
        → Linear(256) → LayerNorm → GELU → Dropout(0.1)
        → Linear(|𝒫|)   [logits; sigmoid applied at inference]
```

### Label smoothing
Targets are smoothed to ỹ = y · (1−ε) + ε/2 before BCE, preventing
overconfident predictions that hurt calibration.

---

## Layer 3: Runtime Risk Estimator

### Instantaneous risk
```
ρᵢᵗ = (1 / |𝒫ᵢᵗ|) Σ_{p ∈ 𝒫ᵢᵗ} |uᵢ,ₚᵗ − p̂ᵢ,ₚ|
```
where 𝒫ᵢᵗ = {p : uᵢ,ₚᵗ > 0} is the set of actively invoked permissions.

### EMA smoothing
```
ρ̄ᵢᵗ = α · ρᵢᵗ + (1−α) · ρ̄ᵢᵗ⁻¹    (α = 0.3)
```
Suppresses noise from single-use legitimate permission accesses (e.g., a
camera app opening the camera once per session).

### Composite per-permission risk
For targeted enforcement decisions:
```
ϱ_composite(p, fᵢ) = |uᵢ,ₚᵗ − p̂ᵢ,ₚ| · (1 − p̂ᵢ,ₚ)
```
This down-weights deviations for permissions that are semantically expected
(high p̂), reducing false revocations on ambiguous apps.

---

## Layer 4: Multi-Agent Governance

### Dec-POMDP structure

| Agent | k | Observation | Action |
|-------|---|-------------|--------|
| Monitoring | 1 | Raw usage counts, system load | {IDLE, SAMPLE} |
| Risk-Analysis | 2 | Usage delta, predicted probs, EMA risks | {DEFER, ANALYSE} |
| Enforcement | 3 | EMA risks, belief state, enforcement history | {no_op, alert, rate_limit, revoke} |

### Shared belief state
```
bₜ = f_ψ(bₜ₋₁, o¹ₜ, o²ₜ, o³ₜ)
```
GRU-based encoder with orthogonal weight initialisation. Each agent's
observation is independently projected to a common embed_dim=128 before
concatenation as GRU input.

### CTDE training
- **Training**: Centralised critic V_ψ(s) with full global state access.
- **Execution**: Each agent uses only its local observation o^k.
- **Advantage**: GAE-λ (λ=0.95, γ=0.99).
- **Policy update**: PPO-clip (ε=0.2), 10 epochs per rollout.

### Lagrangian safety constraint
```
ℒ(θ, μ) = 𝔼[Σ γᵗ rₜ] − μ · max(0, c̄ − ε_safe)

μ ← max(0, μ + η_μ · (c̄ − ε_safe))    η_μ = 1e-3
```
`ε_safe = 0.025` bounds the false-revocation rate at 2.5% over any rollout.

---

## Reward Function

```
rₜ = Σᵢ (ρ̄ᵢᵗ⁻¹ − ρ̄ᵢᵗ)           ← risk reduction (λ₀ = 1.0)
   − λ₁ · Σᵢ 𝟙[false_revoke_i]   ← λ₁ = 5.0
   − λ₂ · C_enforcement           ← λ₂ = 0.01
```

The high λ₁ penalty ensures that false revocations are strongly discouraged
even before the Lagrangian constraint kicks in, providing a smooth optimisation
landscape.

---

## Enforcement Action Space

| Action | Target | Effect |
|--------|--------|--------|
| `no_op` | — | No change |
| `alert` | App | Display user notification |
| `rate_limit` | App + Permissions | Halve invocation probability |
| `revoke` | App + Permissions | Block all future invocations |

`rate_limit` and `revoke` are only applied when `EMA_risk > risk_threshold`
(default 0.5), implemented as a risk-gate in `EnforcementHead.select_action`.
