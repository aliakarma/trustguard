# TrustGuard Dashboard — Comprehensive Design & Implementation Guide

> **Scope**: This document specifies every screen, every component, every interactive feature, every data binding, and every aesthetic decision required to build a publication-ready, AAAI-grade interactive dashboard for the TrustGuard project. After following this guide the final product must be: (1) visually stunning enough for a live demo in front of 500+ researchers, (2) fully bilingual (English LTR / Arabic RTL), (3) switchable between Dark and Light themes, (4) interactive — allowing a user to configure inputs, run live simulations, inspect agent internals, and explore all paper results.

---

## Table of Contents

1. [Design Philosophy & Visual Identity](#1-design-philosophy--visual-identity)
2. [Technology Stack & Architecture](#2-technology-stack--architecture)
3. [Bilingual Support: English & Arabic](#3-bilingual-support-english--arabic)
4. [Dynamic Theming: Dark Mode & Light Mode](#4-dynamic-theming-dark-mode--light-mode)
5. [Global Navigation & App Shell](#5-global-navigation--app-shell)
6. [Screen 1 — Command Center (Overview)](#6-screen-1--command-center-overview)
7. [Screen 2 — Live Simulation Playground](#7-screen-2--live-simulation-playground)
8. [Screen 3 — Agent Inspector (Dec-POMDP Deep Dive)](#8-screen-3--agent-inspector-dec-pomdp-deep-dive)
9. [Screen 4 — Semantic Encoder Visualizer](#9-screen-4--semantic-encoder-visualizer)
10. [Screen 5 — Results Explorer (Paper Tables & Charts)](#10-screen-5--results-explorer-paper-tables--charts)
11. [Screen 6 — Adversarial & Robustness Lab](#11-screen-6--adversarial--robustness-lab)
12. [Screen 7 — Sensitivity & Ablation Studio](#12-screen-7--sensitivity--ablation-studio)
13. [Screen 8 — Training Monitor](#13-screen-8--training-monitor)
14. [Screen 9 — Real-Device Pilot Report](#14-screen-9--real-device-pilot-report)
15. [Screen 10 — Dataset & PermissionBench Explorer](#15-screen-10--dataset--permissionbench-explorer)
16. [Reusable Component Library](#16-reusable-component-library)
17. [Micro-Animations & Motion Design](#17-micro-animations--motion-design)
18. [Backend API Specification (FastAPI)](#18-backend-api-specification-fastapi)
19. [Project File Structure](#19-project-file-structure)
20. [Implementation Roadmap](#20-implementation-roadmap)
21. [Quality Assurance & Deployment Checklist](#21-quality-assurance--deployment-checklist)

---

## 1. Design Philosophy & Visual Identity

### 1.1 Core Principles

| Principle | What It Means for This Dashboard |
|-----------|----------------------------------|
| **Academic Gravitas** | Every number must trace to a paper table/equation. Tooltips cite equation numbers (e.g., "Eq. 3 — §5.3"). |
| **Narrative Flow** | Screens are ordered to tell the paper's story: problem → architecture → results → stress tests → limitations. |
| **Interactive Inquiry** | Users must be able to ask "what if?" — change hyperparameters, inject attacks, toggle ablation components — and see the effect live. |
| **Zero Placeholders** | Every image, icon, and illustration must be a real generated asset or an actual visualization of data. No stock images, no Lorem Ipsum. |
| **Print-Ready** | Light-mode charts must be high-contrast enough to print in grayscale for supplementary material. |

### 1.2 Color Palette

#### Dark Mode (Default)

| Role | Hex | HSL | Usage |
|------|-----|-----|-------|
| Background Primary | `#0A0F1A` | `220, 38%, 7%` | Page background |
| Background Panel | `#111827` | `220, 33%, 11%` | Cards, panels |
| Surface Glass | `rgba(30, 41, 59, 0.65)` | — | Glassmorphism panels (with `backdrop-filter: blur(16px)`) |
| Border Subtle | `rgba(148, 163, 184, 0.12)` | — | Card borders |
| Text Primary | `#F1F5F9` | `210, 40%, 96%` | Headings, values |
| Text Secondary | `#94A3B8` | `215, 17%, 65%` | Labels, captions |
| Accent — Monitoring (Agent 1) | `#38BDF8` | `198, 93%, 60%` | All Agent-1 visuals |
| Accent — Risk (Agent 2) | `#FBBF24` | `45, 97%, 56%` | All Agent-2 visuals |
| Accent — Enforcement (Agent 3) | `#F43F5E` | `347, 90%, 60%` | All Agent-3 visuals |
| Safe / Benign | `#10B981` | `160, 84%, 39%` | Safe status, benign apps |
| Danger / Malicious | `#EF4444` | `0, 84%, 60%` | Danger status, malicious apps |
| Constraint Budget Line | `#A78BFA` | `263, 93%, 76%` | ε_safe = 2.5% threshold |

#### Light Mode

| Role | Hex |
|------|-----|
| Background Primary | `#F8FAFC` |
| Background Panel | `#FFFFFF` |
| Surface Glass | `rgba(255, 255, 255, 0.85)` with `box-shadow: 0 4px 24px rgba(0,0,0,0.06)` |
| Text Primary | `#0F172A` |
| Text Secondary | `#475569` |
| Accents | Same hues, saturation reduced by 10% |

### 1.3 Typography

| Context | English Font | Arabic Font | Weight | Size |
|---------|-------------|-------------|--------|------|
| Page Title | **Inter** (Google Fonts) | **Cairo** (Google Fonts) | 700 | 28–32px |
| Section Heading | Inter | Cairo | 600 | 20–24px |
| Card Label | Inter | Cairo | 500 | 13–14px |
| Stat Value | **JetBrains Mono** | **IBM Plex Arabic** | 700 | 36–48px |
| Body Text | Inter | Tajawal | 400 | 14–16px |
| Code / Equations | JetBrains Mono | JetBrains Mono | 400 | 13px |
| Tooltip | Inter | Tajawal | 400 | 12px |

### 1.4 Glassmorphism Specification

Every floating card and panel must apply:

```css
.glass-panel {
  background: var(--surface-glass);
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid var(--border-subtle);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}
```

---

## 2. Technology Stack & Architecture

### 2.1 Frontend

| Layer | Technology | Reason |
|-------|-----------|--------|
| Framework | **Next.js 14+ (App Router)** | SSR for SEO, file-based routing, API routes |
| Styling | **Vanilla CSS + CSS Modules** | Full control over glassmorphism, RTL/LTR, micro-animations |
| State Management | **Zustand** | Lightweight, TypeScript-native, minimal boilerplate |
| Data Visualization | **D3.js v7** | Fully custom animated SVG charts (no library look-and-feel) |
| Animation | **Framer Motion** | Physics-based spring animations, layout transitions, gesture support |
| Real-Time | **Socket.io client** | Bi-directional WebSocket communication with FastAPI backend |
| i18n | **next-intl** | ICU message format, server components support, direction-aware |
| Theming | **next-themes** | Flash-free SSR-compatible dark/light mode |
| Math Rendering | **KaTeX** | Inline equation rendering (e.g., ρᵢᵗ, ℒ(θ,μ)) in tooltips and headers |
| Icons | **Lucide React** | MIT-licensed, tree-shakeable, consistent stroke icons |

### 2.2 Backend (Python Bridge)

| Layer | Technology | Reason |
|-------|-----------|--------|
| API Server | **FastAPI** | Async, auto-docs, pydantic validation |
| WebSocket Streaming | **FastAPI WebSocket + asyncio** | Stream simulation steps at configurable tick rate |
| ML Inference | **PyTorch 2.1** (existing TrustGuard stack) | Load checkpoints, run forward passes |
| Serialization | **orjson** | 10× faster than stdlib json for large tensors |
| CORS | **fastapi.middleware.cors** | Allow dashboard origin |

### 2.3 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend (Browser)                     │
│  ┌────────────┐  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │ Zustand    │  │ Socket.io      │  │ D3.js + Framer Motion       │ │
│  │ Global     │←→│ Client         │  │ Visualization Layer         │ │
│  │ Store      │  │ (real-time)    │  │                             │ │
│  └────────────┘  └───────┬────────┘  └─────────────────────────────┘ │
└──────────────────────────┼───────────────────────────────────────────┘
                           │ WebSocket (ws://)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Python)                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ REST         │  │ WebSocket        │  │ Simulation Runner      │  │
│  │ /api/results │  │ /ws/simulation   │  │ (PermissionEnv +       │  │
│  │ /api/config  │  │ /ws/training     │  │  3 Agents + Critic)    │  │
│  └──────────────┘  └──────────────────┘  └────────────────────────┘  │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ TrustGuard Python Package                                      │   │
│  │ trustguard/agents  trustguard/models  trustguard/environment   │   │
│  │ trustguard/marl    trustguard/utils   trustguard/dataset       │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Bilingual Support: English & Arabic

### 3.1 RTL Layout Strategy

All CSS must use **logical properties** exclusively:

| ❌ Physical | ✅ Logical |
|-------------|-----------|
| `margin-left` | `margin-inline-start` |
| `padding-right` | `padding-inline-end` |
| `text-align: left` | `text-align: start` |
| `float: left` | `float: inline-start` |
| `border-left` | `border-inline-start` |

The HTML root must dynamically set `dir="rtl"` and `lang="ar"` when Arabic is active.

### 3.2 Translation File Structure

```
messages/
├── en.json       ← English translations
└── ar.json       ← Arabic translations
```

**Sample translation keys** (both files must cover every key):

```json
{
  "nav.command_center": "Command Center",
  "nav.simulation": "Live Simulation",
  "nav.agent_inspector": "Agent Inspector",
  "nav.results": "Results Explorer",
  "nav.adversarial": "Adversarial Lab",
  "nav.sensitivity": "Sensitivity Studio",
  "nav.training": "Training Monitor",
  "nav.pilot": "Real-Device Pilot",
  "nav.dataset": "PermissionBench",

  "metrics.aipr": "Anomalous Invocation Prevention Rate",
  "metrics.epr": "Exfiltration Prevention Rate",
  "metrics.frr": "False Revocation Rate",
  "metrics.fir": "False Intervention Rate",
  "metrics.prr": "Privacy Risk Reduction",
  "metrics.auroc": "Area Under ROC",
  "metrics.macro_f1": "Macro F1-Score",
  "metrics.latency": "Enforcement Latency",

  "agents.monitoring": "Monitoring Agent (k=1)",
  "agents.risk": "Risk-Analysis Agent (k=2)",
  "agents.enforcement": "Enforcement Agent (k=3)",

  "actions.no_op": "No Action",
  "actions.alert": "Alert",
  "actions.rate_limit": "Rate Limit",
  "actions.revoke": "Revoke",
  "actions.sample": "Sample",
  "actions.idle": "Idle",
  "actions.analyse": "Analyse",
  "actions.defer": "Defer",

  "terms.belief_state": "Shared Belief State bₜ",
  "terms.ema_risk": "EMA-Smoothed Risk ρ̄ᵢᵗ",
  "terms.lagrangian": "Lagrangian Multiplier μ",
  "terms.eps_safe": "Safety Budget ε_safe",
  "terms.dec_pomdp": "Dec-POMDP",
  "terms.ctde": "Centralized Training / Decentralized Execution"
}
```

Arabic translations must be prepared by a domain expert. Technical RL terms (e.g., "MAPPO", "PPO-clip", "GAE-λ") stay in English even in Arabic mode but are wrapped in `<bdi>` tags for proper bidirectional isolation.

### 3.3 Chart RTL Behavior

- **Time-series charts**: X-axis origin remains at the left (time flows left-to-right universally).
- **Bar charts**: Bar labels and legend text flip to RTL.
- **Radial gauges**: No directional change needed.
- **Tables**: Column order reverses; header text right-aligns.

---

## 4. Dynamic Theming: Dark Mode & Light Mode

### 4.1 CSS Variable Architecture

All colors are defined as CSS custom properties on `:root` and overridden by `[data-theme="light"]`:

```css
:root {
  /* Dark Mode (default) */
  --bg-primary: #0A0F1A;
  --bg-panel: #111827;
  --surface-glass: rgba(30, 41, 59, 0.65);
  --border-subtle: rgba(148, 163, 184, 0.12);
  --text-primary: #F1F5F9;
  --text-secondary: #94A3B8;
  --accent-monitor: #38BDF8;
  --accent-risk: #FBBF24;
  --accent-enforce: #F43F5E;
  --accent-safe: #10B981;
  --accent-danger: #EF4444;
  --accent-constraint: #A78BFA;
  --chart-grid: rgba(148, 163, 184, 0.08);
  --glow-monitor: 0 0 20px rgba(56, 189, 248, 0.3);
  --glow-risk: 0 0 20px rgba(251, 191, 36, 0.3);
  --glow-enforce: 0 0 20px rgba(244, 63, 94, 0.3);
}

[data-theme="light"] {
  --bg-primary: #F8FAFC;
  --bg-panel: #FFFFFF;
  --surface-glass: rgba(255, 255, 255, 0.85);
  --border-subtle: rgba(0, 0, 0, 0.06);
  --text-primary: #0F172A;
  --text-secondary: #475569;
  --chart-grid: rgba(0, 0, 0, 0.06);
  --glow-monitor: none;
  --glow-risk: none;
  --glow-enforce: none;
}
```

### 4.2 Theme Toggle Component

A pill-shaped toggle in the header bar with sun/moon icons. On click, it triggers a smooth 300ms CSS transition on `background-color`, `color`, and `border-color` for all elements using CSS variables. Use Framer Motion's `layout` prop on the toggle knob for physics-based sliding.

---

## 5. Global Navigation & App Shell

### 5.1 Sidebar Navigation

A collapsible sidebar (280px expanded, 64px collapsed) containing:

| Icon | Label | Route |
|------|-------|-------|
| 🎯 | Command Center | `/` |
| ▶️ | Live Simulation | `/simulation` |
| 🤖 | Agent Inspector | `/agents` |
| 🧠 | Semantic Encoder | `/encoder` |
| 📊 | Results Explorer | `/results` |
| 🛡️ | Adversarial Lab | `/adversarial` |
| 🔬 | Sensitivity Studio | `/sensitivity` |
| 📈 | Training Monitor | `/training` |
| 📱 | Real-Device Pilot | `/pilot` |
| 🗂️ | PermissionBench | `/dataset` |

At the bottom of the sidebar:
- **Language Toggle**: 🌐 EN / AR switch
- **Theme Toggle**: ☀️ / 🌙 switch
- **GitHub Link**: External link to repository

### 5.2 Header Bar

A sticky top bar (height: 56px) with:
- **Page Title**: Current screen name (localized)
- **Breadcrumb**: Home → Current Screen → Sub-section
- **Live Clock**: Current simulation time (when simulation is running)
- **Connection Status**: Green dot + "Connected" / Red dot + "Disconnected" for WebSocket

---

## 6. Screen 1 — Command Center (Overview)

**Purpose**: Provide a single-glance summary of TrustGuard's capabilities and headline paper results. This is the "hero" screen that must immediately impress.

### 6.1 Hero Metrics Row (Top)

Five glass cards in a horizontal row, each showing:

| Card | Value | Source |
|------|-------|--------|
| **AIPR** | `63.4% ± 1.6` | `results/task2_enforcement.json` → TrustGuard (ours) |
| **EPR** | `71.8% ± 2.5` | Same file |
| **FRR** | `2.1% ± 0.3` | Same file |
| **Macro-F1** | `0.939 ± 0.004` | `results/task1_prediction.json` → TrustGuard |
| **Latency** | `1.9s` | `results/task2_enforcement.json` → TrustGuard |

Each card has:
- A small colored bar at the top matching the metric's semantic meaning (green for AIPR/EPR, amber for FRR staying below budget, blue for F1, etc.)
- The number animates up from 0 using a counter animation (500ms ease-out)
- A tooltip showing the full metric name, equation reference, and what the number means
- The ± std is rendered in smaller `text-secondary` font beside the main value

### 6.2 FRR Safety Gauge

A large circular gauge (200×200px) showing:
- Current FRR (2.1%) as a green arc
- Safety budget ε_safe (2.5%) as a dashed purple line
- The gap (0.4pp) labeled "Within Budget ✓"
- Below the gauge: a KaTeX-rendered equation: `FRR(π) = E[Σ c^fr_t] / max(1, E[Σ c^rv_t]) ≤ ε_safe = 0.025` (from Eq. 2 of the paper)

### 6.3 Architecture Diagram (Interactive)

A stylized, animated version of Figure 1 from the paper (the TikZ architecture diagram), built in SVG with D3:

- Each module (Semantic Encoder, Permission Predictor, Risk Estimator, 3 Agents, Belief Encoder) is a clickable node
- Data flow arrows animate with traveling dots
- Clicking a node navigates to the corresponding detail screen (e.g., clicking Enforcement Agent goes to `/agents#enforcement`)
- On hover, nodes glow with their agent accent color

### 6.4 System Health Grid

A 2×3 grid of mini-cards showing:
- **Apps Monitored**: "700 (500B / 200M)" with benign/malicious icon split
- **Permissions Tracked**: "42 dangerous permissions (API 34)"
- **Episode Length**: "72h (4,320 steps × Δ=60s)"
- **Seeds**: "5 seeds: {7, 42, 123, 777, 2024}"
- **Training Compute**: "4× A100 · ~96h MARL + 6h pre-training"
- **Model Size**: "~220M params (Semantic Encoder)"

### 6.5 Quick Comparison Bar Chart

A grouped horizontal bar chart comparing TrustGuard vs. all baselines on AIPR and EPR (from `task2_enforcement.json`). Bars animate in sequentially with staggered delays. TrustGuard's bar glows with a gradient. Baselines that cannot enforce at runtime (DREBIN, MaMaDroid, DexRay, MaskDroid) show a grayed-out "N/A — install-time only" label.

---

## 7. Screen 2 — Live Simulation Playground

**Purpose**: The crown jewel of interactivity. Users configure a simulation scenario, press "Run", and watch TrustGuard's three agents operate in real-time, making enforcement decisions step by step.

### 7.1 Configuration Panel (Left Sidebar, 320px)

A form panel where users set simulation parameters before running:

#### Environment Settings
| Input | Type | Default | Range | Maps To |
|-------|------|---------|-------|---------|
| Benign Apps | Number Slider | 50 | 10–500 | `EnvConfig.num_benign_apps` |
| Malicious Apps | Number Slider | 10 | 1–200 | `EnvConfig.num_malicious_apps` |
| Simulation Steps | Number Slider | 200 | 50–4320 | `EnvConfig.max_steps` |
| Playback Speed | Dropdown | 1× | 0.5×, 1×, 2×, 5×, 10× | WebSocket tick rate |

#### Reward Weights
| Input | Type | Default | Range | Maps To |
|-------|------|---------|-------|---------|
| λ₁ (false revoc penalty) | Slider | 10.0 | 1.0–30.0 | `EnvConfig.false_revoc_penalty` |
| λ₂ (enforcement cost) | Slider | 0.1 | 0.01–1.0 | `EnvConfig.enforce_cost` |
| λ₃ (risk reduction weight) | Slider | 1.0 | 0.1–5.0 | `EnvConfig.risk_reduction_weight` |

#### Safety Constraint
| Input | Type | Default | Range | Maps To |
|-------|------|---------|-------|---------|
| ε_safe (FRR budget) | Slider | 0.025 | 0.005–0.10 | `lagrangian.eps_safe` |

#### Risk Estimator
| Input | Type | Default | Range | Maps To |
|-------|------|---------|-------|---------|
| EMA α | Slider | 0.3 | 0.05–0.9 | `risk_estimator.ema_alpha` |
| Risk Threshold (τ) | Slider | 0.5 | 0.1–0.9 | `enforcement_agent.risk_threshold` |

#### Agent Mode
| Input | Type | Default | Options |
|-------|------|---------|---------|
| Execution Mode | Radio Group | Stochastic | Stochastic / Deterministic |
| Checkpoint | Dropdown | `best` | `best`, `latest`, uploaded `.pt` |

**"▶ Run Simulation" Button**: Large green button at the bottom. Sends all parameters to the backend via REST, which spawns a `PermissionEnv`, loads the checkpoint, and begins streaming step data over WebSocket.

### 7.2 Main Visualization Area (Center)

#### 7.2.1 Live Risk Heatmap (Primary Visual)

A large animated heatmap (N_apps rows × timestep columns, scrolling horizontally):
- Each cell is colored by the app's EMA risk ρ̄ᵢᵗ (green 0.0 → yellow 0.3 → red 0.7 → deep red 1.0)
- Malicious apps are marked with a 🔴 icon on the row label
- When the Enforcement Agent fires a `revoke` on an app, a ⚡ icon appears on that cell
- When a `rate_limit` is applied, a 🔻 icon appears
- The current timestep column is highlighted with a brighter border

#### 7.2.2 Real-Time Metric Sparklines (Below Heatmap)

Four small spark-line charts updating in real-time:
1. **Cumulative AIPR%** — anomalous invocations blocked so far / total anomalous invocations so far
2. **Cumulative FRR%** — false revocations / total revocations, with the ε_safe horizontal line
3. **Episode Reward** — cumulative rₜ over time
4. **Mean EMA Risk** — average ρ̄ across all apps

#### 7.2.3 Agent Action Timeline (Below Sparklines)

A three-lane horizontal timeline showing each agent's decisions as colored dots:
- **Lane 1 (Agent k=1, Monitoring)**: Blue dots = SAMPLE_NOW, gray dots = IDLE
- **Lane 2 (Agent k=2, Risk Analysis)**: Amber dots = ANALYSE, gray dots = DEFER
- **Lane 3 (Agent k=3, Enforcement)**: No color = no_op, yellow = alert, orange = rate_limit, red = revoke

Clicking any dot opens a detail popover showing:
- The agent's observation vector at that timestep
- The policy distribution (probabilities for each action)
- The selected action and its log-probability

### 7.3 Live Enforcement Log (Right Panel, 280px)

A scrolling log of enforcement events:

```
[t=42]  ⚡ REVOKE  com.suspicious.flashlight
        Permissions: CAMERA, READ_CONTACTS
        EMA Risk: 0.78  |  False: No ✓
        Latency: 1.2s from anomaly onset

[t=45]  ⚠ ALERT   com.social.messenger
        EMA Risk: 0.52
        → No revocation; risk below threshold

[t=51]  🔻 RATE_LIMIT  com.game.puzzle
        Permissions: ACCESS_FINE_LOCATION
        EMA Risk: 0.61  |  False: Yes ✗ (benign app)
```

Each entry is clickable and expands to show:
- The per-permission risk vector ϱ(p, fᵢ) = |uᵢ,ₚᵗ − p̂ᵢ,ₚ|
- The belief state bₜ as a mini bar chart of its 256 dimensions
- The Lagrange multiplier μ at that timestep

### 7.4 Simulation Controls (Bottom Bar)

- **⏸ Pause / ▶ Resume**: Pause/resume the WebSocket stream
- **⏹ Stop**: End the simulation early and display final summary
- **📊 Show Summary**: Opens a modal with the final `StepInfo` metrics
- **💾 Export**: Download the full simulation trace as JSON

### 7.5 Post-Simulation Summary Modal

When the simulation ends (or the user clicks Stop), a full-screen modal appears:

| Metric | Value | Paper Reference |
|--------|-------|-----------------|
| AIPR | Computed from this run | 63.4% (paper) |
| EPR | Computed from this run | 71.8% (paper) |
| FRR | Computed from this run | 2.1% (paper) |
| Total Revocations | N | — |
| False Revocations | N | — |
| Mean Reward | Computed | — |
| ε_safe Budget | Was it met? ✓/✗ | 2.5% |

Plus a mini bar chart comparing this run's metrics to the paper's reported values.

---

## 8. Screen 3 — Agent Inspector (Dec-POMDP Deep Dive)

**Purpose**: Let users deeply understand how each of the three agents works — their observations, policies, actions, and how they coordinate through the shared belief state.

### 8.1 Three-Column Agent Layout

The screen is divided into three equal columns, one per agent, each with the agent's accent color as a top border.

#### Column 1 — Monitoring Agent (k=1, Blue)

**Observation Space Card**:
- Shows `obs_dim = |𝒫| + 2 = 44`
- Diagram: `[mean_usage_per_permission (42), time_since_sample (1), system_load (1)]`
- Live values if simulation is running

**Action Space Card**:
- Binary: `{IDLE (0), SAMPLE_NOW (1)}`
- Shows the current policy distribution as two stacked bars (π(IDLE|o¹) vs π(SAMPLE|o¹))

**Role Description Card**:
- "Adaptively samples permission-usage observations. Balances monitoring coverage against device overhead by learning when to trigger a full environment snapshot."
- Key insight: "By learning an adaptive schedule, the agent front-loads monitoring effort toward anomalous applications while idling on stable ones."

**Network Architecture Card**:
- Diagram: `obs (44) → Linear(256) → LayerNorm → Tanh → Dropout → Linear(256) → LayerNorm → Tanh → Linear(2) → Categorical`
- Weight count and parameter summary

#### Column 2 — Risk-Analysis Agent (k=2, Amber)

**Observation Space Card**:
- Shows `obs_dim = 2×|𝒫| + 1 = 85`
- Diagram: `[mean_usage_delta (42), predicted_probs (42), mean_ema_risk (1)]`
- The predicted probs `p̂ᵢ,ₚ` are highlighted as coming from the Permission Prediction Model gθ

**Action Space Card**:
- Binary: `{DEFER (0), ANALYSE (1)}`
- Distribution bars

**Interaction Card**:
- "Receives δuᵢᵗ from the Monitoring Agent and produces updated risk estimates, incorporating both instantaneous deviation and semantic risk."
- Data flow arrow from Column 1 → Column 2

#### Column 3 — Enforcement Agent (k=3, Red)

**Observation Space Card**:
- Shows `obs_dim = 1 + belief_dim + 2 = 259`
- Diagram: `[mean_ema_risk (1), belief_state (256), revoke_rate_history (1), alert_rate_history (1)]`
- The belief state `bₜ` is highlighted as coming from the GRU Belief Encoder

**Action Space Card (Two-Headed)**:
- **Head 1 — Action Type**: 4-way categorical `{no_op, alert, rate_limit, revoke}`
- **Head 2 — Permission Targets**: Binary Bernoulli mask over 42 permissions
- Interactive: User can hover over each action to see its cost `c(a)` = {0, 0.2, 0.5, 1.0}
- Risk-gate explanation: "Actions other than no_op are masked out when EMA risk < τ = 0.5"

**Safety Integration Card**:
- "During training, the Lagrange multiplier μ is passed as an auxiliary scalar observation, allowing the policy to internalize the false-revocation penalty."
- Shows the Lagrangian: `ℒ(θ, μ) = 𝔼[Σ γᵗ rₜ] − μ(FRR(π) − ε_safe)` rendered via KaTeX

### 8.2 Shared Belief State Panel (Full Width Below Columns)

**GRU Belief Encoder Visualization**:
- Architecture diagram: `o¹ₜ → Proj(44→128) ─┐`
                          `o²ₜ → Proj(85→128) ─┤→ Concat(384) → GRU(512) → Linear(256) → Tanh → bₜ`
                          `o³ₜ → Proj(259→128)─┘`
- If simulation is running: a live 256-dimensional bar chart of the current bₜ values, colored by magnitude (near 0 = gray, near ±1 = accent)
- Hidden state carry: "State hₜ is carried across 128-step segments within 72-hour episodes"

### 8.3 Interactive "What Does This Agent Do?" Panel

A dropdown selector: "Select an agent to inspect". When selected:
- Shows the agent's `forward()` method call graph as a simplified flowchart
- Allows the user to input a custom observation vector (manually set each dimension), click "Compute Action", and see the resulting policy distribution
- This calls the backend `/api/agent_forward` endpoint

---

## 9. Screen 4 — Semantic Encoder Visualizer

**Purpose**: Let users understand how app metadata is fused into the 256-dimensional embedding ϕ(fᵢ).

### 9.1 Interactive App Input Form

Users can type in app details and see the encoding process:

| Input | Type | Example |
|-------|------|---------|
| App Description | Textarea | "A flashlight app that illuminates your screen" |
| App Category | Dropdown | Tools, Communication, Games, ... (33 categories) |
| Declared Permissions | Multi-select checkboxes | CAMERA, READ_CONTACTS, ACCESS_FINE_LOCATION, ... (all 42) |
| API Features | Textarea | "android.hardware.Camera.open(), android.location.LocationManager.getLastKnownLocation()" |

**"Encode" Button**: Sends data to `/api/encode` and returns:
- The ϕ(fᵢ) vector (256 dims), visualized as a heat bar
- The predicted permission probabilities p̂ᵢ,ₚ for all 42 permissions
- The semantic risk scores ϱ(p, fᵢ) = 1 − p̂ᵢ,ₚ for all declared permissions

### 9.2 Three-Modality Fusion Diagram

An animated SVG showing:
1. **BERT branch**: Description text → Tokenizer → BERT → CLS token → Projection(768→768) → `e_text`
2. **GAT branch**: API call graph → GATv2 (2 layers, 4 heads) → Global Mean+Max Pool → Projection → `e_graph`
3. **CodeBERT branch**: API feature text → CodeBERT → CLS → Projection → `e_code`
4. **Fusion**: `[e_text ‖ e_graph ‖ e_code]` → MLP(2304→512→256) → LayerNorm → L2-normalize → ϕ(fᵢ)

Each branch animates sequentially when the user clicks "Encode".

### 9.3 Permission Risk Table

A styled table showing all 42 Android dangerous permissions with:
- Permission name
- Permission group (Calendar, Camera, Contacts, Location, Microphone, Phone, Sensors, SMS, Storage, Bluetooth, etc.)
- Predicted probability p̂ᵢ,ₚ (colored bar, green for high, red for low)
- Semantic risk 1 − p̂ᵢ,ₚ (colored bar, inverse)
- Whether this permission was declared by the input app (checkmark)
- Whether it's flagged as "unexpectedly present" (risk > 0.5 and declared)

### 9.4 Side-by-Side App Comparison

Users can encode TWO apps simultaneously and see their ϕ vectors side by side:
- Cosine similarity between the two embeddings
- Highlighted dimensions where they diverge most
- Permission prediction differences

---

## 10. Screen 5 — Results Explorer (Paper Tables & Charts)

**Purpose**: Present every experimental result from the paper in interactive, filterable, sortable form.

### 10.1 Task Selector Tabs

Three prominent tabs at the top:
- **Task 1 — Permission Risk Prediction** (from `task1_prediction.json`)
- **Task 2 — Autonomous Enforcement** (from `task2_enforcement.json`)
- **Task 3 — Adversarial Robustness** (from `adversarial_robustness.json`)

### 10.2 Task 1 Panel

**Interactive Table**:
| Method | Macro-F1 | AUROC | PR-AUC | Ext. AUROC |
|--------|----------|-------|--------|------------|
| Static Policy | 0.501 | — | — | — |
| DREBIN | 0.864 ± 0.005 | 0.907 ± 0.004 | 0.851 ± 0.006 | 0.858 ± 0.007 |
| MaMaDroid | 0.877 ± 0.005 | 0.921 ± 0.004 | 0.869 ± 0.005 | 0.874 ± 0.006 |
| DexRay | 0.919 ± 0.004 | 0.961 ± 0.003 | 0.923 ± 0.004 | 0.917 ± 0.005 |
| MaskDroid | 0.924 ± 0.004 | **0.967** ± 0.003 | 0.928 ± 0.004 | 0.926 ± 0.005 |
| Rule Threshold | 0.888 ± 0.004 | 0.913 ± 0.003 | 0.879 ± 0.004 | 0.881 ± 0.006 |
| Single-Agent RL | 0.918 ± 0.005 | 0.947 ± 0.004 | 0.903 ± 0.005 | 0.901 ± 0.006 |
| SA PPO-Lagrangian | 0.919 ± 0.005 | 0.948 ± 0.004 | 0.905 ± 0.005 | 0.903 ± 0.006 |
| MAPPO-Lagrangian | 0.928 ± 0.004 | 0.955 ± 0.003 | 0.916 ± 0.004 | 0.912 ± 0.005 |
| **TrustGuard** | **0.939** ± 0.004 | 0.963 ± 0.003 | **0.931** ± 0.004 | 0.921 ± 0.006 |

- Click any column header to sort
- Click any row to highlight it
- TrustGuard row is permanently highlighted with a gradient background
- **Best values** in each column are auto-bolded
- Toggle "Show ± std" on/off

**Grouped Bar Chart**: Same data as animated bars, grouped by method.

**Key Findings Panel**: Render the `key_findings` array from the JSON as a bullet list with alert styling.

### 10.3 Task 2 Panel

Same structure, but with metrics: AIPR%, EPR%, AET-R%, PRR%, FRR%, FIR%, Latency(s).

Additional visual: **Radar Chart** comparing TrustGuard vs. MAPPO-Lagrangian vs. Single-Agent RL across all 7 metrics simultaneously. Each axis is normalized to [0, 1] (FRR inverted so higher = better).

### 10.4 Task 3 Panel

**Heatmap Table**: Methods × Attack scenarios (Clean, MM, RTMA, MM+RTMA), cells colored by AUROC value.

**Degradation Waterfall Chart**: For TrustGuard specifically, show:
- Clean: 0.963
- After MM: 0.891 (−7.2)
- After RTMA: 0.847 (−11.6)
- After MM+RTMA: 0.802 (−16.1)
as a cascading waterfall where each attack chips away at the bar.

### 10.5 Statistical Significance Panel

From `statistical_tests.json`:
- DeLong test p-values displayed as a matrix
- Bootstrap 95% CIs for key metrics
- "Not statistically significant" indicators where p > 0.05

---

## 11. Screen 6 — Adversarial & Robustness Lab

**Purpose**: Interactive adversarial attack simulation. Users configure attacks and see their effect on TrustGuard's performance.

### 11.1 Attack Configuration Panel

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| Attack Type | Radio | None | None, Manifest Mimicry (MM), RTMA, MM+RTMA |
| MM Intensity | Slider | Medium | Low (add 2 perms), Medium (5), High (10) |
| RTMA Timing Profile | Dropdown | Category-matched | Category-matched, Random, Extreme |
| Target Category | Dropdown | All | Specific category to attack |

### 11.2 Before/After Comparison

Split screen showing:
- **Left**: Benign version of an app with its permissions, risk scores, and agent decisions
- **Right**: The same app after the attack is applied — showing the changed manifest, the modified timing profile, and how the agents respond differently

### 11.3 AUROC Degradation Tracker

As the user adjusts attack parameters, a real-time AUROC estimate updates:
- Uses cached model predictions and re-scores under the modified features
- Shows a live gauge: "AUROC: 0.847 (−11.6 from clean)"

### 11.4 Temporal Hold-Out Results

From `temporal_holdout.json`:
- **Timeline Chart**: Training period (2012–2018) → Test period (2019–2021) with a visible gap
- Table of all methods' AUROC and AIPR under temporal drift
- Annotation: "TrustGuard degrades only 2.2 AUROC points across the gap — the smallest among evaluated methods"

### 11.5 Stress Test Results

From `task2_stress_tests.json`:
- **In-Generator vs. AASE-B vs. 2% Prevalence** comparison cards
- Recalibration before/after: show FRR dropping from 4.7% → 2.4% after 72h dual recalibration
- Key insight card: "The safety budget is a distribution-conditional property requiring recalibration under shift"

---

## 12. Screen 7 — Sensitivity & Ablation Studio

**Purpose**: Interactive exploration of all hyperparameter sensitivity analyses and ablation studies from the paper.

### 12.1 Sensitivity Grids (Interactive)

#### 12.1.1 Reward Weight Grid (λ₁ × λ₂)

From `sensitivity_analyses.json` → `reward_weights_lambda`:

An interactive 3×3 heatmap where:
- Rows = λ₁ ∈ {5, 10, 20}
- Columns = λ₂ ∈ {0.05, 0.1, 0.2}
- Cell color = AIPR% (green scale)
- Cell overlay = FRR% (text, red if > ε_safe)

Clicking a cell shows a detail popover with the full metric values and which paper table it maps to.

The user can also use the sliders in the Simulation Playground to run a live simulation with any λ₁, λ₂ combination and compare against the precomputed grid.

#### 12.1.2 Annotation Threshold Grid (τ_low × τ_high)

From `sensitivity_analyses.json` → `annotation_thresholds_tau`:

A 3×3 heatmap of AUROC values.

#### 12.1.3 EMA-α Sensitivity

From `sensitivity_analyses.json` → `ema_alpha`:

A line chart with α ∈ {0.1, 0.3, 0.5, 0.7} on the X-axis, and three lines:
- PRR% (primary Y-axis)
- FRR% (secondary Y-axis)
- Median Latency (secondary Y-axis)

Interactive: dragging a slider along α shows the interpolated values.

#### 12.1.4 Encoder Modality Ablation

From `sensitivity_analyses.json` → `encoder_modalities`:

A bar chart showing AUROC for each modality combination: T, C, G, T+C, T+G, C+G, T+C+G. The full model is highlighted.

### 12.2 Ablation Studies

From `ablations.json`:

#### 12.2.1 Factorial 2×2 Ablation

| Configuration | AIPR% | PRR% | FRR% |
|---------------|-------|------|------|
| Multi + Constraint (**TrustGuard**) | **63.4** | **41.3** | **2.1** |
| Multi − Constraint | 65.1 | 43.1 | 8.9 ✗ |
| Single + Constraint | 49.8 | 33.6 | 2.4 |
| Single − Constraint | 51.2 | 34.9 | 6.8 ✗ |

Visualized as a 2×2 grid of cards. The card without constraint shows a red "FRR Violated" badge.

**Key Interactive Feature**: Toggle between "with constraint" and "without constraint" to see FRR change from 2.1% → 8.9%. A Framer Motion animation shows the FRR gauge needle swinging past the budget line.

#### 12.2.2 Component Ablations

Bar chart comparing:
- Full TrustGuard: 63.4% AIPR
- Local per-agent beliefs (strictly decentralized): 59.7%
- w/o semantic encoder: 44.6%
- w/o runtime traces: 40.9%

Each bar is annotated with "−X.X pp" relative to full.

#### 12.2.3 Null Ablations

- Homogeneous agents + belief encoder: 61.5% AIPR
- Fixed Agents 1–2 (always-on): 62.9% AIPR

Annotation: "Fixing Agents 1–2 always-on costs only 0.5 AIPR points (within noise)"

### 12.3 Constraint Dynamics

From `constraint_dynamics.json`:

A dual-axis time-series chart showing training iterations on X-axis:
- **Left Y-axis**: Rolling FRR% (line, starts at 9.6%, converges to 2.1%)
- **Right Y-axis**: Lagrange multiplier μ (line, starts at 0.21, stabilizes at 0.85)
- **Horizontal dashed line**: ε_safe = 2.5%
- **Annotation marker**: "FRR converges below ε_safe near iteration 2,100"

Per-Category FRR table below:
- Communication: 1.4% (lowest)
- Personalization: 3.8% (highest, exceeds budget ⚠)
- Tools: 3.5% (high variance ⚠)

---

## 13. Screen 8 — Training Monitor

**Purpose**: Visualize the MAPPO training process, including learning curves, loss components, and gradient statistics.

### 13.1 Training Curve Dashboard

If a training run is active (connected via WebSocket to `/ws/training`), show live-updating charts:

- **Episode Reward** (smoothed EMA): Line chart, X = iteration, Y = mean reward
- **Policy Loss per Agent**: Three lines (blue, amber, red)
- **Value Loss**: Centralized critic loss
- **Entropy**: Per-agent entropy (should decrease over training)
- **Lagrange Multiplier μ**: Live value with historical line
- **FRR (rolling)**: With ε_safe threshold
- **Gradient Norm**: Per-agent, showing if clipping is active

### 13.2 Checkpoint Browser

A table of saved checkpoints:
| Checkpoint | Iteration | Reward | FRR | AIPR | Actions |
|------------|-----------|--------|-----|------|---------|
| `best` | 4,200 | 12.4 | 2.1% | 63.4% | [Load] [Download] |
| `latest` | 5,000 | 11.8 | 2.1% | 62.9% | [Load] [Download] |

"Load" sets the checkpoint for the Simulation Playground.

### 13.3 Hyperparameter Summary

Render the full YAML configs as a styled, collapsible tree:
- `marl.yaml` → MAPPO section, Lagrangian section, Rollout section, Supervised section
- `model.yaml` → All model dimensions
- `training.yaml` → Training schedule

---

## 14. Screen 9 — Real-Device Pilot Report

**Purpose**: Present the 14-day, 35-device pilot study results transparently, including where the system falls short.

### 14.1 Pilot Overview Cards

| Metric | Value |
|--------|-------|
| Duration | 14 days |
| Devices | 35 (4 OEMs, Android 11–14) |
| Median Apps/Device | 64 |
| Mode | Notification-only (no revocation) |
| IRB | Approved |

### 14.2 Key Results

| Metric | Field Value | Simulation Value | Gap |
|--------|------------|------------------|-----|
| False Alert Rate | **3.4%** [2.9%, 3.9%] | 2.5% (budget) | **+0.9pp ⚠** |
| Median Alert Latency | 4.8s | 1.9s | +2.9s |
| P95 Alert Latency | 12.3s | — | — |
| Battery Overhead | 18.6 mWh/hour | — | No control |
| Notifications Acceptable | 71% | — | — |
| Notifications Intrusive | 12% | — | — |

### 14.3 Sim-to-Real Gap Analysis

A visual comparison chart showing simulation FRR (2.1%) vs. field false-alert rate (3.4%), with an explanation card:
- "The constraint does not transfer unmodified"
- "Deployment requires ~72-hour on-device dual recalibration"
- "Latency gap stems from AppOpsManager background execution limits"

### 14.4 Limitations Callout

A prominent `[!WARNING]` styled card:
> This pilot ran in notification-only mode. The enforcement policy was NOT field-tested. The 3.4% field false-alert rate exceeds the 2.5% simulated budget. These are honest limitations the paper discloses.

---

## 15. Screen 10 — Dataset & PermissionBench Explorer

**Purpose**: Let users explore the PermissionBench dataset structure, statistics, and annotation protocol.

### 15.1 Dataset Statistics

| Split | Benign | Malicious | Total |
|-------|--------|-----------|-------|
| Train (70%) | 43,288 | 10,158 | 53,446 |
| Val (10%) | 6,184 | 1,451 | 7,635 |
| Test (20%) | 12,368 | 2,903 | 15,271 |
| **Total** | **61,840** | **14,512** | **76,352** |

Visualized as stacked bars with benign (green) and malicious (red) segments.

### 15.2 Permission Distribution Chart

A horizontal bar chart showing the 42 dangerous permissions ranked by frequency of appearance in the dataset. Color-coded by permission group (Calendar, Camera, Contacts, etc.).

### 15.3 Annotation Protocol Flowchart

An interactive Mermaid-style flowchart showing:
1. Seed predictor gθ⁽⁰⁾ trained on 5,560 Drebin + 5,560 matched benign
2. Three criteria: TaintDroid flag → Override | gθ < τ_low=0.05 → Anomalous | gθ > τ_high=0.70 → Legitimate | Ambiguous → Human annotators (κ=0.84)
3. The ~8% resolved by humans highlighted

### 15.4 Category Browser

A grid of 33 Google Play category cards, each showing:
- Category name
- Number of benign/malicious apps
- Most common permissions in that category
- Average risk score

---

## 16. Reusable Component Library

Every visual element should be a reusable React component:

### 16.1 Data Display Components

| Component | Props | Description |
|-----------|-------|-------------|
| `<StatCard>` | `label, value, std, unit, accentColor, icon, tooltip, trend` | Glowing metric card with counter animation |
| `<GaugeChart>` | `value, max, threshold, label, size` | Circular gauge with threshold line |
| `<SparkLine>` | `data[], color, height, width, showDots` | Inline mini line chart |
| `<HeatmapGrid>` | `data[][], rowLabels, colLabels, colorScale` | Interactive heatmap with tooltips |
| `<RadarChart>` | `axes[], datasets[], size` | Multi-axis radar/spider chart |
| `<WaterfallChart>` | `segments[{label, delta}], baseColor` | Cascading waterfall for degradation |
| `<TimelineTrack>` | `events[], laneCount, colorMap` | Multi-lane horizontal timeline |

### 16.2 Interactive Components

| Component | Props | Description |
|-----------|-------|-------------|
| `<ParamSlider>` | `label, min, max, step, default, unit, onChange` | Labeled slider with live value display |
| `<PermissionSelector>` | `permissions[], selected[], onChange` | Multi-select checkbox grid grouped by permission group |
| `<AgentActionPopover>` | `observation, policyDist, selectedAction, logProb` | Detailed action inspection popover |
| `<RunButton>` | `onClick, isLoading, label` | Animated run button with loading spinner |

### 16.3 Layout Components

| Component | Description |
|-----------|-------------|
| `<GlassPanel>` | Wrapper applying glassmorphism CSS |
| `<SplitView>` | Resizable left/right or top/bottom panels |
| `<TabBar>` | Animated tab selector with underline indicator |
| `<CollapsibleSection>` | Section with smooth expand/collapse |

---

## 17. Micro-Animations & Motion Design

### 17.1 Page Transitions

Each screen uses Framer Motion's `AnimatePresence` with:
```
initial={{ opacity: 0, y: 20 }}
animate={{ opacity: 1, y: 0 }}
exit={{ opacity: 0, y: -10 }}
transition={{ duration: 0.3, ease: "easeOut" }}
```

### 17.2 Number Counter Animation

All stat values animate from 0 to their target using:
```
transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}  // custom ease curve
```

### 17.3 Chart Draw-In

Line charts draw their paths using SVG `stroke-dashoffset` animation:
```css
.line-path {
  stroke-dasharray: var(--total-length);
  stroke-dashoffset: var(--total-length);
  animation: draw 1.2s ease-out forwards;
}
@keyframes draw {
  to { stroke-dashoffset: 0; }
}
```

Bar charts grow from 0 height with staggered delays (50ms per bar).

### 17.4 Glow Effects (Dark Mode Only)

Active metric cards pulse gently:
```css
@keyframes glow-pulse {
  0%, 100% { box-shadow: var(--glow-monitor); }
  50% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.5); }
}
.stat-card--active {
  animation: glow-pulse 3s ease-in-out infinite;
}
```

### 17.5 Hover Interactions

- Cards: `transform: translateY(-2px)` + shadow deepens
- Buttons: `transform: scale(1.03)` + brightness increase
- Chart elements: Fade other elements to 30% opacity, highlight hovered element to 100%

---

## 18. Backend API Specification (FastAPI)

### 18.1 REST Endpoints

#### `GET /api/results/{task}`
Returns the contents of `results/{task}.json`.
- **Path params**: `task` ∈ {`task1_prediction`, `task2_enforcement`, `adversarial_robustness`, `ablations`, `sensitivity_analyses`, `constraint_dynamics`, `temporal_holdout`, `pilot_summary`, `task2_stress_tests`, `statistical_tests`}
- **Response**: Raw JSON from the file

#### `GET /api/config/{file}`
Returns the YAML config parsed to JSON.
- **Path params**: `file` ∈ {`model`, `marl`, `training`, `dataset`}

#### `POST /api/encode`
Runs the App Semantic Encoder on user-provided input.
- **Body**: `{ description: str, category: str, permissions: str[], api_features: str }`
- **Response**: `{ embedding: float[256], predicted_probs: float[42], risk_scores: float[42] }`

#### `POST /api/agent_forward`
Runs a single agent's forward pass on a custom observation.
- **Body**: `{ agent: "monitoring"|"risk"|"enforcement", observation: float[], deterministic: bool }`
- **Response**: `{ action: int, action_name: str, log_prob: float, policy_distribution: float[] }`

#### `POST /api/simulation/start`
Starts a new simulation with the given parameters.
- **Body**: `{ num_benign: int, num_malicious: int, max_steps: int, eps_safe: float, ema_alpha: float, risk_threshold: float, lambda1: float, lambda2: float, lambda3: float, checkpoint: str, seed: int, deterministic: bool }`
- **Response**: `{ session_id: str, ws_url: str }`

#### `POST /api/simulation/{session_id}/stop`
Stops a running simulation.

#### `GET /api/checkpoints`
Lists available model checkpoints.
- **Response**: `[{ name: str, path: str, iteration: int, metrics: {...} }]`

### 18.2 WebSocket Endpoints

#### `WS /ws/simulation/{session_id}`
Streams simulation step data.

**Message format** (sent every tick):
```json
{
  "step": 42,
  "usage_matrix": [[0,1,0,...], ...],
  "ema_risk": [0.12, 0.05, 0.78, ...],
  "agent_actions": {
    "monitoring": {"action": 1, "action_name": "SAMPLE_NOW", "log_prob": -0.23, "distribution": [0.2, 0.8]},
    "risk": {"action": 1, "action_name": "ANALYSE", "log_prob": -0.34, "distribution": [0.3, 0.7]},
    "enforcement": {"action_type": 3, "action_name": "revoke", "perm_targets": [0,0,1,0,...], "log_prob": -1.2, "distribution": [0.1, 0.1, 0.2, 0.6]}
  },
  "belief_state": [0.12, -0.34, ...],
  "lagrange_mu": 0.85,
  "reward": 1.23,
  "cumulative_reward": 45.6,
  "info": {
    "risk_reduction": 0.15,
    "false_revocations": 0,
    "total_revocations": 2,
    "enforcement_cost": 1.0,
    "privacy_risk": 0.23,
    "frr_cumulative": 0.021,
    "aipr_cumulative": 0.58,
    "done": false
  }
}
```

#### `WS /ws/training`
Streams training metrics if a training run is active.

---

## 19. Project File Structure

```
trustguard-dashboard/
├── app/                         # Next.js App Router
│   ├── layout.tsx               # Root layout (sidebar, header, providers)
│   ├── page.tsx                 # Screen 1: Command Center
│   ├── simulation/
│   │   └── page.tsx             # Screen 2: Live Simulation Playground
│   ├── agents/
│   │   └── page.tsx             # Screen 3: Agent Inspector
│   ├── encoder/
│   │   └── page.tsx             # Screen 4: Semantic Encoder Visualizer
│   ├── results/
│   │   └── page.tsx             # Screen 5: Results Explorer
│   ├── adversarial/
│   │   └── page.tsx             # Screen 6: Adversarial Lab
│   ├── sensitivity/
│   │   └── page.tsx             # Screen 7: Sensitivity Studio
│   ├── training/
│   │   └── page.tsx             # Screen 8: Training Monitor
│   ├── pilot/
│   │   └── page.tsx             # Screen 9: Real-Device Pilot
│   └── dataset/
│       └── page.tsx             # Screen 10: PermissionBench Explorer
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── AppShell.tsx
│   ├── data-display/
│   │   ├── StatCard.tsx
│   │   ├── GaugeChart.tsx
│   │   ├── SparkLine.tsx
│   │   ├── HeatmapGrid.tsx
│   │   ├── RadarChart.tsx
│   │   ├── WaterfallChart.tsx
│   │   └── TimelineTrack.tsx
│   ├── interactive/
│   │   ├── ParamSlider.tsx
│   │   ├── PermissionSelector.tsx
│   │   ├── AgentActionPopover.tsx
│   │   └── RunButton.tsx
│   ├── common/
│   │   ├── GlassPanel.tsx
│   │   ├── SplitView.tsx
│   │   ├── TabBar.tsx
│   │   ├── CollapsibleSection.tsx
│   │   └── KaTeXBlock.tsx
│   └── charts/
│       ├── D3LineChart.tsx
│       ├── D3BarChart.tsx
│       ├── D3Heatmap.tsx
│       └── D3Gauge.tsx
├── hooks/
│   ├── useSimulation.ts         # WebSocket connection + state
│   ├── useTraining.ts           # Training WebSocket
│   ├── useResults.ts            # Fetch + cache paper results
│   └── useTheme.ts              # Theme helpers
├── stores/
│   ├── simulationStore.ts       # Zustand store for simulation state
│   └── globalStore.ts           # Zustand store for app-wide state
├── lib/
│   ├── api.ts                   # Fetch wrappers for REST endpoints
│   ├── websocket.ts             # Socket.io client setup
│   └── constants.ts             # Permission names, action names, colors
├── styles/
│   ├── globals.css              # CSS variables, reset, glassmorphism
│   ├── layout.module.css
│   └── components/              # Per-component CSS modules
├── messages/
│   ├── en.json                  # English translations
│   └── ar.json                  # Arabic translations
├── public/
│   └── fonts/                   # Self-hosted Inter, Cairo, JetBrains Mono
├── next.config.js
├── package.json
└── tsconfig.json

backend/
├── main.py                      # FastAPI app entry point
├── routers/
│   ├── results.py               # /api/results endpoints
│   ├── config.py                # /api/config endpoints
│   ├── encode.py                # /api/encode endpoint
│   ├── simulation.py            # /api/simulation + /ws/simulation
│   └── training.py              # /ws/training
├── services/
│   ├── simulation_runner.py     # Wraps PermissionEnv + agents
│   ├── encoder_service.py       # Wraps AppSemanticEncoder
│   └── checkpoint_manager.py    # Load/list checkpoints
├── schemas/
│   ├── simulation.py            # Pydantic models for simulation
│   └── encoding.py              # Pydantic models for encoding
└── requirements.txt
```

---

## 20. Implementation Roadmap

### Phase 1 — Foundation (Days 1–3)
- [x] Initialize Next.js 14 project with App Router
- [x] Set up CSS variables for both themes (dark/light)
- [x] Set up logical CSS properties for RTL/LTR support
- [x] Configure `next-intl` with EN and AR translation files
- [x] Configure `next-themes` for dark/light toggle
- [x] Build the AppShell: Sidebar, Header, route structure
- [x] Install and configure D3.js, Framer Motion, KaTeX, Socket.io, Zustand

### Phase 2 — Component Library (Days 4–7)
- [x] Build all reusable components (StatCard, GaugeChart, SparkLine, etc.)
- [x] Implement glassmorphism panel with CSS
- [x] Implement all micro-animations (counter, draw-in, glow)
- [x] Build ParamSlider, PermissionSelector, RunButton
- [x] Verify all components render correctly in both themes and both languages
- [x] Load Google Fonts (Inter, Cairo, JetBrains Mono, IBM Plex Arabic, Tajawal)

### Phase 3 — Static Screens (Days 8–12)
- [x] Build Screen 1 (Command Center) with static data from `results/` JSON
- [x] Build Screen 5 (Results Explorer) with all three task tabs
- [x] Build Screen 7 (Sensitivity & Ablation Studio) with all grids and charts
- [x] Build Screen 9 (Real-Device Pilot Report)
- [x] Build Screen 10 (PermissionBench Explorer)
- [x] Ensure all numbers match the paper exactly

### Phase 4 — Backend & Interactive Screens (Days 13–18)
- [x] Build FastAPI backend with all REST endpoints
- [x] Implement `/api/encode` with PyTorch model loading
- [x] Implement `/api/simulation/start` and WebSocket streamer
- [x] Build Screen 2 (Live Simulation Playground) — config panel + heatmap + sparklines + action timeline + enforcement log
- [x] Build Screen 3 (Agent Inspector) — three-column layout + belief state panel + interactive forward pass
- [x] Build Screen 4 (Semantic Encoder Visualizer) — input form + encoding flow + risk table
- [x] Build Screen 6 (Adversarial Lab) — attack configuration + before/after comparison

### Phase 5 — Training Monitor & Integration (Days 19–21)
- [x] Implement `/ws/training` WebSocket
- [x] Build Screen 8 (Training Monitor) — live curves + checkpoint browser + config viewer
- [x] Connect all screens to backend APIs
- [x] Implement session persistence (Zustand middleware)

### Phase 6 — Polish & QA (Days 22–25)
- [x] Full RTL audit: verify every screen in Arabic mode
- [x] Full theme audit: verify every screen in light mode
- [x] Performance audit: ensure all animations run at 60fps
- [x] Accessibility audit: WCAG 2.1 AA compliance (contrast ratios, keyboard navigation, ARIA labels)
- [x] Cross-browser testing: Chrome, Firefox, Safari, Edge
- [x] Responsive testing: 1920px, 1440px, 1024px, 768px breakpoints
- [x] Print stylesheet for light-mode charts
- [x] Final data verification: every number traces to its source JSON

---

## 21. Quality Assurance & Deployment Checklist

### 21.1 Data Integrity

- [x] Every metric value displayed matches the corresponding `results/*.json` file exactly
- [x] Every equation rendered via KaTeX matches the paper's LaTeX source
- [x] Every architectural description matches the actual Python source code
- [x] Every hyperparameter default matches `configs/*.yaml`
- [x] Agent observation dimensions match: Monitoring=44, Risk=85, Enforcement=259

### 21.2 Visual Quality

- [x] All glassmorphism panels have consistent blur, saturation, and border
- [x] All charts use the defined accent color palette (no off-brand colors)
- [x] All fonts load correctly (no fallback flicker)
- [x] All micro-animations are smooth (no jank or layout shift)
- [x] Dark mode has no white flashes on page load
- [x] Light mode charts print legibly in grayscale

### 21.3 Interactivity

- [x] Simulation Playground runs correctly with all parameter combinations
- [x] Agent Inspector's custom observation input produces correct action distributions
- [x] Semantic Encoder input form handles edge cases (empty description, no permissions, etc.)
- [x] WebSocket reconnects gracefully on disconnection
- [x] Simulation can be paused, resumed, and stopped cleanly

### 21.4 Internationalization

- [x] All visible text uses translation keys (no hardcoded strings)
- [x] Arabic layout is fully RTL with correct text alignment
- [x] Charts display correctly in both directions
- [x] Technical terms in Arabic mode are wrapped in `<bdi>` tags
- [x] Font sizes accommodate Arabic text (which is typically ~10% taller)

### 21.5 Deployment

- [x] Frontend builds with `next build` without errors
- [x] Backend starts with `uvicorn main:app` and passes health check
- [x] CORS is configured for the deployment domain
- [x] Environment variables documented: `CHECKPOINT_PATH`, `RESULTS_DIR`, `ALLOWED_ORIGINS`
- [x] Docker Compose file provided for one-command startup

---

> **Final Note**: This dashboard is not merely a visualization tool — it is an interactive, bilingual, publication-grade research artifact designed to communicate the full depth of TrustGuard's architecture, training dynamics, experimental rigor, and honest limitations. Every screen tells a chapter of the paper's story, and every interactive feature invites the audience to verify the claims themselves.
