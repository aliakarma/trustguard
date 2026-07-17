# TrustGuard Dashboard

Interactive research dashboard for the TrustGuard multi-agent permission-governance
framework. Built with Next.js 16, React 19, and Tailwind v4.

## Architecture — two processes

The dashboard is a thin client. Static result pages (Command Center, Results,
Sensitivity, Adversarial, Pilot, Dataset) read bundled JSON and work standalone.
The **interactive** pages (Live Simulation, Training Monitor, Agent Inspector,
Semantic Encoder) call a small Python inference backend over HTTP + WebSocket.

```
Next.js frontend  :3000   ──►   FastAPI backend  :8001
(this folder)                   (../backend/main.py)
```

If the backend isn't running, the interactive pages show a clear
"Backend unavailable" banner telling you how to start it.

## Getting started

**1. Start the inference backend** (from the repo root, needs the `trustguard`
Python env — torch, fastapi, uvicorn):

```bash
python backend/main.py           # serves http://127.0.0.1:8001
```

**2. Start the dashboard** (from this `dashboard/` folder):

```bash
npm install
npm run dev                      # serves http://localhost:3000
```

Open http://localhost:3000.

### Configuration

The backend URL can be overridden with environment variables (defaults shown):

```
NEXT_PUBLIC_API_BASE=http://localhost:8001
NEXT_PUBLIC_WS_BASE=ws://localhost:8001
```

## Features

- **Command Center** — headline enforcement metrics + cross-method comparison
- **Live Simulation** — streams a live permission-governance episode (real `PermissionEnv`)
- **Training Monitor** — live MAPPO-Lagrangian training curves
- **Agent Inspector** — forward passes for the three cooperative policies
- **Semantic Encoder** — fuse app metadata into ϕ(fᵢ) and predict per-permission risk
- **Results / Adversarial / Sensitivity / Pilot / Dataset** — paper results explorers
- **Theme + language** — light/dark and English/Arabic (full RTL), both persisted
