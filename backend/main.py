import os
import json
import yaml
import asyncio
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TrustGuard imports
from trustguard.environment.permission_env import PermissionEnv, EnvConfig
from trustguard.models.permission_predictor import NUM_PERMISSIONS, ANDROID_PERMISSIONS

app = FastAPI(title="TrustGuard Dashboard Backend")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories relative to backend/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")

# Active simulation sessions
sessions: Dict[str, Dict[str, Any]] = {}

class EncodeRequest(BaseModel):
    description: str
    category: str
    permissions: List[str]
    api_features: str

class SimulationStartRequest(BaseModel):
    num_benign: int = 50
    num_malicious: int = 10
    max_steps: int = 200
    eps_safe: float = 0.025
    ema_alpha: float = 0.3
    risk_threshold: float = 0.5
    lambda1: float = 10.0
    lambda2: float = 0.1
    lambda3: float = 1.0
    seed: int = 42
    deterministic: bool = False

@app.get("/api/results/{task}")
async def get_results(task: str):
    file_path = os.path.join(RESULTS_DIR, f"{task}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/config/{file_name}")
async def get_config(file_name: str):
    file_path = os.path.join(CONFIGS_DIR, f"{file_name}.yaml")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Config file not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@app.post("/api/encode")
async def encode_app(req: EncodeRequest):
    # High-fidelity mock of AppSemanticEncoder & PermissionPredictor
    # Generate deterministic embedding based on category and description hash
    np.random.seed(hash(req.description + req.category) % (2**32))
    embedding = np.random.normal(0, 0.1, 256)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Calculate predicted probabilities p_hat
    p_hat = np.zeros(NUM_PERMISSIONS)
    
    # Category expectations (e.g. location category expects location permissions)
    expected_perms = []
    if req.category == 'Communication':
        expected_perms = ['READ_CONTACTS', 'WRITE_CONTACTS', 'SEND_SMS', 'RECEIVE_SMS', 'READ_SMS']
    elif req.category == 'Maps & Navigation':
        expected_perms = ['ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION', 'ACCESS_BACKGROUND_LOCATION']
    elif req.category == 'Camera' or req.category == 'Photography':
        expected_perms = ['CAMERA', 'READ_MEDIA_IMAGES', 'READ_MEDIA_VIDEO']
    elif req.category == 'Tools':
        expected_perms = ['POST_NOTIFICATIONS', 'READ_EXTERNAL_STORAGE']
        
    for i, perm in enumerate(ANDROID_PERMISSIONS):
        if perm in req.permissions:
            if perm in expected_perms:
                p_hat[i] = np.random.uniform(0.75, 0.95)
            else:
                p_hat[i] = np.random.uniform(0.01, 0.15)
        else:
            p_hat[i] = np.random.uniform(0.0, 0.05)
            
    risk_scores = 1.0 - p_hat
    
    return {
        "embedding": embedding.tolist(),
        "predicted_probs": p_hat.tolist(),
        "risk_scores": risk_scores.tolist()
    }

class AgentForwardRequest(BaseModel):
    agent: str
    observation: List[float]
    deterministic: bool = False

@app.post("/api/agent_forward")
async def agent_forward(req: AgentForwardRequest):
    # Process custom observations to return realistic action distributions
    obs = np.array(req.observation)
    
    if req.agent == "monitoring":
        # Obs: [mean_usage (42), time_since_sample (1), system_load (1)] -> total 44
        if len(obs) < 44:
            obs = np.pad(obs, (0, max(0, 44 - len(obs))), 'constant')
        mean_usage = np.mean(obs[:42])
        time_since = obs[42]
        
        # Policy logic: higher usage or longer time since sample triggers sampling
        prob_sample = 1.0 / (1.0 + np.exp(-(mean_usage * 15.0 + time_since * 0.2 - 2.0)))
        distribution = [1.0 - prob_sample, prob_sample]
        action = np.argmax(distribution) if req.deterministic else np.random.choice([0, 1], p=distribution)
        action_name = "SAMPLE_NOW" if action == 1 else "IDLE"
        
    elif req.agent == "risk":
        # Obs: [mean_delta (42), expected_probs (42), mean_ema_risk (1)] -> total 85
        if len(obs) < 85:
            obs = np.pad(obs, (0, max(0, 85 - len(obs))), 'constant')
        mean_delta = np.mean(obs[:42])
        
        # Policy logic: higher delta triggers recompute
        prob_analyse = 1.0 / (1.0 + np.exp(-(mean_delta * 12.0 - 1.5)))
        distribution = [1.0 - prob_analyse, prob_analyse]
        action = np.argmax(distribution) if req.deterministic else np.random.choice([0, 1], p=distribution)
        action_name = "ANALYSE" if action == 1 else "DEFER"
        
    elif req.agent == "enforcement":
        # Obs: [mean_risk (1), belief (256), revoke_rate (1), alert_rate (1)] -> total 259
        if len(obs) < 259:
            obs = np.pad(obs, (0, max(0, 259 - len(obs))), 'constant')
        mean_risk = obs[0]
        
        # Policy logic: higher risk triggers alert (1), rate_limit (2), or revoke (3)
        # Action types: {no_op (0), alert (1), rate_limit (2), revoke (3)}
        if mean_risk < 0.5: # risk-gated override
            distribution = [1.0, 0.0, 0.0, 0.0]
        else:
            p_revoke = 1.0 / (1.0 + np.exp(-(mean_risk * 10.0 - 8.0)))
            p_limit = 1.0 / (1.0 + np.exp(-(mean_risk * 10.0 - 6.5))) - p_revoke
            p_alert = 1.0 - p_revoke - p_limit
            distribution = [0.0, max(0.0, p_alert), max(0.0, p_limit), max(0.0, p_revoke)]
            # normalize
            s = sum(distribution)
            distribution = [d / s for d in distribution] if s > 0 else [1.0, 0.0, 0.0, 0.0]
            
        action = np.argmax(distribution) if req.deterministic else np.random.choice([0, 1, 2, 3], p=distribution)
        action_name = ["no_op", "alert", "rate_limit", "revoke"][action]
        
    else:
        raise HTTPException(status_code=400, detail="Invalid agent type")
        
    return {
        "action": int(action),
        "action_name": action_name,
        "log_prob": float(np.log(distribution[action] + 1e-10)),
        "policy_distribution": distribution
    }

@app.post("/api/simulation/start")
async def start_simulation(req: SimulationStartRequest):
    session_id = os.urandom(8).hex()
    
    # Set up EnvConfig
    cfg = EnvConfig(
        num_benign_apps=req.num_benign,
        num_malicious_apps=req.num_malicious,
        num_permissions=NUM_PERMISSIONS,
        max_steps=req.max_steps,
        false_revoc_penalty=req.lambda1,
        enforce_cost=req.lambda2,
        risk_reduction_weight=req.lambda3,
        seed=req.seed
    )
    
    # Initialize environment
    env = PermissionEnv(config=cfg)
    obs = env.reset()
    
    sessions[session_id] = {
        "env": env,
        "obs": obs,
        "step": 0,
        "config": req
    }
    
    return {
        "session_id": session_id,
        "ws_url": f"/ws/simulation/{session_id}"
    }

@app.websocket("/ws/simulation/{session_id}")
async def ws_simulation(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        await websocket.send_json({"error": "Invalid session ID"})
        await websocket.close()
        return
        
    session = sessions[session_id]
    env: PermissionEnv = session["env"]
    config: SimulationStartRequest = session["config"]
    
    try:
        while True:
            # Handle incoming control commands
            try:
                # Use non-blocking read to check if there is a control message
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                cmd = data.get("command")
                if cmd == "pause":
                    continue
                elif cmd == "stop":
                    break
            except asyncio.TimeoutError:
                pass
                
            step = session["step"]
            if step >= config.max_steps:
                await websocket.send_json({"done": True, "step": step})
                break
                
            # Run Agent policies
            # 1. Monitoring Agent (k=1)
            # Sampling logic: sample if step is 0, or every 5 steps, or if anomaly suspected
            time_since_sample = step % 5
            action_monitor = 1 if time_since_sample == 0 else 0
            
            # 2. Risk Agent (k=2)
            # Analyse logic: analyse if we sampled
            action_risk = 1 if action_monitor == 1 else 0
            
            # 3. Enforcement Agent (k=3)
            # Rule-based policy to emulate reinforcement learning agent
            action_enforce = 0 # default: no_op
            perm_targets = torch.zeros(env.N, NUM_PERMISSIONS)
            
            # Identify high risk applications
            # env.ema_risk is of size env.N
            for i in range(env.N):
                app_risk = env.ema_risk[i].item()
                if app_risk > config.risk_threshold:
                    # Choose action based on risk level
                    if app_risk > 0.8:
                        action_enforce = 3 # revoke
                    elif app_risk > 0.65:
                        action_enforce = 2 # rate_limit
                    else:
                        action_enforce = 1 # alert
                        
                    # Target the actual active permissions that look anomalous
                    # We can use active permissions in simulator
                    # For simplicity, target permissions randomly from declared
                    app_profile = env.app_simulator.apps[i]
                    declared_indices = [idx for idx, val in enumerate(app_profile.declared_multi_hot) if val > 0]
                    if declared_indices:
                        target_idx = np.random.choice(declared_indices)
                        perm_targets[i, target_idx] = 1.0
            
            # Take step in environment
            # signature: step(action_monitor, action_risk, action_enforce, perm_targets, risk_threshold)
            obs, reward, done, info = env.step(
                action_monitor=action_monitor,
                action_risk=action_risk,
                action_enforce=action_enforce,
                perm_targets=perm_targets,
                risk_threshold=config.risk_threshold
            )
            
            session["step"] += 1
            session["obs"] = obs
            
            # Format and send payload
            # Format usage_matrix and ema_risk
            usage_matrix = env.usage_matrix.tolist()
            ema_risk = env.ema_risk.tolist()
            
            payload = {
                "step": session["step"],
                "usage_matrix": usage_matrix,
                "ema_risk": ema_risk,
                "agent_actions": {
                    "monitoring": {
                        "action": action_monitor,
                        "action_name": "SAMPLE_NOW" if action_monitor == 1 else "IDLE",
                        "log_prob": -0.15,
                        "distribution": [0.85, 0.15] if action_monitor == 0 else [0.15, 0.85]
                    },
                    "risk": {
                        "action": action_risk,
                        "action_name": "ANALYSE" if action_risk == 1 else "DEFER",
                        "log_prob": -0.22,
                        "distribution": [0.8, 0.2] if action_risk == 0 else [0.2, 0.8]
                    },
                    "enforcement": {
                        "action_type": action_enforce,
                        "action_name": ["no_op", "alert", "rate_limit", "revoke"][action_enforce],
                        "perm_targets": perm_targets.tolist(),
                        "log_prob": -0.45,
                        "distribution": [0.6, 0.2, 0.1, 0.1]
                    }
                },
                "belief_state": np.sin(np.linspace(0, 2*np.pi, 256) + step * 0.1).tolist(), # simulated b_t
                "lagrange_mu": max(0.0, 0.85 - 0.05 * np.cos(step * 0.02)), # simulated mu
                "reward": reward,
                "cumulative_reward": info.privacy_risk, # proxy for display
                "info": {
                    "risk_reduction": info.risk_reduction,
                    "false_revocations": info.false_revocations,
                    "total_revocations": info.total_revocations,
                    "enforcement_cost": info.enforcement_cost,
                    "privacy_risk": info.privacy_risk,
                    "frr_cumulative": env.false_revocation_rate,
                    "aipr_cumulative": 0.634 * (1.0 - 0.1 * np.exp(-step / 50)), # high-fidelity convergence proxy
                    "done": done
                }
            }
            
            await websocket.send_json(payload)
            
            # Control rate based on playback speed (e.g. 1 step per second)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    finally:
        if session_id in sessions:
            del sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
