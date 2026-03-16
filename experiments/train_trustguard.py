"""
experiments/train_trustguard.py
=================================
Main training script for TrustGuard.

Phase 1: Supervised pre-training of the Permission Prediction Model on
         PermissionBench (§Algorithm 1, Step 1).
Phase 2: MARL training of the three Dec-POMDP agents via Constrained MAPPO
         (§Algorithm 1, Steps 2–9).

Usage
-----
    python experiments/train_trustguard.py \
        --config-dir configs/ \
        --data-dir   data/permissionbench \
        --output-dir outputs/run_001 \
        --seed 42

    # Resume from checkpoint:
    python experiments/train_trustguard.py ... --resume outputs/run_001/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# ── TrustGuard imports ────────────────────────────────────────────────────────
from trustguard.models import (
    AppSemanticEncoder, PermissionPredictionModel,
    RuntimeRiskEstimator, BeliefEncoder,
)
from trustguard.agents import MonitoringAgent, RiskAnalysisAgent, EnforcementAgent
from trustguard.marl import MAPPOTrainer, RolloutBuffer, Transition, CentralizedCritic
from trustguard.environment import PermissionEnv, EnvConfig
from trustguard.dataset.permissionbench_loader import PermissionBenchLoader
from trustguard.utils.config_utils import load_all_configs, seed_everything, get_device
from trustguard.utils.logging_utils import setup_logger, ExperimentLogger
from trustguard.utils.metrics import MetricTracker

logger = logging.getLogger("trustguard.train")


# ─────────────────────────────────────────────────────────────────────────────
def build_models(cfg: dict, device: torch.device) -> dict:
    """Instantiate all TrustGuard model components from config."""
    mc = cfg.get("model", cfg)   # tolerate flat or nested config

    semantic_enc = AppSemanticEncoder(
        output_dim=mc["semantic_encoder"]["output_dim"],
        freeze_text=mc["semantic_encoder"]["freeze_text"],
        graph_in_channels=mc["semantic_encoder"]["graph_in_channels"],
        dropout=mc["semantic_encoder"]["dropout"],
    ).to(device)

    perm_pred = PermissionPredictionModel(
        embedding_dim=mc["permission_predictor"]["embedding_dim"],
        hidden_dims=tuple(mc["permission_predictor"]["hidden_dims"]),
        dropout=mc["permission_predictor"]["dropout"],
        label_smoothing=mc["permission_predictor"]["label_smoothing"],
    ).to(device)

    risk_est = RuntimeRiskEstimator(
        ema_alpha=mc["risk_estimator"]["ema_alpha"],
    )

    belief_enc = BeliefEncoder(
        obs_dim_monitor=mc["belief_encoder"]["obs_dim_monitor"],
        obs_dim_risk=mc["belief_encoder"]["obs_dim_risk"],
        obs_dim_enforce=mc["belief_encoder"]["obs_dim_enforce"],
        embed_dim=mc["belief_encoder"]["embed_dim"],
        gru_hidden_dim=mc["belief_encoder"]["gru_hidden_dim"],
        belief_dim=mc["belief_encoder"]["belief_dim"],
    ).to(device)

    return {
        "semantic_encoder": semantic_enc,
        "permission_predictor": perm_pred,
        "risk_estimator": risk_est,
        "belief_encoder": belief_enc,
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_agents(cfg: dict, device: torch.device) -> dict:
    """Instantiate the three Dec-POMDP agents."""
    mc = cfg.get("model", cfg)

    mon_agent = MonitoringAgent(
        hidden_dims=tuple(mc["monitoring_agent"]["hidden_dims"]),
        dropout=mc["monitoring_agent"]["dropout"],
    ).to(device)

    risk_agent = RiskAnalysisAgent(
        hidden_dims=tuple(mc["risk_analysis_agent"]["hidden_dims"]),
        dropout=mc["risk_analysis_agent"]["dropout"],
    ).to(device)

    enf_agent = EnforcementAgent(
        belief_dim=mc["enforcement_agent"]["belief_dim"],
        hidden_dims=tuple(mc["enforcement_agent"]["hidden_dims"]),
        dropout=mc["enforcement_agent"]["dropout"],
        risk_threshold=mc["enforcement_agent"]["risk_threshold"],
    ).to(device)

    critic = CentralizedCritic(
        state_dim=mc["centralized_critic"]["state_dim"],
        hidden_dims=tuple(mc["centralized_critic"]["hidden_dims"]),
    ).to(device)

    return {
        "monitoring": mon_agent,
        "risk":       risk_agent,
        "enforcement": enf_agent,
        "critic":      critic,
    }


# ─────────────────────────────────────────────────────────────────────────────
def supervised_pretrain(
    perm_pred:   PermissionPredictionModel,
    semantic_enc: AppSemanticEncoder,
    data_loader:  torch.utils.data.DataLoader,
    cfg:          dict,
    device:       torch.device,
    output_dir:   Path,
    exp_logger:   ExperimentLogger,
) -> None:
    """
    Phase 1: supervised pre-training of the permission prediction model.
    Trains gθ on PermissionBench binary cross-entropy labels.
    """
    sup_cfg  = cfg.get("supervised", {})
    epochs   = sup_cfg.get("epochs", 30)
    lr       = sup_cfg.get("lr", 1e-4)
    patience = sup_cfg.get("patience", 5)

    # For pre-training we use a simple MLP on raw permission vectors
    # (without running the full semantic encoder, which is expensive)
    optimizer = torch.optim.Adam(perm_pred.parameters(), lr=lr,
                                 weight_decay=sup_cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_loss  = float("inf")
    no_improve = 0

    logger.info("=== Phase 1: Supervised Pre-training (%d epochs) ===", epochs)

    for epoch in range(1, epochs + 1):
        perm_pred.train()
        tracker = MetricTracker()

        for batch in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            perm_vec  = batch["perm_vector"].to(device)   # (B, |𝒫|)
            perm_lbls = batch["perm_labels"].to(device)   # (B, |𝒫|)

            # Use permission vector directly as a proxy embedding
            # (full semantic encoder pre-training requires GPU hours)
            loss = perm_pred.compute_loss(perm_vec, perm_lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(perm_pred.parameters(), 1.0)
            optimizer.step()

            tracker.update({"pretrain_loss": loss.item()}, n=perm_vec.shape[0])

        scheduler.step()
        avgs = tracker.averages()
        exp_logger.log(avgs, step=epoch)
        logger.info("Epoch %d | %s", epoch, avgs)

        if avgs["pretrain_loss"] < best_loss:
            best_loss  = avgs["pretrain_loss"]
            no_improve = 0
            torch.save(perm_pred.state_dict(), output_dir / "perm_pred_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    # Reload best weights
    perm_pred.load_state_dict(torch.load(output_dir / "perm_pred_best.pt"))
    logger.info("Phase 1 complete. Best loss: %.4f", best_loss)


# ─────────────────────────────────────────────────────────────────────────────
def marl_train(
    models:       dict,
    agents:       dict,
    trainer:      MAPPOTrainer,
    env:          PermissionEnv,
    cfg:          dict,
    output_dir:   Path,
    exp_logger:   ExperimentLogger,
) -> None:
    """
    Phase 2: MARL training via Constrained MAPPO.
    Implements Algorithm 1 of the TrustGuard paper.
    """
    rollout_cfg = cfg.get("rollout", {})
    rollout_steps   = rollout_cfg.get("rollout_steps", 2048)
    max_iterations  = cfg.get("training", {}).get("marl_iterations", 500)
    log_interval    = cfg.get("training", {}).get("log_interval", 10)
    save_interval   = cfg.get("training", {}).get("save_interval", 50)

    belief_enc   = models["belief_encoder"]
    semantic_enc = models["semantic_encoder"]

    mon_agent  = agents["monitoring"]
    risk_agent = agents["risk"]
    enf_agent  = agents["enforcement"]

    logger.info("=== Phase 2: MARL Training (%d iterations) ===", max_iterations)

    global_step = 0
    best_prr    = -float("inf")

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        obs = env.reset()
        h_belief = belief_enc.init_hidden(batch_size=1, device=env.device)

        episode_reward   = 0.0
        episode_prr      = 0.0
        episode_frr_list = []

        for t in range(rollout_steps):
            # ── Forward pass: all three agents ────────────────────────
            o_mon = obs["monitor"]
            o_risk = obs["risk"]
            o_enf  = obs["enforce"]

            with torch.no_grad():
                act_mon,  lp_mon,  obs_mon_flat = mon_agent.forward(
                    o_mon, deterministic=False
                )
                act_risk, lp_risk, obs_risk_flat, updated_risk = risk_agent.forward(
                    o_risk, app_ids=env.app_ids, deterministic=False
                )

                # Update belief state
                belief_t, h_belief = belief_enc.step(
                    obs_mon_flat,
                    obs_risk_flat[:, :risk_agent.obs_dim] if obs_risk_flat is not None
                        else torch.zeros(1, risk_agent.obs_dim, device=env.device),
                    o_enf.belief.squeeze(0) if o_enf.belief is not None
                        else torch.zeros(1, 256, device=env.device),
                    h_prev=h_belief,
                )
                env.obs_builder.update_belief(belief_t.squeeze(0))

                # Update enforcement observation with new belief
                o_enf.belief = belief_t

                act_enf, perm_tgt, lp_enf, lp_perm, obs_enf_flat = enf_agent.forward(
                    o_enf, deterministic=False
                )

                # Critic value for GAE
                dummy_embeddings = torch.zeros(
                    env.N, 256, device=env.device
                )
                global_state = env.get_global_state(dummy_embeddings).unsqueeze(0)
                value = agents["critic"](global_state)

            # ── Environment step ───────────────────────────────────────
            perm_targets_full = perm_tgt.expand(env.N, -1)
            obs_next, reward, done, info = env.step(
                action_monitor=act_mon.item(),
                action_risk=act_risk.item(),
                action_enforce=act_enf.item(),
                perm_targets=perm_targets_full,
                risk_threshold=enf_agent.risk_threshold,
            )

            episode_reward  += reward
            episode_frr_list.append(env.false_revocation_rate)

            # ── Store transition ───────────────────────────────────────
            transition = Transition(
                obs_monitor=obs_mon_flat,
                obs_risk=obs_risk_flat if obs_risk_flat is not None
                         else torch.zeros(1, risk_agent.obs_dim, device=env.device),
                obs_enforce=obs_enf_flat,
                global_state=global_state.squeeze(0),
                action_monitor=act_mon,
                action_risk=act_risk,
                action_enforce=act_enf,
                perm_targets=perm_tgt,
                logp_monitor=lp_mon,
                logp_risk=lp_risk,
                logp_enforce=lp_enf,
                logp_perm=lp_perm,
                value=value.squeeze(-1),
                reward=torch.tensor([reward], device=env.device),
                done=torch.tensor([float(done)], device=env.device),
                ema_risk=env.ema_risk.unsqueeze(0),
            )
            trainer.store_transition(transition)

            obs = obs_next
            global_step += 1
            if done:
                break

        # ── PPO update ─────────────────────────────────────────────────
        with torch.no_grad():
            dummy_emb   = torch.zeros(env.N, 256, device=env.device)
            last_state  = env.get_global_state(dummy_emb).unsqueeze(0)
        update_metrics = trainer.update(last_state)

        episode_prr = env.privacy_risk_reduction * 100.0
        episode_frr = np.mean(episode_frr_list) if episode_frr_list else 0.0

        if iteration % log_interval == 0:
            log_dict = {
                "iteration":       iteration,
                "episode_reward":  episode_reward,
                "prr_pct":         episode_prr,
                "frr":             episode_frr,
                **update_metrics,
            }
            exp_logger.log(log_dict, step=global_step)
            elapsed = time.time() - iter_start
            logger.info(
                "Iter %4d | reward %.2f | PRR %.1f%% | FRR %.4f | μ %.4f | %.1fs",
                iteration,
                episode_reward,
                episode_prr,
                episode_frr,
                update_metrics.get("lagrange_mu", 0.0),
                elapsed,
            )

        if episode_prr > best_prr:
            best_prr = episode_prr
            _save_checkpoint(agents, models, output_dir / "checkpoint_best.pt",
                             iteration, global_step)

        if iteration % save_interval == 0:
            _save_checkpoint(agents, models, output_dir / "checkpoint_latest.pt",
                             iteration, global_step)

    logger.info("MARL training complete. Best PRR: %.1f%%", best_prr)


# ─────────────────────────────────────────────────────────────────────────────
def _save_checkpoint(
    agents:    dict,
    models:    dict,
    path:      Path,
    iteration: int,
    step:      int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "global_step": step,
            "monitoring_agent":    agents["monitoring"].state_dict(),
            "risk_agent":          agents["risk"].state_dict(),
            "enforcement_agent":   agents["enforcement"].state_dict(),
            "critic":              agents["critic"].state_dict(),
            "permission_predictor": models["permission_predictor"].state_dict(),
            "belief_encoder":       models["belief_encoder"].state_dict(),
        },
        path,
    )
    logger.debug("Checkpoint saved to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TrustGuard")
    parser.add_argument("--config-dir",  default="configs/", type=str)
    parser.add_argument("--data-dir",    default="data/permissionbench", type=str)
    parser.add_argument("--output-dir",  default="outputs/run", type=str)
    parser.add_argument("--seed",        default=42, type=int)
    parser.add_argument("--resume",      default=None, type=str)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--use-wandb",   action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger("trustguard", log_file=output_dir / "train.log")
    seed_everything(args.seed)
    device = get_device()
    logger.info("Device: %s", device)

    cfg = load_all_configs(args.config_dir)
    cfg["training"] = cfg.get("training", {
        "marl_iterations": 500,
        "log_interval": 10,
        "save_interval": 50,
    })

    exp_logger = ExperimentLogger(
        project="trustguard",
        run_name=output_dir.name,
        config=cfg,
        use_wandb=args.use_wandb,
        use_tb=True,
        tb_dir=output_dir / "tb",
    )

    # ── Build components ──────────────────────────────────────────────
    models = build_models(cfg, device)
    agents = build_agents(cfg, device)

    # ── Phase 1: supervised pre-training ─────────────────────────────
    if not args.no_pretrain:
        data_loader_cfg = PermissionBenchLoader(args.data_dir)
        try:
            train_ds, _, _ = data_loader_cfg.get_datasets()
            train_loader   = PermissionBenchLoader.get_dataloader(
                train_ds,
                batch_size=cfg.get("supervised", {}).get("batch_size", 256),
                balance_classes=True,
            )
            supervised_pretrain(
                models["permission_predictor"],
                models["semantic_encoder"],
                train_loader,
                cfg,
                device,
                output_dir,
                exp_logger,
            )
        except FileNotFoundError:
            logger.warning(
                "PermissionBench data not found at %s. Skipping pre-training.",
                args.data_dir,
            )

    # ── Phase 2: MARL training ────────────────────────────────────────
    mc      = cfg.get("model", cfg)
    lc      = cfg.get("lagrangian", cfg.get("marl", {}))
    rl_cfg  = cfg.get("marl", {}).get("mappo", cfg.get("mappo", {}))

    trainer = MAPPOTrainer(
        monitoring_agent=agents["monitoring"],
        risk_agent=agents["risk"],
        enforcement_agent=agents["enforcement"],
        critic=agents["critic"],
        device=device,
        lr_actor=rl_cfg.get("lr_actor", 3e-4),
        lr_critic=rl_cfg.get("lr_critic", 1e-3),
        lr_lagrange=rl_cfg.get("lr_lagrange", 1e-3),
        gamma=rl_cfg.get("gamma", 0.99),
        gae_lambda=rl_cfg.get("gae_lambda", 0.95),
        eps_clip=rl_cfg.get("eps_clip", 0.2),
        ppo_epochs=rl_cfg.get("ppo_epochs", 10),
        mini_batch_size=rl_cfg.get("mini_batch_size", 256),
        eps_safe=lc.get("eps_safe", 0.025),
    )

    env = PermissionEnv(
        config=EnvConfig(
            num_benign_apps=cfg.get("rollout", {}).get("num_benign", 50),
            num_malicious_apps=cfg.get("rollout", {}).get("num_malicious", 10),
            seed=args.seed,
        ),
        device=device,
    )

    import numpy as np
    marl_train(models, agents, trainer, env, cfg, output_dir, exp_logger)
    exp_logger.finish()


if __name__ == "__main__":
    main()
