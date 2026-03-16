"""
TrustGuard
==========
A Multi-Agent Reinforcement Learning Framework for Autonomous Permission
Governance in Mobile Ecosystems.

Dec-POMDP formulation with Centralised Training / Decentralised Execution
(CTDE) via MAPPO and a Lagrangian safety constraint on false revocations.

Reference
---------
Akarma, A., Jan, S., & Syed, T. A. (2025).
TrustGuard: A Multi-Agent Reinforcement Learning Framework for Autonomous
Permission Governance in Mobile Ecosystems.
"""

__version__ = "1.0.0"
__author__ = "Ali Akarma, Salman Jan, Toqeer Ali Syed"
__license__ = "MIT"

# ── Public API surface ────────────────────────────────────────────────────────
from trustguard.models.semantic_encoder import AppSemanticEncoder
from trustguard.models.permission_predictor import PermissionPredictionModel
from trustguard.models.runtime_risk_estimator import RuntimeRiskEstimator
from trustguard.models.belief_encoder import BeliefEncoder

from trustguard.agents.monitoring_agent import MonitoringAgent
from trustguard.agents.risk_analysis_agent import RiskAnalysisAgent
from trustguard.agents.enforcement_agent import EnforcementAgent

__all__ = [
    # Models
    "AppSemanticEncoder",
    "PermissionPredictionModel",
    "RuntimeRiskEstimator",
    "BeliefEncoder",
    # Agents
    "MonitoringAgent",
    "RiskAnalysisAgent",
    "EnforcementAgent",
]
