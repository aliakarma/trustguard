"""
trustguard/models/permission_predictor.py
==========================================
Layer 2 of TrustGuard: Permission Prediction Model.

Implements the multi-label neural network

    gθ : ℝ²⁵⁶ → [0, 1]^|𝒫|

that estimates the conditional probability that each permission p is
legitimately warranted given the application embedding ϕ(fᵢ):

    p̂ᵢ,ₚ = P(p ∈ 𝒫ᵢ* | ϕ(fᵢ); θ)

Training objective: binary cross-entropy with label smoothing ε = 0.1.

Reference: §5.2 of the TrustGuard paper (Eq. 4–5).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Android dangerous permissions (full set as of API 34) ────────────────────
ANDROID_PERMISSIONS: list[str] = [
    # Calendar
    "READ_CALENDAR", "WRITE_CALENDAR",
    # Call log
    "READ_CALL_LOG", "WRITE_CALL_LOG", "PROCESS_OUTGOING_CALLS",
    # Camera
    "CAMERA",
    # Contacts
    "READ_CONTACTS", "WRITE_CONTACTS", "GET_ACCOUNTS",
    # Location
    "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
    "ACCESS_BACKGROUND_LOCATION",
    # Microphone
    "RECORD_AUDIO",
    # Phone
    "READ_PHONE_STATE", "READ_PHONE_NUMBERS", "CALL_PHONE",
    "ANSWER_PHONE_CALLS", "ADD_VOICEMAIL", "USE_SIP",
    # Sensors
    "BODY_SENSORS", "BODY_SENSORS_BACKGROUND",
    # SMS
    "SEND_SMS", "RECEIVE_SMS", "READ_SMS", "RECEIVE_WAP_PUSH",
    "RECEIVE_MMS",
    # Storage
    "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
    "READ_MEDIA_IMAGES", "READ_MEDIA_VIDEO", "READ_MEDIA_AUDIO",
    # Bluetooth
    "BLUETOOTH_SCAN", "BLUETOOTH_CONNECT", "BLUETOOTH_ADVERTISE",
    # Activity recognition
    "ACTIVITY_RECOGNITION",
    # Nearby WiFi devices
    "NEARBY_WIFI_DEVICES",
    # Notifications
    "POST_NOTIFICATIONS",
]

NUM_PERMISSIONS = len(ANDROID_PERMISSIONS)
PERMISSION_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(ANDROID_PERMISSIONS)}


# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    """
    Binary cross-entropy loss with label smoothing for multi-label targets.

    Smooth the binary targets:
        ỹ = y * (1 − ε) + ε / 2

    where ε is the smoothing coefficient, then compute standard BCE.

    Parameters
    ----------
    smoothing : float
        Label-smoothing coefficient ε (default: 0.1).
    reduction : str
        'mean', 'sum', or 'none'.
    pos_weight : Tensor, optional
        Per-permission class imbalance weights.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing}")
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    # ------------------------------------------------------------------
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits : Tensor  shape (B, |𝒫|)   — raw (pre-sigmoid) scores
        targets : Tensor shape (B, |𝒫|)   — binary ground-truth labels

        Returns
        -------
        Tensor  scalar loss value
        """
        # Smooth labels
        targets_smooth = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
class PermissionPredictionModel(nn.Module):
    """
    Multi-label MLP that maps application embeddings to permission probability
    distributions.

    Architecture
    ------------
    ϕ (B, 256) → Linear(256, 512) → GELU → Dropout
               → Linear(512, 256) → GELU → Dropout
               → Linear(256, |𝒫|) → [sigmoid at inference]

    The final sigmoid is applied outside the forward pass (i.e. the model
    returns raw logits) so that ``LabelSmoothingBCE`` can use numerically
    stable ``binary_cross_entropy_with_logits``.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input embedding (must match AppSemanticEncoder.output_dim).
    num_permissions : int
        Total number of permissions |𝒫|.
    hidden_dims : tuple[int, ...]
        Widths of the MLP hidden layers.
    dropout : float
        Dropout probability between layers.
    label_smoothing : float
        Label smoothing ε for the training loss.

    Example
    -------
    >>> model = PermissionPredictionModel()
    >>> phi = torch.randn(4, 256)
    >>> probs = model.predict_proba(phi)
    >>> probs.shape
    torch.Size([4, 42])
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_permissions: int = NUM_PERMISSIONS,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_permissions = num_permissions

        # ── Build MLP dynamically from hidden_dims ────────────────────
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_permissions))

        self.mlp = nn.Sequential(*layers)

        # ── Permission-vector input adapter ───────────────────────────
        # The model is designed to consume the semantic embedding
        # φ(fᵢ) ∈ ℝ^embedding_dim. Supervised pre-training and the Task-1
        # evaluation use a lightweight proxy: the declared-permission
        # multi-hot vector (dim = num_permissions) instead of running the
        # ~220M-param semantic encoder. This adapter projects that proxy
        # into the embedding space so both input kinds share one MLP head.
        # It is only applied when a permission-vector-sized input is passed;
        # genuine embedding inputs (dim == embedding_dim) bypass it.
        if num_permissions != embedding_dim:
            self.perm_adapter: Optional[nn.Module] = nn.Linear(
                num_permissions, embedding_dim
            )
        else:
            self.perm_adapter = None

        # ── Loss function ─────────────────────────────────────────────
        self.loss_fn = LabelSmoothingBCE(smoothing=label_smoothing)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        modules = list(self.mlp.modules())
        if self.perm_adapter is not None:
            modules.append(self.perm_adapter)
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, phi: Tensor) -> Tensor:
        """
        Compute raw logits for all permissions.

        Accepts either the semantic embedding φ(fᵢ) of shape
        ``(B, embedding_dim)`` or, for the lightweight pre-training/eval
        proxy path, the declared-permission multi-hot vector of shape
        ``(B, num_permissions)``. Permission-vector inputs are projected into
        the embedding space by ``perm_adapter`` before the shared MLP head.

        Parameters
        ----------
        phi : Tensor  shape (B, embedding_dim) or (B, num_permissions)

        Returns
        -------
        logits : Tensor  shape (B, num_permissions)
            Raw (pre-sigmoid) scores. Use ``predict_proba`` for probabilities.
        """
        if (
            self.perm_adapter is not None
            and phi.shape[-1] == self.num_permissions
            and phi.shape[-1] != self.embedding_dim
        ):
            phi = self.perm_adapter(phi)
        return self.mlp(phi)

    # ------------------------------------------------------------------
    def predict_proba(self, phi: Tensor) -> Tensor:
        """
        Return permission probabilities in [0, 1].

        Parameters
        ----------
        phi : Tensor  shape (B, embedding_dim)

        Returns
        -------
        probs : Tensor  shape (B, num_permissions)
        """
        return torch.sigmoid(self.forward(phi))

    # ------------------------------------------------------------------
    def compute_loss(self, phi: Tensor, targets: Tensor) -> Tensor:
        """
        Compute label-smoothed BCE loss.

        Parameters
        ----------
        phi : Tensor     shape (B, embedding_dim)
        targets : Tensor shape (B, num_permissions)  — binary labels

        Returns
        -------
        loss : Tensor  scalar
        """
        logits = self.forward(phi)
        return self.loss_fn(logits, targets)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def risk_scores(self, phi: Tensor) -> Tensor:
        """
        Compute per-permission semantic risk scores: ϱ(p, fᵢ) = 1 − p̂ᵢ,ₚ.

        Parameters
        ----------
        phi : Tensor  shape (B, embedding_dim)

        Returns
        -------
        risk : Tensor  shape (B, num_permissions)
            Higher value → permission is less expected for this app type.
        """
        return 1.0 - self.predict_proba(phi)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_permissions={self.num_permissions}"
        )
