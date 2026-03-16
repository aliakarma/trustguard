"""
trustguard/models/semantic_encoder.py
======================================
Layer 1 of TrustGuard: App Semantic Encoder.

Fuses three modalities into a unified application embedding ϕ(fᵢ) ∈ ℝ²⁵⁶:

  1. App-store description text  → BERT CLS embedding
  2. API call graph              → Graph Attention Network (GAT) node pooling
  3. API usage feature vector    → CodeBERT CLS embedding

The three representations are concatenated and projected to ℝ²⁵⁶ via a
two-layer MLP with LayerNorm and dropout.

Reference: §5.1 of the TrustGuard paper (Eq. for ϕ(fᵢ)).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch as GeoBatch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BERT_MODEL = "bert-base-uncased"
_CODEBERT_MODEL = "microsoft/codebert-base"
_HIDDEN_DIM = 768          # BERT / CodeBERT hidden size
_GRAPH_HIDDEN = 256        # GAT intermediate dimension
_GRAPH_HEADS = 4           # number of GAT attention heads
_FUSION_DIM = 512          # intermediate fusion MLP width
_OUTPUT_DIM = 256          # final embedding dimension


# ─────────────────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Wraps a HuggingFace transformer and extracts the [CLS] token embedding.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: bert-base-uncased).
    freeze : bool
        If True, all transformer weights are frozen (fine-tune only the
        projection head). Useful for fast experimentation.
    output_dim : int
        Projection output dimension.
    dropout : float
        Dropout probability on the projected output.
    """

    def __init__(
        self,
        model_name: str = _BERT_MODEL,
        freeze: bool = False,
        output_dim: int = _HIDDEN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("TextEncoder (%s): transformer weights frozen.", model_name)

        hidden = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    def encode_raw(self, texts: list[str], device: torch.device) -> Tensor:
        """
        Tokenise a list of strings and return CLS embeddings.

        Parameters
        ----------
        texts : list[str]
            Raw text strings (one per application).
        device : torch.device

        Returns
        -------
        Tensor  shape (B, hidden_size)
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        out = self.transformer(**encoding)
        return out.last_hidden_state[:, 0, :]   # CLS token

    # ------------------------------------------------------------------
    def forward(self, texts: list[str], device: torch.device) -> Tensor:
        """
        Parameters
        ----------
        texts : list[str]
        device : torch.device

        Returns
        -------
        Tensor  shape (B, output_dim)
        """
        cls = self.encode_raw(texts, device)
        return self.projection(cls)


# ─────────────────────────────────────────────────────────────────────────────
class APICallGraphEncoder(nn.Module):
    """
    Graph Attention Network encoder for application API-call graphs.

    Each node represents an API call; edges represent caller → callee
    relationships extracted from the APK's call graph.

    Architecture
    ------------
    GATv2 (2 layers, ``_GRAPH_HEADS`` heads each) → global mean+max pooling
    → Linear projection to ``output_dim``.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features (e.g. API embedding index).
    hidden_channels : int
        Width of GATv2 intermediate layers.
    output_dim : int
        Projection output dimension.
    heads : int
        Number of attention heads in each GATv2 layer.
    dropout : float
        Dropout on attention coefficients.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = _GRAPH_HIDDEN,
        output_dim: int = _HIDDEN_DIM,
        heads: int = _GRAPH_HEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
        )
        # mean‖max pooled vector has 2 × hidden_channels dims
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    def forward(self, graph_batch: GeoBatch) -> Tensor:
        """
        Parameters
        ----------
        graph_batch : torch_geometric.data.Batch
            Batched PyG graph. Expected attributes:
            - ``x``       : node feature matrix  (N_total, in_channels)
            - ``edge_index`` : COO edge indices    (2, E_total)
            - ``batch``   : node-to-graph mapping (N_total,)

        Returns
        -------
        Tensor  shape (B, output_dim)
        """
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        # Global pooling: concatenate mean and max for richer representation
        x_mean = global_mean_pool(x, batch)   # (B, hidden)
        x_max  = global_max_pool(x, batch)    # (B, hidden)
        x_pool = torch.cat([x_mean, x_max], dim=-1)  # (B, 2*hidden)

        return self.projection(x_pool)


# ─────────────────────────────────────────────────────────────────────────────
class AppSemanticEncoder(nn.Module):
    """
    Full three-modality encoder that produces ϕ(fᵢ) ∈ ℝ²⁵⁶.

    Modalities
    ----------
    1. ``description``  : app-store text → BERT CLS embedding
    2. ``api_graph``    : API call graph → GATv2 graph embedding
    3. ``api_features`` : API usage text / code snippet → CodeBERT embedding

    All three projected embeddings are concatenated to form a (B, 3*H) vector,
    which is then fused by a two-layer MLP into the final (B, 256) embedding.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the final embedding (default 256).
    freeze_text : bool
        Freeze BERT and CodeBERT transformer weights.
    graph_in_channels : int
        Input node-feature dimension for the API call graph.
    dropout : float
        Dropout applied inside the fusion MLP.

    Example
    -------
    >>> encoder = AppSemanticEncoder()
    >>> phi = encoder(
    ...     descriptions=["A flashlight app"],
    ...     api_graph_batch=graph_batch,
    ...     api_feature_texts=["android.hardware.Camera.open()"],
    ... )
    >>> phi.shape
    torch.Size([1, 256])
    """

    def __init__(
        self,
        output_dim: int = _OUTPUT_DIM,
        freeze_text: bool = False,
        graph_in_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # ── Three modality encoders ───────────────────────────────────
        self.text_encoder = TextEncoder(
            model_name=_BERT_MODEL,
            freeze=freeze_text,
            output_dim=_HIDDEN_DIM,
            dropout=dropout,
        )
        self.graph_encoder = APICallGraphEncoder(
            in_channels=graph_in_channels,
            output_dim=_HIDDEN_DIM,
            dropout=dropout,
        )
        self.code_encoder = TextEncoder(
            model_name=_CODEBERT_MODEL,
            freeze=freeze_text,
            output_dim=_HIDDEN_DIM,
            dropout=dropout,
        )

        # ── Fusion MLP: 3 × 768 → 512 → 256 ─────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * _HIDDEN_DIM, _FUSION_DIM),
            nn.LayerNorm(_FUSION_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_FUSION_DIM, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialisation for the fusion MLP linear layers."""
        for module in self.fusion_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        descriptions: list[str],
        api_graph_batch: GeoBatch,
        api_feature_texts: list[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Compute unified application embedding ϕ(fᵢ).

        Parameters
        ----------
        descriptions : list[str]
            App-store description strings, one per application in the batch.
        api_graph_batch : torch_geometric.data.Batch
            Batched API call graphs.
        api_feature_texts : list[str]
            Code-level API usage descriptions (e.g. concatenated API call
            names) — one string per application.
        device : torch.device, optional
            Target device. Inferred from the graph batch if not provided.

        Returns
        -------
        phi : Tensor  shape (B, 256)
            Normalised application semantic embeddings.
        """
        if device is None:
            device = api_graph_batch.x.device

        # ── Modality embeddings ───────────────────────────────────────
        e_text  = self.text_encoder(descriptions, device)        # (B, 768)
        e_graph = self.graph_encoder(api_graph_batch)            # (B, 768)
        e_code  = self.code_encoder(api_feature_texts, device)   # (B, 768)

        # ── Fusion ───────────────────────────────────────────────────
        fused = torch.cat([e_text, e_graph, e_code], dim=-1)    # (B, 2304)
        phi   = self.fusion_mlp(fused)                           # (B, 256)

        # L2-normalise for downstream cosine-distance computations
        phi = F.normalize(phi, p=2, dim=-1)
        return phi

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_batch(
        self,
        descriptions: list[str],
        api_graph_batch: GeoBatch,
        api_feature_texts: list[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Convenience wrapper for inference (no gradient computation).
        Identical signature to ``forward``.
        """
        self.eval()
        return self.forward(descriptions, api_graph_batch, api_feature_texts, device)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"output_dim={self.output_dim}"
