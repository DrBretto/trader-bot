"""
Regime classification model using GRU architecture.

This model takes context features (market conditions) as input and outputs
regime classification (5 classes) with probabilities and embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


# Regime labels
REGIME_LABELS = [
    'calm_uptrend',
    'risk_on_trend',
    'risk_off_trend',
    'choppy',
    'high_vol_panic'
]

# Context features used as input
CONTEXT_FEATURES = [
    'spy_return_1d',
    'spy_return_21d',
    'spy_vol_21d',
    'rate_2y',
    'rate_10y',
    'yield_slope',
    'credit_spread_proxy',
    'risk_off_proxy',
    'vixy_return_21d',
    'gdelt_avg_tone',
]


class RegimeGRU(nn.Module):
    """
    GRU-based regime classification model.

    Takes a sequence of context features and outputs regime classification.
    Uses GRU to capture temporal patterns in market conditions.
    """

    def __init__(
        self,
        input_dim: int = len(CONTEXT_FEATURES),
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = len(REGIME_LABELS),
        embedding_dim: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention mechanism for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Embedding layer (for downstream use)
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths for packed sequences

        Returns:
            Dict with keys:
                - logits: (batch_size, num_classes)
                - probs: (batch_size, num_classes)
                - embedding: (batch_size, embedding_dim)
                - attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch, seq, hidden)
        x = F.relu(x)

        # GRU encoding
        if lengths is not None:
            # Pack sequences for variable length
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, hidden = self.gru(x_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(x)  # (batch, seq, hidden)

        # Attention-weighted aggregation
        attn_scores = self.attention(gru_out).squeeze(-1)  # (batch, seq)

        if lengths is not None:
            # Mask padding positions
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq)

        # Weighted sum of GRU outputs
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)  # (batch, hidden)

        # Compute embedding
        embedding = self.embedding_layer(context)  # (batch, embedding_dim)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'embedding': embedding,
            'attention_weights': attn_weights
        }

    def predict(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Make prediction for a single input or batch.

        Args:
            x: Input tensor

        Returns:
            Dict with regime_label, regime_probs, regime_embedding
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension

            output = self.forward(x)

            # Get predicted class
            pred_idx = output['probs'].argmax(dim=-1)

            results = []
            for i in range(x.shape[0]):
                probs_dict = {
                    REGIME_LABELS[j]: output['probs'][i, j].item()
                    for j in range(self.num_classes)
                }
                results.append({
                    'regime_label': REGIME_LABELS[pred_idx[i].item()],
                    'regime_probs': probs_dict,
                    'regime_embedding': output['embedding'][i].tolist()
                })

            return results[0] if len(results) == 1 else results


class RegimeTransformer(nn.Module):
    """
    Transformer-based regime classification model.

    Alternative architecture using self-attention for capturing
    complex temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int = len(CONTEXT_FEATURES),
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = len(REGIME_LABELS),
        embedding_dim: int = 8,
        dropout: float = 0.2,
        max_seq_len: int = 63  # ~3 months of trading days
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Embedding layer
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dict with logits, probs, embedding
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch, seq, hidden)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, hidden)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use CLS token output for classification
        cls_output = x[:, 0, :]  # (batch, hidden)

        # Compute embedding
        embedding = self.embedding_layer(cls_output)

        # Classification
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'embedding': embedding
        }

    def predict(self, x: torch.Tensor) -> Dict[str, any]:
        """Make prediction for a single input or batch."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)

            output = self.forward(x)
            pred_idx = output['probs'].argmax(dim=-1)

            results = []
            for i in range(x.shape[0]):
                probs_dict = {
                    REGIME_LABELS[j]: output['probs'][i, j].item()
                    for j in range(self.num_classes)
                }
                results.append({
                    'regime_label': REGIME_LABELS[pred_idx[i].item()],
                    'regime_probs': probs_dict,
                    'regime_embedding': output['embedding'][i].tolist()
                })

            return results[0] if len(results) == 1 else results


def create_regime_model(
    model_type: str = 'gru',
    **kwargs
) -> nn.Module:
    """
    Factory function to create regime model.

    Args:
        model_type: 'gru' or 'transformer'
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    if model_type == 'gru':
        return RegimeGRU(**kwargs)
    elif model_type == 'transformer':
        return RegimeTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
