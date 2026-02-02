"""
Ensemble regime classification with disagreement detection.

Loads both GRU and Transformer regime models, runs inference on both,
and computes ensemble prediction with agreement/disagreement metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import pickle


class EnsembleRegimeModel:
    """
    Ensemble of GRU and Transformer regime models.

    Combines predictions from both models and tracks disagreement
    as an uncertainty signal.
    """

    def __init__(
        self,
        gru_model=None,
        transformer_model=None,
        gru_weight: float = 0.5,
        transformer_weight: float = 0.5,
        disagreement_threshold: float = 0.3,
        device: str = 'auto'
    ):
        self.gru_model = gru_model
        self.transformer_model = transformer_model
        self.gru_weight = gru_weight
        self.transformer_weight = transformer_weight
        self.disagreement_threshold = disagreement_threshold

        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Move models to device if they exist
        if self.gru_model is not None:
            self.gru_model = self.gru_model.to(self.device)
            self.gru_model.eval()
        if self.transformer_model is not None:
            self.transformer_model = self.transformer_model.to(self.device)
            self.transformer_model.eval()

    @classmethod
    def load_from_files(
        cls,
        gru_path: str,
        transformer_path: str,
        device: str = 'auto'
    ) -> 'EnsembleRegimeModel':
        """Load ensemble from saved model files."""
        with open(gru_path, 'rb') as f:
            gru_data = pickle.load(f)

        with open(transformer_path, 'rb') as f:
            transformer_data = pickle.load(f)

        return cls(
            gru_model=gru_data['model'],
            transformer_model=transformer_data['model'],
            device=device
        )

    def compute_disagreement(
        self,
        probs_gru: torch.Tensor,
        probs_transformer: torch.Tensor
    ) -> float:
        """
        Compute disagreement between models using cosine similarity.

        Returns:
            Disagreement score (0 = perfect agreement, 1 = complete disagreement)
        """
        # Flatten if needed
        p1 = probs_gru.flatten()
        p2 = probs_transformer.flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()

        # Convert to disagreement (0 = agreement, 1 = disagreement)
        return 1.0 - cos_sim

    def predict(self, context_data: torch.Tensor) -> Dict:
        """
        Make ensemble prediction.

        Args:
            context_data: Input tensor (batch_size, seq_len, features) or dict/Series

        Returns:
            Dict with:
                - regime_label: Ensemble predicted regime
                - regime_probs: Ensemble probabilities
                - regime_embedding: Ensemble embedding (average)
                - gru_prediction: GRU model prediction
                - transformer_prediction: Transformer model prediction
                - disagreement: Disagreement score (0-1)
                - confidence: Ensemble confidence
                - position_size_multiplier: Suggested sizing adjustment
        """
        # Handle different input types
        if isinstance(context_data, dict):
            context_data = self._dict_to_tensor(context_data)
        elif not isinstance(context_data, torch.Tensor) and hasattr(context_data, 'get'):  # pandas Series
            context_data = self._series_to_tensor(context_data)

        # Ensure tensor is on device
        if isinstance(context_data, torch.Tensor):
            context_data = context_data.to(self.device)

            # Add batch/sequence dimensions if needed
            if context_data.dim() == 1:
                context_data = context_data.unsqueeze(0).unsqueeze(0)
            elif context_data.dim() == 2:
                context_data = context_data.unsqueeze(0)

        # Get predictions from both models
        with torch.no_grad():
            if self.gru_model is not None:
                gru_output = self.gru_model(context_data)
                gru_probs = gru_output['probs'][0]
                gru_embedding = gru_output['embedding'][0]
            else:
                gru_probs = None
                gru_embedding = None

            if self.transformer_model is not None:
                trans_output = self.transformer_model(context_data)
                trans_probs = trans_output['probs'][0]
                trans_embedding = trans_output['embedding'][0]
            else:
                trans_probs = None
                trans_embedding = None

        # Compute ensemble prediction
        if gru_probs is not None and trans_probs is not None:
            # Weighted average of probabilities
            ensemble_probs = (
                self.gru_weight * gru_probs +
                self.transformer_weight * trans_probs
            )

            # Average embeddings
            ensemble_embedding = (gru_embedding + trans_embedding) / 2

            # Compute disagreement
            disagreement = self.compute_disagreement(gru_probs, trans_probs)

        elif gru_probs is not None:
            ensemble_probs = gru_probs
            ensemble_embedding = gru_embedding
            disagreement = 0.0
        elif trans_probs is not None:
            ensemble_probs = trans_probs
            ensemble_embedding = trans_embedding
            disagreement = 0.0
        else:
            raise ValueError("No models loaded")

        # Get regime labels
        from training.models.regime_transformer import REGIME_LABELS

        # Ensemble prediction
        ensemble_idx = ensemble_probs.argmax().item()
        ensemble_label = REGIME_LABELS[ensemble_idx]
        ensemble_confidence = ensemble_probs[ensemble_idx].item()

        # Individual predictions
        gru_label = None
        gru_confidence = 0.0
        if gru_probs is not None:
            gru_idx = gru_probs.argmax().item()
            gru_label = REGIME_LABELS[gru_idx]
            gru_confidence = gru_probs[gru_idx].item()

        trans_label = None
        trans_confidence = 0.0
        if trans_probs is not None:
            trans_idx = trans_probs.argmax().item()
            trans_label = REGIME_LABELS[trans_idx]
            trans_confidence = trans_probs[trans_idx].item()

        # Compute position size multiplier based on confidence and agreement
        position_size_multiplier = self._compute_position_multiplier(
            ensemble_confidence, disagreement
        )

        return {
            'regime_label': ensemble_label,
            'regime_probs': {
                REGIME_LABELS[i]: ensemble_probs[i].item()
                for i in range(len(REGIME_LABELS))
            },
            'regime_embedding': ensemble_embedding.cpu().tolist() if ensemble_embedding is not None else [],
            'confidence': ensemble_confidence,
            'disagreement': disagreement,
            'agreement': 1.0 - disagreement,
            'position_size_multiplier': position_size_multiplier,
            'gru_prediction': {
                'label': gru_label,
                'confidence': gru_confidence,
                'probs': {
                    REGIME_LABELS[i]: gru_probs[i].item()
                    for i in range(len(REGIME_LABELS))
                } if gru_probs is not None else {}
            },
            'transformer_prediction': {
                'label': trans_label,
                'confidence': trans_confidence,
                'probs': {
                    REGIME_LABELS[i]: trans_probs[i].item()
                    for i in range(len(REGIME_LABELS))
                } if trans_probs is not None else {}
            }
        }

    def _compute_position_multiplier(
        self,
        confidence: float,
        disagreement: float
    ) -> float:
        """
        Compute position size multiplier based on model confidence and agreement.

        Returns:
            Multiplier between 0.5 and 1.0
        """
        # Base multiplier from confidence
        conf_mult = 0.5 + 0.5 * confidence

        # Reduce if models disagree
        if disagreement > self.disagreement_threshold:
            # Scale down based on how much disagreement exceeds threshold
            disagreement_penalty = (disagreement - self.disagreement_threshold) / (1 - self.disagreement_threshold)
            conf_mult *= (1 - 0.5 * disagreement_penalty)

        return max(0.5, min(1.0, conf_mult))

    def _dict_to_tensor(self, data: dict) -> torch.Tensor:
        """Convert dict to tensor."""
        from training.models.regime_transformer import CONTEXT_FEATURES
        values = [data.get(f, 0) for f in CONTEXT_FEATURES]
        return torch.tensor(values, dtype=torch.float32)

    def _series_to_tensor(self, series) -> torch.Tensor:
        """Convert pandas Series to tensor."""
        from training.models.regime_transformer import CONTEXT_FEATURES
        values = [series.get(f, 0) for f in CONTEXT_FEATURES]
        return torch.tensor(values, dtype=torch.float32)


def baseline_ensemble_regime(context_data) -> Dict:
    """
    Baseline ensemble regime using deterministic rules.

    Used when ML models aren't available. Returns same format as ensemble.
    """
    from src.models.baseline_regime import baseline_regime_model

    result = baseline_regime_model(context_data)

    return {
        'regime_label': result['regime_label'],
        'regime_probs': result['regime_probs'],
        'regime_embedding': result['regime_embedding'],
        'confidence': 1.0,  # Baseline is always "confident"
        'disagreement': 0.0,
        'agreement': 1.0,
        'position_size_multiplier': 1.0,
        'gru_prediction': {
            'label': result['regime_label'],
            'confidence': 1.0,
            'probs': result['regime_probs']
        },
        'transformer_prediction': {
            'label': result['regime_label'],
            'confidence': 1.0,
            'probs': result['regime_probs']
        },
        'is_baseline': True
    }
