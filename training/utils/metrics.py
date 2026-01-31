"""
Evaluation metrics for training.

Provides metrics for both regime classification and health scoring models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


class RegimeMetrics:
    """Metrics for regime classification model."""

    def __init__(self, regime_labels: List[str]):
        self.regime_labels = regime_labels
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update metrics with batch results.

        Args:
            predictions: Model output logits or probs (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            loss: Optional batch loss
        """
        if predictions.dim() > 1:
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions

        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(targets, preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(targets, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
        }

        if self.losses:
            metrics['loss'] = np.mean(self.losses)

        # Per-class metrics
        for i, label in enumerate(self.regime_labels):
            class_mask = targets == i
            if class_mask.sum() > 0:
                class_preds = preds[class_mask]
                class_targets = targets[class_mask]
                metrics[f'accuracy_{label}'] = (class_preds == class_targets).mean()

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(
            self.targets,
            self.predictions,
            labels=list(range(len(self.regime_labels)))
        )

    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.regime_labels,
            zero_division=0
        )


class HealthMetrics:
    """Metrics for health scoring autoencoder."""

    def __init__(self, vol_buckets: List[str] = None, behaviors: List[str] = None):
        self.vol_buckets = vol_buckets or ['low', 'med', 'high']
        self.behaviors = behaviors or ['momentum', 'mean_reversion', 'mixed']
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.reconstruction_errors = []
        self.health_predictions = []
        self.health_targets = []
        self.vol_predictions = []
        self.vol_targets = []
        self.behavior_predictions = []
        self.behavior_targets = []
        self.losses = defaultdict(list)

    def update(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        health_pred: Optional[torch.Tensor] = None,
        health_target: Optional[torch.Tensor] = None,
        vol_pred: Optional[torch.Tensor] = None,
        vol_target: Optional[torch.Tensor] = None,
        behavior_pred: Optional[torch.Tensor] = None,
        behavior_target: Optional[torch.Tensor] = None,
        losses: Optional[Dict[str, float]] = None
    ):
        """Update metrics with batch results."""
        # Reconstruction error
        recon_error = torch.mean((reconstruction - original) ** 2, dim=-1)
        self.reconstruction_errors.extend(recon_error.detach().cpu().numpy().tolist())

        # Health score
        if health_pred is not None:
            self.health_predictions.extend(health_pred.detach().cpu().numpy().flatten().tolist())
        if health_target is not None:
            self.health_targets.extend(health_target.detach().cpu().numpy().flatten().tolist())

        # Volatility bucket
        if vol_pred is not None:
            if vol_pred.dim() > 1:
                preds = vol_pred.argmax(dim=-1)
            else:
                preds = vol_pred
            self.vol_predictions.extend(preds.detach().cpu().numpy().tolist())
        if vol_target is not None:
            self.vol_targets.extend(vol_target.detach().cpu().numpy().tolist())

        # Behavior
        if behavior_pred is not None:
            if behavior_pred.dim() > 1:
                preds = behavior_pred.argmax(dim=-1)
            else:
                preds = behavior_pred
            self.behavior_predictions.extend(preds.cpu().numpy().tolist())
        if behavior_target is not None:
            self.behavior_targets.extend(behavior_target.cpu().numpy().tolist())

        # Losses
        if losses:
            for k, v in losses.items():
                self.losses[k].append(v)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {
            'reconstruction_mse': np.mean(self.reconstruction_errors),
            'reconstruction_std': np.std(self.reconstruction_errors),
        }

        # Health score metrics
        if self.health_predictions and self.health_targets:
            preds = np.array(self.health_predictions)
            targets = np.array(self.health_targets)
            metrics['health_mse'] = np.mean((preds - targets) ** 2)
            metrics['health_mae'] = np.mean(np.abs(preds - targets))
            metrics['health_correlation'] = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0

        # Volatility bucket metrics
        if self.vol_predictions and self.vol_targets:
            preds = np.array(self.vol_predictions)
            targets = np.array(self.vol_targets)
            metrics['vol_accuracy'] = accuracy_score(targets, preds)
            metrics['vol_f1'] = f1_score(targets, preds, average='macro', zero_division=0)

        # Behavior metrics
        if self.behavior_predictions and self.behavior_targets:
            preds = np.array(self.behavior_predictions)
            targets = np.array(self.behavior_targets)
            metrics['behavior_accuracy'] = accuracy_score(targets, preds)
            metrics['behavior_f1'] = f1_score(targets, preds, average='macro', zero_division=0)

        # Aggregate losses
        for k, v in self.losses.items():
            metrics[f'loss_{k}'] = np.mean(v)

        return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            is_better = self.best_score is None or score < self.best_score - self.min_delta
        else:
            is_better = self.best_score is None or score > self.best_score + self.min_delta

        if is_better:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def compute_regime_labels_from_baseline(context_df) -> np.ndarray:
    """
    Generate regime labels using baseline model logic.

    This creates pseudo-labels for self-supervised or semi-supervised training.
    """
    labels = []

    vol_p85 = 0.25
    vol_p50 = 0.18
    vol_p40 = 0.15

    for _, row in context_df.iterrows():
        spy_21d_ret = row.get('spy_return_21d', 0)
        spy_21d_vol = row.get('spy_vol_21d', 0.15)
        credit_stress = row.get('credit_spread_proxy', 0)
        vixy_21d_ret = row.get('vixy_return_21d', 0)

        if (vixy_21d_ret > 0.20 or spy_21d_vol > vol_p85) and spy_21d_ret < -0.08:
            label = 'high_vol_panic'
        elif spy_21d_ret < -0.05 or credit_stress < -0.03:
            label = 'risk_off_trend'
        elif spy_21d_ret > 0.06 and spy_21d_vol < vol_p40:
            label = 'calm_uptrend'
        elif abs(spy_21d_ret) < 0.02 and spy_21d_vol > vol_p50:
            label = 'choppy'
        else:
            label = 'risk_on_trend'

        labels.append(label)

    return np.array(labels)


def compute_health_labels_from_baseline(features_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate health labels using baseline model logic.

    Returns:
        (health_scores, vol_buckets, behaviors)
    """
    health_scores = []
    vol_buckets = []
    behaviors = []

    # Group by date for cross-sectional ranking
    for date, group in features_df.groupby('date'):
        # Compute ranks
        mom_63_rank = group['return_63d'].rank(pct=True) if 'return_63d' in group.columns else 0.5
        mom_21_rank = group['return_21d'].rank(pct=True) if 'return_21d' in group.columns else 0.5
        vol_63_rank = group['vol_63d'].rank(pct=True) if 'vol_63d' in group.columns else 0.5
        dd_63_rank = (-group['drawdown_63d']).rank(pct=True) if 'drawdown_63d' in group.columns else 0.5
        rs_63_rank = group['rel_strength_63d'].rank(pct=True) if 'rel_strength_63d' in group.columns else 0.5

        # Composite scores
        momentum = 0.6 * mom_63_rank + 0.4 * mom_21_rank
        risk = 0.6 * vol_63_rank + 0.4 * (1 - dd_63_rank)

        # Health score
        health = (0.45 * momentum + 0.35 * rs_63_rank + 0.20 * (1 - risk)).clip(0, 1)
        health_scores.extend(health.tolist())

        # Vol bucket
        vol_bucket = np.digitize(vol_63_rank, [0.33, 0.67])
        vol_buckets.extend(vol_bucket.tolist())

        # Behavior
        behavior = np.where(
            (mom_63_rank > 0.66) & (dd_63_rank > 0.5),
            0,  # momentum
            2   # mixed
        )
        behaviors.extend(behavior.tolist())

    return np.array(health_scores), np.array(vol_buckets), np.array(behaviors)
