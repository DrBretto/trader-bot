"""Cross-sectional ranking evaluation metrics.

Evaluates a ranking model's predictions against realized forward returns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


def compute_ranking_metrics(
    predicted_scores: Dict[str, float],
    realized_returns: Dict[str, float],
) -> Dict[str, Any]:
    """Compare predicted ranking scores against realized forward returns.

    Args:
        predicted_scores: {symbol: predicted_score} — higher = better expected
        realized_returns: {symbol: realized_forward_return}

    Returns:
        Dictionary with ranking quality metrics.
    """
    common = sorted(set(predicted_scores) & set(realized_returns))
    if len(common) < 5:
        return {
            'spearman_rho': 0.0,
            'spearman_p': 1.0,
            'top_quintile_return': 0.0,
            'bottom_quintile_return': 0.0,
            'quintile_spread': 0.0,
            'n_assets': len(common),
            'valid': False,
        }

    pred = np.array([predicted_scores[s] for s in common])
    real = np.array([realized_returns[s] for s in common])

    # Spearman rank correlation
    rho, p_value = stats.spearmanr(pred, real)

    # Quintile analysis
    n = len(common)
    q_size = max(1, n // 5)

    # Sort by predicted score (descending)
    order = np.argsort(-pred)
    top_idx = order[:q_size]
    bottom_idx = order[-q_size:]

    top_return = float(np.mean(real[top_idx]))
    bottom_return = float(np.mean(real[bottom_idx]))
    spread = top_return - bottom_return

    return {
        'spearman_rho': float(rho) if np.isfinite(rho) else 0.0,
        'spearman_p': float(p_value) if np.isfinite(p_value) else 1.0,
        'top_quintile_return': top_return,
        'bottom_quintile_return': bottom_return,
        'quintile_spread': spread,
        'n_assets': n,
        'valid': True,
    }


def aggregate_ranking_metrics(
    per_date_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-date ranking metrics across a walk-forward window."""
    valid = [m for m in per_date_metrics if m.get('valid')]
    if not valid:
        return {
            'mean_spearman_rho': 0.0,
            'mean_quintile_spread': 0.0,
            'hit_rate': 0.0,
            'n_dates': 0,
            'information_ratio': 0.0,
        }

    rhos = [m['spearman_rho'] for m in valid]
    spreads = [m['quintile_spread'] for m in valid]

    mean_rho = float(np.mean(rhos))
    mean_spread = float(np.mean(spreads))
    hit_rate = float(np.mean([1.0 if s > 0 else 0.0 for s in spreads]))

    std_spread = float(np.std(spreads)) if len(spreads) > 1 else 1.0
    ir = mean_spread / std_spread if std_spread > 1e-8 else 0.0

    return {
        'mean_spearman_rho': mean_rho,
        'mean_quintile_spread': mean_spread,
        'hit_rate': hit_rate,
        'n_dates': len(valid),
        'information_ratio': ir,
    }
