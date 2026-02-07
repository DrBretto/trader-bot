"""Entropy / Distribution-Shift Score expert module.

Produces entropy_score and entropy_shift_flag from rolling return
histogram analysis. High entropy = return distribution is changing,
patterns may be breaking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def compute_entropy_shift(
    spy_returns: pd.Series,
    window: int = 60,
    num_bins: int = 20,
    z_threshold: float = 1.5,
    consecutive_days_required: int = 3,
    prev_consecutive_days: int = 0,
    prev_above_threshold: bool = False,
) -> Dict[str, Any]:
    """
    Compute entropy-based distribution shift detection.

    Args:
        spy_returns: SPY daily returns, at least `window` observations.
        window: Rolling window for histogram (default: 60 trading days).
        num_bins: Number of bins for return histogram (default: 20).
        z_threshold: Z-score threshold for shift detection (default: 1.5).
        consecutive_days_required: Days above threshold to trigger flag.
        prev_consecutive_days: Yesterday's consecutive-days counter.
        prev_above_threshold: Whether yesterday was above threshold.

    Returns:
        Dict with entropy_score, entropy_shift_flag, and diagnostics.
    """
    if spy_returns is None or len(spy_returns) < window:
        return _neutral_result('Insufficient SPY return history')

    spy_returns = spy_returns.dropna()
    if len(spy_returns) < window:
        return _neutral_result('Insufficient non-null SPY returns')

    # Current window returns
    current_returns = spy_returns.iloc[-window:]

    # Compute Shannon entropy of return distribution
    entropy = _shannon_entropy(current_returns.values, num_bins)
    max_entropy = np.log(num_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Compute rolling entropy for z-score
    # Use all available data for a robust z-score
    lookback = min(len(spy_returns), 252)
    rolling_entropies = []
    for i in range(window, lookback + 1):
        start = max(0, len(spy_returns) - lookback + i - window)
        end = start + window
        if end <= len(spy_returns):
            chunk = spy_returns.iloc[start:end].values
            rolling_entropies.append(_shannon_entropy(chunk, num_bins))

    if len(rolling_entropies) >= 10:
        ent_mean = np.mean(rolling_entropies)
        ent_std = np.std(rolling_entropies)
        entropy_z_score = (entropy - ent_mean) / ent_std if ent_std > 0 else 0.0
    else:
        entropy_z_score = 0.0

    # Update consecutive-days counter
    above_threshold = abs(entropy_z_score) > z_threshold
    if above_threshold and prev_above_threshold:
        consecutive_days = prev_consecutive_days + 1
    elif above_threshold:
        consecutive_days = 1
    else:
        consecutive_days = 0

    entropy_shift_flag = consecutive_days >= consecutive_days_required

    return {
        'entropy_score': float(normalized_entropy),
        'entropy_z_score': float(entropy_z_score),
        'entropy_shift_flag': bool(entropy_shift_flag),
        'entropy_consecutive_days': int(consecutive_days),
        'entropy_above_threshold': bool(above_threshold),
    }


def _shannon_entropy(values: np.ndarray, num_bins: int) -> float:
    """Compute Shannon entropy of a distribution using histogram binning."""
    counts, _ = np.histogram(values, bins=num_bins)
    probs = counts / counts.sum()
    # Filter zero-probability bins
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _neutral_result(reason: str) -> Dict[str, Any]:
    """Return neutral entropy result when computation is not possible."""
    return {
        'entropy_score': 0.5,
        'entropy_z_score': 0.0,
        'entropy_shift_flag': False,
        'entropy_consecutive_days': 0,
        'entropy_above_threshold': False,
        'degraded_reason': reason,
    }
