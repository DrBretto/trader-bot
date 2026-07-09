"""Baseline regime model using deterministic rules."""

import pandas as pd
from typing import Dict, Any, List


# Regime labels
REGIMES = [
    'calm_uptrend',
    'risk_on_trend',
    'risk_off_trend',
    'choppy',
    'high_vol_panic'
]

# Volatility percentile thresholds (calibrated to recent market data)
# These represent typical market conditions 2024-2026
VOL_P40 = 0.08  # 8% annualized (low vol)
VOL_P50 = 0.10  # 10% annualized (normal)
VOL_P85 = 0.14  # 14% annualized (elevated)


def baseline_regime_model(context: pd.Series) -> Dict[str, Any]:
    """
    Deterministic baseline regime classification.

    Uses simple rules based on SPY returns, volatility, credit spreads,
    and VIX proxy to classify the current market regime.

    Args:
        context: Single row from context DataFrame with keys:
            - spy_return_21d
            - spy_vol_21d
            - credit_spread_proxy
            - vixy_return_21d

    Returns:
        {
            'regime_label': str,
            'regime_probs': dict[str, float],
            'regime_embedding': list[float]  # dummy for Phase 1
        }
    """
    # Extract context values with defaults
    spy_21d_ret = context.get('spy_return_21d', 0) or 0
    spy_21d_vol = context.get('spy_vol_21d', 0) or 0
    credit_stress = context.get('credit_spread_proxy', 0) or 0
    vixy_21d_ret = context.get('vixy_return_21d', 0) or 0

    # Classification rules (in priority order)

    # 1. High Vol Panic: VIX spike + high vol + negative returns
    # Calibrated: vixy > 10% (spike) OR vol > P85, AND returns < -2%
    if (vixy_21d_ret > 0.10 or spy_21d_vol > VOL_P85) and spy_21d_ret < -0.02:
        label = 'high_vol_panic'

    # 2. Risk-Off Trend: Negative returns or credit stress
    # Calibrated: returns < -1% OR credit stress < -1%
    elif spy_21d_ret < -0.01 or credit_stress < -0.01:
        label = 'risk_off_trend'

    # 3. Calm Uptrend: Strong positive returns + low vol
    # Calibrated: returns > 3% AND vol < P40
    elif spy_21d_ret > 0.03 and spy_21d_vol < VOL_P40:
        label = 'calm_uptrend'

    # 4. Choppy: Low absolute returns + elevated vol
    # Calibrated: |returns| < 1% AND vol > P50
    elif abs(spy_21d_ret) < 0.01 and spy_21d_vol > VOL_P50:
        label = 'choppy'

    # 5. Default: Risk-On Trend
    else:
        label = 'risk_on_trend'

    # Generate probabilities (deterministic for baseline)
    probs = {regime: 0.0 for regime in REGIMES}
    probs[label] = 1.0

    # Dummy embedding vector (8 dimensions, will be real in Phase 2)
    embedding = [0.0] * 8

    return {
        'regime_label': label,
        'regime_probs': probs,
        'regime_embedding': embedding
    }


def get_regime_description(regime_label: str) -> str:
    """Get human-readable description of a regime."""
    descriptions = {
        'calm_uptrend': 'Calm uptrend with low volatility - favor equities',
        'risk_on_trend': 'Risk-on trend - moderate equity allocation',
        'risk_off_trend': 'Risk-off trend - favor bonds, reduce equities',
        'choppy': 'Choppy/sideways market - reduce position sizes',
        'high_vol_panic': 'High volatility panic - defensive posture'
    }
    return descriptions.get(regime_label, 'Unknown regime')


def get_regime_risk_level(regime_label: str) -> int:
    """Get risk level (1-5) for a regime."""
    risk_levels = {
        'calm_uptrend': 1,
        'risk_on_trend': 2,
        'choppy': 3,
        'risk_off_trend': 4,
        'high_vol_panic': 5
    }
    return risk_levels.get(regime_label, 3)
