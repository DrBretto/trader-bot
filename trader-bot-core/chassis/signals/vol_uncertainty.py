"""Vol-of-Vol / Options Uncertainty Score expert module.

Produces vol_uncertainty_score in [0, 1] and vol_regime_label from
VIX, VVIX, and SKEW index data. Detects "panic before panic" via
options-market stress signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# Hardcoded percentile thresholds (based on historical distributions)
# VIX: median ~17, 80th ~25, 95th ~30
VIX_THRESHOLDS = {'p20': 13, 'p50': 17, 'p80': 25, 'p95': 30}
# VVIX: median ~85, 80th ~105, 95th ~120
VVIX_THRESHOLDS = {'p20': 75, 'p50': 85, 'p80': 105, 'p95': 120}
# SKEW: median ~125, low ~110 (complacent), high ~145 (tail fear)
SKEW_THRESHOLDS = {'p20': 115, 'p50': 125, 'p80': 140, 'p95': 150}


def _percentile_score(value: float, thresholds: dict) -> float:
    """Convert a value to a 0-1 percentile score using threshold breakpoints."""
    if value <= thresholds['p20']:
        return 0.1
    elif value <= thresholds['p50']:
        return 0.2 + 0.3 * (value - thresholds['p20']) / (thresholds['p50'] - thresholds['p20'])
    elif value <= thresholds['p80']:
        return 0.5 + 0.3 * (value - thresholds['p50']) / (thresholds['p80'] - thresholds['p50'])
    elif value <= thresholds['p95']:
        return 0.8 + 0.15 * (value - thresholds['p80']) / (thresholds['p95'] - thresholds['p80'])
    else:
        return 0.95


def compute_vol_uncertainty(
    vix: float,
    vvix: Optional[float] = None,
    skew: Optional[float] = None,
    vix_history: Optional[pd.Series] = None,
    vvix_history: Optional[pd.Series] = None,
    skew_history: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute volatility uncertainty score and regime label.

    Args:
        vix: Current VIX value
        vvix: Current VVIX value (None if unavailable)
        skew: Current SKEW value (None if unavailable)
        vix_history: Historical VIX values for dynamic percentile (optional)
        vvix_history: Historical VVIX values for dynamic percentile (optional)

    Returns:
        Dict with vol_uncertainty_score, vol_regime_label, and diagnostics.
    """
    params = params or {}
    vix_thresholds = params.get('vix_thresholds', VIX_THRESHOLDS)
    vvix_thresholds = params.get('vvix_thresholds', VVIX_THRESHOLDS)
    skew_thresholds = params.get('skew_thresholds', SKEW_THRESHOLDS)
    dynamic_history_min_obs = int(params.get('dynamic_history_min_obs', 60))
    regime_thresholds = params.get('regime_thresholds', {})
    panic_thresholds = regime_thresholds.get('panic', {})
    unstable_thresholds = regime_thresholds.get('unstable', {})
    panic_vix_pctile = float(panic_thresholds.get('vix_pctile', 0.80))
    panic_vvix_pctile = float(panic_thresholds.get('vvix_pctile', 0.80))
    unstable_vvix_pctile = float(unstable_thresholds.get('vvix_pctile', 0.80))
    unstable_vix_pctile_max = float(unstable_thresholds.get('vix_pctile_max', 0.60))

    # Compute percentiles
    if vix_history is not None and len(vix_history) >= dynamic_history_min_obs:
        vix_pctile = float((vix_history < vix).mean())
    else:
        vix_pctile = _percentile_score(vix, vix_thresholds)

    if vvix is not None:
        if vvix_history is not None and len(vvix_history) >= dynamic_history_min_obs:
            vvix_pctile = float((vvix_history < vvix).mean())
        else:
            vvix_pctile = _percentile_score(vvix, vvix_thresholds)
    else:
        vvix_pctile = 0.5  # neutral if unavailable

    if skew is not None:
        if skew_history is not None and len(skew_history) >= dynamic_history_min_obs:
            skew_pctile = float((skew_history < skew).mean())
        else:
            skew_pctile = _percentile_score(skew, skew_thresholds)
    else:
        skew_pctile = 0.5  # neutral if unavailable

    # Composite score (reweight based on availability)
    if vvix is not None and skew is not None:
        score = 0.45 * vix_pctile + 0.35 * vvix_pctile + 0.20 * skew_pctile
    elif vvix is not None:
        score = 0.55 * vix_pctile + 0.45 * vvix_pctile
    elif skew is not None:
        score = 0.65 * vix_pctile + 0.35 * skew_pctile
    else:
        score = vix_pctile

    score = float(np.clip(score, 0.0, 1.0))

    # Regime label
    if vix_pctile > panic_vix_pctile and vvix_pctile > panic_vvix_pctile:
        vol_regime_label = 'panic'
    elif vvix_pctile > unstable_vvix_pctile and vix_pctile < unstable_vix_pctile_max:
        vol_regime_label = 'unstable_calm'
    else:
        vol_regime_label = 'calm'

    return {
        'vol_uncertainty_score': score,
        'vol_regime_label': vol_regime_label,
        'vix_percentile': vix_pctile,
        'vvix_percentile': vvix_pctile,
        'skew_percentile': skew_pctile,
        'vix_value': vix,
        'vvix_value': vvix if vvix is not None else 0.0,
        'skew_value': skew if skew is not None else 0.0,
    }
