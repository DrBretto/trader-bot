"""Macro & Credit Regime Score expert module.

Produces a macro_credit_score in [-1, +1] from yield curve slope and
high-yield credit spread proxy. Positive = favorable macro backdrop,
negative = deteriorating.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# Historical norms for z-score computation (10Y-3M spread)
# Long-run mean ~1.5%, std ~1.0% based on 1982-2024 data
SLOPE_MEAN = 1.50
SLOPE_STD = 1.00

# HY spread proxy: HYG-IEF 21d return difference
# Centered near 0, std ~2% annualized
HY_SPREAD_MEAN = 0.0
HY_SPREAD_STD = 0.02


def compute_macro_credit(
    rate_10y: float,
    rate_3m: float,
    hyg_prices: Optional[pd.Series] = None,
    ief_prices: Optional[pd.Series] = None,
    rate_2y: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute macro/credit regime score.

    Args:
        rate_10y: 10-Year Treasury yield
        rate_3m: 3-Month Treasury yield (0 if unavailable)
        hyg_prices: Recent HYG close prices (Series indexed by date)
        ief_prices: Recent IEF close prices (Series indexed by date)
        rate_2y: 2-Year Treasury yield (fallback if 3M unavailable)

    Returns:
        Dict with macro_credit_score and diagnostics.
    """
    # Yield curve slope
    if rate_3m and rate_3m != 0:
        yield_slope = rate_10y - rate_3m
    elif rate_2y and rate_2y != 0:
        yield_slope = rate_10y - rate_2y
    else:
        yield_slope = 0.0

    slope_z = (yield_slope - SLOPE_MEAN) / SLOPE_STD if SLOPE_STD > 0 else 0.0

    # HY spread proxy from ETF returns
    hy_spread_proxy = 0.0
    if hyg_prices is not None and ief_prices is not None:
        if len(hyg_prices) >= 21 and len(ief_prices) >= 21:
            hyg_ret = (hyg_prices.iloc[-1] / hyg_prices.iloc[-21]) - 1.0
            ief_ret = (ief_prices.iloc[-1] / ief_prices.iloc[-21]) - 1.0
            hy_spread_proxy = hyg_ret - ief_ret

    hy_spread_z = (
        (hy_spread_proxy - HY_SPREAD_MEAN) / HY_SPREAD_STD
        if HY_SPREAD_STD > 0 else 0.0
    )

    # Combine: positive slope + tight spreads = bullish
    score = 0.6 * np.tanh(slope_z) + 0.4 * np.tanh(hy_spread_z)
    score = float(np.clip(score, -1.0, 1.0))

    return {
        'macro_credit_score': score,
        'yield_slope_10y_3m': yield_slope,
        'hy_spread_proxy': hy_spread_proxy,
        'slope_z_score': slope_z,
        'hy_spread_z_score': hy_spread_z,
    }
