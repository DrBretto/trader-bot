"""Cross-Asset Fragility Score expert module.

Produces fragility_score in [0, 1] from rolling cross-asset correlation
and PCA absorption. High fragility = markets tightly coupled, shocks
propagate quickly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


PANEL_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'EFA', 'EEM']
MIN_SYMBOLS = 6
MIN_DAYS = 40

# Historical norms for normalization.
#
# These constants are the **production defaults**. They are kept at their
# original values for backward compatibility; operator opt-in to the
# 2026-04-30 recalibration is done by promoting a decision-params bundle
# whose `signals.fragility` block sets the values from RECALIBRATED_2026_04_30
# below (see `config/decision_params.recalibrated_2026_04_30.json`). See
# `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md` for the
# empirical-re-derivation evidence and `docs/POSTMORTEMS.md` →
# "stale-historical-norm" for the failure mode this gate exists to prevent.
#
# Original provenance: introduced in commit b9317e1 (2026-02-06) without a
# documented source dataset; values approximated as "Average pairwise
# correlation: mean ~0.30, std ~0.15" / "PC1 explained variance: mean ~0.45,
# std ~0.12". Re-validation against 900-day empirical (2022-09 → 2026-04)
# panel close prices found the originals to be at the empirical 5th
# percentile (avg_corr_mean) and below the empirical minimum (pc1_mean).
AVG_CORR_MEAN = 0.30
AVG_CORR_STD = 0.15
PC1_MEAN = 0.45
PC1_STD = 0.12

# Empirical 900-day re-derivation, panel = PANEL_SYMBOLS, window = 60 trading
# days, source = yfinance closes 2022-09 → 2026-04. Cross-validated against
# production signals.parquet to ±0.005 absolute. See Phase 2 audit doc.
# Operator promotes to active by deploying
# config/decision_params.recalibrated_2026_04_30.json or by setting these
# values in the runtime params bundle's `signals.fragility` block.
RECALIBRATED_2026_04_30 = {
    'AVG_CORR_MEAN': 0.4774,
    'AVG_CORR_STD': 0.0979,
    'PC1_MEAN': 0.6007,
    'PC1_STD': 0.0658,
    'source_dataset': 'yfinance close, PANEL_SYMBOLS, 900 trading days ending 2026-04-29',
    'derivation_date': '2026-04-30',
    're_validation_cadence': 'quarterly; see test_fragility_calibration.py',
}


def compute_fragility(
    prices_df: pd.DataFrame,
    panel_symbols: Optional[List[str]] = None,
    window: int = 60,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute cross-asset fragility score.

    Args:
        prices_df: Multi-symbol daily prices with columns [date, symbol, close]
                   or [date, symbol, open, high, low, close, volume]
        panel_symbols: List of symbols to use (default: PANEL_SYMBOLS)
        window: Rolling window in trading days (default: 60)

    Returns:
        Dict with fragility_score and diagnostics.
    """
    params = params or {}
    min_symbols = int(params.get('MIN_SYMBOLS', MIN_SYMBOLS))
    min_days = int(params.get('MIN_DAYS', MIN_DAYS))
    avg_corr_mean = float(params.get('AVG_CORR_MEAN', AVG_CORR_MEAN))
    avg_corr_std = float(params.get('AVG_CORR_STD', AVG_CORR_STD))
    pc1_mean = float(params.get('PC1_MEAN', PC1_MEAN))
    pc1_std = float(params.get('PC1_STD', PC1_STD))
    if 'window_days' in params:
        window = int(params.get('window_days', window))

    if panel_symbols is None:
        panel_symbols = PANEL_SYMBOLS

    # Pivot to wide format: date x symbol close prices
    available = prices_df[prices_df['symbol'].isin(panel_symbols)].copy()
    available['date'] = pd.to_datetime(available['date'])

    if 'close' not in available.columns:
        return _neutral_result('No close column in prices_df')

    pivot = available.pivot_table(index='date', columns='symbol', values='close')
    pivot = pivot.sort_index()

    # Check minimum data requirements
    valid_symbols = pivot.columns[pivot.tail(window).notna().sum() >= min_days]
    if len(valid_symbols) < min_symbols:
        return _neutral_result(
            f'Insufficient symbols: {len(valid_symbols)} < {min_symbols}'
        )

    # Compute daily returns for the window
    returns = pivot[valid_symbols].tail(window + 1).pct_change().dropna()

    if len(returns) < min_days:
        return _neutral_result(f'Insufficient return history: {len(returns)} < {min_days}')

    # Correlation matrix
    corr_matrix = returns.corr().values
    n = corr_matrix.shape[0]

    # Average pairwise correlation (upper triangle, excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    avg_correlation = float(np.mean(upper_tri))

    # PCA via eigendecomposition of correlation matrix
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    total_var = eigenvalues.sum()

    if total_var > 0:
        pc1_explained = float(eigenvalues[0] / total_var)
        pc2_explained = float(eigenvalues[1] / total_var) if len(eigenvalues) > 1 else 0.0
    else:
        pc1_explained = 0.0
        pc2_explained = 0.0

    # Normalize to [0, 1] using tanh of z-scores
    corr_z = (avg_correlation - avg_corr_mean) / avg_corr_std if avg_corr_std > 0 else 0.0
    pc1_z = (pc1_explained - pc1_mean) / pc1_std if pc1_std > 0 else 0.0

    # Higher correlation and higher PC1 absorption = more fragile
    norm_corr = (np.tanh(corr_z) + 1) / 2  # map tanh [-1,1] to [0,1]
    norm_pc1 = (np.tanh(pc1_z) + 1) / 2

    fragility_score = float(np.clip(0.5 * norm_corr + 0.5 * norm_pc1, 0.0, 1.0))

    return {
        'fragility_score': fragility_score,
        'avg_correlation': avg_correlation,
        'pc1_explained': pc1_explained,
        'pc2_explained': pc2_explained,
        'symbols_used': len(valid_symbols),
    }


def _neutral_result(reason: str) -> Dict[str, Any]:
    """Return neutral fragility result when computation is not possible."""
    return {
        'fragility_score': 0.5,
        'avg_correlation': 0.0,
        'pc1_explained': 0.0,
        'pc2_explained': 0.0,
        'symbols_used': 0,
        'degraded_reason': reason,
    }
