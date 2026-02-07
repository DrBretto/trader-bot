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

# Historical norms for normalization
# Average pairwise correlation: mean ~0.30, std ~0.15
AVG_CORR_MEAN = 0.30
AVG_CORR_STD = 0.15
# PC1 explained variance: mean ~0.45, std ~0.12
PC1_MEAN = 0.45
PC1_STD = 0.12


def compute_fragility(
    prices_df: pd.DataFrame,
    panel_symbols: Optional[List[str]] = None,
    window: int = 60,
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
    valid_symbols = pivot.columns[pivot.tail(window).notna().sum() >= MIN_DAYS]
    if len(valid_symbols) < MIN_SYMBOLS:
        return _neutral_result(
            f'Insufficient symbols: {len(valid_symbols)} < {MIN_SYMBOLS}'
        )

    # Compute daily returns for the window
    returns = pivot[valid_symbols].tail(window + 1).pct_change().dropna()

    if len(returns) < MIN_DAYS:
        return _neutral_result(f'Insufficient return history: {len(returns)} < {MIN_DAYS}')

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
    corr_z = (avg_correlation - AVG_CORR_MEAN) / AVG_CORR_STD if AVG_CORR_STD > 0 else 0.0
    pc1_z = (pc1_explained - PC1_MEAN) / PC1_STD if PC1_STD > 0 else 0.0

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
