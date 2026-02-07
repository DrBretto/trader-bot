"""Expert signals orchestrator.

Runs all four expert signal modules and returns a combined result dict.
Each module is wrapped in try/except with neutral fallback values so the
pipeline never breaks due to signal computation failure.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from src.signals.macro_credit import compute_macro_credit
from src.signals.vol_uncertainty import compute_vol_uncertainty
from src.signals.fragility import compute_fragility
from src.signals.entropy_shift import compute_entropy_shift
from src.utils.s3_client import S3Client


def _get_latest_close(index_df: pd.DataFrame) -> Optional[float]:
    """Extract the latest close value from an index DataFrame."""
    if index_df is None or len(index_df) == 0:
        return None
    return float(index_df.sort_values('date')['close'].iloc[-1])


def _get_close_series(index_df: pd.DataFrame) -> Optional[pd.Series]:
    """Extract close price series from an index DataFrame."""
    if index_df is None or len(index_df) == 0:
        return None
    sorted_df = index_df.sort_values('date')
    return sorted_df.set_index('date')['close']


def _get_symbol_closes(prices_df: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract close price series for a symbol from the main prices DataFrame."""
    symbol_data = prices_df[prices_df['symbol'] == symbol].sort_values('date')
    if len(symbol_data) == 0:
        return pd.Series(dtype=float)
    return symbol_data.set_index('date')['close']


def _load_prev_entropy_state(s3: Optional[S3Client]) -> dict:
    """Load previous day's entropy consecutive-days counter from timeseries."""
    if s3 is None:
        return {'prev_consecutive_days': 0, 'prev_above_threshold': False}

    try:
        ts_df = s3.read_parquet('dashboard/timeseries.parquet')
        if len(ts_df) == 0:
            return {'prev_consecutive_days': 0, 'prev_above_threshold': False}

        ts_df = ts_df.sort_values('date')
        last_row = ts_df.iloc[-1]
        return {
            'prev_consecutive_days': int(last_row.get('entropy_consecutive_days', 0)),
            'prev_above_threshold': bool(last_row.get('entropy_above_threshold', False)),
        }
    except Exception:
        return {'prev_consecutive_days': 0, 'prev_above_threshold': False}


def run(
    prices_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    context_df: pd.DataFrame,
    vvix_data: Optional[pd.DataFrame] = None,
    skew_data: Optional[pd.DataFrame] = None,
    s3_client: Optional[S3Client] = None,
) -> Dict[str, Any]:
    """
    Run all expert signal modules.

    Args:
        prices_df: Multi-symbol daily prices
        fred_df: FRED macro data with columns [date, series_id, value]
        context_df: Single-row market context DataFrame
        vvix_data: VVIX index data from Stooq (or None)
        skew_data: SKEW index data from Stooq (or None)
        s3_client: S3Client for loading previous state (or None)

    Returns:
        Dict with all expert signal outputs and metadata.
    """
    print("Computing expert signals...")

    result = {
        'computed_at': datetime.now().isoformat(),
    }

    # Extract context values
    ctx = context_df.iloc[0] if len(context_df) > 0 else {}

    # --- 1. Macro/Credit ---
    try:
        from src.steps.ingest_fred import get_latest_values
        fred_latest = get_latest_values(fred_df) if len(fred_df) > 0 else {}

        hyg_closes = _get_symbol_closes(prices_df, 'HYG')
        ief_closes = _get_symbol_closes(prices_df, 'IEF')

        macro = compute_macro_credit(
            rate_10y=fred_latest.get('DGS10', 0),
            rate_3m=fred_latest.get('DGS3MO', 0),
            hyg_prices=hyg_closes if len(hyg_closes) > 0 else None,
            ief_prices=ief_closes if len(ief_closes) > 0 else None,
            rate_2y=fred_latest.get('DGS2', 0),
        )
        result['macro_credit'] = macro
        print(f"  Macro/Credit score: {macro['macro_credit_score']:.3f}")
    except Exception as e:
        print(f"  Macro/Credit FAILED: {e}")
        result['macro_credit'] = {
            'macro_credit_score': 0.0,
            'yield_slope_10y_3m': 0.0,
            'hy_spread_proxy': 0.0,
            'slope_z_score': 0.0,
            'hy_spread_z_score': 0.0,
            'degraded_reason': str(e),
        }

    # --- 2. Vol Uncertainty ---
    try:
        vix_value = fred_latest.get('VIXCLS', 0) if 'fred_latest' in dir() else 0
        # Also try context_df
        if vix_value == 0:
            vix_value = float(ctx.get('vixy_return_21d', 0)) if hasattr(ctx, 'get') else 0

        # Get VIX from FRED history for percentile computation
        vix_history = None
        if len(fred_df) > 0:
            vix_data = fred_df[fred_df['series_id'] == 'VIXCLS'].sort_values('date')
            if len(vix_data) >= 60:
                vix_history = vix_data['value']
                vix_value = float(vix_data['value'].iloc[-1])

        vvix_value = _get_latest_close(vvix_data)
        skew_value = _get_latest_close(skew_data)

        vvix_history = _get_close_series(vvix_data)

        vol = compute_vol_uncertainty(
            vix=vix_value,
            vvix=vvix_value,
            skew=skew_value,
            vix_history=vix_history,
            vvix_history=vvix_history,
        )
        result['vol_uncertainty'] = vol
        print(f"  Vol Uncertainty score: {vol['vol_uncertainty_score']:.3f} ({vol['vol_regime_label']})")
    except Exception as e:
        print(f"  Vol Uncertainty FAILED: {e}")
        result['vol_uncertainty'] = {
            'vol_uncertainty_score': 0.5,
            'vol_regime_label': 'calm',
            'vix_percentile': 0.5,
            'vvix_percentile': 0.5,
            'skew_percentile': 0.5,
            'vix_value': 0.0,
            'vvix_value': 0.0,
            'skew_value': 0.0,
            'degraded_reason': str(e),
        }

    # --- 3. Fragility ---
    try:
        frag = compute_fragility(prices_df)
        result['fragility'] = frag
        print(f"  Fragility score: {frag['fragility_score']:.3f}")
    except Exception as e:
        print(f"  Fragility FAILED: {e}")
        result['fragility'] = {
            'fragility_score': 0.5,
            'avg_correlation': 0.0,
            'pc1_explained': 0.0,
            'pc2_explained': 0.0,
            'symbols_used': 0,
            'degraded_reason': str(e),
        }

    # --- 4. Entropy Shift ---
    try:
        spy_closes = _get_symbol_closes(prices_df, 'SPY')
        spy_returns = spy_closes.pct_change().dropna() if len(spy_closes) > 1 else pd.Series(dtype=float)

        prev_state = _load_prev_entropy_state(s3_client)

        ent = compute_entropy_shift(
            spy_returns=spy_returns,
            prev_consecutive_days=prev_state['prev_consecutive_days'],
            prev_above_threshold=prev_state['prev_above_threshold'],
        )
        result['entropy_shift'] = ent
        flag_str = 'SHIFT DETECTED' if ent['entropy_shift_flag'] else 'normal'
        print(f"  Entropy score: {ent['entropy_score']:.3f} ({flag_str})")
    except Exception as e:
        print(f"  Entropy Shift FAILED: {e}")
        result['entropy_shift'] = {
            'entropy_score': 0.5,
            'entropy_z_score': 0.0,
            'entropy_shift_flag': False,
            'entropy_consecutive_days': 0,
            'entropy_above_threshold': False,
            'degraded_reason': str(e),
        }

    return result
