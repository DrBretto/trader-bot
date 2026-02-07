"""Feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features for a single asset.

    Input:
        df: DataFrame with columns [date, open, high, low, close, volume]

    Output:
        df with additional feature columns
    """
    df = df.sort_values('date').copy()

    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_21d'] = df['close'].pct_change(21)
    df['return_63d'] = df['close'].pct_change(63)

    # Volatility (annualized)
    df['vol_21d'] = df['return_1d'].rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['return_1d'].rolling(63).std() * np.sqrt(252)

    # Drawdown
    df['peak_63d'] = df['close'].rolling(63, min_periods=1).max()
    df['drawdown_63d'] = (df['close'] - df['peak_63d']) / df['peak_63d']

    # Trend (log slope over 63 days)
    df['log_price'] = np.log(df['close'].clip(lower=0.01))
    df['trend_63d'] = df['log_price'].diff(63) / 63

    # Liquidity proxy
    df['volume_ma21'] = df['volume'].rolling(21).median()

    # Drop intermediate columns
    df = df.drop(columns=['log_price', 'peak_63d'], errors='ignore')

    return df


def compute_relative_strength(
    asset_df: pd.DataFrame,
    spy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add relative strength vs SPY.

    Both DataFrames must have computed features (return_21d, return_63d).
    """
    # Ensure date columns are datetime
    asset_df = asset_df.copy()
    spy_df = spy_df.copy()
    asset_df['date'] = pd.to_datetime(asset_df['date'])
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    # Merge with SPY data
    merged = asset_df.merge(
        spy_df[['date', 'return_21d', 'return_63d']],
        on='date',
        suffixes=('', '_spy'),
        how='left'
    )

    # Compute relative strength
    merged['rel_strength_21d'] = merged['return_21d'] - merged['return_21d_spy']
    merged['rel_strength_63d'] = merged['return_63d'] - merged['return_63d_spy']

    # Drop SPY columns
    merged = merged.drop(columns=['return_21d_spy', 'return_63d_spy'])

    return merged


def compute_context_features(
    spy_df: pd.DataFrame,
    tlt_df: pd.DataFrame,
    hyg_df: pd.DataFrame,
    ief_df: pd.DataFrame,
    vixy_df: pd.DataFrame,
    fred_data: Dict[str, float],
    gdelt_data: Dict[str, float],
    target_date: Optional[pd.Timestamp] = None,
    vvix_value: Optional[float] = None,
    skew_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Compute market-wide context features.

    Returns:
        Single-row DataFrame with context for the target date
    """
    if target_date is None:
        target_date = spy_df['date'].max()

    # Get latest values for each proxy
    def get_latest(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        row = df[df['date'] == date]
        if len(row) == 0:
            # Fall back to most recent data
            row = df[df['date'] <= date].tail(1)
        if len(row) == 0:
            return pd.Series()
        return row.iloc[0]

    spy_latest = get_latest(spy_df, target_date)
    tlt_latest = get_latest(tlt_df, target_date)
    hyg_latest = get_latest(hyg_df, target_date)
    ief_latest = get_latest(ief_df, target_date)
    vixy_latest = get_latest(vixy_df, target_date)

    # Rates from FRED
    rate_2y = fred_data.get('DGS2', 0)
    rate_3m = fred_data.get('DGS3MO', 0)
    rate_10y = fred_data.get('DGS10', 0)
    yield_slope = rate_10y - rate_2y
    yield_slope_10y_3m = rate_10y - rate_3m

    # Credit spread proxy (HYG - IEF performance)
    credit_spread_proxy = 0
    if len(hyg_latest) > 0 and len(ief_latest) > 0:
        hyg_ret = hyg_latest.get('return_21d', 0) or 0
        ief_ret = ief_latest.get('return_21d', 0) or 0
        credit_spread_proxy = hyg_ret - ief_ret

    # Risk-off proxy (SPY - TLT trend)
    risk_off_proxy = 0
    if len(spy_latest) > 0 and len(tlt_latest) > 0:
        spy_ret = spy_latest.get('return_21d', 0) or 0
        tlt_ret = tlt_latest.get('return_21d', 0) or 0
        risk_off_proxy = spy_ret - tlt_ret

    context = {
        'date': target_date,
        'spy_return_1d': spy_latest.get('return_1d', 0) or 0,
        'spy_return_21d': spy_latest.get('return_21d', 0) or 0,
        'spy_vol_21d': spy_latest.get('vol_21d', 0) or 0,
        'rate_2y': rate_2y,
        'rate_3m': rate_3m,
        'rate_10y': rate_10y,
        'yield_slope': yield_slope,
        'yield_slope_10y_3m': yield_slope_10y_3m,
        'credit_spread_proxy': credit_spread_proxy,
        'risk_off_proxy': risk_off_proxy,
        'vixy_return_21d': vixy_latest.get('return_21d', 0) if len(vixy_latest) > 0 else 0,
        'vvix_value': vvix_value if vvix_value is not None else 0,
        'skew_value': skew_value if skew_value is not None else 0,
        'gdelt_doc_count': gdelt_data.get('gdelt_doc_count', 0),
        'gdelt_avg_tone': gdelt_data.get('gdelt_avg_tone', 0),
        'gdelt_tone_std': gdelt_data.get('gdelt_tone_std', 0),
        'gdelt_neg_tone_share': gdelt_data.get('gdelt_neg_tone_share', 0)
    }

    return pd.DataFrame([context])


def compute_cross_sectional_ranks(
    features_df: pd.DataFrame,
    target_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute cross-sectional percentile ranks for a given date.

    Used by the asset health model.
    """
    # Filter to target date
    latest = features_df[features_df['date'] == target_date].copy()

    if len(latest) == 0:
        return pd.DataFrame()

    # Compute ranks (0-1 percentile)
    latest['mom_63_rank'] = latest['return_63d'].rank(pct=True)
    latest['mom_21_rank'] = latest['return_21d'].rank(pct=True)
    latest['vol_63_rank'] = latest['vol_63d'].rank(pct=True)
    latest['dd_63_rank'] = (-latest['drawdown_63d']).rank(pct=True)  # Less negative = better
    latest['rs_63_rank'] = latest['rel_strength_63d'].rank(pct=True)
    latest['rs_21_rank'] = latest['rel_strength_21d'].rank(pct=True)

    return latest
