"""Data validation utilities for price and macro data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple


def validate_price_data(
    prices_df: pd.DataFrame,
    universe: List[str],
    max_missing_pct: float = 0.05,
    max_staleness_days: int = 2
) -> Dict[str, Any]:
    """
    Validate price data quality.

    Args:
        prices_df: DataFrame with columns [date, symbol, open, high, low, close, volume]
        universe: List of expected symbols
        max_missing_pct: Maximum allowed missing symbol percentage
        max_staleness_days: Maximum days since last data

    Returns:
        Dict with validation results
    """
    issues = []

    # Check if DataFrame is empty
    if len(prices_df) == 0:
        return {
            'valid': False,
            'issues': ['No price data available'],
            'coverage': 0.0,
            'latest_date': None,
            'missing_symbols': universe,
            'degraded_mode': True
        }

    # Check symbol coverage
    available_symbols = set(prices_df['symbol'].unique())
    missing_symbols = [s for s in universe if s not in available_symbols]
    coverage = 1 - (len(missing_symbols) / len(universe))

    if coverage < (1 - max_missing_pct):
        issues.append(f"Symbol coverage {coverage:.1%} below threshold {1 - max_missing_pct:.1%}")

    # Check data freshness
    latest_date = prices_df['date'].max()
    if isinstance(latest_date, str):
        latest_date = pd.to_datetime(latest_date)

    days_stale = (pd.Timestamp.now() - latest_date).days

    # Account for weekends
    if pd.Timestamp.now().dayofweek in [5, 6]:  # Saturday or Sunday
        days_stale -= (pd.Timestamp.now().dayofweek - 4)

    if days_stale > max_staleness_days:
        issues.append(f"Data is {days_stale} days stale (max {max_staleness_days})")

    # Check for required columns
    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in prices_df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check for NaN values in OHLCV
    if 'close' in prices_df.columns:
        nan_count = prices_df['close'].isna().sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in close prices")

    # Check for zero or negative volumes
    if 'volume' in prices_df.columns:
        zero_vol = (prices_df['volume'] <= 0).sum()
        if zero_vol > 0:
            issues.append(f"{zero_vol} rows with zero or negative volume")

    # Check critical symbols
    critical_symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'HYG', 'LQD', 'GLD', 'VIXY']
    missing_critical = [s for s in critical_symbols if s not in available_symbols]
    if missing_critical:
        issues.append(f"Missing critical symbols: {missing_critical}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'coverage': coverage,
        'latest_date': latest_date.isoformat() if latest_date else None,
        'missing_symbols': missing_symbols,
        'missing_critical': missing_critical,
        'degraded_mode': len(missing_critical) > 0 or coverage < 0.80
    }


def validate_fred_data(
    fred_df: pd.DataFrame,
    required_series: List[str],
    max_staleness_days: int = 5
) -> Dict[str, Any]:
    """
    Validate FRED macroeconomic data.

    Args:
        fred_df: DataFrame with columns [date, series_id, value]
        required_series: List of required FRED series IDs
        max_staleness_days: Maximum days since last data

    Returns:
        Dict with validation results
    """
    issues = []

    if len(fred_df) == 0:
        return {
            'valid': False,
            'issues': ['No FRED data available'],
            'missing_series': required_series,
            'degraded_mode': True
        }

    # Check series coverage
    available_series = set(fred_df['series_id'].unique())
    missing_series = [s for s in required_series if s not in available_series]

    if missing_series:
        issues.append(f"Missing FRED series: {missing_series}")

    # Check freshness for each series
    stale_series = []
    for series in available_series:
        series_data = fred_df[fred_df['series_id'] == series]
        latest = series_data['date'].max()
        if isinstance(latest, str):
            latest = pd.to_datetime(latest)

        days_stale = (pd.Timestamp.now() - latest).days
        if days_stale > max_staleness_days:
            stale_series.append((series, days_stale))

    if stale_series:
        issues.append(f"Stale FRED series: {stale_series}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'missing_series': missing_series,
        'stale_series': stale_series,
        'degraded_mode': len(missing_series) > 0
    }


def check_data_quality(
    prices_df: pd.DataFrame,
    symbol: str
) -> Tuple[bool, List[str]]:
    """
    Check data quality for a single symbol.

    Returns:
        (is_valid, list of issues)
    """
    issues = []

    symbol_data = prices_df[prices_df['symbol'] == symbol]

    if len(symbol_data) == 0:
        return False, ['No data for symbol']

    # Check for sufficient history (63 days for features)
    if len(symbol_data) < 63:
        issues.append(f'Insufficient history: {len(symbol_data)} days (need 63)')

    # Check for gaps (more than 5 consecutive missing days)
    symbol_data = symbol_data.sort_values('date')
    dates = pd.to_datetime(symbol_data['date'])
    gaps = dates.diff().dt.days
    if gaps.max() > 7:  # Allow for weekends + holidays
        issues.append(f'Large gap in data: {gaps.max()} days')

    # Check for suspicious price jumps (> 50% in one day)
    returns = symbol_data['close'].pct_change()
    max_jump = returns.abs().max()
    if max_jump > 0.50:
        issues.append(f'Suspicious price jump: {max_jump:.1%}')

    return len(issues) == 0, issues
