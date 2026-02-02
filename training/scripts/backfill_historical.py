"""
Historical data backfill for ensemble training.

Downloads 3 years of historical data (price + macro) and computes context features.
Output: training/data/historical_context.parquet
"""

import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")


# Symbols for context features
PRICE_SYMBOLS = ['SPY', 'TLT', 'HYG', 'IEF', 'VIXY']

# FRED series for macro data
FRED_SERIES = ['DGS2', 'DGS10', 'VIXCLS']


def fetch_price_data(
    symbols: list,
    start_date: str,
    end_date: str,
    delay: float = 0.5
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using yfinance.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        delay: Delay between requests to avoid rate limits

    Returns:
        DataFrame with columns [date, symbol, open, high, low, close, volume]
    """
    all_data = []

    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if len(df) == 0:
                print(f"    WARNING: No data for {symbol}")
                continue

            # Rename columns to lowercase
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Add symbol column
            df['symbol'] = symbol

            # Select columns
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            # Ensure date is datetime without timezone
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            all_data.append(df)
            print(f"    {len(df)} rows fetched")

            time.sleep(delay)

        except Exception as e:
            print(f"    ERROR fetching {symbol}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def fetch_fred_data(
    series_ids: list,
    api_key: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch FRED data for multiple series.

    Args:
        series_ids: List of FRED series IDs
        api_key: FRED API key
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns [date, series_id, value]
    """
    all_data = []

    for series_id in series_ids:
        print(f"  Fetching {series_id}...")
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if 'observations' not in data:
                print(f"    WARNING: No observations for {series_id}")
                continue

            records = []
            for obs in data['observations']:
                if obs['value'] != '.':
                    records.append({
                        'date': pd.to_datetime(obs['date']),
                        'series_id': series_id,
                        'value': float(obs['value'])
                    })

            df = pd.DataFrame(records)
            all_data.append(df)
            print(f"    {len(df)} observations fetched")

        except Exception as e:
            print(f"    ERROR fetching {series_id}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def forward_fill_fred(df: pd.DataFrame, max_fill_days: int = 5) -> pd.DataFrame:
    """Forward-fill FRED data for weekends/holidays."""
    if len(df) == 0:
        return df

    filled_dfs = []

    for series_id in df['series_id'].unique():
        series_data = df[df['series_id'] == series_id].copy()
        series_data = series_data.set_index('date').sort_index()

        # Create full date range
        date_range = pd.date_range(
            start=series_data.index.min(),
            end=series_data.index.max(),
            freq='D'
        )

        # Reindex and forward fill
        series_data = series_data.reindex(date_range)
        series_data['value'] = series_data['value'].ffill(limit=max_fill_days)
        series_data['series_id'] = series_id

        # Reset index
        series_data = series_data.reset_index()
        series_data = series_data.rename(columns={'index': 'date'})
        series_data = series_data.dropna(subset=['value'])

        filled_dfs.append(series_data)

    return pd.concat(filled_dfs, ignore_index=True)


def compute_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for a single asset (replicates feature_utils logic).
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

    # Trend
    df['log_price'] = np.log(df['close'].clip(lower=0.01))
    df['trend_63d'] = df['log_price'].diff(63) / 63

    df = df.drop(columns=['log_price', 'peak_63d'], errors='ignore')

    return df


def build_context_features(
    prices_df: pd.DataFrame,
    fred_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build time-series context features from price and FRED data.

    Returns:
        DataFrame with one row per trading day
    """
    print("Computing context features...")

    # Compute features for each symbol
    symbol_features = {}
    for symbol in prices_df['symbol'].unique():
        symbol_data = prices_df[prices_df['symbol'] == symbol].copy()
        if len(symbol_data) >= 63:  # Need minimum history for 63-day features
            symbol_features[symbol] = compute_asset_features(symbol_data)
            print(f"  {symbol}: {len(symbol_features[symbol])} rows with features")

    # Check required symbols
    required = ['SPY', 'TLT', 'HYG', 'IEF', 'VIXY']
    missing = [s for s in required if s not in symbol_features]
    if missing:
        print(f"  WARNING: Missing symbols: {missing}")

    # Get SPY dates as the base (trading days)
    if 'SPY' not in symbol_features:
        raise ValueError("SPY data is required for context features")

    spy_df = symbol_features['SPY']
    dates = spy_df['date'].values

    # Pivot FRED data
    fred_pivot = {}
    if len(fred_df) > 0 and 'series_id' in fred_df.columns:
        for series_id in fred_df['series_id'].unique():
            series_data = fred_df[fred_df['series_id'] == series_id].set_index('date')['value']
            fred_pivot[series_id] = series_data

    # Build context for each date
    context_rows = []

    for date in dates:
        # SPY features
        spy_row = spy_df[spy_df['date'] == date]
        if len(spy_row) == 0:
            continue
        spy_row = spy_row.iloc[0]

        # Get latest FRED values (as of date)
        def get_fred_value(series_id: str, as_of: pd.Timestamp) -> float:
            if series_id not in fred_pivot:
                return 0.0
            series = fred_pivot[series_id]
            valid = series[series.index <= as_of]
            if len(valid) == 0:
                return 0.0
            return valid.iloc[-1]

        rate_2y = get_fred_value('DGS2', date)
        rate_10y = get_fred_value('DGS10', date)
        yield_slope = rate_10y - rate_2y

        # Credit spread proxy: HYG - IEF 21-day return
        credit_spread_proxy = 0.0
        if 'HYG' in symbol_features and 'IEF' in symbol_features:
            hyg_row = symbol_features['HYG'][symbol_features['HYG']['date'] == date]
            ief_row = symbol_features['IEF'][symbol_features['IEF']['date'] == date]
            if len(hyg_row) > 0 and len(ief_row) > 0:
                hyg_ret = hyg_row.iloc[0].get('return_21d', 0) or 0
                ief_ret = ief_row.iloc[0].get('return_21d', 0) or 0
                credit_spread_proxy = hyg_ret - ief_ret

        # Risk-off proxy: SPY - TLT 21-day return
        risk_off_proxy = 0.0
        if 'TLT' in symbol_features:
            tlt_row = symbol_features['TLT'][symbol_features['TLT']['date'] == date]
            if len(tlt_row) > 0:
                spy_ret = spy_row.get('return_21d', 0) or 0
                tlt_ret = tlt_row.iloc[0].get('return_21d', 0) or 0
                risk_off_proxy = spy_ret - tlt_ret

        # VIXY return
        vixy_return_21d = 0.0
        if 'VIXY' in symbol_features:
            vixy_row = symbol_features['VIXY'][symbol_features['VIXY']['date'] == date]
            if len(vixy_row) > 0:
                vixy_return_21d = vixy_row.iloc[0].get('return_21d', 0) or 0

        context_rows.append({
            'date': date,
            'spy_return_1d': spy_row.get('return_1d', 0) or 0,
            'spy_return_21d': spy_row.get('return_21d', 0) or 0,
            'spy_vol_21d': spy_row.get('vol_21d', 0) or 0,
            'rate_2y': rate_2y,
            'rate_10y': rate_10y,
            'yield_slope': yield_slope,
            'credit_spread_proxy': credit_spread_proxy,
            'risk_off_proxy': risk_off_proxy,
            'vixy_return_21d': vixy_return_21d,
            'gdelt_avg_tone': 0.0  # Placeholder - historical tone not available
        })

    context_df = pd.DataFrame(context_rows)

    # Drop rows with NaN in key features (first 63 days will have NaNs)
    context_df = context_df.dropna(subset=['spy_return_21d', 'spy_vol_21d'])

    print(f"  Context features computed: {len(context_df)} rows")

    return context_df


def run_backfill(
    output_path: str,
    fred_api_key: Optional[str] = None,
    years: int = 3
) -> pd.DataFrame:
    """
    Run the full backfill pipeline.

    Args:
        output_path: Path to save the parquet file
        fred_api_key: FRED API key (optional if env var FRED_API_KEY is set)
        years: Number of years of history to fetch

    Returns:
        Context features DataFrame
    """
    # Get FRED API key
    if fred_api_key is None:
        fred_api_key = os.environ.get('FRED_API_KEY')

    if not fred_api_key:
        print("WARNING: No FRED API key provided. FRED data will be empty.")
        print("Set FRED_API_KEY environment variable or pass --fred-api-key")

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365 + 90)).strftime('%Y-%m-%d')
    # Extra 90 days for warmup period (63-day features)

    print(f"Backfilling data from {start_date} to {end_date}...")
    print()

    # Fetch price data
    print("Fetching price data (yfinance)...")
    prices_df = fetch_price_data(PRICE_SYMBOLS, start_date, end_date)
    print(f"  Total: {len(prices_df)} price records")
    print()

    # Fetch FRED data
    fred_df = pd.DataFrame()
    if fred_api_key:
        print("Fetching FRED data...")
        fred_df = fetch_fred_data(FRED_SERIES, fred_api_key, start_date, end_date)
        if len(fred_df) > 0:
            fred_df = forward_fill_fred(fred_df)
            print(f"  Total: {len(fred_df)} FRED records (after fill)")
        print()

    # Build context features
    context_df = build_context_features(prices_df, fred_df)

    # Validate data quality
    print()
    print("Data quality check:")
    print(f"  Date range: {context_df['date'].min()} to {context_df['date'].max()}")
    print(f"  Total rows: {len(context_df)}")

    # Check for excessive NaNs
    nan_counts = context_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN counts: {nan_counts[nan_counts > 0].to_dict()}")

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    context_df.to_parquet(output_path, index=False)
    print()
    print(f"Saved to {output_path}")

    return context_df


def main():
    parser = argparse.ArgumentParser(description='Backfill historical data for training')
    parser.add_argument(
        '--output',
        type=str,
        default='training/data/historical_context.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--fred-api-key',
        type=str,
        help='FRED API key (or set FRED_API_KEY env var)'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Years of history to fetch'
    )

    args = parser.parse_args()

    run_backfill(
        output_path=args.output,
        fred_api_key=args.fred_api_key,
        years=args.years
    )


if __name__ == '__main__':
    main()
