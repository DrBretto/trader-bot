"""FRED macroeconomic data ingestion."""

import pandas as pd
import requests
from typing import Dict, List, Optional


# Required FRED series
FRED_SERIES = {
    'DGS2': '2-Year Treasury Yield',
    'DGS3MO': '3-Month Treasury Yield',
    'DGS10': '10-Year Treasury Yield',
    'VIXCLS': 'VIX Close',
    'DCOILWTICO': 'WTI Oil Price',
    'DEXUSEU': 'USD/EUR Exchange Rate'
}


def fetch_fred_series(
    series_id: str,
    api_key: str,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Fetch daily series from FRED.

    Args:
        series_id: FRED series ID (e.g., 'DGS10')
        api_key: FRED API key
        lookback_days: Days of history to fetch

    Returns:
        DataFrame with columns: date, series_id, value
    """
    url = "https://api.stlouisfed.org/fred/series/observations"

    start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if 'observations' not in data:
            print(f"FRED: No observations for {series_id}")
            return pd.DataFrame()

        records = []
        for obs in data['observations']:
            # FRED uses '.' for missing values
            if obs['value'] != '.':
                records.append({
                    'date': pd.to_datetime(obs['date']),
                    'series_id': series_id,
                    'value': float(obs['value'])
                })

        return pd.DataFrame(records)

    except Exception as e:
        print(f"FRED fetch failed for {series_id}: {e}")
        return pd.DataFrame()


def forward_fill_missing(
    df: pd.DataFrame,
    max_fill_days: int = 5
) -> pd.DataFrame:
    """
    Forward-fill missing values (FRED data has gaps for weekends/holidays).

    Args:
        df: DataFrame with date, series_id, value columns
        max_fill_days: Maximum days to forward fill

    Returns:
        DataFrame with filled values
    """
    if len(df) == 0:
        return df

    # Create date range for each series
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

        # Drop rows that couldn't be filled
        series_data = series_data.dropna(subset=['value'])

        filled_dfs.append(series_data)

    if not filled_dfs:
        return pd.DataFrame()

    return pd.concat(filled_dfs, ignore_index=True)


def get_latest_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get the most recent value for each FRED series.

    Returns:
        Dict mapping series_id to latest value
    """
    if len(df) == 0:
        return {}

    latest = {}
    for series_id in df['series_id'].unique():
        series_data = df[df['series_id'] == series_id]
        if len(series_data) > 0:
            latest_row = series_data.sort_values('date').iloc[-1]
            latest[series_id] = latest_row['value']

    return latest


def run(api_key: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Run FRED data ingestion for all required series.

    Args:
        api_key: FRED API key
        lookback_days: Days of history to fetch

    Returns:
        DataFrame with all FRED series data
    """
    all_data = []
    failed_series = []

    print(f"Ingesting FRED data for {len(FRED_SERIES)} series...")

    for series_id, name in FRED_SERIES.items():
        df = fetch_fred_series(series_id, api_key, lookback_days)

        if len(df) > 0:
            all_data.append(df)
            print(f"  {series_id}: {len(df)} observations")
        else:
            failed_series.append(series_id)
            print(f"  {series_id}: FAILED")

    if failed_series:
        print(f"Failed to fetch FRED series: {failed_series}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    # Forward fill missing values
    result = forward_fill_missing(result)

    print(f"FRED ingestion complete: {len(result)} total observations")

    return result
