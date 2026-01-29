"""Price data ingestion from Stooq and Alpha Vantage."""

import pandas as pd
import requests
from io import StringIO
from typing import List, Dict, Optional
import time


def fetch_stooq_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq.

    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        lookback_days: How many days of history (default 365)

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    # Stooq uses US. suffix for US stocks
    stooq_symbol = f"{symbol}.US"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Check for valid data (not empty or error page)
        if len(response.text) < 50 or 'No data' in response.text:
            print(f"Stooq: No data for {symbol}")
            return pd.DataFrame()

        df = pd.read_csv(StringIO(response.text))

        # Handle column names (Stooq may use different cases)
        df.columns = df.columns.str.lower()

        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Stooq: Missing columns for {symbol}")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Keep only recent data
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        df = df[df['date'] >= cutoff]

        # Add symbol column
        df['symbol'] = symbol

        return df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Stooq fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_alphavantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Fallback price source for critical symbols.

    Args:
        symbol: Ticker symbol
        api_key: Alpha Vantage API key

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            # Check for rate limit message
            if 'Note' in data or 'Information' in data:
                print(f"Alpha Vantage rate limited for {symbol}")
            else:
                print(f"Alpha Vantage: No data for {symbol}")
            return pd.DataFrame()

        ts = data['Time Series (Daily)']

        records = []
        for date_str, values in ts.items():
            records.append({
                'date': pd.to_datetime(date_str),
                'symbol': symbol,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })

        df = pd.DataFrame(records).sort_values('date')
        return df.tail(365)  # Keep last year

    except Exception as e:
        print(f"Alpha Vantage fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def run(
    universe: List[Dict],
    alphavantage_key: Optional[str] = None,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Run price ingestion for the entire universe.

    Args:
        universe: List of dicts with 'symbol' key
        alphavantage_key: API key for Alpha Vantage fallback
        lookback_days: Days of history to fetch

    Returns:
        Combined DataFrame with all symbols' price data
    """
    all_data = []
    critical_symbols = {'SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'HYG', 'LQD', 'GLD', 'VIXY'}

    symbols = [u['symbol'] if isinstance(u, dict) else u for u in universe]
    failed_symbols = []

    print(f"Ingesting prices for {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        if i > 0 and i % 10 == 0:
            print(f"  Progress: {i}/{len(symbols)}")
            # Small delay to avoid rate limiting
            time.sleep(0.5)

        # Try Stooq first
        df = fetch_stooq_daily(symbol, lookback_days)

        # Fallback to Alpha Vantage for critical symbols
        if len(df) == 0 and symbol in critical_symbols and alphavantage_key:
            print(f"  Trying Alpha Vantage fallback for {symbol}")
            df = fetch_alphavantage_daily(symbol, alphavantage_key)
            time.sleep(1)  # Alpha Vantage rate limit

        if len(df) > 0:
            all_data.append(df)
        else:
            failed_symbols.append(symbol)

    if failed_symbols:
        print(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    print(f"Ingested {len(result)} price records for {result['symbol'].nunique()} symbols")

    return result
