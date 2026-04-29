"""Price data ingestion from Stooq, Alpha Vantage, and yfinance."""

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


def fetch_stooq_index(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch daily close data for a Stooq index (e.g., ^VVIX, ^SKEW).

    Index symbols use no .US suffix and typically only have close data.

    Args:
        symbol: Stooq index symbol (e.g., '^VVIX', '^SKEW')
        lookback_days: Days of history to fetch

    Returns:
        DataFrame with columns: date, symbol, close
        Empty DataFrame on failure.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if len(response.text) < 50 or 'No data' in response.text:
            print(f"Stooq index: No data for {symbol}")
            return pd.DataFrame()

        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.lower()

        if 'date' not in df.columns or 'close' not in df.columns:
            print(f"Stooq index: Missing columns for {symbol}")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        df = df[df['date'] >= cutoff]

        df['symbol'] = symbol

        return df[['date', 'symbol', 'close']].reset_index(drop=True)

    except Exception as e:
        print(f"Stooq index fetch failed for {symbol}: {e}")
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


def fetch_alpaca_daily(
    symbol: str,
    key_id: str,
    secret_key: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Fallback daily price source via Alpaca historical bars.

    Uses the market-data endpoint rather than the broker endpoint so the night
    pipeline can fetch stable daily bars without depending on public scrapers.
    """
    if not key_id or not secret_key:
        return pd.DataFrame()

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    headers = {
        'APCA-API-KEY-ID': key_id,
        'APCA-API-SECRET-KEY': secret_key,
    }
    params = {
        'timeframe': '1Day',
        'start': start,
        'end': end,
        'limit': 10000,
        'adjustment': 'all',
        'feed': 'iex',  # free tier; Alpaca may also accept 'sip' with subscription
    }

    for attempt_feed in ['iex', None]:
        attempt_params = dict(params)
        if attempt_feed is None:
            attempt_params.pop('feed', None)
        try:
            response = requests.get(url, headers=headers, params=attempt_params, timeout=15)
            response.raise_for_status()
            data = response.json()
            bars = data.get('bars', [])
            if not bars:
                if attempt_feed == 'iex':
                    print(f"Alpaca: No IEX bars for {symbol}, retrying without feed")
                    continue
                print(f"Alpaca: No bars for {symbol}")
                return pd.DataFrame()

            records = []
            for bar in bars:
                records.append({
                    'date': pd.to_datetime(bar['t']).tz_localize(None),
                    'symbol': symbol,
                    'open': float(bar['o']),
                    'high': float(bar['h']),
                    'low': float(bar['l']),
                    'close': float(bar['c']),
                    'volume': int(bar['v']),
                })

            df = pd.DataFrame(records).sort_values('date')
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            result = df[df['date'] >= cutoff]
            if len(result) > 0:
                print(f"Alpaca: {len(result)} bars for {symbol} (feed={attempt_feed})")
                return result
        except Exception as e:
            print(f"Alpaca daily fetch failed for {symbol} (feed={attempt_feed}): {e}")
            if attempt_feed == 'iex':
                continue
            return pd.DataFrame()

    return pd.DataFrame()


def fetch_yfinance_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fallback daily price source via yfinance.

    Args:
        symbol: Ticker symbol
        lookback_days: How many days of history to keep

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        print(f"yfinance not available for daily fallback on {symbol}")
        return pd.DataFrame()

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="2y", interval="1d", auto_adjust=False)
        if len(history) == 0:
            print(f"yfinance: No daily history for {symbol}")
            return pd.DataFrame()

        history = history.reset_index()
        history.columns = history.columns.str.lower()
        if 'date' not in history.columns:
            history = history.rename(columns={'datetime': 'date'})

        if 'date' not in history.columns:
            print(f"yfinance: Missing date column for {symbol}")
            return pd.DataFrame()

        history['date'] = pd.to_datetime(history['date'])
        if getattr(history['date'].dt, 'tz', None) is not None:
            history['date'] = history['date'].dt.tz_localize(None)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in history.columns]
        if missing_cols:
            print(f"yfinance: Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        history = history[history['date'] >= cutoff].copy()
        if len(history) == 0:
            return pd.DataFrame()

        history['symbol'] = symbol
        return history[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"yfinance daily fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def run(
    universe: List[Dict],
    alphavantage_key: Optional[str] = None,
    alpaca_key_id: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
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

        # Primary authenticated fallback for Lambda/runtime use.
        if len(df) == 0 and alpaca_key_id and alpaca_secret_key:
            print(f"  Trying Alpaca market-data fallback for {symbol}")
            df = fetch_alpaca_daily(symbol, alpaca_key_id, alpaca_secret_key, lookback_days)

        # Final fallback to yfinance daily history. This keeps the night
        # pipeline aligned with the already-resilient morning quote path.
        if len(df) == 0:
            print(f"  Trying yfinance fallback for {symbol}")
            df = fetch_yfinance_daily(symbol, lookback_days)

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


def fetch_morning_quotes(symbols: List[str], broker=None) -> pd.DataFrame:
    """
    Fetch current/morning prices for a small set of symbols.

    Source priority:
    1. Broker snapshot API (Alpaca data API when broker supports it)
    2. yfinance intraday quote
    3. Stooq latest daily close

    Used by the morning execution run to get fresh market prices.
    Designed for ~10-20 symbols (held positions + intent symbols).

    Args:
        symbols: List of ticker symbols
        broker: Optional broker adapter with get_snapshots() support

    Returns:
        DataFrame with columns: symbol, price, open, high, low, volume, timestamp
        Empty DataFrame if all fetches fail.
    """
    records = []
    covered_symbols = set()

    # Primary source: broker snapshot API (Alpaca data API)
    if broker is not None:
        try:
            snapshots = broker.get_snapshots(symbols)
            for snap in snapshots:
                records.append(snap)
                covered_symbols.add(snap['symbol'])
            if snapshots:
                print(f"Broker snapshots: {len(snapshots)}/{len(symbols)} symbols")
        except Exception as e:
            print(f"Broker snapshot fetch failed: {e}")

    missing_symbols = [s for s in symbols if s not in covered_symbols]

    # Fallback 1: yfinance (intraday quote)
    if missing_symbols:
        yf_missing = []
        try:
            import yfinance as yf
            for symbol in missing_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if len(hist) > 0:
                        latest = hist.iloc[-1]
                        records.append({
                            'symbol': symbol,
                            'price': float(latest['Close']),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'volume': int(latest['Volume']),
                            'timestamp': hist.index[-1].isoformat()
                        })
                        covered_symbols.add(symbol)
                    else:
                        yf_missing.append(symbol)
                except Exception as e:
                    print(f"yfinance quote failed for {symbol}: {e}")
                    yf_missing.append(symbol)
        except ImportError:
            print("yfinance not available, falling back to Stooq close data")
            yf_missing = list(missing_symbols)
        missing_symbols = yf_missing

    # Fallback 2: Stooq latest daily close
    for symbol in missing_symbols:
        try:
            stooq_df = fetch_stooq_daily(symbol, lookback_days=10)
            if len(stooq_df) > 0:
                latest = stooq_df.sort_values('date').iloc[-1]
                close = float(latest['close'])
                records.append({
                    'symbol': symbol,
                    'price': close,
                    'open': float(latest.get('open', close)),
                    'high': float(latest.get('high', close)),
                    'low': float(latest.get('low', close)),
                    'volume': int(latest.get('volume', 0)),
                    'timestamp': pd.to_datetime(latest['date']).isoformat()
                })
                print(f"Using Stooq fallback quote for {symbol}: ${close:.2f}")
            else:
                print(f"Stooq fallback quote unavailable for {symbol}")
        except Exception as e:
            print(f"Stooq fallback failed for {symbol}: {e}")

    if not records:
        print("Warning: No morning quotes fetched")
        return pd.DataFrame()

    # Keep one row per symbol (prefer first source that succeeded)
    result = pd.DataFrame(records).drop_duplicates(subset=['symbol'], keep='first')
    return result.reset_index(drop=True)
