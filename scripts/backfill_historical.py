#!/usr/bin/env python3
"""
Backfill historical data to bootstrap the system.

Downloads historical prices from Stooq, FRED data, and generates
features for each day. Saves to S3 in the same format as the daily pipeline.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

from src.utils.s3_client import S3Client
from src.steps import ingest_prices, ingest_fred, build_features
from src.models import baseline_regime_model, baseline_health_model


def get_default_universe() -> List[str]:
    """Get default universe of symbols to backfill."""
    return [
        # Core ETFs
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Sector ETFs
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLRE',
        # Fixed income
        'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'TIP',
        # Commodities
        'GLD', 'SLV', 'USO',
        # Volatility
        'VIXY',
        # International
        'EFA', 'EEM', 'VEU',
    ]


def fetch_historical_prices(symbols: List[str], days: int = 365) -> pd.DataFrame:
    """
    Fetch historical prices from Stooq.

    Stooq provides free historical data without API key.
    """
    all_prices = []

    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            # Stooq URL format for historical data
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
            df = pd.read_csv(url)

            if len(df) > 0 and 'Date' in df.columns:
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                df['symbol'] = symbol
                df['date'] = pd.to_datetime(df['date'])

                # Filter to requested days
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df['date'] >= cutoff]

                all_prices.append(df)

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"    Warning: Failed to fetch {symbol}: {e}")

    if all_prices:
        return pd.concat(all_prices, ignore_index=True)
    return pd.DataFrame()


def fetch_historical_fred(fred_key: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical FRED data in long format."""
    from src.steps.ingest_fred import FRED_SERIES, fetch_fred_series

    all_data = []

    for series_id, series_name in FRED_SERIES.items():
        print(f"  Fetching FRED {series_id}...")
        try:
            df = fetch_fred_series(series_id, fred_key, days + 30)
            if len(df) > 0:
                all_data.append(df)
            time.sleep(0.3)
        except Exception as e:
            print(f"    Warning: Failed to fetch {series_id}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def generate_daily_artifacts(
    date_str: str,
    prices_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    s3: S3Client
) -> bool:
    """Generate and save daily artifacts for a single date."""

    date = pd.to_datetime(date_str)

    # Filter data up to this date
    prices_to_date = prices_df[prices_df['date'] <= date].copy()
    fred_to_date = fred_df[fred_df['date'] <= date].copy()

    if len(prices_to_date) == 0:
        return False

    # Get just this day's prices for the artifact
    day_prices = prices_df[prices_df['date'] == date].copy()
    if len(day_prices) == 0:
        return False

    # Build features using historical data
    gdelt_data = {'avg_tone': 0, 'num_articles': 0}  # Default GDELT

    try:
        features_df, context_df = build_features.run(
            prices_to_date,
            fred_to_date,
            gdelt_data
        )
    except Exception as e:
        print(f"    Warning: Feature build failed for {date_str}: {e}")
        return False

    if len(features_df) == 0 or len(context_df) == 0:
        return False

    # Filter to just this date
    day_features = features_df[features_df['date'] == date].copy()
    day_context = context_df[context_df['date'] == date].copy()

    if len(day_features) == 0 or len(day_context) == 0:
        return False

    # Run inference with baseline models
    regime_result = baseline_regime_model(day_context.iloc[0])
    health_df = baseline_health_model(day_features, date)

    # Merge health scores into features
    if len(health_df) > 0:
        day_features = day_features.merge(
            health_df[['symbol', 'health_score', 'vol_bucket']],
            on='symbol',
            how='left'
        )

    # Add regime to context
    day_context['regime'] = regime_result['regime_label']

    # Create portfolio state (initial state)
    portfolio_state = {
        'date': date_str,
        'portfolio_value': 100000,
        'cash': 100000,
        'invested': 0,
        'holdings': [],
        'benchmark_value': 100000
    }

    # Save artifacts
    base_path = f"daily/{date_str}"

    try:
        s3.write_parquet(day_prices, f"{base_path}/prices.parquet")
        s3.write_parquet(day_features, f"{base_path}/features.parquet")
        s3.write_parquet(day_context, f"{base_path}/context.parquet")
        s3.write_json(portfolio_state, f"{base_path}/portfolio_state.json")
        s3.write_json({
            'regime': regime_result,
            'health_summary': {'count': len(health_df)}
        }, f"{base_path}/inference.json")

        return True
    except Exception as e:
        print(f"    Warning: Failed to save artifacts for {date_str}: {e}")
        return False


def backfill(
    bucket: str,
    region: str = 'us-east-1',
    days: int = 180,
    fred_key: str = '',
    symbols: Optional[List[str]] = None
):
    """
    Backfill historical data to S3.

    Args:
        bucket: S3 bucket name
        region: AWS region
        days: Number of days to backfill
        fred_key: FRED API key
        symbols: List of symbols (uses default if None)
    """
    print(f"Starting backfill of {days} days to s3://{bucket}")

    s3 = S3Client(bucket, region)
    symbols = symbols or get_default_universe()

    # Fetch historical prices
    print(f"\n1. Fetching historical prices for {len(symbols)} symbols...")
    prices_df = fetch_historical_prices(symbols, days + 90)  # Extra for feature calculation
    print(f"   Got {len(prices_df)} price records")

    if len(prices_df) == 0:
        print("ERROR: No price data fetched. Aborting.")
        return

    # Fetch historical FRED data
    print("\n2. Fetching historical FRED data...")
    if fred_key:
        fred_df = fetch_historical_fred(fred_key, days + 90)
        print(f"   Got {len(fred_df)} FRED records")
    else:
        print("   No FRED key provided, using synthetic data")
        # Create minimal synthetic FRED data in long format [date, series_id, value]
        from src.steps.ingest_fred import FRED_SERIES
        dates = sorted(prices_df['date'].unique())
        fred_records = []
        for series_id in FRED_SERIES.keys():
            # Generate reasonable synthetic values for each series
            base_values = {
                'DGS2': 4.0,    # 2-year Treasury
                'DGS10': 4.5,   # 10-year Treasury
                'VIXCLS': 18.0, # VIX
                'DCOILWTICO': 75.0,  # Oil price
                'DEXUSEU': 1.08      # USD/EUR
            }
            base = base_values.get(series_id, 1.0)
            np.random.seed(42)  # Reproducible
            for date in dates:
                fred_records.append({
                    'date': date,
                    'series_id': series_id,
                    'value': base + np.random.randn() * (base * 0.05)
                })
        fred_df = pd.DataFrame(fred_records)

    # Get unique dates to process
    all_dates = sorted(prices_df['date'].unique())

    # Only process dates within the backfill window
    cutoff = datetime.now() - timedelta(days=days)
    dates_to_process = [d for d in all_dates if pd.to_datetime(d) >= cutoff]

    print(f"\n3. Generating daily artifacts for {len(dates_to_process)} days...")

    success_count = 0
    for i, date in enumerate(dates_to_process):
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

        if (i + 1) % 10 == 0:
            print(f"   Processing {i + 1}/{len(dates_to_process)}: {date_str}")

        if generate_daily_artifacts(date_str, prices_df, fred_df, s3):
            success_count += 1

    print(f"\n4. Backfill complete!")
    print(f"   Successfully created artifacts for {success_count}/{len(dates_to_process)} days")

    # Update latest.json
    if success_count > 0:
        latest_date = max(dates_to_process)
        latest_date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
        s3.write_json({
            'date': latest_date_str,
            'timestamp': datetime.now().isoformat(),
            'backfilled': True,
            'days_backfilled': success_count
        }, 'daily/latest.json')
        print(f"   Updated daily/latest.json to {latest_date_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill historical data')
    parser.add_argument('--bucket', type=str, default='investment-system-data')
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--days', type=int, default=180, help='Days to backfill')
    parser.add_argument('--fred-key', type=str, default='', help='FRED API key (optional)')
    args = parser.parse_args()

    backfill(
        bucket=args.bucket,
        region=args.region,
        days=args.days,
        fred_key=args.fred_key
    )
