"""
GDELT historical data backfill for training.

Downloads GKG files from GDELT and extracts tone/sentiment metrics.
Samples 4 files per day (every 6 hours) to keep download size manageable.

Output: training/data/historical_gdelt.parquet
"""

import os
import time
import argparse
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# GKG file timestamps to sample (every 6 hours)
SAMPLE_HOURS = ['000000', '060000', '120000', '180000']

# Column index for TONE in GKG 2.0 format
TONE_COLUMN_INDEX = 15


def parse_tone_field(tone_str: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse GDELT TONE field.

    Format: tone,positive,negative,polarity,activity,selfref,wordcount

    Returns:
        Tuple of (tone, positive_score, negative_score) or None if invalid
    """
    if not tone_str or pd.isna(tone_str):
        return None

    try:
        parts = str(tone_str).split(',')
        if len(parts) >= 3:
            tone = float(parts[0])
            positive = float(parts[1])
            negative = float(parts[2])
            return (tone, positive, negative)
    except (ValueError, IndexError):
        pass

    return None


def fetch_gkg_file(url: str, timeout: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch and parse a single GKG file.

    Returns:
        DataFrame with columns [tone, positive, negative] or None if failed
    """
    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            return None

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Find the CSV file in the zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None

            with z.open(csv_files[0]) as f:
                # GKG is tab-separated, no header
                # We only need the TONE column (index 15)
                df = pd.read_csv(
                    f,
                    sep='\t',
                    header=None,
                    usecols=[TONE_COLUMN_INDEX],
                    names=['tone_raw'],
                    on_bad_lines='skip',
                    dtype=str
                )

        # Parse tone fields
        records = []
        for tone_raw in df['tone_raw']:
            parsed = parse_tone_field(tone_raw)
            if parsed:
                records.append({
                    'tone': parsed[0],
                    'positive': parsed[1],
                    'negative': parsed[2]
                })

        if not records:
            return None

        return pd.DataFrame(records)

    except zipfile.BadZipFile:
        return None
    except Exception as e:
        return None


def fetch_day_gdelt(date: datetime, delay: float = 0.5) -> Dict[str, float]:
    """
    Fetch GDELT data for a single day by sampling 4 files.

    Returns:
        Dict with aggregated metrics for the day
    """
    date_str = date.strftime('%Y%m%d')

    all_tones = []
    all_positive = []
    all_negative = []
    doc_count = 0

    for hour in SAMPLE_HOURS:
        url = f"http://data.gdeltproject.org/gdeltv2/{date_str}{hour}.gkg.csv.zip"

        df = fetch_gkg_file(url)

        if df is not None and len(df) > 0:
            all_tones.extend(df['tone'].tolist())
            all_positive.extend(df['positive'].tolist())
            all_negative.extend(df['negative'].tolist())
            doc_count += len(df)

        time.sleep(delay)

    if not all_tones:
        return {
            'date': date,
            'gdelt_doc_count': 0,
            'gdelt_avg_tone': np.nan,
            'gdelt_tone_std': np.nan,
            'gdelt_neg_tone_share': np.nan,
            'gdelt_available': False
        }

    # Compute aggregates
    tones = np.array(all_tones)

    return {
        'date': date,
        'gdelt_doc_count': doc_count,
        'gdelt_avg_tone': float(np.mean(tones)),
        'gdelt_tone_std': float(np.std(tones)),
        'gdelt_neg_tone_share': float(np.sum(tones < 0) / len(tones)),
        'gdelt_available': True
    }


def run_backfill(
    output_path: str,
    start_date: str,
    end_date: str,
    workers: int = 4,
    delay: float = 0.3
) -> pd.DataFrame:
    """
    Run GDELT historical backfill.

    Args:
        output_path: Path to save parquet file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        workers: Number of parallel workers
        delay: Delay between requests per worker

    Returns:
        DataFrame with daily GDELT metrics
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Generate date range
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    print(f"Fetching GDELT data for {len(dates)} days...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Workers: {workers}")
    print()

    results = []
    failed_dates = []

    # Process dates with progress indicator
    for i, date in enumerate(dates):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(dates)} ({date.strftime('%Y-%m-%d')})")

        try:
            result = fetch_day_gdelt(date, delay=delay)
            results.append(result)

            if not result['gdelt_available']:
                failed_dates.append(date.strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"  ERROR on {date.strftime('%Y-%m-%d')}: {e}")
            failed_dates.append(date.strftime('%Y-%m-%d'))
            results.append({
                'date': date,
                'gdelt_doc_count': 0,
                'gdelt_avg_tone': np.nan,
                'gdelt_tone_std': np.nan,
                'gdelt_neg_tone_share': np.nan,
                'gdelt_available': False
            })

    df = pd.DataFrame(results)

    # Summary
    print()
    print("Backfill complete:")
    print(f"  Total days: {len(df)}")
    print(f"  Days with data: {df['gdelt_available'].sum()}")
    print(f"  Days missing: {len(failed_dates)}")

    if df['gdelt_available'].sum() > 0:
        valid = df[df['gdelt_available']]
        print(f"  Avg tone range: {valid['gdelt_avg_tone'].min():.2f} to {valid['gdelt_avg_tone'].max():.2f}")
        print(f"  Avg negative share: {valid['gdelt_neg_tone_share'].mean():.2%}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print()
    print(f"Saved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Backfill GDELT historical data')
    parser.add_argument(
        '--output',
        type=str,
        default='training/data/historical_gdelt.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2021-12-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between requests (seconds)'
    )

    args = parser.parse_args()

    run_backfill(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
