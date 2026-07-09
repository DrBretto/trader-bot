"""GDELT sentiment data ingestion."""

import pandas as pd
import requests
import zipfile
from io import BytesIO
from typing import Dict, Optional


def fetch_gdelt_daily_aggregate(date: pd.Timestamp) -> Dict[str, float]:
    """
    Fetch GDELT daily aggregate features.

    Note: For Phase 1, GDELT is optional. If the fetch fails, we return
    placeholder values and mark gdelt_available=False.

    Args:
        date: Date to fetch

    Returns:
        Dict with keys: gdelt_doc_count, gdelt_avg_tone, gdelt_tone_std, gdelt_neg_tone_share
    """
    date_str = date.strftime('%Y%m%d')

    # Default values if fetch fails
    default_values = {
        'gdelt_doc_count': 0,
        'gdelt_avg_tone': 0.0,
        'gdelt_tone_std': 0.0,
        'gdelt_neg_tone_share': 0.0,
        'gdelt_available': False
    }

    # Try to fetch GKG counts file
    url = f"http://data.gdeltproject.org/gdeltv2/{date_str}.gkgcounts.csv.zip"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"GDELT: No data for {date_str} (status {response.status_code})")
            return default_values

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # List files in the zip
            file_list = z.namelist()

            # Find the CSV file
            csv_file = None
            for f in file_list:
                if f.endswith('.csv'):
                    csv_file = f
                    break

            if csv_file is None:
                print(f"GDELT: No CSV found in zip for {date_str}")
                return default_values

            with z.open(csv_file) as f:
                # GKG counts file is tab-separated with specific columns
                df = pd.read_csv(
                    f,
                    sep='\t',
                    header=None,
                    names=['date', 'count', 'type', 'fips', 'name'],
                    on_bad_lines='skip'
                )

        # Aggregate to daily features
        doc_count = len(df)

        return {
            'gdelt_doc_count': doc_count,
            'gdelt_avg_tone': 0.0,  # Placeholder - tone data requires gkg.csv parsing
            'gdelt_tone_std': 0.0,
            'gdelt_neg_tone_share': 0.0,
            'gdelt_available': True
        }

    except zipfile.BadZipFile:
        print(f"GDELT: Invalid zip file for {date_str}")
        return default_values
    except Exception as e:
        print(f"GDELT fetch failed for {date_str}: {e}")
        return default_values


def run(run_date: str) -> Dict[str, float]:
    """
    Run GDELT data ingestion for the given date.

    For Phase 1, this is a simplified implementation that just fetches
    basic document counts. Tone analysis is deferred to later phases.

    Args:
        run_date: Date string in YYYY-MM-DD format

    Returns:
        Dict with GDELT aggregate features
    """
    date = pd.to_datetime(run_date)

    print(f"Ingesting GDELT data for {run_date}...")

    # Try to fetch for the target date
    result = fetch_gdelt_daily_aggregate(date)

    # If today's data isn't available yet, try yesterday
    if not result.get('gdelt_available', False):
        yesterday = date - pd.Timedelta(days=1)
        print(f"  Trying previous day: {yesterday.strftime('%Y-%m-%d')}")
        result = fetch_gdelt_daily_aggregate(yesterday)

    if result.get('gdelt_available', False):
        print(f"  GDELT doc count: {result['gdelt_doc_count']}")
    else:
        print("  GDELT data unavailable - using placeholders")

    return result
