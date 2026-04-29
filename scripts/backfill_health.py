#!/usr/bin/env python3
"""Backfill per-symbol asset_health into historical inference.json artifacts.

Reads daily/{date}/features.parquet, runs baseline_health_model, and writes
the health scores into existing inference.json files that lack asset_health.

Usage:
    source .venv/bin/activate
    AWS_PROFILE=personal python scripts/backfill_health.py \
        --bucket investment-system-data --region us-east-1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.utils.s3_client import S3Client
from src.models.baseline_health import baseline_health_model


def main():
    parser = argparse.ArgumentParser(description='Backfill asset_health into historical inference')
    parser.add_argument('--bucket', default='investment-system-data')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without writing')
    args = parser.parse_args()

    os.environ.setdefault('AWS_DEFAULT_REGION', args.region)
    s3 = S3Client(args.bucket)

    dates = sorted(s3.list_daily_dates(max_days=900))
    print(f'Found {len(dates)} dates')

    enriched = 0
    skipped = 0
    already_ok = 0

    for i, date_str in enumerate(dates):
        inference = s3.read_json(f'daily/{date_str}/inference.json')
        if not inference or not isinstance(inference, dict):
            skipped += 1
            continue

        # Check if already has valid asset_health
        ah = inference.get('asset_health')
        if isinstance(ah, list) and len(ah) > 0 and isinstance(ah[0], dict) and 'symbol' in ah[0]:
            already_ok += 1
            continue

        # Load features and compute baseline health
        features = s3.read_parquet(f'daily/{date_str}/features.parquet')
        if features is None or len(features) == 0:
            skipped += 1
            continue

        latest_date = features['date'].max()
        health_df = baseline_health_model(features, latest_date)
        if len(health_df) == 0:
            skipped += 1
            continue

        asset_health = []
        for _, row in health_df.iterrows():
            asset_health.append({
                'symbol': row['symbol'],
                'health_score': float(row['health_score']),
                'vol_bucket': str(row['vol_bucket']),
                'behavior': row['behavior'],
                'latent': row['latent'],
            })

        inference['asset_health'] = asset_health

        # Normalize regime keys while we're here
        regime = inference.get('regime', {})
        if 'label' not in regime and 'regime_label' in regime:
            regime['label'] = regime['regime_label']
        if 'probs' not in regime and 'regime_probs' in regime:
            regime['probs'] = regime['regime_probs']

        if args.dry_run:
            print(f'  [{i+1}/{len(dates)}] {date_str}: would enrich with {len(asset_health)} health scores')
        else:
            s3.write_json(inference, f'daily/{date_str}/inference.json')
            if (enriched + 1) % 20 == 0 or i == len(dates) - 1:
                print(f'  [{i+1}/{len(dates)}] {date_str}: enriched {len(asset_health)} symbols')

        enriched += 1

    print(f'\nDone. Enriched: {enriched}, Already OK: {already_ok}, Skipped: {skipped}')


if __name__ == '__main__':
    main()
