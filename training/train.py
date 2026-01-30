#!/usr/bin/env python3
"""
Main training orchestrator.

Downloads historical data from S3, trains regime and health models,
uploads trained models to S3, and updates the latest.json pointer.

Usage:
    python train.py --config config/aws_config.json
    python train.py --bucket investment-system-data --region us-east-1
"""

import os
import sys
import json
import pickle
import argparse
import boto3
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_regime import train_regime_model, save_regime_model
from training.train_health import train_health_model, save_health_model
from training.utils.data_loader import S3DataDownloader


def download_historical_data(
    bucket: str,
    region: str,
    max_days: int = 365,
    cache_dir: str = '/tmp/training_cache'
):
    """Download historical data from S3."""
    print(f"Downloading historical data from s3://{bucket}...")

    downloader = S3DataDownloader(bucket, region, cache_dir)

    # Try to get consolidated historical data first
    try:
        prices_df = downloader.download_parquet('data/prices_historical.parquet')
        features_df = downloader.download_parquet('data/features_historical.parquet')
        context_df = downloader.download_parquet('data/context_historical.parquet')

        if not prices_df.empty and not features_df.empty and not context_df.empty:
            print(f"Loaded consolidated historical data")
            return prices_df, features_df, context_df
    except Exception as e:
        print(f"No consolidated data found, building from daily artifacts...")

    # Build from daily artifacts
    prices_df, features_df, context_df = downloader.build_historical_dataset(max_days=max_days)

    return prices_df, features_df, context_df


def upload_model_to_s3(
    s3_client,
    bucket: str,
    local_path: str,
    s3_key: str
):
    """Upload model file to S3."""
    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}...")
    s3_client.upload_file(local_path, bucket, s3_key)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser(description='Train investment system models')
    parser.add_argument('--config', type=str, help='Path to AWS config JSON')
    parser.add_argument('--bucket', type=str, default='investment-system-data', help='S3 bucket name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--max-days', type=int, default=365, help='Max days of historical data')
    parser.add_argument('--regime-type', type=str, default='gru', choices=['gru', 'transformer'])
    parser.add_argument('--health-type', type=str, default='autoencoder', choices=['autoencoder', 'vae'])
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--skip-upload', action='store_true', help='Skip S3 upload (local testing)')
    parser.add_argument('--output-dir', type=str, default='/tmp/models', help='Local output directory')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        bucket = config.get('s3_bucket', args.bucket)
        region = config.get('aws_region', args.region)
    else:
        bucket = args.bucket
        region = args.region

    print("=" * 60)
    print("Investment System Model Training")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"S3 Bucket: {bucket}")
    print(f"Region: {region}")
    print(f"Max Days: {args.max_days}")
    print("=" * 60)

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=region)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Version string
    version = datetime.now().strftime('%Y%m%d')

    # Download historical data
    print("\n[1/5] Downloading historical data...")
    prices_df, features_df, context_df = download_historical_data(
        bucket, region, args.max_days
    )

    if context_df.empty:
        print("ERROR: No context data available. Cannot train regime model.")
        print("Please ensure daily pipeline has been running to generate data.")
        sys.exit(1)

    if features_df.empty:
        print("ERROR: No features data available. Cannot train health model.")
        print("Please ensure daily pipeline has been running to generate data.")
        sys.exit(1)

    print(f"Data loaded: {len(context_df)} context rows, {len(features_df)} feature rows")

    # Train regime model
    print("\n[2/5] Training regime model...")
    regime_model, regime_history = train_regime_model(
        context_df=context_df,
        model_type=args.regime_type,
        epochs=args.epochs,
        save_dir=os.path.join(args.output_dir, 'regime_checkpoints')
    )

    regime_path = os.path.join(args.output_dir, f'regime_v{version}.pkl')
    save_regime_model(regime_model, regime_history, regime_path, version)

    # Train health model
    print("\n[3/5] Training health model...")
    health_model, health_history = train_health_model(
        features_df=features_df,
        model_type=args.health_type,
        epochs=args.epochs,
        save_dir=os.path.join(args.output_dir, 'health_checkpoints')
    )

    health_path = os.path.join(args.output_dir, f'health_v{version}.pkl')
    save_health_model(health_model, health_history, health_path, version)

    # Upload to S3
    if not args.skip_upload:
        print("\n[4/5] Uploading models to S3...")

        # Upload regime model
        s3_regime_key = f'models/regime_v{version}.pkl'
        upload_model_to_s3(s3, bucket, regime_path, s3_regime_key)

        # Upload health model
        s3_health_key = f'models/health_v{version}.pkl'
        upload_model_to_s3(s3, bucket, health_path, s3_health_key)

        # Update latest.json pointer
        print("\n[5/5] Updating latest.json...")
        latest_config = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'regime_model': s3_regime_key,
            'health_model': s3_health_key,
            'regime_metrics': {
                'test_accuracy': regime_history.get('test_accuracy'),
                'test_f1_macro': regime_history.get('test_f1_macro'),
                'model_type': regime_history.get('model_type')
            },
            'health_metrics': {
                'test_recon_mse': health_history.get('test_recon_mse'),
                'test_health_correlation': health_history.get('test_health_correlation'),
                'model_type': health_history.get('model_type')
            }
        }

        latest_path = os.path.join(args.output_dir, 'latest.json')
        with open(latest_path, 'w') as f:
            json.dump(latest_config, f, indent=2)

        s3.upload_file(latest_path, bucket, 'models/latest.json')
        print(f"  Updated models/latest.json")
    else:
        print("\n[4/5] Skipping S3 upload (--skip-upload)")
        print("[5/5] Skipping latest.json update")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Version: {version}")
    print(f"\nRegime Model:")
    print(f"  Type: {regime_history.get('model_type')}")
    print(f"  Test Accuracy: {regime_history.get('test_accuracy', 0):.4f}")
    print(f"  Test F1 (macro): {regime_history.get('test_f1_macro', 0):.4f}")
    print(f"  Best Epoch: {regime_history.get('best_epoch')}")

    print(f"\nHealth Model:")
    print(f"  Type: {health_history.get('model_type')}")
    print(f"  Test Recon MSE: {health_history.get('test_recon_mse', 0):.4f}")
    print(f"  Test Health Corr: {health_history.get('test_health_correlation', 0):.4f}")
    print(f"  Best Epoch: {health_history.get('best_epoch')}")

    if not args.skip_upload:
        print(f"\nModels uploaded to:")
        print(f"  s3://{bucket}/models/regime_v{version}.pkl")
        print(f"  s3://{bucket}/models/health_v{version}.pkl")
        print(f"  s3://{bucket}/models/latest.json")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
