"""
Data loading utilities for training.

Handles downloading historical data from S3, preparing sequences,
and creating PyTorch DataLoaders.
"""

import os
import json
import boto3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import io


class S3DataDownloader:
    """Download and cache historical data from S3."""

    def __init__(self, bucket: str, region: str = 'us-east-1', cache_dir: str = '/tmp/training_cache'):
        self.bucket = bucket
        self.region = region
        self.cache_dir = cache_dir
        self.s3 = boto3.client('s3', region_name=region)
        os.makedirs(cache_dir, exist_ok=True)

    def download_parquet(self, key: str, use_cache: bool = True) -> pd.DataFrame:
        """Download parquet file from S3."""
        cache_path = os.path.join(self.cache_dir, key.replace('/', '_'))

        if use_cache and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()))

            if use_cache:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_parquet(cache_path)

            return df
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return pd.DataFrame()

    def download_json(self, key: str) -> dict:
        """Download JSON file from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return {}

    def list_daily_artifacts(self, prefix: str = 'daily/', days: int = 365) -> List[str]:
        """List available daily artifact dates."""
        dates = []
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
            for obj in page.get('CommonPrefixes', []):
                date_str = obj['Prefix'].replace(prefix, '').rstrip('/')
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date_str)
                except ValueError:
                    continue

        return sorted(dates)[-days:]

    def build_historical_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_days: int = 365
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build historical dataset from daily artifacts.

        Returns:
            (prices_df, features_df, context_df)
        """
        dates = self.list_daily_artifacts(days=max_days)

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        print(f"Building dataset from {len(dates)} days...")

        prices_list = []
        features_list = []
        context_list = []

        for date in dates:
            # Try to load parquet files
            try:
                prices = self.download_parquet(f'daily/{date}/prices.parquet', use_cache=True)
                if not prices.empty:
                    prices['date'] = pd.to_datetime(date)
                    prices_list.append(prices)
            except:
                pass

            try:
                features = self.download_parquet(f'daily/{date}/features.parquet', use_cache=True)
                if not features.empty:
                    features['date'] = pd.to_datetime(date)
                    features_list.append(features)
            except:
                pass

            try:
                context = self.download_parquet(f'daily/{date}/context.parquet', use_cache=True)
                if not context.empty:
                    context_list.append(context)
            except:
                pass

        prices_df = pd.concat(prices_list, ignore_index=True) if prices_list else pd.DataFrame()
        features_df = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
        context_df = pd.concat(context_list, ignore_index=True) if context_list else pd.DataFrame()

        print(f"Loaded: {len(prices_df)} price rows, {len(features_df)} feature rows, {len(context_df)} context rows")

        return prices_df, features_df, context_df


class RegimeDataset(Dataset):
    """
    PyTorch Dataset for regime classification.

    Creates sequences of context features for training the regime model.
    """

    def __init__(
        self,
        context_df: pd.DataFrame,
        feature_cols: List[str],
        seq_length: int = 21,
        label_col: Optional[str] = None,
        regime_labels: Optional[List[str]] = None
    ):
        """
        Args:
            context_df: DataFrame with context features
            feature_cols: List of feature column names to use
            seq_length: Sequence length for GRU/Transformer
            label_col: Column name for regime labels (if supervised)
            regime_labels: List of regime label strings
        """
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.regime_labels = regime_labels or []

        # Sort by date
        context_df = context_df.sort_values('date').reset_index(drop=True)

        # Extract features
        self.features = context_df[feature_cols].values.astype(np.float32)
        self.dates = context_df['date'].values

        # Extract labels if available
        if label_col and label_col in context_df.columns:
            self.labels = context_df[label_col].values
            self.label_to_idx = {l: i for i, l in enumerate(regime_labels)}
        else:
            self.labels = None

        # Normalize features
        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero

        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)

        # Calculate valid indices (need seq_length history)
        self.valid_indices = list(range(seq_length - 1, len(self.features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        # Get sequence
        start_idx = actual_idx - self.seq_length + 1
        seq = self.features[start_idx:actual_idx + 1]

        # Normalize
        seq = (seq - self.feature_mean) / self.feature_std

        result = {
            'features': torch.tensor(seq, dtype=torch.float32),
            'date': str(self.dates[actual_idx])
        }

        if self.labels is not None:
            label = self.labels[actual_idx]
            if isinstance(label, str):
                label_idx = self.label_to_idx.get(label, 0)
            else:
                label_idx = int(label)
            result['label'] = torch.tensor(label_idx, dtype=torch.long)

        return result


class HealthDataset(Dataset):
    """
    PyTorch Dataset for asset health scoring.

    Creates batches of asset features for training the autoencoder.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        label_cols: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            features_df: DataFrame with asset features
            feature_cols: List of feature column names to use
            label_cols: Dict mapping output names to column names
                        e.g., {'health': 'health_score', 'vol': 'vol_bucket'}
        """
        self.feature_cols = feature_cols
        self.label_cols = label_cols or {}

        # Extract features
        self.features = features_df[feature_cols].values.astype(np.float32)

        # Normalize features
        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0

        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)

        # Extract labels
        self.labels = {}
        for name, col in self.label_cols.items():
            if col in features_df.columns:
                self.labels[name] = features_df[col].values

        # Store metadata
        if 'symbol' in features_df.columns:
            self.symbols = features_df['symbol'].values
        else:
            self.symbols = None

        if 'date' in features_df.columns:
            self.dates = features_df['date'].values
        else:
            self.dates = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Normalize features
        features = (self.features[idx] - self.feature_mean) / self.feature_std

        result = {
            'features': torch.tensor(features, dtype=torch.float32)
        }

        for name, values in self.labels.items():
            if isinstance(values[idx], (int, float)):
                result[name] = torch.tensor(values[idx], dtype=torch.float32)
            else:
                result[name] = values[idx]

        return result

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """Return normalization parameters for inference."""
        return {
            'mean': self.feature_mean,
            'std': self.feature_std
        }


def create_regime_dataloaders(
    context_df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 21,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    label_col: Optional[str] = None,
    regime_labels: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for regime model.

    Uses chronological split (no data leakage).
    """
    # Sort by date
    context_df = context_df.sort_values('date').reset_index(drop=True)

    n = len(context_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = context_df.iloc[:train_end]
    val_df = context_df.iloc[train_end:val_end]
    test_df = context_df.iloc[val_end:]

    train_dataset = RegimeDataset(
        train_df, feature_cols, seq_length, label_col, regime_labels
    )
    val_dataset = RegimeDataset(
        val_df, feature_cols, seq_length, label_col, regime_labels
    )
    test_dataset = RegimeDataset(
        test_df, feature_cols, seq_length, label_col, regime_labels
    )

    # Use training stats for normalization on val/test
    val_dataset.feature_mean = train_dataset.feature_mean
    val_dataset.feature_std = train_dataset.feature_std
    test_dataset.feature_mean = train_dataset.feature_mean
    test_dataset.feature_std = train_dataset.feature_std

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def create_health_dataloaders(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    label_cols: Optional[Dict[str, str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for health model.

    Uses chronological split based on dates.
    """
    # Sort by date
    features_df = features_df.sort_values('date').reset_index(drop=True)

    # Get unique dates and split
    dates = features_df['date'].unique()
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_df = features_df[features_df['date'].isin(train_dates)]
    val_df = features_df[features_df['date'].isin(val_dates)]
    test_df = features_df[features_df['date'].isin(test_dates)]

    train_dataset = HealthDataset(train_df, feature_cols, label_cols)
    val_dataset = HealthDataset(val_df, feature_cols, label_cols)
    test_dataset = HealthDataset(test_df, feature_cols, label_cols)

    # Use training stats for normalization
    val_dataset.feature_mean = train_dataset.feature_mean
    val_dataset.feature_std = train_dataset.feature_std
    test_dataset.feature_mean = train_dataset.feature_mean
    test_dataset.feature_std = train_dataset.feature_std

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
