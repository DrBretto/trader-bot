"""Baseline asset health model using rank-based scoring."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def baseline_health_model(
    features_df: pd.DataFrame,
    latest_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Deterministic baseline health scoring using cross-sectional ranks.

    Args:
        features_df: DataFrame with all symbols' features
        latest_date: Date to score

    Returns:
        DataFrame with columns: symbol, health_score, vol_bucket, behavior, latent
    """
    # Filter to latest date
    latest = features_df[features_df['date'] == latest_date].copy()

    if len(latest) == 0:
        # Try to get the most recent data available
        if len(features_df) > 0:
            latest_date = features_df['date'].max()
            latest = features_df[features_df['date'] == latest_date].copy()

    if len(latest) == 0:
        return pd.DataFrame(columns=['symbol', 'health_score', 'vol_bucket', 'behavior', 'latent'])

    # Compute cross-sectional ranks (0-1 percentile)
    latest['mom_63_rank'] = latest['return_63d'].rank(pct=True)
    latest['mom_21_rank'] = latest['return_21d'].rank(pct=True)
    latest['vol_63_rank'] = latest['vol_63d'].rank(pct=True)
    latest['dd_63_rank'] = (-latest['drawdown_63d']).rank(pct=True)  # Less negative = better
    latest['rs_63_rank'] = latest['rel_strength_63d'].rank(pct=True)

    # Fill NaN ranks with 0.5 (neutral)
    rank_cols = ['mom_63_rank', 'mom_21_rank', 'vol_63_rank', 'dd_63_rank', 'rs_63_rank']
    for col in rank_cols:
        latest[col] = latest[col].fillna(0.5)

    # Composite scores
    latest['momentum'] = 0.6 * latest['mom_63_rank'] + 0.4 * latest['mom_21_rank']
    latest['risk'] = 0.6 * latest['vol_63_rank'] + 0.4 * (1 - latest['dd_63_rank'])

    # Health formula:
    # 45% momentum + 35% relative strength + 20% inverse risk
    latest['health_score'] = (
        0.45 * latest['momentum'] +
        0.35 * latest['rs_63_rank'] +
        0.20 * (1 - latest['risk'])
    ).clip(0, 1)

    # Volatility bucket
    latest['vol_bucket'] = pd.cut(
        latest['vol_63_rank'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['low', 'med', 'high'],
        include_lowest=True
    )

    # Fill missing vol_bucket with 'med'
    latest['vol_bucket'] = latest['vol_bucket'].fillna('med')

    # Behavior classification
    def classify_behavior(row):
        mom_rank = row.get('mom_63_rank', 0.5)
        dd_rank = row.get('dd_63_rank', 0.5)
        vol_rank = row.get('vol_63_rank', 0.5)

        if mom_rank > 0.66 and dd_rank > 0.5:
            return 'momentum'
        elif vol_rank < 0.33 and mom_rank > 0.5:
            return 'quality'
        elif dd_rank < 0.33:
            return 'distressed'
        else:
            return 'mixed'

    latest['behavior'] = latest.apply(classify_behavior, axis=1)

    # Dummy latent vector (16 dimensions, will be real autoencoder output in Phase 2)
    latest['latent'] = [[0.0] * 16] * len(latest)

    return latest[['symbol', 'health_score', 'vol_bucket', 'behavior', 'latent']]


def get_health_tier(health_score: float) -> str:
    """Get tier label for a health score."""
    if health_score >= 0.75:
        return 'excellent'
    elif health_score >= 0.60:
        return 'good'
    elif health_score >= 0.45:
        return 'fair'
    elif health_score >= 0.35:
        return 'poor'
    else:
        return 'critical'


def get_top_candidates(
    health_df: pd.DataFrame,
    n: int = 10,
    min_health: float = 0.50
) -> pd.DataFrame:
    """Get top N candidates by health score."""
    candidates = health_df[health_df['health_score'] >= min_health].copy()
    candidates = candidates.sort_values('health_score', ascending=False)
    return candidates.head(n)
