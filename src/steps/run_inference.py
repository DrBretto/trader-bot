"""Inference step - runs regime and health models."""

import pandas as pd
from typing import Dict, Any
from datetime import datetime

from src.models.baseline_regime import baseline_regime_model
from src.models.baseline_health import baseline_health_model


def run(
    features_df: pd.DataFrame,
    context_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run model inference to get regime and asset health predictions.

    Args:
        features_df: Asset features DataFrame
        context_df: Market context DataFrame
        config: Configuration dict

    Returns:
        Inference output with regime and asset_health
    """
    print("Running inference...")

    # Get the latest date
    if len(context_df) > 0:
        latest_date = pd.to_datetime(context_df['date'].iloc[0])
    elif len(features_df) > 0:
        latest_date = features_df['date'].max()
    else:
        latest_date = pd.Timestamp.now()

    # Get context row for regime model
    if len(context_df) > 0:
        context_row = context_df.iloc[0]
    else:
        # Create empty context
        context_row = pd.Series({
            'spy_return_21d': 0,
            'spy_vol_21d': 0.15,
            'credit_spread_proxy': 0,
            'vixy_return_21d': 0
        })

    # Run regime model
    regime_output = baseline_regime_model(context_row)
    print(f"  Regime: {regime_output['regime_label']}")

    # Run health model
    health_df = baseline_health_model(features_df, latest_date)
    print(f"  Health scores computed for {len(health_df)} symbols")

    # Convert health DataFrame to list of dicts
    asset_health = []
    for _, row in health_df.iterrows():
        asset_health.append({
            'symbol': row['symbol'],
            'health_score': float(row['health_score']),
            'vol_bucket': str(row['vol_bucket']),
            'behavior': row['behavior'],
            'latent': row['latent']
        })

    # Build inference output
    inference_output = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'model_versions': {
            'regime': 'baseline_v1',
            'health': 'baseline_v1'
        },
        'regime': {
            'label': regime_output['regime_label'],
            'probs': regime_output['regime_probs'],
            'embedding': regime_output['regime_embedding']
        },
        'asset_health': asset_health
    }

    return inference_output
