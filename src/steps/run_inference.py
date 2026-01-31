"""Inference step - runs regime and health models."""

import pandas as pd
from typing import Dict, Any
from datetime import datetime

from src.models.baseline_regime import baseline_regime_model
from src.models.baseline_health import baseline_health_model
from src.models.loader import ModelLoader
from src.utils.s3_client import S3Client


def run(
    features_df: pd.DataFrame,
    context_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run model inference to get regime and asset health predictions.

    Supports ensemble regime models with disagreement detection.

    Args:
        features_df: Asset features DataFrame
        context_df: Market context DataFrame
        config: Configuration dict

    Returns:
        Inference output with regime, asset_health, and ensemble metrics
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

    # Try to load trained models from S3
    bucket = config.get('s3_bucket', 'investment-system-data')
    s3 = S3Client(bucket)
    loader = ModelLoader(s3)
    model_versions = loader.load_models()

    # Run regime model (uses ensemble if available)
    regime_output = loader.predict_regime(context_row)
    print(f"  Regime: {regime_output['regime_label']}")

    # Log ensemble info if available
    if model_versions.get('ensemble'):
        print(f"  Ensemble confidence: {regime_output.get('confidence', 1.0):.2f}")
        print(f"  Ensemble agreement: {regime_output.get('agreement', 1.0):.2f}")
        if regime_output.get('disagreement', 0) > 0.3:
            print(f"  WARNING: High model disagreement ({regime_output['disagreement']:.2f})")

    # Run health model
    if loader.using_trained_models and loader.health_model:
        health_df = loader.predict_health(features_df, latest_date)
    else:
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

    # Build inference output with ensemble metrics
    inference_output = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'model_versions': model_versions,
        'regime': {
            'label': regime_output['regime_label'],
            'probs': regime_output.get('regime_probs', {}),
            'embedding': regime_output.get('regime_embedding', []),
            'confidence': regime_output.get('confidence', 1.0),
            'disagreement': regime_output.get('disagreement', 0.0),
            'agreement': regime_output.get('agreement', 1.0),
            'position_size_multiplier': regime_output.get('position_size_multiplier', 1.0),
        },
        'asset_health': asset_health
    }

    # Add individual model predictions if ensemble
    if model_versions.get('ensemble'):
        inference_output['regime']['gru_prediction'] = regime_output.get('gru_prediction', {})
        inference_output['regime']['transformer_prediction'] = regime_output.get('transformer_prediction', {})

    return inference_output
