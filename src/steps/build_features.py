"""Feature engineering step for the daily pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

from src.utils.feature_utils import (
    compute_asset_features,
    compute_relative_strength,
    compute_context_features
)
from src.steps.ingest_fred import get_latest_values


def run(
    prices_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    gdelt_data: Dict[str, float],
    vvix_value: float = None,
    skew_value: float = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features from raw price and macro data.

    Args:
        prices_df: Price data with columns [date, symbol, open, high, low, close, volume]
        fred_df: FRED data with columns [date, series_id, value]
        gdelt_data: GDELT aggregate features dict
        vvix_value: Latest VVIX index value (from Stooq), or None
        skew_value: Latest SKEW index value (from Stooq), or None

    Returns:
        Tuple of (features_df, context_df)
    """
    print("Building features...")

    if len(prices_df) == 0:
        print("  No price data available")
        return pd.DataFrame(), pd.DataFrame()

    # Get list of symbols
    symbols = prices_df['symbol'].unique()
    print(f"  Processing {len(symbols)} symbols")

    # Compute features for each symbol
    features_list = []

    for symbol in symbols:
        symbol_data = prices_df[prices_df['symbol'] == symbol].copy()

        if len(symbol_data) < 10:  # Need minimum history
            continue

        # Compute technical features
        symbol_features = compute_asset_features(symbol_data)
        symbol_features['symbol'] = symbol

        features_list.append(symbol_features)

    if not features_list:
        print("  No features computed")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all symbol features
    features_df = pd.concat(features_list, ignore_index=True)

    # Get SPY data for relative strength
    spy_features = features_df[features_df['symbol'] == 'SPY'].copy()

    if len(spy_features) > 0:
        # Compute relative strength for all symbols
        rel_strength_list = []

        for symbol in symbols:
            if symbol == 'SPY':
                # SPY relative strength to itself is 0
                symbol_data = features_df[features_df['symbol'] == symbol].copy()
                symbol_data['rel_strength_21d'] = 0.0
                symbol_data['rel_strength_63d'] = 0.0
                rel_strength_list.append(symbol_data)
            else:
                symbol_data = features_df[features_df['symbol'] == symbol].copy()
                if len(symbol_data) > 0:
                    symbol_data = compute_relative_strength(symbol_data, spy_features)
                    rel_strength_list.append(symbol_data)

        features_df = pd.concat(rel_strength_list, ignore_index=True)
    else:
        # If SPY is missing, set relative strength to 0
        features_df['rel_strength_21d'] = 0.0
        features_df['rel_strength_63d'] = 0.0

    # Get proxy ETF data for context features
    proxy_symbols = {
        'SPY': features_df[features_df['symbol'] == 'SPY'],
        'TLT': features_df[features_df['symbol'] == 'TLT'],
        'HYG': features_df[features_df['symbol'] == 'HYG'],
        'IEF': features_df[features_df['symbol'] == 'IEF'],
        'VIXY': features_df[features_df['symbol'] == 'VIXY']
    }

    # Get latest FRED values
    fred_latest = get_latest_values(fred_df) if len(fred_df) > 0 else {}

    # Compute context features
    context_df = compute_context_features(
        spy_df=proxy_symbols['SPY'],
        tlt_df=proxy_symbols['TLT'],
        hyg_df=proxy_symbols['HYG'],
        ief_df=proxy_symbols['IEF'],
        vixy_df=proxy_symbols['VIXY'],
        fred_data=fred_latest,
        gdelt_data=gdelt_data,
        vvix_value=vvix_value,
        skew_value=skew_value
    )

    # Clean up features_df - keep only latest values per symbol for decision making
    features_df = features_df.sort_values(['symbol', 'date'])

    # Select output columns
    feature_columns = [
        'date', 'symbol', 'close',
        'return_1d', 'return_5d', 'return_21d', 'return_63d',
        'vol_21d', 'vol_63d', 'drawdown_63d', 'trend_63d',
        'volume_ma21', 'rel_strength_21d', 'rel_strength_63d'
    ]

    # Only keep columns that exist
    feature_columns = [c for c in feature_columns if c in features_df.columns]
    features_df = features_df[feature_columns]

    print(f"  Features computed: {len(features_df)} rows, {len(feature_columns)} columns")
    print(f"  Context features: {list(context_df.columns)}")

    return features_df, context_df
