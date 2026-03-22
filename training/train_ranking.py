"""Train the cross-sectional ranking MLP on historical asset features.

Training target: percentile rank of 21-day forward return within the universe
on each date. The model learns to predict which assets will outperform peers.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES

logger = logging.getLogger(__name__)

FORWARD_DAYS = 21


def _prepare_ranking_data(
    features_df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    regime_labels: Optional[Dict[str, str]] = None,
    use_relative_targets: bool = False,
    universe_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Build cross-sectional ranking training data from historical features.

    For each date, compute the forward return rank of each asset,
    then pair current features with the realized rank as target.

    Args:
        regime_labels: Optional {date_str: regime_label} for regime conditioning.
        use_relative_targets: If True, target is rank of (asset_return - universe_mean)
            instead of rank of raw asset_return.

    Returns (X, y, normalization) where X is (N, F), y is (N, 1) in [0, 1].
    """
    df = features_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    # Compute forward returns per symbol
    df['fwd_return'] = df.groupby('symbol')['close'].transform(
        lambda s: s.shift(-forward_days) / s - 1
    )
    df = df.dropna(subset=['fwd_return'])

    if use_relative_targets:
        # Remove common market component: asset return minus universe mean on same date
        df['universe_mean'] = df.groupby('date')['fwd_return'].transform('mean')
        df['fwd_return_relative'] = df['fwd_return'] - df['universe_mean']
        df['fwd_rank'] = df.groupby('date')['fwd_return_relative'].rank(pct=True)
    else:
        df['fwd_rank'] = df.groupby('date')['fwd_return'].rank(pct=True)

    # Filter to rows with all required features
    for feat in RANKING_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    valid = df.dropna(subset=RANKING_FEATURES + ['fwd_rank'])
    if len(valid) == 0:
        raise ValueError("No valid training rows after filtering")

    # Compute normalization stats from training data
    normalization = {}
    for feat in RANKING_FEATURES:
        vals = valid[feat].values.astype(float)
        normalization[feat] = {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
        }

    # Build arrays
    X_raw = valid[RANKING_FEATURES].values.astype(np.float32)
    y = valid['fwd_rank'].values.astype(np.float32).reshape(-1, 1)

    # Normalize features
    X = np.zeros_like(X_raw)
    for i, feat in enumerate(RANKING_FEATURES):
        mean = normalization[feat]['mean']
        std = normalization[feat]['std']
        if std < 1e-8:
            std = 1.0
        X[:, i] = (X_raw[:, i] - mean) / std

    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Add regime one-hot conditioning if regime labels provided
    regime_classes = ['calm_uptrend', 'risk_on_trend', 'risk_off_trend', 'choppy', 'high_vol_panic']
    if regime_labels is not None:
        date_strs = valid['date'].dt.strftime('%Y-%m-%d').values
        regime_onehot = np.zeros((len(valid), len(regime_classes)), dtype=np.float32)
        for j, ds in enumerate(date_strs):
            regime = regime_labels.get(ds, 'choppy')
            if regime in regime_classes:
                regime_onehot[j, regime_classes.index(regime)] = 1.0
            else:
                regime_onehot[j, regime_classes.index('choppy')] = 1.0
        X = np.concatenate([X, regime_onehot], axis=1)
        # Record regime feature names in normalization for reference
        for rc in regime_classes:
            normalization[f'regime_{rc}'] = {'mean': 0.0, 'std': 1.0}
        logger.info("Added %d regime conditioning features", len(regime_classes))

    # Add structural asset identity features if universe metadata provided
    if universe_meta is not None:
        asset_classes = ['equity', 'bond', 'commodity', 'fx', 'vol']
        sector_groups = [
            'eq_broad', 'eq_sector', 'eq_intl', 'eq_factor',
            'eq_industry', 'eq_style', 'eq_theme',
            'bond_treas', 'bond_credit', 'bond_other',
            'commodity', 'fx', 'vol',
        ]
        _sector_to_group = {}
        for sector in universe_meta.values():
            s = sector.get('sector', '')
            if s.startswith('sector_') or s in ('broad', 'global'):
                _sector_to_group[s] = 'eq_broad' if s in ('broad', 'global') else 'eq_sector'
            elif s.startswith('international_') or s.startswith('country_') or s.startswith('region_'):
                _sector_to_group[s] = 'eq_intl'
            elif s.startswith('factor_'):
                _sector_to_group[s] = 'eq_factor'
            elif s.startswith('industry_'):
                _sector_to_group[s] = 'eq_industry'
            elif s.startswith('style_'):
                _sector_to_group[s] = 'eq_style'
            elif s.startswith('theme_'):
                _sector_to_group[s] = 'eq_theme'
            elif s.startswith('treas_'):
                _sector_to_group[s] = 'bond_treas'
            elif s.startswith('credit_'):
                _sector_to_group[s] = 'bond_credit'
            elif s in ('aggregate', 'muni'):
                _sector_to_group[s] = 'bond_other'
            elif s in ('gold', 'silver', 'broad_commodities', 'oil', 'natural_gas'):
                _sector_to_group[s] = 'commodity'
            elif s in ('usd', 'eur'):
                _sector_to_group[s] = 'fx'
            elif s == 'volatility':
                _sector_to_group[s] = 'vol'
            else:
                _sector_to_group[s] = 'eq_broad'

        symbols = valid['symbol'].values
        ac_onehot = np.zeros((len(valid), len(asset_classes)), dtype=np.float32)
        sg_onehot = np.zeros((len(valid), len(sector_groups)), dtype=np.float32)
        for j, sym in enumerate(symbols):
            meta = universe_meta.get(sym, {})
            ac = meta.get('asset_class', 'equity')
            sec = meta.get('sector', 'broad')
            if ac in asset_classes:
                ac_onehot[j, asset_classes.index(ac)] = 1.0
            sg = _sector_to_group.get(sec, 'eq_broad')
            if sg in sector_groups:
                sg_onehot[j, sector_groups.index(sg)] = 1.0

        X = np.concatenate([X, ac_onehot, sg_onehot], axis=1)
        for ac in asset_classes:
            normalization[f'ac_{ac}'] = {'mean': 0.0, 'std': 1.0}
        for sg in sector_groups:
            normalization[f'sg_{sg}'] = {'mean': 0.0, 'std': 1.0}
        logger.info("Added %d asset-class + %d sector-group features", len(asset_classes), len(sector_groups))

    logger.info(
        "Prepared %d training samples from %d dates, %d symbols (input_dim=%d)",
        len(X), valid['date'].nunique(), valid['symbol'].nunique(), X.shape[1],
    )
    return X, y, normalization


def train_ranking_model(
    features_df: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    save_dir: Optional[str] = None,
    forward_days: int = FORWARD_DAYS,
    regime_labels: Optional[Dict[str, str]] = None,
    use_relative_targets: bool = False,
    universe_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[RankingMLP, Dict[str, Any]]:
    """Train the ranking MLP and return (model, history)."""
    X, y, normalization = _prepare_ranking_data(
        features_df, forward_days,
        regime_labels=regime_labels,
        use_relative_targets=use_relative_targets,
        universe_meta=universe_meta,
    )

    # Train/val split (last 20% of samples, respecting time order)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = X.shape[1]  # RANKING_FEATURES + optional regime one-hot
    model = RankingMLP(input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d: train_loss=%.6f, val_loss=%.6f",
                epoch + 1, train_loss, val_loss,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    history['normalization'] = normalization
    history['best_val_loss'] = best_val_loss
    history['train_samples'] = len(X_train)
    history['val_samples'] = len(X_val)
    history['forward_days'] = forward_days

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / 'ranking_mlp.pt')
        with open(save_path / 'ranking_history.json', 'w') as f:
            json.dump({
                k: v for k, v in history.items()
                if k != 'normalization'
            }, f, indent=2, default=str)
        with open(save_path / 'ranking_normalization.json', 'w') as f:
            json.dump(normalization, f, indent=2)
        logger.info("Saved ranking model to %s", save_path)

    return model, history
