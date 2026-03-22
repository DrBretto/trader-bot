"""Build extended historical training data for the ranking model.

Downloads 2+ years of OHLCV for the universe, computes features,
assigns heuristic regime labels, and trains an expanded conditioned model.

Usage:
    AWS_PROFILE=personal python scripts/build_ranking_history.py
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def _load_universe(path='config/universe.csv'):
    """Load universe symbols."""
    symbols = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('eligible', '1') == '1':
                symbols.append(row['symbol'])
    return symbols


def _download_history(symbols, start='2023-06-01', end='2026-03-21'):
    """Download OHLCV history via yfinance."""
    import yfinance as yf
    logger.info("Downloading %d symbols from %s to %s...", len(symbols), start, end)
    data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker')
    frames = []
    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            else:
                df = data[sym][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna(subset=['Close'])
            if len(df) < 50:
                continue
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['symbol'] = sym
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            frames.append(df)
        except Exception as e:
            logger.warning("Skip %s: %s", sym, e)
    if not frames:
        raise ValueError("No data downloaded")
    result = pd.concat(frames, ignore_index=True)
    logger.info("Downloaded %d rows for %d symbols", len(result), result['symbol'].nunique())
    return result


def _compute_features(prices_df):
    """Compute per-asset features from OHLCV data."""
    from training.models.ranking_mlp import RANKING_FEATURES

    df = prices_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    # Compute returns and vol per symbol
    for sym, grp in df.groupby('symbol'):
        idx = grp.index
        close = grp['close']
        df.loc[idx, 'return_1d'] = close.pct_change(1)
        df.loc[idx, 'return_5d'] = close.pct_change(5)
        df.loc[idx, 'return_21d'] = close.pct_change(21)
        df.loc[idx, 'return_63d'] = close.pct_change(63)
        df.loc[idx, 'vol_21d'] = close.pct_change(1).rolling(21).std() * np.sqrt(252)
        df.loc[idx, 'vol_63d'] = close.pct_change(1).rolling(63).std() * np.sqrt(252)
        rolling_max = close.rolling(63).max()
        df.loc[idx, 'drawdown_63d'] = (close - rolling_max) / rolling_max
        sma63 = close.rolling(63).mean()
        df.loc[idx, 'trend_63d'] = (close - sma63) / sma63

    # Relative strength vs SPY
    spy = df[df['symbol'] == 'SPY'][['date', 'return_21d', 'return_63d']].copy()
    spy.columns = ['date', 'spy_ret_21d', 'spy_ret_63d']
    df = df.merge(spy, on='date', how='left')
    df['rel_strength_21d'] = df['return_21d'] - df['spy_ret_21d'].fillna(0)
    df['rel_strength_63d'] = df['return_63d'] - df['spy_ret_63d'].fillna(0)
    df = df.drop(columns=['spy_ret_21d', 'spy_ret_63d'], errors='ignore')

    # Drop rows without enough features
    df = df.dropna(subset=RANKING_FEATURES)
    logger.info("Computed features: %d rows, %d symbols", len(df), df['symbol'].nunique())
    return df


def _assign_heuristic_regimes(features_df):
    """Assign regime labels from SPY price characteristics.

    This is a simple heuristic for historical dates that lack pipeline
    regime classification. It approximates the 5-class regime from
    observable market characteristics.
    """
    spy = features_df[features_df['symbol'] == 'SPY'][['date', 'return_21d', 'vol_21d']].copy()
    spy['date'] = pd.to_datetime(spy['date'])

    labels = {}
    for _, row in spy.iterrows():
        ret = row.get('return_21d', 0) or 0
        vol = row.get('vol_21d', 0) or 0
        ds = row['date'].strftime('%Y-%m-%d')

        if ret < -0.05 and vol > 0.25:
            labels[ds] = 'high_vol_panic'
        elif ret < -0.02 or vol > 0.22:
            labels[ds] = 'risk_off_trend'
        elif vol > 0.16:
            labels[ds] = 'choppy'
        elif ret > 0.02:
            labels[ds] = 'risk_on_trend'
        else:
            labels[ds] = 'calm_uptrend'

    logger.info("Assigned heuristic regimes: %d dates", len(labels))
    regime_counts = {}
    for v in labels.values():
        regime_counts[v] = regime_counts.get(v, 0) + 1
    for r, c in sorted(regime_counts.items()):
        logger.info("  %s: %d", r, c)
    return labels


def main():
    from optimizer.config import OptimizerConfig
    from optimizer.data_access import load_optimizer_dataset
    from optimizer.walk_forward import build_walk_forward_plan
    from optimizer.ranking_eval import compute_ranking_metrics, aggregate_ranking_metrics
    from training.train_ranking import train_ranking_model
    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES

    logger.info("=== Ranking History Expansion ===")

    # Step 1: Download extended history
    symbols = _load_universe()
    prices = _download_history(symbols, start='2023-06-01', end='2026-03-21')

    # Step 2: Compute features
    features = _compute_features(prices)

    # Step 3: Assign heuristic regime labels
    regime_labels = _assign_heuristic_regimes(features)

    # Step 4: Define gate cutoff (same as pipeline eval)
    gate_cutoff = '2026-01-15'
    train_features = features[features['date'] < gate_cutoff]
    logger.info("Training data: %d rows before %s", len(train_features), gate_cutoff)
    logger.info("  Symbols: %d, Dates: %d",
                train_features['symbol'].nunique(),
                train_features['date'].nunique())

    # Step 5: Train conditioned model on expanded history
    logger.info("=== Training EXPANDED CONDITIONED model ===")
    model_exp_cond, hist_exp_cond = train_ranking_model(
        features_df=train_features, epochs=150, batch_size=128,
        save_dir='models/ranking_expanded_conditioned',
        regime_labels=regime_labels,
        use_relative_targets=True,
    )
    norm_exp = hist_exp_cond['normalization']
    logger.info("Expanded conditioned: val_loss=%.6f, samples=%d",
                hist_exp_cond['best_val_loss'], hist_exp_cond['train_samples'])

    # Step 6: Train unconditioned control on expanded history
    logger.info("=== Training EXPANDED UNCONDITIONED control ===")
    model_exp_uncond, hist_exp_uncond = train_ranking_model(
        features_df=train_features, epochs=150, batch_size=128,
        save_dir='models/ranking_expanded_unconditioned',
        use_relative_targets=True,
    )
    norm_exp_uncond = hist_exp_uncond['normalization']
    logger.info("Expanded unconditioned: val_loss=%.6f", hist_exp_uncond['best_val_loss'])

    # Step 7: Evaluate on gate segment using pipeline data
    logger.info("=== Loading pipeline dataset for gate evaluation ===")
    cfg = OptimizerConfig(
        bucket='investment-system-data', region='us-east-1',
        max_days=900, train_days=42, test_days=42, step_days=21, gate_days=42,
    )
    dataset = load_optimizer_dataset(cfg)
    plan = build_walk_forward_plan(dataset.dates, cfg.train_days, cfg.test_days, cfg.step_days, cfg.gate_days)

    # Compute forward returns from pipeline data
    fwd_returns = {}
    dates = dataset.dates
    snap_map = dataset.by_date()
    for i, date in enumerate(dates):
        target_idx = i + 21
        if target_idx >= len(dates):
            break
        snap_now = snap_map.get(date)
        snap_fwd = snap_map.get(dates[target_idx])
        if not snap_now or not snap_fwd:
            continue
        if snap_now.features_df is None or snap_fwd.features_df is None:
            continue
        now_px = dict(zip(snap_now.features_df['symbol'], snap_now.features_df['close']))
        fwd_px = dict(zip(snap_fwd.features_df['symbol'], snap_fwd.features_df['close']))
        rets = {}
        for sym in now_px:
            if sym in fwd_px and now_px[sym] > 0:
                rets[sym] = fwd_px[sym] / now_px[sym] - 1
        if rets:
            fwd_returns[date] = rets

    # Get pipeline regime labels for gate eval
    pipeline_regimes = {}
    for snap in dataset.snapshots:
        if snap.inference and 'regime' in snap.inference:
            r = snap.inference['regime']
            pipeline_regimes[snap.date] = r.get('label', 'choppy') if isinstance(r, dict) else str(r)

    def _eval_on_gate(model, norm, label, use_regime=False, regime_src=None):
        metrics_list = []
        for date in plan.gate_dates:
            if date not in fwd_returns:
                continue
            snap = snap_map.get(date)
            if snap is None or snap.features_df is None:
                continue
            rl = (regime_src or {}).get(date, 'choppy') if use_regime else ''
            scores = {}
            for _, row in snap.features_df.iterrows():
                sym = row.get('symbol')
                if not sym:
                    continue
                feat_dict = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
                scores[sym] = model.predict_scores(feat_dict, norm, regime_label=rl)
            realized = fwd_returns[date]
            m = compute_ranking_metrics(scores, realized)
            if m['valid']:
                metrics_list.append(m)
        agg = aggregate_ranking_metrics(metrics_list)
        logger.info("%s: rho=%.4f, spread=%.4f, hit=%.0f%%, IR=%.4f, dates=%d",
                    label, agg['mean_spearman_rho'], agg['mean_quintile_spread'],
                    agg['hit_rate'] * 100, agg['information_ratio'], agg['n_dates'])
        return agg

    logger.info("=== Gate Evaluation ===")

    # Load the previous conditioned champion for comparison
    import torch
    prev_model = RankingMLP(input_dim=15)
    prev_path = Path('models/ranking_conditioned/ranking_mlp.pt')
    if prev_path.exists():
        prev_model.load_state_dict(torch.load(prev_path, weights_only=True))
        with open('models/ranking_conditioned/ranking_normalization.json') as f:
            prev_norm = json.load(f)
        agg_prev = _eval_on_gate(prev_model, prev_norm, "PREV_CONDITIONED", use_regime=True, regime_src=pipeline_regimes)
    else:
        logger.warning("No previous conditioned model found")
        agg_prev = {'mean_spearman_rho': 0, 'mean_quintile_spread': 0, 'hit_rate': 0, 'information_ratio': 0, 'n_dates': 0}

    agg_exp_cond = _eval_on_gate(model_exp_cond, norm_exp, "EXP_CONDITIONED", use_regime=True, regime_src=pipeline_regimes)
    agg_exp_uncond = _eval_on_gate(model_exp_uncond, norm_exp_uncond, "EXP_UNCONDITIONED", use_regime=False)

    # Save results
    results = {
        'prev_conditioned': agg_prev,
        'expanded_conditioned': agg_exp_cond,
        'expanded_unconditioned': agg_exp_uncond,
        'training': {
            'expanded_conditioned': {
                'train_samples': hist_exp_cond['train_samples'],
                'val_samples': hist_exp_cond['val_samples'],
                'best_val_loss': hist_exp_cond['best_val_loss'],
            },
            'expanded_unconditioned': {
                'train_samples': hist_exp_uncond['train_samples'],
                'val_samples': hist_exp_uncond['val_samples'],
                'best_val_loss': hist_exp_uncond['best_val_loss'],
            },
        },
        'history': {
            'total_rows': len(features),
            'total_symbols': int(features['symbol'].nunique()),
            'total_dates': int(features['date'].nunique()),
            'train_rows': len(train_features),
            'train_dates': int(train_features['date'].nunique()),
        },
    }

    output_path = 'runs/ranking_history_expansion_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== COMPARISON ===")
    print(f"PREV_CONDITIONED:    rho={agg_prev['mean_spearman_rho']:+.4f}, spread={agg_prev['mean_quintile_spread']:+.4f}, IR={agg_prev['information_ratio']:+.4f}, hit={agg_prev['hit_rate']*100:.0f}%")
    print(f"EXP_CONDITIONED:     rho={agg_exp_cond['mean_spearman_rho']:+.4f}, spread={agg_exp_cond['mean_quintile_spread']:+.4f}, IR={agg_exp_cond['information_ratio']:+.4f}, hit={agg_exp_cond['hit_rate']*100:.0f}%")
    print(f"EXP_UNCONDITIONED:   rho={agg_exp_uncond['mean_spearman_rho']:+.4f}, spread={agg_exp_uncond['mean_quintile_spread']:+.4f}, IR={agg_exp_uncond['information_ratio']:+.4f}, hit={agg_exp_uncond['hit_rate']*100:.0f}%")

    exp_better = agg_exp_cond['mean_spearman_rho'] > agg_prev['mean_spearman_rho'] + 0.02
    print(f"\nExpanded history improved conditioned model: {'YES' if exp_better else 'NO'}")


if __name__ == '__main__':
    main()
