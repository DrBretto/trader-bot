"""Train the ranking model and compare baseline hybrid vs incumbent.

Usage:
    AWS_PROFILE=personal python scripts/run_ranking_comparison.py
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from optimizer.config import OptimizerConfig
from optimizer.data_access import load_optimizer_dataset
from optimizer.walk_forward import build_walk_forward_plan
from optimizer.replay import run_replay_for_dates
from optimizer.fitness import compute_segment_metrics
from optimizer.ranking_eval import compute_ranking_metrics, aggregate_ranking_metrics
from training.train_ranking import train_ranking_model
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def _build_features_df(dataset):
    """Concatenate all per-date features into one DataFrame."""
    frames = []
    for snap in dataset.snapshots:
        if snap.features_df is not None and len(snap.features_df) > 0:
            df = snap.features_df.copy()
            df['date'] = snap.date
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _compute_forward_returns(dataset, forward_days=21):
    """Compute realized forward returns per asset per date."""
    fwd = {}
    dates = dataset.dates
    snap_map = dataset.by_date()
    for i, date in enumerate(dates):
        target_idx = i + forward_days
        if target_idx >= len(dates):
            break
        target_date = dates[target_idx]
        snap_now = snap_map.get(date)
        snap_fwd = snap_map.get(target_date)
        if snap_now is None or snap_fwd is None:
            continue
        if snap_now.features_df is None or snap_fwd.features_df is None:
            continue

        now_prices = dict(zip(
            snap_now.features_df['symbol'],
            snap_now.features_df['close'],
        ))
        fwd_prices = dict(zip(
            snap_fwd.features_df['symbol'],
            snap_fwd.features_df['close'],
        ))

        returns = {}
        for sym in now_prices:
            if sym in fwd_prices and now_prices[sym] > 0:
                returns[sym] = fwd_prices[sym] / now_prices[sym] - 1
        if returns:
            fwd[date] = returns
    return fwd


def _score_assets_with_ranking(model, features_df, normalization, date, regime_label='', universe_meta=None):
    """Score all assets for a given date using the ranking model."""
    date_features = features_df[features_df['date'] == date] if 'date' in features_df.columns else features_df
    scores = {}
    for _, row in date_features.iterrows():
        sym = row.get('symbol')
        if not sym:
            continue
        feat_dict = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
        meta = (universe_meta or {}).get(sym, {})
        score = model.predict_scores(
            feat_dict, normalization,
            regime_label=regime_label,
            asset_class=meta.get('asset_class', ''),
            sector=meta.get('sector', ''),
        )
        scores[sym] = score
    return scores


def _load_universe_meta(path='config/universe.csv'):
    """Load universe metadata from CSV."""
    import csv
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            meta[row['symbol']] = {
                'asset_class': row.get('asset_class', ''),
                'sector': row.get('sector', ''),
            }
    return meta


def _get_regime_labels(dataset):
    """Extract regime labels from inference snapshots."""
    labels = {}
    for snap in dataset.snapshots:
        if snap.inference and 'regime' in snap.inference:
            regime = snap.inference['regime']
            if isinstance(regime, dict):
                labels[snap.date] = regime.get('label', 'choppy')
            elif isinstance(regime, str):
                labels[snap.date] = regime
    return labels


def main():
    logger.info("=== Ranking Head Comparison: Baseline Hybrid vs Incumbent ===")

    cfg = OptimizerConfig(
        bucket='investment-system-data',
        region='us-east-1',
        max_days=900,
        train_days=42,
        test_days=42,
        step_days=21,
        gate_days=42,
    )

    logger.info("Loading dataset...")
    dataset = load_optimizer_dataset(cfg)
    logger.info("Loaded %d snapshots (%s to %s)", len(dataset.snapshots), dataset.dates[0], dataset.dates[-1])

    # Build walk-forward plan
    plan = build_walk_forward_plan(dataset.dates, cfg.train_days, cfg.test_days, cfg.step_days, cfg.gate_days)
    logger.info("Walk-forward: %d folds, gate %s-%s", len(plan.folds), plan.gate_dates[0], plan.gate_dates[-1])

    # Prepare full features
    logger.info("Building features dataframe...")
    all_features = _build_features_df(dataset)
    logger.info("Features: %d rows, %d symbols", len(all_features), all_features['symbol'].nunique())

    # Get regime labels for conditioning
    regime_labels = _get_regime_labels(dataset)
    logger.info("Regime labels: %d dates", len(regime_labels))

    # Load universe metadata for structural features
    universe_meta = _load_universe_meta()
    logger.info("Universe meta: %d symbols", len(universe_meta))

    # Train ranking model on pre-gate data
    train_cutoff = plan.gate_dates[0]
    train_features = all_features[all_features['date'] < train_cutoff]
    logger.info("Training on %d rows before %s", len(train_features), train_cutoff)

    # --- BASELINE MODEL (no conditioning, raw targets) ---
    logger.info("=== Training BASELINE model (no conditioning, raw targets) ===")
    model_base, hist_base = train_ranking_model(
        features_df=train_features, epochs=100, batch_size=64,
        save_dir='models/ranking_baseline',
    )
    norm_base = hist_base['normalization']
    logger.info("Baseline: val_loss=%.6f", hist_base['best_val_loss'])

    # --- CONDITIONED MODEL (regime + relative targets) ---
    logger.info("=== Training CONDITIONED model (regime conditioning + relative targets) ===")
    model_cond, hist_cond = train_ranking_model(
        features_df=train_features, epochs=100, batch_size=64,
        save_dir='models/ranking_conditioned',
        regime_labels=regime_labels,
        use_relative_targets=True,
    )
    norm_cond = hist_cond['normalization']
    logger.info("Conditioned: val_loss=%.6f", hist_cond['best_val_loss'])

    # --- STRUCTURAL MODEL (regime + relative targets + asset class + sector group) ---
    logger.info("=== Training STRUCTURAL model (regime + relative + asset identity) ===")
    model_struct, hist_struct = train_ranking_model(
        features_df=train_features, epochs=100, batch_size=64,
        save_dir='models/ranking_structural',
        regime_labels=regime_labels,
        use_relative_targets=True,
        universe_meta=universe_meta,
    )
    norm_struct = hist_struct['normalization']
    logger.info("Structural: val_loss=%.6f", hist_struct['best_val_loss'])

    # Compute forward returns for evaluation
    logger.info("Computing forward returns...")
    fwd_returns = _compute_forward_returns(dataset)
    logger.info("Forward returns computed for %d dates", len(fwd_returns))

    # --- Evaluate BOTH models on gate segment ---
    def _eval_model(model, norm, label, use_regime=False, use_struct=False):
        metrics_list = []
        for date in plan.gate_dates:
            if date not in fwd_returns:
                continue
            snap = dataset.by_date().get(date)
            if snap is None or snap.features_df is None:
                continue
            rl = regime_labels.get(date, 'choppy') if use_regime else ''
            um = universe_meta if use_struct else None
            scores = _score_assets_with_ranking(model, snap.features_df, norm, date, regime_label=rl, universe_meta=um)
            realized = fwd_returns[date]
            metrics = compute_ranking_metrics(scores, realized)
            if metrics['valid']:
                metrics_list.append(metrics)
        agg = aggregate_ranking_metrics(metrics_list)
        logger.info("%s gate metrics:", label)
        logger.info("  Spearman rho: %.4f", agg['mean_spearman_rho'])
        logger.info("  Quintile spread: %.4f", agg['mean_quintile_spread'])
        logger.info("  Hit rate: %.2f%%", agg['hit_rate'] * 100)
        logger.info("  IR: %.4f", agg['information_ratio'])
        logger.info("  Dates: %d", agg['n_dates'])
        return agg

    logger.info("=== BASELINE Ranking Quality on Gate ===")
    agg_base = _eval_model(model_base, norm_base, "BASELINE", use_regime=False)

    logger.info("=== CONDITIONED Ranking Quality on Gate ===")
    agg_cond = _eval_model(model_cond, norm_cond, "CONDITIONED", use_regime=True)

    logger.info("=== STRUCTURAL Ranking Quality on Gate ===")
    agg_struct = _eval_model(model_struct, norm_struct, "STRUCTURAL", use_regime=True, use_struct=True)

    # Run incumbent replay on gate
    logger.info("=== Running Incumbent Replay on Gate ===")
    with open('config/decision_params.json') as f:
        incumbent_params = json.load(f)

    incumbent_result = run_replay_for_dates(
        dataset=dataset,
        decision_dates=plan.gate_dates,
        candidate_bundle=incumbent_params,
        random_seed=42,
        initial_capital=100000.0,
    )
    incumbent_metrics = compute_segment_metrics(incumbent_result.steps, incumbent_result.fills, plan.gate_dates)

    logger.info("Incumbent gate metrics:")
    logger.info("  Ann. return: %.2f%%", incumbent_metrics.annualized_return * 100)
    logger.info("  Sharpe: %.4f", incumbent_metrics.sharpe)
    logger.info("  Max DD: %.2f%%", incumbent_metrics.max_drawdown * 100)
    logger.info("  Round trips: %d", incumbent_metrics.realized_round_trips)

    # Output summary
    from dataclasses import asdict
    results = {
        'baseline_ranking': agg_base,
        'conditioned_ranking': agg_cond,
        'structural_ranking': agg_struct,
        'incumbent_gate': asdict(incumbent_metrics),
        'training_baseline': {
            'train_samples': hist_base['train_samples'],
            'val_samples': hist_base['val_samples'],
            'best_val_loss': hist_base['best_val_loss'],
        },
        'training_conditioned': {
            'train_samples': hist_cond['train_samples'],
            'val_samples': hist_cond['val_samples'],
            'best_val_loss': hist_cond['best_val_loss'],
        },
        'training_structural': {
            'train_samples': hist_struct['train_samples'],
            'val_samples': hist_struct['val_samples'],
            'best_val_loss': hist_struct['best_val_loss'],
        },
    }

    output_path = 'runs/ranking_comparison_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    print("\n=== COMPARISON SUMMARY ===")
    print(f"BASELINE:     rho={agg_base['mean_spearman_rho']:+.4f}, spread={agg_base['mean_quintile_spread']:+.4f}, IR={agg_base['information_ratio']:+.4f}, hit={agg_base['hit_rate']*100:.0f}%")
    print(f"CONDITIONED:  rho={agg_cond['mean_spearman_rho']:+.4f}, spread={agg_cond['mean_quintile_spread']:+.4f}, IR={agg_cond['information_ratio']:+.4f}, hit={agg_cond['hit_rate']*100:.0f}%")
    print(f"STRUCTURAL:   rho={agg_struct['mean_spearman_rho']:+.4f}, spread={agg_struct['mean_quintile_spread']:+.4f}, IR={agg_struct['information_ratio']:+.4f}, hit={agg_struct['hit_rate']*100:.0f}%")
    print(f"INCUMBENT:    ann_return={incumbent_metrics.annualized_return*100:+.2f}%, sharpe={incumbent_metrics.sharpe:+.4f}, max_dd={incumbent_metrics.max_drawdown*100:.2f}%")
    struct_better = agg_struct['mean_spearman_rho'] > agg_cond['mean_spearman_rho'] + 0.02
    print(f"\nStructural features improved over conditioned: {'YES' if struct_better else 'NO'}")


if __name__ == '__main__':
    main()
