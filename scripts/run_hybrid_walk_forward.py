"""Run hybrid walk-forward confirmation: incumbent vs hybrid across folds + gate.

Usage:
    AWS_PROFILE=personal python scripts/run_hybrid_walk_forward.py
"""

import json
import logging
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from optimizer.config import OptimizerConfig
from optimizer.data_access import load_optimizer_dataset
from optimizer.walk_forward import build_walk_forward_plan
from optimizer.replay import run_replay_for_dates
from optimizer.fitness import compute_segment_metrics
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def _load_model(model_dir='models/ranking_expanded_unconditioned'):
    with open(Path(model_dir) / 'ranking_normalization.json') as f:
        norm = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(Path(model_dir) / 'ranking_mlp.pt', weights_only=True))
    model.eval()
    return model, norm


def _run_segment(dataset, dates, params, model=None, norm=None, blend=0.0):
    """Run replay on a segment and return metrics."""
    result = run_replay_for_dates(
        dataset=dataset, decision_dates=dates,
        candidate_bundle=params, random_seed=42, initial_capital=100000.0,
        ranking_model=model, ranking_normalization=norm, ranking_blend=blend,
    )
    return compute_segment_metrics(result.steps, result.fills, dates)


def main():
    logger.info("=== Hybrid Walk-Forward Confirmation ===")

    model, norm = _load_model()
    logger.info("Loaded ranking model")

    cfg = OptimizerConfig(
        bucket='investment-system-data', region='us-east-1',
        max_days=900, train_days=42, test_days=42, step_days=21, gate_days=42,
    )
    dataset = load_optimizer_dataset(cfg)
    plan = build_walk_forward_plan(dataset.dates, cfg.train_days, cfg.test_days, cfg.step_days, cfg.gate_days)
    logger.info("Folds: %d, Gate: %s-%s", len(plan.folds), plan.gate_dates[0], plan.gate_dates[-1])

    with open('config/decision_params.json') as f:
        params = json.load(f)

    blends = [0.0, 0.25, 0.35, 0.50]
    results = {'folds': [], 'gate': {}, 'summary': {}}

    # Run each fold
    for fold in plan.folds:
        fold_results = {}
        for blend in blends:
            label = f"blend_{blend}" if blend > 0 else "incumbent"
            logger.info("Fold %d, %s: test %s-%s", fold.fold_id, label, fold.test_start, fold.test_end)
            m = _run_segment(dataset, fold.test_dates, params,
                            model if blend > 0 else None, norm if blend > 0 else None, blend)
            fold_results[label] = asdict(m)
            logger.info("  return=%.2f%%, sharpe=%.4f, dd=%.2f%%, trips=%d",
                       m.annualized_return * 100, m.sharpe, m.max_drawdown * 100, m.realized_round_trips)
        results['folds'].append({'fold_id': fold.fold_id, 'test_start': fold.test_start,
                                 'test_end': fold.test_end, 'results': fold_results})

    # Run gate
    gate_results = {}
    for blend in blends:
        label = f"blend_{blend}" if blend > 0 else "incumbent"
        logger.info("Gate, %s: %s-%s", label, plan.gate_dates[0], plan.gate_dates[-1])
        m = _run_segment(dataset, plan.gate_dates, params,
                        model if blend > 0 else None, norm if blend > 0 else None, blend)
        gate_results[label] = asdict(m)
        logger.info("  return=%.2f%%, sharpe=%.4f, dd=%.2f%%, trips=%d",
                   m.annualized_return * 100, m.sharpe, m.max_drawdown * 100, m.realized_round_trips)
    results['gate'] = gate_results

    # Compute summary: mean return across folds + gate for each blend
    for blend in blends:
        label = f"blend_{blend}" if blend > 0 else "incumbent"
        fold_returns = [f['results'][label]['annualized_return'] for f in results['folds']]
        gate_return = results['gate'][label]['annualized_return']
        fold_sharpes = [f['results'][label]['sharpe'] for f in results['folds']]
        gate_sharpe = results['gate'][label]['sharpe']
        fold_dds = [f['results'][label]['max_drawdown'] for f in results['folds']]
        gate_dd = results['gate'][label]['max_drawdown']

        results['summary'][label] = {
            'mean_fold_return': sum(fold_returns) / len(fold_returns),
            'gate_return': gate_return,
            'mean_fold_sharpe': sum(fold_sharpes) / len(fold_sharpes),
            'gate_sharpe': gate_sharpe,
            'worst_fold_dd': min(fold_dds),
            'gate_dd': gate_dd,
        }

    output_path = 'runs/hybrid_walk_forward_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== WALK-FORWARD SUMMARY ===")
    print(f"{'Config':<14} {'Fold1 ret':>10} {'Fold2 ret':>10} {'Gate ret':>10} {'Gate Sharpe':>12} {'Gate DD':>8}")
    for blend in blends:
        label = f"blend_{blend}" if blend > 0 else "incumbent"
        f1 = results['folds'][0]['results'][label]['annualized_return'] * 100
        f2 = results['folds'][1]['results'][label]['annualized_return'] * 100
        gr = results['gate'][label]['annualized_return'] * 100
        gs = results['gate'][label]['sharpe']
        gd = results['gate'][label]['max_drawdown'] * 100
        print(f"{label:<14} {f1:>+9.2f}% {f2:>+9.2f}% {gr:>+9.2f}% {gs:>+11.4f} {gd:>+7.2f}%")

    # Check confirmation
    inc_gate = results['gate']['incumbent']['annualized_return']
    hyb_gate = results['gate']['blend_0.35']['annualized_return']
    inc_folds = [f['results']['incumbent']['annualized_return'] for f in results['folds']]
    hyb_folds = [f['results']['blend_0.35']['annualized_return'] for f in results['folds']]
    beats_gate = hyb_gate > inc_gate
    beats_all_folds = all(h > i for h, i in zip(hyb_folds, inc_folds))
    print(f"\nHybrid 0.35 beats incumbent on gate: {'YES' if beats_gate else 'NO'}")
    print(f"Hybrid 0.35 beats incumbent on ALL folds: {'YES' if beats_all_folds else 'NO'}")
    print(f"Walk-forward confirms: {'YES' if beats_gate and beats_all_folds else 'MIXED' if beats_gate else 'NO'}")


if __name__ == '__main__':
    main()
