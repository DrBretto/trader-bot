"""Run hybrid-system (ranking-integrated) vs incumbent gate comparison.

Uses the modified replay harness that passes ranking scores through to
the decision engine's score_candidates function.

Usage:
    AWS_PROFILE=personal python scripts/run_hybrid_gate_comparison.py
"""

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from optimizer.config import OptimizerConfig
from optimizer.data_access import load_optimizer_dataset
from optimizer.walk_forward import build_walk_forward_plan
from optimizer.replay import run_replay_for_dates
from optimizer.fitness import compute_segment_metrics
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def _load_ranking_model(model_dir='models/ranking_expanded_unconditioned'):
    """Load the champion ranking model and normalization."""
    model_path = Path(model_dir) / 'ranking_mlp.pt'
    norm_path = Path(model_dir) / 'ranking_normalization.json'
    with open(norm_path) as f:
        normalization = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, normalization


def main():
    logger.info("=== Hybrid System Gate Comparison ===")

    model, normalization = _load_ranking_model()
    logger.info("Loaded ranking model (%d features)", len(RANKING_FEATURES))

    cfg = OptimizerConfig(
        bucket='investment-system-data', region='us-east-1',
        max_days=900, train_days=42, test_days=42, step_days=21, gate_days=42,
    )
    dataset = load_optimizer_dataset(cfg)
    plan = build_walk_forward_plan(dataset.dates, cfg.train_days, cfg.test_days, cfg.step_days, cfg.gate_days)
    logger.info("Dataset: %d snapshots, gate: %s to %s", len(dataset.snapshots), plan.gate_dates[0], plan.gate_dates[-1])

    with open('config/decision_params.json') as f:
        params = json.load(f)

    # Run incumbent (blend=0)
    logger.info("=== INCUMBENT (no ranking) ===")
    inc_result = run_replay_for_dates(
        dataset=dataset, decision_dates=plan.gate_dates,
        candidate_bundle=params, random_seed=42, initial_capital=100000.0,
    )
    inc_m = compute_segment_metrics(inc_result.steps, inc_result.fills, plan.gate_dates)
    logger.info("INCUMBENT: return=%.2f%%, sharpe=%.4f, dd=%.2f%%, trips=%d",
                inc_m.annualized_return*100, inc_m.sharpe, inc_m.max_drawdown*100, inc_m.realized_round_trips)

    # Run hybrid at several blend levels
    blend_results = {}
    for blend in [0.15, 0.25, 0.35, 0.50]:
        logger.info("=== HYBRID blend=%.2f ===", blend)
        hyb_result = run_replay_for_dates(
            dataset=dataset, decision_dates=plan.gate_dates,
            candidate_bundle=params, random_seed=42, initial_capital=100000.0,
            ranking_model=model, ranking_normalization=normalization, ranking_blend=blend,
        )
        hyb_m = compute_segment_metrics(hyb_result.steps, hyb_result.fills, plan.gate_dates)
        logger.info("HYBRID(%.2f): return=%.2f%%, sharpe=%.4f, dd=%.2f%%, trips=%d",
                    blend, hyb_m.annualized_return*100, hyb_m.sharpe, hyb_m.max_drawdown*100, hyb_m.realized_round_trips)
        blend_results[f'blend_{blend}'] = asdict(hyb_m)

    results = {'incumbent': asdict(inc_m), 'hybrid': blend_results}
    output_path = 'runs/hybrid_gate_comparison_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== COMPARISON ===")
    print(f"INCUMBENT:    return={inc_m.annualized_return*100:+.2f}%, sharpe={inc_m.sharpe:+.4f}, dd={inc_m.max_drawdown*100:.2f}%, trips={inc_m.realized_round_trips}")
    for bk, m in blend_results.items():
        print(f"{bk.upper():14s}return={m['annualized_return']*100:+.2f}%, sharpe={m['sharpe']:+.4f}, dd={m['max_drawdown']*100:.2f}%, trips={m['realized_round_trips']}")

    # Determine best blend
    best_blend = max(blend_results.items(), key=lambda x: x[1]['annualized_return'])
    print(f"\nBest blend: {best_blend[0]} (return={best_blend[1]['annualized_return']*100:+.2f}%)")
    beats_incumbent = best_blend[1]['annualized_return'] > inc_m.annualized_return
    print(f"Hybrid beats incumbent: {'YES' if beats_incumbent else 'NO'}")


if __name__ == '__main__':
    main()
