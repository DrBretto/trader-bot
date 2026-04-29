"""F-7 fragility-relax calibration sweep on the available walk-forward dataset.

Runs the existing optimizer.replay harness across N regime_fusion override
variants on the 3-fold + gate plan that fits 176 aligned daily snapshots
(2025-08-04 → 2026-04-28). Same harness used by run_hybrid_walk_forward.py;
this script swaps the variant axis from ranking blend → fragility-gate config.

Usage:
    AWS_PROFILE=personal .venv/bin/python scripts/run_calibration_sweep_20260429.py
"""

import copy
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from optimizer.config import OptimizerConfig  # noqa: E402
from optimizer.data_access import load_optimizer_dataset  # noqa: E402
from optimizer.fitness import compute_segment_metrics  # noqa: E402
from optimizer.replay import run_replay_for_dates  # noqa: E402
from optimizer.walk_forward import build_walk_forward_plan  # noqa: E402
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES  # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def _load_model(model_dir='models/ranking_expanded_unconditioned'):
    with open(Path(model_dir) / 'ranking_normalization.json') as f:
        norm = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(Path(model_dir) / 'ranking_mlp.pt', weights_only=True))
    model.eval()
    return model, norm


VARIANTS = {
    'current_production':           {},  # exact production: empty regime_fusion
    'thr_085_only':                 {'fragility_threshold': 0.85},
    'relax_conf_080_thr_075':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
        'fragility_threshold': 0.75,
    },
    'relax_conf_080_thr_085':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
        'fragility_threshold': 0.85,
    },
    'relax_conf_075_thr_085':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.75,
        'fragility_threshold': 0.85,
    },
    'relax_conf_085_thr_085':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.85,
        'fragility_threshold': 0.85,
    },
    'relax_off_cap_080':            {
        'fragility_position_cap': 0.80,
    },
    'relax_conf_080_cap_080':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
        'fragility_position_cap': 0.80,
    },
}


def _segment(dataset, dates, params, model, norm, blend):
    res = run_replay_for_dates(
        dataset=dataset,
        decision_dates=dates,
        candidate_bundle=params,
        random_seed=42,
        initial_capital=100000.0,
        ranking_model=model,
        ranking_normalization=norm,
        ranking_blend=blend,
    )
    return compute_segment_metrics(res.steps, res.fills, dates)


def main():
    logger.warning("=== Calibration sweep — F-7 fragility-relax ===")
    model, norm = _load_model()

    cfg = OptimizerConfig(
        bucket='investment-system-data', region='us-east-1',
        max_days=900, train_days=42, test_days=42, step_days=21, gate_days=42,
    )
    dataset = load_optimizer_dataset(cfg)
    plan = build_walk_forward_plan(dataset.dates, cfg.train_days, cfg.test_days, cfg.step_days, cfg.gate_days)
    logger.warning(
        "snapshots=%d folds=%d gate=%s..%s",
        len(dataset.snapshots), len(plan.folds), plan.gate_dates[0], plan.gate_dates[-1],
    )

    with open('config/decision_params.active.json') as f:
        active = json.load(f)
    blend = active.get('decision_engine', {}).get('ranking_blend', 0.35)

    results = {'config': {'folds': len(plan.folds), 'gate_start': plan.gate_dates[0],
                          'gate_end': plan.gate_dates[-1], 'blend': blend},
               'variants': {}}

    for name, rf_overrides in VARIANTS.items():
        bundle = copy.deepcopy(active)
        bundle['regime_fusion'] = rf_overrides
        logger.warning("--- variant: %s rf=%s ---", name, rf_overrides)

        fold_metrics = []
        for fold in plan.folds:
            m = _segment(dataset, fold.test_dates, bundle, model, norm, blend)
            fold_metrics.append(asdict(m))
            logger.warning("  fold %d (%s..%s): ret=%+.2f%% sharpe=%.3f mdd=%+.2f%% trips=%d",
                           fold.fold_id, fold.test_start, fold.test_end,
                           m.annualized_return*100, m.sharpe, m.max_drawdown*100, m.realized_round_trips)
        gate_m = _segment(dataset, plan.gate_dates, bundle, model, norm, blend)
        logger.warning("  gate: ret=%+.2f%% sharpe=%.3f mdd=%+.2f%% trips=%d",
                       gate_m.annualized_return*100, gate_m.sharpe, gate_m.max_drawdown*100, gate_m.realized_round_trips)

        results['variants'][name] = {
            'overrides': rf_overrides,
            'folds': fold_metrics,
            'gate': asdict(gate_m),
            'summary': {
                'mean_fold_return': sum(f['annualized_return'] for f in fold_metrics) / len(fold_metrics),
                'gate_return': gate_m.annualized_return,
                'mean_fold_sharpe': sum(f['sharpe'] for f in fold_metrics) / len(fold_metrics),
                'gate_sharpe': gate_m.sharpe,
                'worst_fold_dd': min(f['max_drawdown'] for f in fold_metrics),
                'gate_dd': gate_m.max_drawdown,
                'gate_trips': gate_m.realized_round_trips,
            },
        }

    out = Path('runs/calibration_sweep_20260429.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.warning("results -> %s", out)

    # Summary table
    header = f"{'variant':<28} {'mean_fold_ret':>12} {'gate_ret':>9} {'gate_sharpe':>11} {'gate_dd':>8} {'trips':>6}"
    print('\n' + header)
    print('-' * len(header))
    for name in VARIANTS:
        s = results['variants'][name]['summary']
        print(f"{name:<28} {s['mean_fold_return']*100:>+11.2f}% {s['gate_return']*100:>+8.2f}% {s['gate_sharpe']:>11.4f} {s['gate_dd']*100:>+7.2f}% {s['gate_trips']:>6d}")


if __name__ == '__main__':
    main()
