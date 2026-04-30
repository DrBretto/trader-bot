"""Walk-forward sweep for the 2026-04-30 fragility recalibration.

Tests current_production vs recalibrated-constants vs recalibration-paired
combinations. Reuses the same harness as run_calibration_sweep_20260429.py;
the new axis is `signals.fragility` constants rather than just `regime_fusion`
gate parameters.

Usage:
    AWS_PROFILE=personal .venv/bin/python scripts/run_recalibration_sweep_20260430.py
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
from src.signals.fragility import RECALIBRATED_2026_04_30  # noqa: E402
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES  # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


RECAL_FRAGILITY = {
    'AVG_CORR_MEAN': RECALIBRATED_2026_04_30['AVG_CORR_MEAN'],
    'AVG_CORR_STD': RECALIBRATED_2026_04_30['AVG_CORR_STD'],
    'PC1_MEAN': RECALIBRATED_2026_04_30['PC1_MEAN'],
    'PC1_STD': RECALIBRATED_2026_04_30['PC1_STD'],
}


VARIANTS = {
    # Control — exact production state.
    'current_production': {
        'signals': {},
        'regime_fusion': {},
    },
    # Phase-4 fixes-only baseline (matches prior audit's recommended ship).
    # Same as current_production for replay purposes (no behavior change in
    # the v3 fusion path under healthy state).
    'fixes_only_no_calibration': {
        'signals': {},
        'regime_fusion': {},
    },
    # Recalibration shipped behind the bundle gate, defaulted off.
    # Behavior is identical to control (no override applied).
    'recalibrated_constants_default_off': {
        'signals': {},
        'regime_fusion': {},
    },
    # Recalibration active.
    'recalibrated_constants_default_on': {
        'signals': {'fragility': RECAL_FRAGILITY},
        'regime_fusion': {},
    },
    # Recalibration + raised gate threshold (so only true extremes throttle).
    'recalibrated_plus_threshold_085': {
        'signals': {'fragility': RECAL_FRAGILITY},
        'regime_fusion': {'fragility_threshold': 0.85},
    },
    # Recalibration + F-7 relax-in-risk-on.
    'recalibrated_plus_F7_relax_on': {
        'signals': {'fragility': RECAL_FRAGILITY},
        'regime_fusion': {
            'fragility_relax_in_risk_on': True,
            'fragility_relax_confidence': 0.80,
        },
    },
    # Recalibration + raised threshold + F-7 relax (most aggressive variant).
    'recalibrated_plus_threshold_085_plus_F7_relax': {
        'signals': {'fragility': RECAL_FRAGILITY},
        'regime_fusion': {
            'fragility_threshold': 0.85,
            'fragility_relax_in_risk_on': True,
            'fragility_relax_confidence': 0.80,
        },
    },
}


def _load_model(model_dir='models/ranking_expanded_unconditioned'):
    with open(Path(model_dir) / 'ranking_normalization.json') as f:
        norm = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(Path(model_dir) / 'ranking_mlp.pt', weights_only=True))
    model.eval()
    return model, norm


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


def _windowed_metrics(dataset, start, end, params, model, norm, blend):
    all_dates = [d for d in dataset.dates if start <= d <= end]
    if len(all_dates) < 2:
        return None
    return _segment(dataset, all_dates, params, model, norm, blend)


def main():
    logger.warning("=== Recalibration sweep — 2026-04-30 fragility constants ===")
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

    # Sub-windows.
    rally_start, rally_end = '2026-04-15', '2026-04-29'
    panic_start, panic_end = '2026-02-11', '2026-03-31'

    with open('config/decision_params.active.json') as f:
        active = json.load(f)
    blend = active.get('decision_engine', {}).get('ranking_blend', 0.35)

    results = {
        'config': {
            'folds': len(plan.folds),
            'gate_start': plan.gate_dates[0],
            'gate_end': plan.gate_dates[-1],
            'rally_start': rally_start, 'rally_end': rally_end,
            'panic_start': panic_start, 'panic_end': panic_end,
            'blend': blend,
        },
        'variants': {},
    }

    for name, overrides in VARIANTS.items():
        bundle = copy.deepcopy(active)
        bundle['signals'] = overrides.get('signals', {})
        bundle['regime_fusion'] = overrides.get('regime_fusion', {})
        logger.warning("--- variant: %s ---", name)
        logger.warning("    overrides=%s", overrides)

        fold_metrics = []
        for fold in plan.folds:
            m = _segment(dataset, fold.test_dates, bundle, model, norm, blend)
            fold_metrics.append(asdict(m))
        gate_m = _segment(dataset, plan.gate_dates, bundle, model, norm, blend)
        rally_m = _windowed_metrics(dataset, rally_start, rally_end, bundle, model, norm, blend)
        panic_m = _windowed_metrics(dataset, panic_start, panic_end, bundle, model, norm, blend)

        logger.warning("  gate: ret=%+.2f%% sharpe=%.3f mdd=%+.2f%% trips=%d",
                       gate_m.annualized_return*100, gate_m.sharpe, gate_m.max_drawdown*100, gate_m.realized_round_trips)
        if rally_m is not None:
            logger.warning("  rally: ret=%+.2f%% sharpe=%.3f mdd=%+.2f%% trips=%d",
                           rally_m.annualized_return*100, rally_m.sharpe, rally_m.max_drawdown*100, rally_m.realized_round_trips)
        if panic_m is not None:
            logger.warning("  panic: ret=%+.2f%% sharpe=%.3f mdd=%+.2f%% trips=%d",
                           panic_m.annualized_return*100, panic_m.sharpe, panic_m.max_drawdown*100, panic_m.realized_round_trips)

        results['variants'][name] = {
            'overrides': overrides,
            'folds': fold_metrics,
            'gate': asdict(gate_m),
            'rally': asdict(rally_m) if rally_m is not None else None,
            'panic': asdict(panic_m) if panic_m is not None else None,
            'summary': {
                'mean_fold_return': sum(f['annualized_return'] for f in fold_metrics) / len(fold_metrics),
                'gate_return': gate_m.annualized_return,
                'rally_return': rally_m.annualized_return if rally_m else None,
                'panic_return': panic_m.annualized_return if panic_m else None,
                'gate_sharpe': gate_m.sharpe,
                'gate_dd': gate_m.max_drawdown,
                'rally_dd': rally_m.max_drawdown if rally_m else None,
                'panic_dd': panic_m.max_drawdown if panic_m else None,
                'gate_trips': gate_m.realized_round_trips,
            },
        }

    out = Path('runs/recalibration_sweep_20260430.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.warning("results -> %s", out)

    # Summary table.
    header = (
        f"{'variant':<48} {'mean_fold_ret':>12} {'gate_ret':>9} {'rally_ret':>10} {'panic_ret':>10} "
        f"{'gate_sharpe':>11} {'gate_dd':>8} {'rally_dd':>8} {'panic_dd':>8}"
    )
    print('\n' + header)
    print('-' * len(header))
    for name in VARIANTS:
        s = results['variants'][name]['summary']
        rally_ret = f"{s['rally_return']*100:>+9.2f}%" if s['rally_return'] is not None else '       n/a'
        panic_ret = f"{s['panic_return']*100:>+9.2f}%" if s['panic_return'] is not None else '       n/a'
        rally_dd = f"{s['rally_dd']*100:>+7.2f}%" if s['rally_dd'] is not None else '     n/a'
        panic_dd = f"{s['panic_dd']*100:>+7.2f}%" if s['panic_dd'] is not None else '     n/a'
        print(
            f"{name:<48} {s['mean_fold_return']*100:>+11.2f}% {s['gate_return']*100:>+8.2f}% "
            f"{rally_ret} {panic_ret} {s['gate_sharpe']:>11.4f} {s['gate_dd']*100:>+7.2f}% "
            f"{rally_dd} {panic_dd}"
        )


if __name__ == '__main__':
    main()
