#!/usr/bin/env python3
"""Gate-only fusion-parameter sensitivity sweep.

Tests multiple fusion parameter variants on the gate segment to determine
whether the current fusion layer is too conservative.

Usage:
    source .venv/bin/activate
    AWS_PROFILE=personal python scripts/run_fusion_sweep.py \
        --config config/optimizer_evaluation.yaml
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.config import load_optimizer_config
from optimizer.data_access import load_optimizer_dataset
from optimizer.fitness import compute_segment_metrics
from optimizer.promote import load_active_bundle
from optimizer.replay import run_replay_for_dates
from optimizer.walk_forward import build_walk_forward_plan


def _run_gate_with_fusion_params(dataset, bundle, plan, config, label, fusion_overrides):
    """Run gate segment replay with custom fusion parameters."""
    if not plan.gate_dates:
        raise ValueError('Gate segment is empty')

    # Create a modified bundle with the fusion overrides
    modified_bundle = copy.deepcopy(bundle)
    existing_fusion = modified_bundle.get('regime_fusion', {})
    existing_fusion.update(fusion_overrides)
    modified_bundle['regime_fusion'] = existing_fusion

    gate_start = plan.gate_dates[0]
    gate_index = plan.dates.index(gate_start)
    replay_dates = plan.dates[:gate_index] + plan.gate_dates

    result = run_replay_for_dates(
        dataset=dataset,
        decision_dates=replay_dates,
        candidate_bundle=modified_bundle,
        random_seed=config.random_seed + 900001,
        initial_capital=config.initial_portfolio_value,
    )

    metrics = compute_segment_metrics(
        steps=result.steps,
        fills=result.fills,
        segment_dates=plan.gate_dates,
    )

    # Count macro_downgrade fires
    macro_fires = 0
    for step in result.steps:
        if hasattr(step, 'regime') and 'choppy' in str(step.regime):
            if str(getattr(step, 'valuation_date', '')) in set(plan.gate_dates):
                macro_fires += 1

    return {
        'label': label,
        'days': metrics.days,
        'annualized_return': round(metrics.annualized_return, 6),
        'sharpe': round(metrics.sharpe, 4),
        'max_drawdown': round(metrics.max_drawdown, 6),
        'win_rate': round(metrics.win_rate, 4),
        'realized_round_trips': metrics.realized_round_trips,
        'wins': metrics.wins,
        'losses': metrics.losses,
        'traded_notional': round(metrics.traded_notional, 2),
        'fold_score': round(metrics.fold_score, 6),
        'macro_downgrade_count': macro_fires,
        'fusion_overrides': fusion_overrides,
    }


def main():
    parser = argparse.ArgumentParser(description='Gate-only fusion sensitivity sweep')
    parser.add_argument('--config', default='config/optimizer_evaluation.yaml')
    args = parser.parse_args()

    config = load_optimizer_config(args.config)
    dataset = load_optimizer_dataset(config)
    bundle = load_active_bundle(config.config_dir_path)
    plan = build_walk_forward_plan(
        dates=dataset.dates,
        train_days=config.train_days,
        test_days=config.test_days,
        step_days=config.step_days,
        gate_days=config.gate_days,
    )

    print(f'Gate: {plan.gate_dates[0]} to {plan.gate_dates[-1]} ({len(plan.gate_dates)} days)')
    print()

    # Define sweep variants
    variants = [
        ('BASELINE', {}),
        ('MACRO_SOFT_-0.75', {'macro_downgrade_threshold': -0.75}),
        ('MACRO_SOFT_-1.00', {'macro_downgrade_threshold': -1.00}),
        ('MACRO_OFF_-2.00', {'macro_downgrade_threshold': -2.00}),
        ('THROTTLE_SOFT_0.30', {'throttle_to_exposure_scale': 0.30}),
        ('COMBINED_SOFT', {
            'macro_downgrade_threshold': -0.75,
            'throttle_to_exposure_scale': 0.30,
        }),
        ('FRAGILITY_SOFT_0.90', {'fragility_threshold': 0.90}),
    ]

    results = []
    for label, overrides in variants:
        print(f'Running: {label}...')
        r = _run_gate_with_fusion_params(dataset, bundle, plan, config, label, overrides)
        results.append(r)
        print(f'  -> return={r["annualized_return"]:+.1%}, rts={r["realized_round_trips"]}, wr={r["win_rate"]:.0%}, macro_fires={r["macro_downgrade_count"]}')

    baseline = results[0]

    print()
    print('=' * 100)
    print('FUSION SENSITIVITY SWEEP — GATE SEGMENT')
    print('=' * 100)
    print(f'{"Variant":<25} {"Ann.Ret":>8} {"Sharpe":>8} {"MaxDD":>8} {"WR":>6} {"RTs":>5} {"Notional":>10} {"MacroFires":>10}')
    print('-' * 100)
    for r in results:
        print(f'{r["label"]:<25} {r["annualized_return"]:>+7.1%} {r["sharpe"]:>+7.2f} {r["max_drawdown"]:>+7.2%} {r["win_rate"]:>5.0%} {r["realized_round_trips"]:>5} {r["traded_notional"]:>10,.0f} {r["macro_downgrade_count"]:>10}')

    print()
    print('DELTAS vs BASELINE:')
    print(f'{"Variant":<25} {"dReturn":>8} {"dSharpe":>8} {"dMaxDD":>8} {"dRTs":>5} {"dWR":>6}')
    print('-' * 70)
    for r in results[1:]:
        dr = r['annualized_return'] - baseline['annualized_return']
        ds = r['sharpe'] - baseline['sharpe']
        dd = r['max_drawdown'] - baseline['max_drawdown']
        drt = r['realized_round_trips'] - baseline['realized_round_trips']
        dwr = r['win_rate'] - baseline['win_rate']
        print(f'{r["label"]:<25} {dr:>+7.1%} {ds:>+7.2f} {dd:>+7.2%} {drt:>+5} {dwr:>+5.0%}')

    # JSON output
    output = {
        'gate_dates': f'{plan.gate_dates[0]} to {plan.gate_dates[-1]}',
        'gate_days': len(plan.gate_dates),
        'default_fusion_params': {
            'macro_downgrade_threshold': -0.50,
            'throttle_to_exposure_scale': 0.50,
            'fragility_threshold': 0.75,
        },
        'results': results,
        'deltas': [
            {
                'variant': r['label'],
                'delta_annualized_return': round(r['annualized_return'] - baseline['annualized_return'], 6),
                'delta_sharpe': round(r['sharpe'] - baseline['sharpe'], 4),
                'delta_max_drawdown': round(r['max_drawdown'] - baseline['max_drawdown'], 6),
                'delta_round_trips': r['realized_round_trips'] - baseline['realized_round_trips'],
                'delta_win_rate': round(r['win_rate'] - baseline['win_rate'], 4),
            }
            for r in results[1:]
        ],
    }
    print()
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
