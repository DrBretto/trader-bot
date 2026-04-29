#!/usr/bin/env python3
"""Gate-only model-family ablation for trader-bot.

Runs the gate segment with each model family disabled and compares
against the full-ensemble baseline.

Usage:
    source .venv/bin/activate
    AWS_PROFILE=personal python scripts/run_gate_ablation.py \
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


def _run_gate(dataset, bundle, plan, config, label, mutate_snapshot=None):
    """Run gate segment replay with optional per-snapshot mutation."""
    if not plan.gate_dates:
        raise ValueError('Gate segment is empty')

    gate_start = plan.gate_dates[0]
    gate_index = plan.dates.index(gate_start)
    replay_dates = plan.dates[:gate_index] + plan.gate_dates

    # If mutation is needed, create a mutated dataset
    if mutate_snapshot:
        import copy as _copy
        mutated_snapshots = []
        for snap in dataset.snapshots:
            mutated = _copy.copy(snap)
            mutated.inference = _copy.deepcopy(snap.inference)
            mutated.signal_row = _copy.deepcopy(snap.signal_row)
            mutate_snapshot(mutated)
            mutated_snapshots.append(mutated)

        from optimizer.data_access import OptimizerDataset
        mutated_dataset = OptimizerDataset(
            snapshots=mutated_snapshots,
            universe_df=dataset.universe_df,
        )
    else:
        mutated_dataset = dataset

    result = run_replay_for_dates(
        dataset=mutated_dataset,
        decision_dates=replay_dates,
        candidate_bundle=bundle,
        random_seed=config.random_seed + 900001,
        initial_capital=config.initial_portfolio_value,
    )

    metrics = compute_segment_metrics(
        steps=result.steps,
        fills=result.fills,
        segment_dates=plan.gate_dates,
    )

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
    }


def main():
    parser = argparse.ArgumentParser(description='Gate-only model-family ablation')
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

    print(f'Gate dates: {plan.gate_dates[0]} to {plan.gate_dates[-1]} ({len(plan.gate_dates)} days)')
    print()

    results = []

    # 1. Baseline (full ensemble)
    print('Running: BASELINE (full ensemble)...')
    baseline = _run_gate(dataset, bundle, plan, config, 'BASELINE')
    results.append(baseline)
    print(f'  -> return={baseline["annualized_return"]:+.1%}, rts={baseline["realized_round_trips"]}, wr={baseline["win_rate"]:.0%}')

    # 2. Health OFF (clear asset_health → decision engine can't score candidates)
    def _no_health(snap):
        snap.inference['asset_health'] = []

    print('Running: HEALTH_OFF...')
    health_off = _run_gate(dataset, bundle, plan, config, 'HEALTH_OFF', mutate_snapshot=_no_health)
    results.append(health_off)
    print(f'  -> return={health_off["annualized_return"]:+.1%}, rts={health_off["realized_round_trips"]}, wr={health_off["win_rate"]:.0%}')

    # 3. Expert signals OFF (neutralize all expert signal values → no fusion overrides)
    def _no_expert_signals(snap):
        snap.signal_row = {
            'date': snap.date,
            'macro_credit_score': 0.0,
            'yield_slope_10y_3m': 1.5,
            'hy_spread_proxy': 0.0,
            'vol_uncertainty_score': 0.3,
            'vol_regime_label': 'calm',
            'vix_percentile': 0.3,
            'vvix_percentile': 0.3,
            'vix_value': 15.0,
            'vvix_value': 80.0,
            'skew_value': 120.0,
            'fragility_score': 0.3,
            'avg_correlation': 0.2,
            'pc1_explained': 0.3,
            'entropy_score': 0.5,
            'entropy_z_score': 0.0,
            'entropy_shift_flag': False,
        }

    print('Running: EXPERT_SIGNALS_OFF...')
    expert_off = _run_gate(dataset, bundle, plan, config, 'EXPERT_SIGNALS_OFF', mutate_snapshot=_no_expert_signals)
    results.append(expert_off)
    print(f'  -> return={expert_off["annualized_return"]:+.1%}, rts={expert_off["realized_round_trips"]}, wr={expert_off["win_rate"]:.0%}')

    # 4. Regime OFF (force neutral 'choppy' regime → removes regime-driven allocation)
    def _no_regime(snap):
        regime = snap.inference.get('regime', {})
        regime['label'] = 'choppy'
        regime['probs'] = {'choppy': 1.0}
        regime['confidence'] = 1.0
        regime['disagreement'] = 0.0
        regime['position_size_multiplier'] = 1.0

    print('Running: REGIME_OFF...')
    regime_off = _run_gate(dataset, bundle, plan, config, 'REGIME_OFF', mutate_snapshot=_no_regime)
    results.append(regime_off)
    print(f'  -> return={regime_off["annualized_return"]:+.1%}, rts={regime_off["realized_round_trips"]}, wr={regime_off["win_rate"]:.0%}')

    # Summary
    print()
    print('=' * 80)
    print('GATE ABLATION RESULTS')
    print('=' * 80)
    print(f'{"Variant":<25} {"Ann.Ret":>8} {"Sharpe":>8} {"MaxDD":>8} {"WR":>6} {"RTs":>5} {"Score":>8}')
    print('-' * 80)
    for r in results:
        print(f'{r["label"]:<25} {r["annualized_return"]:>+7.1%} {r["sharpe"]:>+7.2f} {r["max_drawdown"]:>+7.2%} {r["win_rate"]:>5.0%} {r["realized_round_trips"]:>5} {r["fold_score"]:>7.4f}')

    print()
    print('DELTAS vs BASELINE:')
    print(f'{"Variant":<25} {"dReturn":>8} {"dSharpe":>8} {"dMaxDD":>8} {"dRTs":>5}')
    print('-' * 60)
    for r in results[1:]:
        dr = r['annualized_return'] - baseline['annualized_return']
        ds = r['sharpe'] - baseline['sharpe']
        dd = r['max_drawdown'] - baseline['max_drawdown']
        drt = r['realized_round_trips'] - baseline['realized_round_trips']
        print(f'{r["label"]:<25} {dr:>+7.1%} {ds:>+7.2f} {dd:>+7.2%} {drt:>+5}')

    # Write JSON output
    output = {
        'gate_dates': f'{plan.gate_dates[0]} to {plan.gate_dates[-1]}',
        'gate_days': len(plan.gate_dates),
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
