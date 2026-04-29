"""CLI for offline champion-challenger optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from optimizer.champion_challenger import run_optimizer_cycle
from optimizer.config import load_optimizer_config
from optimizer.data_access import OptimizerDataset, load_optimizer_dataset
from optimizer.guardrails import evaluate_guardrails
from optimizer.persistence import append_lineage_event, read_json, utc_now_iso
from optimizer.promote import load_active_bundle, promote_candidate_bundle
from optimizer.rollback import rollback_to
from optimizer.walk_forward import build_walk_forward_plan, evaluate_gate_segment, evaluate_walk_forward


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _compute_spy_benchmarks(
    dataset: 'OptimizerDataset',
    plan: 'WalkForwardPlan',
) -> Dict[str, Any]:
    """Compute SPY buy-and-hold return over each fold and gate segment."""
    import math

    date_map = dataset.by_date()

    def _spy_return_for_dates(dates: list[str]) -> Dict[str, Any]:
        if len(dates) < 2:
            return {'return': 0.0, 'annualized_return': 0.0, 'days': 0}
        first_date = dates[0]
        last_date = dates[-1]
        spy_start = spy_end = None
        if first_date in date_map:
            prices = date_map[first_date].next_prices_df
            spy_rows = prices[prices['symbol'] == 'SPY'] if 'symbol' in prices.columns else prices.head(0)
            if len(spy_rows) > 0:
                spy_start = float(spy_rows.iloc[0].get('close', spy_rows.iloc[0].get('open', 0)))
        if last_date in date_map:
            prices = date_map[last_date].next_prices_df
            spy_rows = prices[prices['symbol'] == 'SPY'] if 'symbol' in prices.columns else prices.head(0)
            if len(spy_rows) > 0:
                spy_end = float(spy_rows.iloc[0].get('close', spy_rows.iloc[0].get('open', 0)))
        if not spy_start or not spy_end or spy_start <= 0:
            return {'return': 0.0, 'annualized_return': 0.0, 'days': len(dates)}
        total_return = (spy_end / spy_start) - 1.0
        ann_return = 0.0
        if len(dates) > 0 and (1.0 + total_return) > 0:
            ann_return = math.pow(1.0 + total_return, 252.0 / len(dates)) - 1.0
        return {
            'return': round(total_return, 6),
            'annualized_return': round(ann_return, 6),
            'days': len(dates),
            'spy_start': round(spy_start, 2) if spy_start else None,
            'spy_end': round(spy_end, 2) if spy_end else None,
        }

    fold_benchmarks = []
    for fold in plan.folds:
        fold_benchmarks.append({
            'fold_id': fold.fold_id,
            'test_start': fold.test_start,
            'test_end': fold.test_end,
            **_spy_return_for_dates(fold.test_dates),
        })

    gate_benchmark = {
        'segment': 'gate',
        'start': plan.gate_dates[0] if plan.gate_dates else None,
        'end': plan.gate_dates[-1] if plan.gate_dates else None,
        **_spy_return_for_dates(plan.gate_dates),
    }

    return {
        'folds': fold_benchmarks,
        'gate': gate_benchmark,
    }


def cmd_run(args: argparse.Namespace) -> int:
    config = load_optimizer_config(args.config)
    result = run_optimizer_cycle(config)
    _print_json(result)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_optimizer_config(args.config)
    active_bundle = load_active_bundle(config.config_dir_path)

    dataset = load_optimizer_dataset(config)
    plan = build_walk_forward_plan(
        dates=dataset.dates,
        train_days=config.train_days,
        test_days=config.test_days,
        step_days=config.step_days,
        gate_days=config.gate_days,
    )

    wf = evaluate_walk_forward(
        dataset=dataset,
        bundle=active_bundle,
        plan=plan,
        random_seed=config.random_seed,
        initial_capital=config.initial_portfolio_value,
        max_workers=config.max_workers,
    )
    gate = evaluate_gate_segment(
        dataset=dataset,
        bundle=active_bundle,
        plan=plan,
        random_seed=config.random_seed,
        initial_capital=config.initial_portfolio_value,
    )
    guardrails = evaluate_guardrails(
        fold_metrics=wf['fold_metrics'],
        gate_metrics=gate,
        config=config.guardrails,
    )

    # Compute SPY benchmark for each fold and the gate segment
    spy_benchmarks = _compute_spy_benchmarks(dataset, plan)

    payload = {
        'version_id': active_bundle.get('version_id'),
        'aligned_snapshots': len(dataset.snapshots),
        'date_range': f'{dataset.dates[0]} to {dataset.dates[-1]}',
        'wf_objective': wf['wf_objective'],
        'wf_mean': wf['wf_mean'],
        'wf_stability_penalty': wf['wf_stability_penalty'],
        'fold_metrics': [metric.to_dict() for metric in wf['fold_metrics']],
        'gate_metrics': gate.to_dict(),
        'spy_benchmark': spy_benchmarks,
        'guardrails': guardrails,
    }
    _print_json(payload)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    config = load_optimizer_config(args.config)
    promote_result = promote_candidate_bundle(config.config_dir_path)
    to_version = str(promote_result.get('to_version', 'unknown'))
    append_lineage_event(
        config=config,
        event={
            'event_type': 'promotion',
            'timestamp': utc_now_iso(),
            'from_version': promote_result.get('from_version'),
            'to_version': promote_result.get('to_version'),
            'run_id': None,
            'reason': 'manual promote command',
            'operator': 'manual',
        },
        active_version=to_version,
    )
    _print_json(promote_result)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    config = load_optimizer_config(args.config)
    result = rollback_to(config.config_dir_path, args.to)
    append_lineage_event(
        config=config,
        event={
            'event_type': 'rollback',
            'timestamp': utc_now_iso(),
            'from_version': result.get('from_version'),
            'to_version': result.get('to_version'),
            'run_id': None,
            'reason': 'manual rollback',
            'operator': 'manual',
        },
        active_version=str(result.get('to_version', 'unknown')),
    )
    _print_json(result)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = load_optimizer_config(args.config)
    index = read_json(config.run_root_path / 'index.json', default={'runs': [], 'active_version': None})
    active_bundle = load_active_bundle(config.config_dir_path)
    active_version = index.get('active_version') or active_bundle.get('version_id')
    lineage = read_json(
        config.run_root_path / 'active_params_lineage.json',
        default={'history': [], 'active_version': active_version},
    )

    latest = (index.get('runs') or [None])[0]
    payload = {
        'active_version': active_version,
        'last_run': latest,
        'lineage_events': len(lineage.get('history', [])),
        'run_root': str(config.run_root_path),
    }
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Offline champion-challenger optimizer')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', default='config/optimizer.yaml', help='Path to optimizer config')

    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', parents=[common], help='Run one optimization cycle and exit')
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser('evaluate', parents=[common], help='Evaluate active params only')
    eval_parser.set_defaults(func=cmd_evaluate)

    promote_parser = subparsers.add_parser('promote', parents=[common], help='Promote existing candidate bundle')
    promote_parser.set_defaults(func=cmd_promote)

    rollback_parser = subparsers.add_parser('rollback', parents=[common], help='Rollback active params to historical file')
    rollback_parser.add_argument('--to', required=True, help='History file path or name under config/decision_params.history')
    rollback_parser.set_defaults(func=cmd_rollback)

    status_parser = subparsers.add_parser('status', parents=[common], help='Show optimizer status and last run')
    status_parser.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
