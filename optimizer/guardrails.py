"""Hard guardrail checks for champion-challenger promotion safety."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from optimizer.config import GuardrailConfig
from optimizer.fitness import SegmentMetrics


@dataclass
class GuardrailCheck:
    name: str
    passed: bool
    value: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_guardrails(
    fold_metrics: Iterable[SegmentMetrics],
    gate_metrics: SegmentMetrics,
    config: GuardrailConfig,
) -> Dict[str, Any]:
    """Evaluate hard constraints across walk-forward folds and gate segment."""
    folds = list(fold_metrics)

    if not folds:
        return {
            'passed': False,
            'checks': [
                {
                    'name': 'has_folds',
                    'passed': False,
                    'value': 0.0,
                    'threshold': 1.0,
                }
            ],
        }

    worst_drawdown = min([fold.max_drawdown for fold in folds] + [gate_metrics.max_drawdown])
    worst_cost_ratio = max([fold.cost_ratio for fold in folds] + [gate_metrics.cost_ratio])
    total_round_trips = sum(fold.realized_round_trips for fold in folds)
    min_fold_ann_return = min(fold.annualized_return for fold in folds)

    checks: List[GuardrailCheck] = [
        GuardrailCheck(
            name='max_drawdown_cap',
            passed=worst_drawdown >= config.max_drawdown,
            value=worst_drawdown,
            threshold=config.max_drawdown,
        ),
        GuardrailCheck(
            name='cost_ratio_cap',
            passed=worst_cost_ratio <= config.max_cost_ratio,
            value=worst_cost_ratio,
            threshold=config.max_cost_ratio,
        ),
        GuardrailCheck(
            name='min_round_trips_total',
            passed=total_round_trips >= config.min_total_round_trips,
            value=float(total_round_trips),
            threshold=float(config.min_total_round_trips),
        ),
        GuardrailCheck(
            name='min_gate_round_trips',
            passed=gate_metrics.realized_round_trips >= config.min_gate_round_trips,
            value=float(gate_metrics.realized_round_trips),
            threshold=float(config.min_gate_round_trips),
        ),
        GuardrailCheck(
            name='min_gate_win_rate',
            passed=gate_metrics.win_rate >= config.min_gate_win_rate,
            value=gate_metrics.win_rate,
            threshold=config.min_gate_win_rate,
        ),
        GuardrailCheck(
            name='min_fold_ann_return',
            passed=min_fold_ann_return >= config.min_fold_annualized_return,
            value=min_fold_ann_return,
            threshold=config.min_fold_annualized_return,
        ),
    ]

    passed = all(check.passed for check in checks)
    return {
        'passed': passed,
        'checks': [check.to_dict() for check in checks],
    }
