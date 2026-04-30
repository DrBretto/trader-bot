"""Hard guardrail checks for champion-challenger promotion safety."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from optimizer.config import GuardrailConfig
from optimizer.fitness import SegmentMetrics


# Phase 3 of the 2026-04-30 optimizer empirical-mutation fix.
#
# Standard guardrails apply to all challengers. The calibration-only path
# drops the trade-frequency gates because a normalization-constant-only
# delta changes *how much* the bot trades, not *whether* it trades.
# Keeping those gates active for calibration deltas is what caused recent
# weekly runs to reject 100% of calibration-class candidates with zero
# round-trips and -99.9% fold returns even when the calibration was
# directionally correct. The calibration-only path is stricter on outcome
# quality (max_drawdown, cost_ratio, min_fold_ann_return) and looser on
# the gates that condition trades themselves.
CALIBRATION_GUARDRAIL_DROPS = frozenset({
    'min_round_trips_total',
    'min_gate_round_trips',
    'min_gate_win_rate',
})


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
    calibration_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate hard constraints across walk-forward folds and gate segment.

    `calibration_only=True` switches to the calibration-class guardrail
    config: trade-frequency gates are skipped (they're not behaviorally
    conditioned by a normalization-constant-only delta), outcome-quality
    gates remain strict. See module docstring for rationale.
    """
    folds = list(fold_metrics)

    if not folds:
        return {
            'passed': False,
            'calibration_only': calibration_only,
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

    all_checks: List[GuardrailCheck] = [
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

    if calibration_only:
        active_checks = [c for c in all_checks if c.name not in CALIBRATION_GUARDRAIL_DROPS]
    else:
        active_checks = all_checks

    passed = all(check.passed for check in active_checks)
    return {
        'passed': passed,
        'calibration_only': calibration_only,
        'checks': [check.to_dict() for check in active_checks],
    }
