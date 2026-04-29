"""Walk-forward split construction and deterministic candidate evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from optimizer.data_access import OptimizerDataset
from optimizer.fitness import SegmentMetrics, aggregate_walk_forward, compute_segment_metrics
from optimizer.replay import run_replay_for_dates


@dataclass
class FoldWindow:
    fold_id: int
    train_dates: List[str]
    test_dates: List[str]

    @property
    def train_start(self) -> str:
        return self.train_dates[0]

    @property
    def train_end(self) -> str:
        return self.train_dates[-1]

    @property
    def test_start(self) -> str:
        return self.test_dates[0]

    @property
    def test_end(self) -> str:
        return self.test_dates[-1]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                'train_start': self.train_start,
                'train_end': self.train_end,
                'test_start': self.test_start,
                'test_end': self.test_end,
                'test_days': len(self.test_dates),
                'complete_dates': len(self.test_dates),
            }
        )
        return payload


@dataclass
class WalkForwardPlan:
    dates: List[str]
    folds: List[FoldWindow]
    gate_dates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        gate = {
            'start': self.gate_dates[0] if self.gate_dates else None,
            'end': self.gate_dates[-1] if self.gate_dates else None,
            'days': len(self.gate_dates),
        }
        return {
            'folds': [fold.to_dict() for fold in self.folds],
            'gate_segment': gate,
        }


def build_walk_forward_plan(
    dates: List[str],
    train_days: int,
    test_days: int,
    step_days: int,
    gate_days: int,
) -> WalkForwardPlan:
    """Create walk-forward folds and a holdout gate segment."""
    unique_dates = sorted({str(date) for date in dates})
    required = train_days + test_days + gate_days
    if len(unique_dates) < required:
        raise ValueError(
            f'Insufficient dates for walk-forward plan: have={len(unique_dates)} need={required}'
        )

    gate_dates = unique_dates[-gate_days:]
    optimization_dates = unique_dates[:-gate_days]

    folds: List[FoldWindow] = []
    fold_id = 1
    start = 0
    while start + train_days + test_days <= len(optimization_dates):
        train_slice = optimization_dates[start:start + train_days]
        test_slice = optimization_dates[start + train_days:start + train_days + test_days]
        folds.append(FoldWindow(fold_id=fold_id, train_dates=train_slice, test_dates=test_slice))
        fold_id += 1
        start += step_days

    if not folds:
        raise ValueError('Walk-forward configuration produced zero folds')

    return WalkForwardPlan(dates=unique_dates, folds=folds, gate_dates=gate_dates)


def _evaluate_single_fold(
    dataset: OptimizerDataset,
    bundle: Dict[str, Any],
    fold: FoldWindow,
    random_seed: int,
    initial_capital: float,
) -> SegmentMetrics:
    replay_dates = [*fold.train_dates, *fold.test_dates]
    replay_result = run_replay_for_dates(
        dataset=dataset,
        decision_dates=replay_dates,
        candidate_bundle=bundle,
        random_seed=random_seed,
        initial_capital=initial_capital,
    )
    return compute_segment_metrics(
        steps=replay_result.steps,
        fills=replay_result.fills,
        segment_dates=fold.test_dates,
    )


def evaluate_walk_forward(
    dataset: OptimizerDataset,
    bundle: Dict[str, Any],
    plan: WalkForwardPlan,
    random_seed: int,
    initial_capital: float,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Evaluate a parameter bundle across walk-forward test folds."""

    def _fold_seed(fold_id: int) -> int:
        return random_seed + fold_id * 1009

    if max_workers > 1 and len(plan.folds) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_single_fold,
                    dataset,
                    bundle,
                    fold,
                    _fold_seed(fold.fold_id),
                    initial_capital,
                )
                for fold in plan.folds
            ]
            fold_metrics = [future.result() for future in futures]
    else:
        fold_metrics = [
            _evaluate_single_fold(
                dataset=dataset,
                bundle=bundle,
                fold=fold,
                random_seed=_fold_seed(fold.fold_id),
                initial_capital=initial_capital,
            )
            for fold in plan.folds
        ]

    agg = aggregate_walk_forward(fold_metrics)
    return {
        'fold_metrics': fold_metrics,
        'wf_mean': agg['wf_mean'],
        'wf_stability_penalty': agg['wf_stability_penalty'],
        'wf_objective': agg['wf_objective'],
        'total_round_trips': sum(metric.realized_round_trips for metric in fold_metrics),
    }


def evaluate_gate_segment(
    dataset: OptimizerDataset,
    bundle: Dict[str, Any],
    plan: WalkForwardPlan,
    random_seed: int,
    initial_capital: float,
) -> SegmentMetrics:
    """Evaluate holdout promotion gate segment with historical warmup state."""
    if not plan.gate_dates:
        raise ValueError('Gate segment is empty')

    gate_start = plan.gate_dates[0]
    gate_index = plan.dates.index(gate_start)
    replay_dates = plan.dates[:gate_index] + plan.gate_dates

    replay_result = run_replay_for_dates(
        dataset=dataset,
        decision_dates=replay_dates,
        candidate_bundle=bundle,
        random_seed=random_seed + 900001,
        initial_capital=initial_capital,
    )

    return compute_segment_metrics(
        steps=replay_result.steps,
        fills=replay_result.fills,
        segment_dates=plan.gate_dates,
    )
