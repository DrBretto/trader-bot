"""Fitness and metric computation for optimizer walk-forward evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Dict, Iterable, List

from src.utils.dashboard_metrics import _build_trade_summary


@dataclass
class SegmentMetrics:
    """Evaluation metrics for a scored segment (fold test or promotion gate)."""

    days: int
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    realized_round_trips: int
    wins: int
    losses: int
    breakeven: int
    cost_ratio: float
    traded_notional: float
    cumulative_transaction_costs: float
    fold_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_fold_score(
    annualized_return: float,
    sharpe: float,
    max_drawdown: float,
    win_rate: float,
) -> float:
    """Compute normalized fold score from plan-defined formula."""
    sharpe_norm = (_clip(sharpe, -1.0, 3.0) + 1.0) / 4.0
    calmar = annualized_return / max(abs(max_drawdown), 0.01)
    calmar_norm = (_clip(calmar, -1.0, 4.0) + 1.0) / 5.0
    ret_norm = (_clip(annualized_return, -0.50, 0.80) + 0.50) / 1.30
    win_norm = _clip(win_rate, 0.0, 1.0)
    return 0.35 * sharpe_norm + 0.30 * calmar_norm + 0.20 * ret_norm + 0.15 * win_norm


def aggregate_walk_forward(fold_metrics: Iterable[SegmentMetrics]) -> Dict[str, float]:
    """Aggregate fold scores into objective + stability penalty."""
    scores = [metric.fold_score for metric in fold_metrics]
    if not scores:
        return {
            'wf_mean': float('-inf'),
            'wf_stability_penalty': 0.0,
            'wf_objective': float('-inf'),
        }

    wf_mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    wf_stability_penalty = 0.20 * stdev
    wf_objective = wf_mean - wf_stability_penalty
    return {
        'wf_mean': wf_mean,
        'wf_stability_penalty': wf_stability_penalty,
        'wf_objective': wf_objective,
    }


def compute_segment_metrics(
    steps: List[Any],
    fills: List[Dict[str, Any]],
    segment_dates: List[str],
) -> SegmentMetrics:
    """Compute annualized return, Sharpe, drawdown, win rate, and cost metrics."""
    segment_date_set = set(segment_dates)

    segment_steps = [
        step
        for step in steps
        if str(getattr(step, 'valuation_date', '')) in segment_date_set
    ]

    if not segment_steps:
        return SegmentMetrics(
            days=0,
            annualized_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            realized_round_trips=0,
            wins=0,
            losses=0,
            breakeven=0,
            cost_ratio=0.0,
            traded_notional=0.0,
            cumulative_transaction_costs=0.0,
            fold_score=0.0,
        )

    daily_returns: List[float] = []
    equity_points: List[float] = []

    first_start = _safe_float(getattr(segment_steps[0], 'start_value', 0.0), 0.0)
    equity_points.append(first_start)

    for step in segment_steps:
        start_value = _safe_float(getattr(step, 'start_value', 0.0), 0.0)
        end_value = _safe_float(getattr(step, 'end_value', 0.0), 0.0)
        day_return = 0.0
        if start_value > 0:
            day_return = (end_value / start_value) - 1.0
        daily_returns.append(day_return)
        equity_points.append(end_value)

    total_days = len(daily_returns)
    cumulative_return = 0.0
    if equity_points[0] > 0:
        cumulative_return = (equity_points[-1] / equity_points[0]) - 1.0

    annualized_return = 0.0
    if total_days > 0 and (1.0 + cumulative_return) > 0:
        annualized_return = math.pow(1.0 + cumulative_return, 252.0 / total_days) - 1.0

    sharpe = 0.0
    if len(daily_returns) >= 2:
        mean_ret = statistics.fmean(daily_returns)
        stdev_ret = statistics.pstdev(daily_returns)
        if stdev_ret > 1e-12:
            sharpe = (mean_ret / stdev_ret) * math.sqrt(252.0)

    peak = equity_points[0]
    max_drawdown = 0.0
    for value in equity_points:
        peak = max(peak, value)
        if peak <= 0:
            continue
        drawdown = (value / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    segment_fills = [
        fill
        for fill in fills
        if str(fill.get('_valuation_date') or fill.get('_trade_date') or '') in segment_date_set
    ]

    trade_summary = _build_trade_summary(segment_fills)
    traded_notional = sum(abs(_safe_float(fill.get('dollars'), 0.0)) for fill in segment_fills)
    cumulative_costs = _safe_float(trade_summary.get('cumulative_transaction_costs'), 0.0)
    cost_ratio = cumulative_costs / traded_notional if traded_notional > 0 else 0.0

    win_rate = _safe_float(trade_summary.get('win_rate'), 0.0)
    fold_score = compute_fold_score(annualized_return, sharpe, max_drawdown, win_rate)

    return SegmentMetrics(
        days=total_days,
        annualized_return=annualized_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        realized_round_trips=int(trade_summary.get('realized_round_trips', 0) or 0),
        wins=int(trade_summary.get('wins', 0) or 0),
        losses=int(trade_summary.get('losses', 0) or 0),
        breakeven=int(trade_summary.get('breakeven', 0) or 0),
        cost_ratio=cost_ratio,
        traded_notional=traded_notional,
        cumulative_transaction_costs=cumulative_costs,
        fold_score=fold_score,
    )
