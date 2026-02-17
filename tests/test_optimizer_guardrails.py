from optimizer.config import GuardrailConfig
from optimizer.fitness import SegmentMetrics
from optimizer.guardrails import evaluate_guardrails


def _segment(
    annualized_return: float,
    max_drawdown: float,
    win_rate: float,
    round_trips: int,
    cost_ratio: float,
) -> SegmentMetrics:
    return SegmentMetrics(
        days=63,
        annualized_return=annualized_return,
        sharpe=1.0,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        realized_round_trips=round_trips,
        wins=int(round(round_trips * win_rate)),
        losses=max(round_trips - int(round(round_trips * win_rate)), 0),
        breakeven=0,
        cost_ratio=cost_ratio,
        traded_notional=100000.0,
        cumulative_transaction_costs=1000.0,
        fold_score=0.6,
    )


def test_guardrails_pass_happy_path() -> None:
    config = GuardrailConfig()
    folds = [
        _segment(0.15, -0.12, 0.56, 12, 0.009),
        _segment(0.11, -0.18, 0.58, 10, 0.010),
    ]
    gate = _segment(0.09, -0.10, 0.50, 6, 0.011)

    result = evaluate_guardrails(folds, gate, config)

    assert result['passed'] is True
    assert all(check['passed'] for check in result['checks'])


def test_guardrails_fail_on_gate_win_rate() -> None:
    config = GuardrailConfig()
    folds = [
        _segment(0.12, -0.15, 0.55, 15, 0.010),
        _segment(0.08, -0.13, 0.52, 10, 0.009),
    ]
    gate = _segment(0.02, -0.11, 0.30, 7, 0.010)

    result = evaluate_guardrails(folds, gate, config)

    assert result['passed'] is False
    gate_check = [check for check in result['checks'] if check['name'] == 'min_gate_win_rate'][0]
    assert gate_check['passed'] is False
