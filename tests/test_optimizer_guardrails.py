from optimizer.config import GuardrailConfig
from optimizer.fitness import SegmentMetrics
from optimizer.guardrails import CALIBRATION_GUARDRAIL_DROPS, evaluate_guardrails


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


# Phase 3 of the 2026-04-30 optimizer empirical-mutation packet.

def test_calibration_only_drops_trade_frequency_gates() -> None:
    """A challenger with zero round-trips fails the standard path
    (min_round_trips_total, min_gate_round_trips, min_gate_win_rate) but
    should pass the calibration-only path because those gates are
    irrelevant to a normalization-constant-only delta. Outcome-quality
    gates remain strict.
    """
    config = GuardrailConfig()
    folds = [
        _segment(0.0, -0.02, 0.0, 0, 0.0),
        _segment(0.0, -0.01, 0.0, 0, 0.0),
    ]
    gate = _segment(0.0, -0.02, 0.0, 0, 0.0)

    standard = evaluate_guardrails(folds, gate, config, calibration_only=False)
    calibration = evaluate_guardrails(folds, gate, config, calibration_only=True)

    assert standard['passed'] is False
    standard_failures = {c['name'] for c in standard['checks'] if not c['passed']}
    assert 'min_round_trips_total' in standard_failures
    assert 'min_gate_round_trips' in standard_failures
    assert 'min_gate_win_rate' in standard_failures

    assert calibration['passed'] is True
    calibration_check_names = {c['name'] for c in calibration['checks']}
    assert calibration_check_names.isdisjoint(CALIBRATION_GUARDRAIL_DROPS)


def test_calibration_only_keeps_outcome_quality_gates() -> None:
    """Calibration-only path stays strict on max_drawdown_cap,
    cost_ratio_cap, min_fold_ann_return. A blowup in any of those rejects.
    """
    config = GuardrailConfig()
    folds = [_segment(0.10, -0.30, 0.55, 5, 0.010)]
    gate = _segment(0.05, -0.30, 0.55, 5, 0.010)

    result = evaluate_guardrails(folds, gate, config, calibration_only=True)
    assert result['passed'] is False
    failures = {c['name'] for c in result['checks'] if not c['passed']}
    assert 'max_drawdown_cap' in failures


def test_calibration_only_flag_in_payload() -> None:
    """Result payload exposes `calibration_only` so downstream tools can
    distinguish calibration-class rejections from mixed-class rejections.
    """
    config = GuardrailConfig()
    folds = [_segment(0.10, -0.10, 0.55, 12, 0.010)]
    gate = _segment(0.05, -0.05, 0.55, 8, 0.010)

    standard = evaluate_guardrails(folds, gate, config, calibration_only=False)
    calibration = evaluate_guardrails(folds, gate, config, calibration_only=True)

    assert standard['calibration_only'] is False
    assert calibration['calibration_only'] is True
