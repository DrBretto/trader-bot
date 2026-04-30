"""Tests for the persistent-rejection alert (Phase 4 of the
2026-04-30 optimizer empirical-mutation packet)."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.alerts import build_persistent_rejection_alert_body
from optimizer.config import OptimizerConfig
from optimizer.persistence import update_rejection_streak


@pytest.fixture
def cfg(tmp_path) -> OptimizerConfig:
    """OptimizerConfig with a temp-dir run_root so the streak state file
    is sandboxed per-test."""
    config = OptimizerConfig(
        rejection_streak_alert_threshold=3,
    )
    config.repo_root = tmp_path
    config.paths.run_root = 'runs/optimizer'
    (tmp_path / 'runs' / 'optimizer').mkdir(parents=True, exist_ok=True)
    return config


def _summary(run_id: str, decision: str) -> dict:
    return {'run_id': run_id, 'decision': decision}


def test_first_rejection_no_alert(cfg):
    state = update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    assert state['consecutive_rejections'] == 1
    assert state['alert_should_fire'] is False


def test_second_rejection_no_alert(cfg):
    update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('run-2', 'not_promoted'))
    assert state['consecutive_rejections'] == 2
    assert state['alert_should_fire'] is False


def test_third_rejection_fires_alert(cfg):
    update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('run-2', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('run-3', 'not_promoted'))
    assert state['consecutive_rejections'] == 3
    assert state['alert_should_fire'] is True
    assert state['last_alert_run_id'] == 'run-3'


def test_fourth_rejection_does_not_re_fire(cfg):
    """Once alerted for a streak, subsequent rejections in the same
    streak do not re-fire. Otherwise the alert would fire every weekly
    cycle once the streak is long enough.
    """
    update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('run-2', 'not_promoted'))
    update_rejection_streak(cfg, _summary('run-3', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('run-4', 'not_promoted'))
    assert state['consecutive_rejections'] == 4
    assert state['alert_should_fire'] is False
    # last_alert_run_id stays at run-3.
    assert state['last_alert_run_id'] == 'run-3'


def test_promotion_resets_counter(cfg):
    update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('run-2', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('run-3', 'promoted'))
    assert state['consecutive_rejections'] == 0
    assert state['alert_should_fire'] is False
    assert state['last_alert_run_id'] is None


def test_promotion_in_run_2_prevents_run_3_alert(cfg):
    """Per the packet's example: a promotion in run 2 prevents the run-3
    alert. Confirms the streak resets correctly.
    """
    update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('run-2', 'promoted'))
    state = update_rejection_streak(cfg, _summary('run-3', 'not_promoted'))
    assert state['consecutive_rejections'] == 1
    assert state['alert_should_fire'] is False


def test_alert_re_fires_for_new_streak(cfg):
    """After a promotion, a new streak that hits threshold fires the
    alert again (different streak, different alert)."""
    # First streak: 3 rejections, alert fires.
    update_rejection_streak(cfg, _summary('a-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('a-2', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('a-3', 'not_promoted'))
    assert state['alert_should_fire'] is True

    # Promotion resets.
    update_rejection_streak(cfg, _summary('promo', 'promoted'))

    # New streak: alert fires again on the new run-3.
    update_rejection_streak(cfg, _summary('b-1', 'not_promoted'))
    update_rejection_streak(cfg, _summary('b-2', 'not_promoted'))
    state = update_rejection_streak(cfg, _summary('b-3', 'not_promoted'))
    assert state['alert_should_fire'] is True
    assert state['last_alert_run_id'] == 'b-3'


class TestAlertBody:
    """Smoke-test the alert body builder produces something operator-readable."""

    def test_subject_carries_traderbot_prefix(self, cfg):
        update_rejection_streak(cfg, _summary('run-1', 'not_promoted'))
        update_rejection_streak(cfg, _summary('run-2', 'not_promoted'))
        state = update_rejection_streak(cfg, _summary('run-3', 'not_promoted'))

        subject, body = build_persistent_rejection_alert_body(
            streak_state=state,
            config=cfg,
            latest_guardrail_results={'checks': [
                {'name': 'min_round_trips_total', 'passed': False, 'value': 0.0, 'threshold': 20.0},
                {'name': 'max_drawdown_cap', 'passed': True, 'value': -0.05, 'threshold': -0.25},
            ]},
            latest_promotion_payload={
                'run_id': 'run-3',
                'champion_version_before': 'hybrid-ranking-035-v1',
                'challenger_version': 'opt-20260430T...',
                'reason_codes': ['runtime_error'],
            },
            latest_challenger_metrics={
                'wf_objective': 0.123,
                'guardrails': {'passed': False, 'calibration_only': False},
            },
        )

        assert subject.startswith('[TraderBot]')
        assert '3 consecutive cycles' in subject
        # Body must include the operator action runbook command.
        assert '--empirical-mutation-debug' in body
        # Body must list the failed check name.
        assert 'min_round_trips_total' in body
        # Body must reference the verification gate target.
        assert 'AVG_CORR_MEAN' in body
