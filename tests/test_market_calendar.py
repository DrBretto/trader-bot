"""Settled NY trading-day forward stamp (PKT-4)."""
from datetime import datetime, timezone

import pandas as pd

from chassis.utils.market_calendar import (
    is_trading_session,
    latest_settled_session,
    morning_execution_window_open,
    ny_today,
    settled_day_from_prices,
    trading_sessions_between,
)
from chassis.steps.morning_executor import validate_intent_freshness


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_friday_night_utc_resolves_to_friday_not_saturday():
    """The bug: 03:00 UTC Sat is the Friday ET evening -> a Friday key, not Saturday."""
    fri_night = _utc(2026, 6, 27, 3, 0)  # 23:00 ET Fri 06-26
    assert ny_today(fri_night).isoformat() == "2026-06-26"
    assert latest_settled_session(now_utc=fri_night) == "2026-06-26"


def test_weekend_runs_walk_back_to_last_weekday():
    assert latest_settled_session(now_utc=_utc(2026, 6, 27, 14)) == "2026-06-26"  # Sat
    assert latest_settled_session(now_utc=_utc(2026, 6, 28, 14)) == "2026-06-26"  # Sun


def test_settled_day_from_prices_uses_spy_max():
    panel = pd.DataFrame([
        {"symbol": "SPY", "date": "2026-06-25"},
        {"symbol": "SPY", "date": "2026-06-26"},
        {"symbol": "QQQ", "date": "2026-06-26"},
    ])
    assert settled_day_from_prices(panel) == "2026-06-26"


def test_settled_day_from_prices_holiday_frozen_close():
    """A holiday run has no new SPY bar -> the settled day is the prior session,
    which the append-only frontier guard then treats as a no-op (no leaf)."""
    panel = pd.DataFrame([{"symbol": "SPY", "date": "2026-07-02"}])  # 07-03 mkt closed
    assert settled_day_from_prices(panel, now_utc=_utc(2026, 7, 3, 22)) == "2026-07-02"


def test_july_3_2026_is_a_market_holiday_not_an_execution_day():
    assert not is_trading_session("2026-07-03")
    assert latest_settled_session(now_utc=_utc(2026, 7, 3, 14)) == "2026-07-02"


def test_current_session_is_not_settled_before_close_buffer():
    assert latest_settled_session(now_utc=_utc(2026, 7, 14, 15)) == "2026-07-13"
    assert latest_settled_session(now_utc=_utc(2026, 7, 14, 21)) == "2026-07-14"


def test_morning_window_is_dst_aware():
    assert not morning_execution_window_open(_utc(2026, 1, 15, 13, 45))  # 08:45 EST
    assert morning_execution_window_open(_utc(2026, 1, 15, 14, 45))      # 09:45 EST
    assert morning_execution_window_open(_utc(2026, 7, 15, 13, 45))      # 09:45 EDT


def test_intent_freshness_counts_sessions_not_calendar_days():
    intents = {"generated_date": "2026-07-02"}
    assert trading_sessions_between("2026-07-02", "2026-07-06") == 1
    assert validate_intent_freshness(intents, as_of_date="2026-07-06")
    assert not validate_intent_freshness(intents, as_of_date="2026-07-07")


def test_settled_day_from_prices_empty_panel_falls_back_to_session():
    got = settled_day_from_prices(pd.DataFrame(), now_utc=_utc(2026, 6, 27, 3))
    assert got == "2026-06-26"


def test_settled_day_from_prices_no_spy_uses_overall_max():
    panel = pd.DataFrame([{"symbol": "QQQ", "date": "2026-06-24"}])
    assert settled_day_from_prices(panel) == "2026-06-24"
