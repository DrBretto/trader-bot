"""Settled NY trading-day forward stamp (PKT-4)."""
from datetime import datetime, timezone

import pandas as pd

from src.utils.market_calendar import (
    ny_today, latest_settled_session, settled_day_from_prices,
)


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


def test_settled_day_from_prices_empty_panel_falls_back_to_session():
    got = settled_day_from_prices(pd.DataFrame(), now_utc=_utc(2026, 6, 27, 3))
    assert got == "2026-06-26"


def test_settled_day_from_prices_no_spy_uses_overall_max():
    panel = pd.DataFrame([{"symbol": "QQQ", "date": "2026-06-24"}])
    assert settled_day_from_prices(panel) == "2026-06-24"
