"""Settled NY trading-day grid — the forward stamp key (PKT-4).

Every canon leaf / decision / ``daily/<D>/`` folder must be dated by the **settled
NY trading day** — the actual close date the book is marked at — NOT the UTC
wall-clock run date. The night job fires at 03:00 UTC (the prior ET evening), so
``datetime.now()`` in the Lambda (UTC) returns the *next* calendar day: a Friday
settle would land on a Saturday leaf. This module supplies the correct key.

Two sources, most-truthful first:

  * ``settled_day_from_prices(prices_df)`` — the max settled SPY bar date in the
    freshly-ingested price panel. This is holiday- AND weekend-aware by
    construction (a non-trading day simply has no bar) and is the night's source
    of truth ("the actual close date the book is marked at").
  * ``latest_settled_session()`` — the latest regular NYSE session on/before the ET
    date of "now". It is weekend- and holiday-aware and is used by intraday phases
    that hold no fresh settled panel.

The equity ledger's append-only frontier guard (``run_date <= frontier`` is a
no-op) is what turns a duplicate/holiday key into "no leaf" — so keying by the
settled day here is sufficient to guarantee weekends/holidays produce no leaf.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
)
from pandas.tseries.offsets import DateOffset

_NY_TZ = "America/New_York"


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular full-day NYSE closures used by the daily simulator.

    Early closes remain trading sessions, which is correct for this daily system.
    Unscheduled national closures are represented by the settled-price grid and can
    be added here when they occur.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=DateOffset(weekday=MO(3)),
            start_date="1998-01-01",
        ),
        Holiday(
            "Washington's Birthday",
            month=2,
            day=1,
            offset=DateOffset(weekday=MO(3)),
        ),
        GoodFriday,
        Holiday(
            "Memorial Day",
            month=5,
            day=31,
            offset=DateOffset(weekday=MO(-1)),
        ),
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date="2022-01-01",
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@lru_cache(maxsize=16)
def _holiday_dates(year: int) -> frozenset:
    start = pd.Timestamp(year=year - 1, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=12, day=31)
    return frozenset(ts.date() for ts in _NYSEHolidayCalendar().holidays(start, end))


def is_trading_session(value) -> bool:
    """Whether ``value`` is a regular NYSE trading session."""
    d = _as_date(value)
    return d.weekday() < 5 and d not in _holiday_dates(d.year)


def latest_session_on_or_before(value) -> date:
    """Walk backward to the latest regular NYSE session."""
    d = _as_date(value)
    while not is_trading_session(d):
        d -= timedelta(days=1)
    return d


def trading_sessions_between(start, end) -> int:
    """Count NYSE sessions strictly after ``start`` through ``end``."""
    a = _as_date(start)
    b = _as_date(end)
    if b < a:
        return -1
    count = 0
    cur = a
    while cur < b:
        cur += timedelta(days=1)
        if is_trading_session(cur):
            count += 1
    return count


def _now_utc(now_utc: Optional[datetime] = None) -> datetime:
    """The reference instant, coerced to a UTC-aware datetime."""
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)


def ny_now(now_utc: Optional[datetime] = None) -> datetime:
    """The timezone-aware New-York datetime for ``now_utc``."""
    u = _now_utc(now_utc)
    try:
        from zoneinfo import ZoneInfo

        return u.astimezone(ZoneInfo(_NY_TZ))
    except Exception:  # noqa: BLE001 - tzdata missing in the runtime image
        return u.astimezone(timezone(timedelta(hours=-5)))


def ny_today(now_utc: Optional[datetime] = None) -> date:
    """The New-York calendar date of ``now_utc`` (current UTC instant by default).

    DST-aware when the tz database is present (``zoneinfo``); otherwise falls back
    to a fixed −5h offset. At the three cron hours this system fires (night 03:00
    UTC, morning 14:45 UTC, midday 18:00 UTC), −5h and the true ET offset (−4h in
    summer, −5h in winter) land on the SAME ET calendar day, so the fallback is
    date-safe even in EDT.
    """
    return ny_now(now_utc).date()


def latest_settled_session(now_utc: Optional[datetime] = None) -> str:
    """Latest completed regular NYSE session as ``YYYY-MM-DD``.

    A session is not settled merely because its New-York calendar date has
    started. Before 16:15 ET, walk back from today so morning and overnight
    watchdogs never demand an intraday leaf that cannot exist yet.
    """
    local_now = ny_now(now_utc)
    candidate = local_now.date()
    if local_now.time() < time(16, 15):
        candidate -= timedelta(days=1)
    return latest_session_on_or_before(candidate).isoformat()


def morning_execution_window_open(now_utc: Optional[datetime] = None) -> bool:
    """Whether the scheduled morning execution window has opened in New York."""
    return ny_now(now_utc).time() >= time(9, 40)


def settled_day_from_prices(
    prices_df: Optional[pd.DataFrame],
    symbol: str = "SPY",
    now_utc: Optional[datetime] = None,
) -> str:
    """The settled NY trading day from a freshly-ingested price panel.

    Returns the max bar date for ``symbol`` (falling back to the max date across
    the whole panel, then to ``latest_settled_session()`` when the panel is empty
    or dateless). This is the actual close date the book is marked at — the night's
    forward-stamp key.
    """
    fallback = latest_settled_session(now_utc)
    if prices_df is None or len(prices_df) == 0 or "date" not in prices_df.columns:
        return fallback

    df = prices_df
    if "symbol" in df.columns:
        sym = df[df["symbol"] == symbol]
        if len(sym) > 0:
            df = sym

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if len(dates) == 0:
        return fallback
    return dates.max().strftime("%Y-%m-%d")
