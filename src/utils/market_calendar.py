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
  * ``latest_settled_session()`` — the latest NY weekday on/before the ET date of
    "now". A cheap calendar proxy (weekend-aware; conservative on holidays) used by
    the intraday phases that run during the live session and hold no fresh panel.

The equity ledger's append-only frontier guard (``run_date <= frontier`` is a
no-op) is what turns a duplicate/holiday key into "no leaf" — so keying by the
settled day here is sufficient to guarantee weekends/holidays produce no leaf.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

_NY_TZ = "America/New_York"


def _now_utc(now_utc: Optional[datetime] = None) -> datetime:
    """The reference instant, coerced to a UTC-aware datetime."""
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)


def ny_today(now_utc: Optional[datetime] = None) -> date:
    """The New-York calendar date of ``now_utc`` (current UTC instant by default).

    DST-aware when the tz database is present (``zoneinfo``); otherwise falls back
    to a fixed −5h offset. At the three cron hours this system fires (night 03:00
    UTC, morning 14:45 UTC, midday 18:00 UTC), −5h and the true ET offset (−4h in
    summer, −5h in winter) land on the SAME ET calendar day, so the fallback is
    date-safe even in EDT.
    """
    u = _now_utc(now_utc)
    try:
        from zoneinfo import ZoneInfo

        return u.astimezone(ZoneInfo(_NY_TZ)).date()
    except Exception:  # noqa: BLE001 — tzdata missing in the runtime image
        return (u.astimezone(timezone(timedelta(hours=-5)))).date()


def latest_settled_session(now_utc: Optional[datetime] = None) -> str:
    """Latest NY weekday on/before ET-today as ``YYYY-MM-DD``.

    A calendar proxy: weekend-aware, and conservative on holidays (it never
    invents a session that the append-only frontier guard wouldn't already reject).
    Used by the intraday phases; the night prefers ``settled_day_from_prices``.
    """
    d = ny_today(now_utc)
    while d.weekday() >= 5:  # 5 = Sat, 6 = Sun
        d -= timedelta(days=1)
    return d.isoformat()


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
