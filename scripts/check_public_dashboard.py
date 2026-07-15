#!/usr/bin/env python3
"""Dependency-free external health check for the public Trader Bot dashboard."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

BASE_URL = "https://trader-bot.infotrope.io"


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _holidays(year: int) -> set[date]:
    out: set[date] = set()
    for y in range(year - 1, year + 2):
        out.update({
            _observed(date(y, 1, 1)),
            _nth_weekday(y, 1, 0, 3),
            _nth_weekday(y, 2, 0, 3),
            _easter(y) - timedelta(days=2),
            _last_weekday(y, 5, 0),
            _observed(date(y, 7, 4)),
            _nth_weekday(y, 9, 0, 1),
            _nth_weekday(y, 11, 3, 4),
            _observed(date(y, 12, 25)),
        })
        if y >= 2022:
            out.add(_observed(date(y, 6, 19)))
    return out


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in _holidays(d.year)


def expected_settled(now_utc: datetime | None = None) -> date:
    now = now_utc or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("America/New_York"))
    candidate = local.date()
    if local.time() < time(16, 15):
        candidate -= timedelta(days=1)
    while not is_session(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _session_lag(start: str, end: date) -> int:
    current = date.fromisoformat(start[:10])
    if current > end:
        return -1
    lag = 0
    while current < end:
        current += timedelta(days=1)
        if is_session(current):
            lag += 1
    return lag


def _fetch_json(path: str) -> Dict[str, Any]:
    stamp = int(datetime.now(timezone.utc).timestamp())
    request = Request(
        f"{BASE_URL}/{path}?health={stamp}",
        headers={"Cache-Control": "no-cache", "User-Agent": "trader-bot-health/1"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def check(require_morning: bool = False) -> Dict[str, Any]:
    dashboard = _fetch_json("dashboard.json")
    shadow = _fetch_json("shadow_timeseries.json")
    expected = expected_settled()
    expected_str = expected.isoformat()
    errors = []

    curve = dashboard.get("equity_curve") or []
    terminal = curve[-1] if curve else {}
    if terminal.get("date") != expected_str:
        errors.append(f"blue/SPY terminal {terminal.get('date')} != {expected_str}")
    if terminal.get("value") is None or terminal.get("benchmark") is None:
        errors.append("blue or SPY terminal is unpopulated")
    metrics = dashboard.get("metrics") or {}
    if metrics.get("canon_source") != "ledger":
        errors.append(f"canon_source is {metrics.get('canon_source')!r}, not ledger")
    if terminal.get("value") is not None and metrics.get("total_value") is not None:
        if round(float(terminal["value"]), 2) != round(float(metrics["total_value"]), 2):
            errors.append("headline total does not match blue ledger terminal")

    live = shadow.get("live_line") or []
    comparison = shadow.get("shadow_A") or []
    live_terminal = live[-1] if live else [None, None]
    comparison_terminal = comparison[-1] if comparison else [None, None]
    if live_terminal[0] != expected_str or live_terminal[1] is None:
        errors.append(f"blue mirror terminal is {live_terminal}")
    if comparison_terminal[0] != expected_str or comparison_terminal[1] is None:
        errors.append(f"yellow comparison terminal is {comparison_terminal}")
    if terminal.get("value") is not None and live_terminal[1] is not None:
        if round(float(terminal["value"]), 2) != round(float(live_terminal[1]), 2):
            errors.append("blue dashboard terminal and public mirror disagree")

    snapshot = dashboard.get("snapshot") or {}
    snapshot_date = snapshot.get("date")
    if not snapshot_date or _session_lag(snapshot_date, expected) > 0:
        errors.append(f"public operational snapshot is stale: {snapshot_date}")

    local_date = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    ).date()
    if require_morning and is_session(local_date):
        if snapshot_date != local_date.isoformat() or snapshot.get("phase") != "morning":
            errors.append(
                f"post-morning snapshot is {snapshot_date}/{snapshot.get('phase')}"
            )

    return {
        "ok": not errors,
        "expected_settled": expected_str,
        "snapshot": {"date": snapshot_date, "phase": snapshot.get("phase")},
        "blue": terminal.get("value"),
        "spy": terminal.get("benchmark"),
        "yellow": comparison_terminal[1],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-morning", action="store_true")
    args = parser.parse_args()
    report = check(require_morning=args.require_morning)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
