"""Fail-loud substrate-currency gate (PKT-TRADER-BOT-FORECAST-SPINE-RELOCATE, P3).

Relocated VERBATIM from ``src/brain/runtime.py`` (the local-HEAD version). This is
the ENFORCING half of the freshness gate: it reads the store watermark, computes
the verdict (``store.ohlcv_store._freshness_gate_verdict``), and — when the
substrate is not current — RAISES ``StaleSubstrateError`` after firing an SNS
CRITICAL. ``decide.cutover.run_cutover``'s forecaster try/except catches the raise
-> ok=False -> the incumbent intents are retained (fail-safe).

The single worst failure this project has shipped (ISSUE-01/02): the in-Lambda
OHLCV store silently stopped advancing, so `mu` froze and the SAME concentrated
line published every night for two weeks with zero alarm — because every monitor
keyed on PUBLISH recency (fresh leaf written nightly) not SUBSTRATE currency (the
OHLCV panel behind mu). This gate keys on substrate currency.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from store.ohlcv_store import _freshness_gate_verdict, _ohlcv_watermark


class StaleSubstrateError(RuntimeError):
    """Raised by the freshness gate when the OHLCV substrate is not current.
    run_cutover's forecaster try/except catches it -> ok=False -> the incumbent
    intents are retained (fail-safe) and an SNS CRITICAL is raised."""


def _latest_settled_trading_day(today: Optional[str] = None) -> str:
    """Latest weekday on/before `today` (NY). The night forecasts off the last
    settled session; the OHLCV store's max bar should track this."""
    import datetime as _dt
    d = _dt.date.fromisoformat(today[:10]) if today else _dt.datetime.now(
        _dt.timezone.utc).astimezone(_dt.timezone(-_dt.timedelta(hours=5))).date()
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    return d.isoformat()


def _alert_stale_substrate(verdict: dict, fresh: dict) -> None:
    reason = "; ".join(verdict.get("reasons") or ["stale"])
    body = (
        "FAIL-LOUD STALENESS GATE FIRED — the night was ABORTED before shipping a "
        "forecast over a stale substrate (the frozen-`mu` failure can no longer ship "
        "silently).\n\n"
        f"run_date                 = {verdict.get('run_date')}\n"
        f"OHLCV store max bar      = {verdict.get('ohlcv_max_date')}\n"
        f"bars spliced this run    = {verdict.get('bars_added')}\n"
        f"newer daily dates (gap)  = {verdict.get('gap_len')}\n"
        f"reason                   = {reason}\n\n"
        "extend errors:\n  " + ("\n  ".join(fresh.get("errors") or ["(none)"])) + "\n\n"
        "The incumbent intents are retained (fail-safe). Check CloudWatch "
        "/aws/lambda/investment-system-daily-pipeline for the [FRESHNESS] lines.")
    try:
        from src.utils.sns_alerts import send_alert
        send_alert(
            subject="[TraderBot] CRITICAL: forecast substrate STALE — night ABORTED (fail-loud gate)",
            body=body)
    except Exception as e:  # noqa: BLE001 — alerting must never crash the check
        print(f"  [FRESHNESS] stale-substrate alert failed (non-fatal): {e}")


def _assert_substrate_current(run_date: str, SL, fresh: dict) -> None:
    """The enforced gate. Raises StaleSubstrateError (+ SNS alert) when the OHLCV
    substrate is not current, so a frozen forecast cannot ship."""
    ohlcv_max, _rows = _ohlcv_watermark(SL)
    verdict = _freshness_gate_verdict(run_date, ohlcv_max, fresh)
    print(f"  [FRESHNESS] gate verdict: {verdict}")
    if verdict["stale"]:
        _alert_stale_substrate(verdict, fresh)
        raise StaleSubstrateError("; ".join(verdict["reasons"]))
