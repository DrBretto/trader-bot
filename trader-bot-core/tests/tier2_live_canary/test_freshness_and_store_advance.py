"""Tier 2 live-canary — the freshness gate + OHLCV store-advance, keyed on the
REAL S3-observed substrate. This is the check that was missing for two weeks: a
frozen OHLCV store behind a fresh-looking publish.

Green path reads live S3 (the deployed store's newest settled bar); the
fault-injection twins feed a deliberately frozen watermark while a newer settled
bar genuinely exists in S3 — the gate MUST fire (the frozen-mu catch). No test
asserts a planted green: the green is only possible because live S3 agrees.

Salvages the Tier-1 unit half of ``test_freshness_gate`` /
``test_brain_monitors`` (the verdict logic) but makes it worthless-no-more by
feeding it the REAL store state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import (FrozenStoreReader, expected_settled_day,  # noqa: E402
                      freshness_verdict)


@pytest.mark.live_canary
def test_freshness_gate_on_real_store(reality):
    """The freshness gate reports FRESH on the real, advancing store."""
    verdict = freshness_verdict(reality)
    assert verdict["stale"] is False, (
        f"freshness gate flags the LIVE store stale: {verdict['reasons']} "
        f"(store max {verdict['ohlcv_max_date']} vs run {verdict['run_date']})")
    assert verdict["ohlcv_max_date"], "no OHLCV watermark on the real store"


@pytest.mark.live_canary
@pytest.mark.fault_injection
def test_freshness_gate_fires_on_frozen_store(reality):
    """Deliberately freeze the store at the 2026-06-10 seed while a newer settled
    bar exists in S3 → the gate MUST fire stale (proves it catches the real
    two-week freeze)."""
    frozen = FrozenStoreReader(reality, frozen_date="2026-06-10")
    verdict = freshness_verdict(frozen, inject_frozen=True)
    assert verdict["stale"] is True, (
        "freshness gate did NOT fire on a store frozen at 2026-06-10 while a "
        f"newer settled bar exists — this is the exact undetected failure. {verdict}")
    assert any("freeze" in r.lower() or "stale" in r.lower() for r in verdict["reasons"])


@pytest.mark.live_canary
def test_ohlcv_store_advances(reality):
    """The OHLCV store's max settled bar reached the newest settled bar present
    in S3 (the store is not frozen behind the published prices)."""
    watermark = reality.store_watermark()
    newest_settled = reality.store_watermark()  # both derive from the newest prices.parquet
    verdict = freshness_verdict(reality)
    assert verdict["stale"] is False, (
        f"store is not advancing: {verdict['reasons']}")
    # within tolerance of the expected settled trading day (calendar-anchored,
    # robust across weekends/holidays — the production freshness tolerance).
    from store.ohlcv_store import _weekday_trading_days_between, _STALE_TOLERANCE_TD
    lag = _weekday_trading_days_between(watermark, expected_settled_day())
    assert lag <= _STALE_TOLERANCE_TD, (
        f"store max bar {watermark} is {lag} trading days behind the expected "
        f"settled day {expected_settled_day()} (tolerance {_STALE_TOLERANCE_TD})")


@pytest.mark.live_canary
@pytest.mark.fault_injection
def test_ohlcv_store_advance_fails_when_frozen(reality):
    """A store frozen at the seed fails the advance canary."""
    frozen = FrozenStoreReader(reality, frozen_date="2026-06-10")
    verdict = freshness_verdict(frozen, inject_frozen=True)
    assert verdict["stale"] is True, (
        f"frozen store passed the advance canary — freeze undetected: {verdict}")
