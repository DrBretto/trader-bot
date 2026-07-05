"""Tier 1 unit-fixture — the production price parser maps a RECORDED REAL Yahoo
v8 payload to its known real close.

Deterministic (no network), but the input is a DATED REAL RECORDING committed at
``tests/fixtures/yahoo_v8_SPY_recorded.json`` (captured live from
``query1.finance.yahoo.com/v8/finance/chart/SPY``), and the expected close is the
real close from that same recording — NOT a value the test planted. This
exercises the real ``feeds.prices.fetch_yahoo_v8_daily`` parse path (the crumb/
cookie handshake is stubbed; only the network is removed, never the parser).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FIXTURES  # noqa: E402

from feeds import prices  # noqa: E402


class _FakeResp:
    def __init__(self, blob: bytes):
        self._blob = blob

    def read(self):
        return self._blob

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeOpener:
    def __init__(self, blob: bytes):
        self._blob = blob

    def open(self, req, timeout=None):
        return _FakeResp(self._blob)


@pytest.mark.unit
def test_price_parser_on_recorded_fixture(monkeypatch):
    rec = json.loads((FIXTURES / "yahoo_v8_SPY_recorded.json").read_text())
    blob = json.dumps(rec["payload"]).encode("utf-8")
    known = rec["known_real_close"]

    # Remove ONLY the network: serve the recorded payload; keep the real parser.
    monkeypatch.setattr(prices, "_ensure_yahoo_session",
                        lambda timeout, force=False: (_FakeOpener(blob), "recorded-crumb"))

    df = prices.fetch_yahoo_v8_daily("SPY", lookback_days=3650)
    assert not df.empty, "parser returned no rows from a real recorded payload"

    import pandas as pd
    df = df.assign(_d=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    row = df[df["_d"] == known["date"]]
    assert not row.empty, f"recorded bar {known['date']} missing from parsed output"
    parsed = float(row["close"].iloc[0])
    assert abs(parsed - known["close"]) < 1e-6, (
        f"parsed close {parsed} != known real close {known['close']} for {known['date']}")
