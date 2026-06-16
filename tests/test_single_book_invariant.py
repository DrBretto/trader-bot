"""Single-book invariant (PKT-TB-001 regression lock).

There is exactly ONE portfolio book in this system: the canon line
(optimized-champion replay, published as dashboard.json metrics.total_value /
equity_curve). The dead paper-broker account survives only as an internal
intent-sizing simulation whose state must never be published under a
live-book name.

These tests run the REAL publish paths against a capture stub and fail if
any freshly produced daily artifact carries a top-level portfolio-value-shaped
field that is not derived from the canon line. If a second live-looking book
ever reappears in daily/latest.json, daily/<date>/*.json, or the rolling
timeseries, this file is what breaks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src.steps import publish_artifacts, paper_trader
from src.utils.sns_alerts import format_morning_summary, format_night_summary


# Field names that read as "the portfolio's value". None of these may appear
# at the top level of any freshly produced daily artifact — the only published
# portfolio value is the canon line's, in dashboard.json metrics.total_value.
PORTFOLIO_VALUE_SHAPED = {
    "portfolio_value",
    "account_value",
    "broker_total_value",
    "total_value",
    "portfolio_total",
    "book_value",
    "equity",
    "nav",
}

MINIMAL_EXPERT_SIGNALS = {
    "macro_credit": {},
    "vol_uncertainty": {},
    "fragility": {},
    "entropy_shift": {},
}

CANON_TOTAL = 114256.0

DEAD_BOOK_STATE = {
    # Realistic dead-book shape: the value that kept leaking (~$95k), plus the
    # legacy broker-era keys live pre-purge state files still carry.
    "portfolio_value": 95565.86,
    "broker_total_value": 95565.86,
    "broker_reconciled": True,
    "cash": 1200.50,
    "benchmark_value": 104210.11,
    "holdings": [
        {"symbol": "ARKK", "shares": 321, "entry_price": 76.93, "market_value": 24694.53},
    ],
    "trades_today": [],
}


class CaptureS3:
    """Capture stub recording every JSON/JSONL/parquet write."""

    def __init__(self, bucket: str = "test-bucket"):
        self.bucket = bucket
        self.s3 = None  # raw boto client slot; extender is stubbed in tests
        self.json_writes: Dict[str, Any] = {}
        self.jsonl_writes: Dict[str, List[Dict[str, Any]]] = {}
        self.parquet_writes: Dict[str, pd.DataFrame] = {}

    def write_json(self, obj: Any, key: str) -> bool:
        self.json_writes[key] = obj
        return True

    def read_json(self, key: str):
        return self.json_writes.get(key)

    def append_jsonl(self, obj: Dict[str, Any], key: str) -> bool:
        self.jsonl_writes.setdefault(key, []).append(obj)
        return True

    def read_jsonl(self, key: str):
        return self.jsonl_writes.get(key, [])

    def write_parquet(self, df: pd.DataFrame, key: str) -> bool:
        self.parquet_writes[key] = df
        return True

    def read_parquet(self, key: str) -> pd.DataFrame:
        return self.parquet_writes.get(key, pd.DataFrame())

    def list_daily_dates(self, max_days: int = 365):
        return []


def _stamping_extender(s3_client, dash):
    """Simulate a successful canon extension (what the real extender stamps).

    PKT-TB-012: the canon is now the New Brain (native two-stage engine), not the
    optimized champion."""
    dash.setdefault("metrics", {})
    dash["metrics"]["canon_source"] = "new_brain"
    dash["metrics"]["total_value"] = CANON_TOTAL
    dash["timeline_correction"] = {
        "version": "lambda-new-brain-canon-v1",
        "canon_source": "new_brain",
        "champion_frontier": "2026-06-09",
    }
    return dash


def _non_stamping_extender(s3_client, dash):
    """Simulate a silently failed extension (guard must hold the publish)."""
    return dash


def _run_night(monkeypatch, s3: CaptureS3, extender_stub) -> Dict[str, Any]:
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    monkeypatch.setattr(
        "src.utils.three_line_replay.extender.extend_dashboard", extender_stub
    )
    return publish_artifacts.run(
        bucket="test-bucket",
        run_date="2026-06-09",
        prices_df=pd.DataFrame(),
        context_df=pd.DataFrame(),
        features_df=pd.DataFrame(),
        inference_output={"regime": {"label": "risk_on_trend", "probs": {}}},
        llm_risks={},
        decisions={"actions": [], "expert_metrics": {}},
        portfolio_state=dict(DEAD_BOOK_STATE),
        trades=[],
        weather={"headline": "OK"},
        validation={},
        expert_signals=MINIMAL_EXPERT_SIGNALS,
    )


def _run_morning(monkeypatch, s3: CaptureS3, extender_stub) -> Dict[str, Any]:
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    monkeypatch.setattr(
        "src.utils.three_line_replay.extender.extend_dashboard", extender_stub
    )
    return publish_artifacts.publish_morning_artifacts(
        bucket="test-bucket",
        run_date="2026-06-09",
        portfolio_state=dict(DEAD_BOOK_STATE),
        trades=[],
        morning_execution={"run_date": "2026-06-09", "trades_executed": 0},
        night_inference={"regime": {"label": "risk_on_trend", "probs": {}}},
        night_decisions={},
        night_weather={},
        expert_signals=MINIMAL_EXPERT_SIGNALS,
        morning_prices=pd.DataFrame(),
    )


def _assert_no_second_book(s3: CaptureS3):
    """Core invariant: scan every freshly produced daily artifact."""
    for key, obj in s3.json_writes.items():
        if not (key.startswith("daily/") and key.endswith(".json")):
            continue
        if not isinstance(obj, dict):
            continue
        leaked = PORTFOLIO_VALUE_SHAPED & set(obj.keys())
        assert not leaked, (
            f"{key} carries portfolio-value-shaped top-level field(s) {leaked} "
            f"— a second live-looking book reappeared (PKT-TB-001)"
        )

    # Rolling timeseries rows (published fresh daily) must not carry the
    # dead book's value either.
    for key in ("dashboard/timeseries.json", "dashboard/data/timeseries.json"):
        rows = s3.json_writes.get(key) or []
        for row in rows:
            assert "portfolio_value" not in row, (
                f"{key} row {row.get('date')} carries portfolio_value"
            )
    for key, df in s3.parquet_writes.items():
        if key.endswith("signals.parquet") or key.endswith("timeseries.parquet"):
            assert "portfolio_value" not in df.columns, (
                f"{key} carries a portfolio_value column"
            )


class TestNightPublish:
    def test_no_second_book_in_daily_artifacts(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_night(monkeypatch, s3, _stamping_extender)
        assert result["success"] is True
        _assert_no_second_book(s3)

    def test_latest_json_is_pointer_only(self, monkeypatch):
        s3 = CaptureS3()
        _run_night(monkeypatch, s3, _stamping_extender)
        latest = s3.json_writes["daily/latest.json"]
        assert "portfolio_value" not in latest
        assert "positions_count" not in latest
        assert latest["date"] == "2026-06-09"
        assert latest["intents_date"] == "2026-06-09"

    def test_state_file_is_marked_internal_sim(self, monkeypatch):
        s3 = CaptureS3()
        _run_night(monkeypatch, s3, _stamping_extender)
        state = s3.json_writes["daily/2026-06-09/portfolio_state.json"]
        assert "portfolio_value" not in state
        assert state["sim_book_value"] == pytest.approx(95565.86)
        assert state["book_role"] == paper_trader.SIM_BOOK_ROLE
        assert "NOT a portfolio" in state["book_note"]

    def test_canon_value_returned_for_email(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_night(monkeypatch, s3, _stamping_extender)
        assert result["canon_total_value"] == pytest.approx(CANON_TOTAL)

    def test_held_dashboard_yields_no_canon_value(self, monkeypatch):
        """When the advance guard holds, no value is available — the email
        must say 'unavailable', never substitute the dead book."""
        s3 = CaptureS3()
        result = _run_night(monkeypatch, s3, _non_stamping_extender)
        assert result["canon_total_value"] is None
        assert "daily/latest.json" not in s3.json_writes
        _assert_no_second_book(s3)


class TestMorningPublish:
    def test_no_second_book_in_daily_artifacts(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_morning(monkeypatch, s3, _stamping_extender)
        assert result["success"] is True
        _assert_no_second_book(s3)
        assert result["canon_total_value"] == pytest.approx(CANON_TOTAL)

    def test_legacy_latest_fields_are_scrubbed(self, monkeypatch):
        """A pre-purge latest.json still carries the dead value; the morning
        merge-update must scrub it rather than preserve it forever."""
        s3 = CaptureS3()
        s3.json_writes["daily/latest.json"] = {
            "date": "2026-06-08",
            "intents_date": "2026-06-09",
            "portfolio_value": 95565.86,
            "positions_count": 5,
        }
        _run_morning(monkeypatch, s3, _stamping_extender)
        latest = s3.json_writes["daily/latest.json"]
        assert "portfolio_value" not in latest
        assert "positions_count" not in latest
        assert latest["intents_date"] == "2026-06-09"  # pointer preserved
        _assert_no_second_book(s3)

    def test_state_file_is_marked_internal_sim(self, monkeypatch):
        s3 = CaptureS3()
        _run_morning(monkeypatch, s3, _stamping_extender)
        state = s3.json_writes["daily/2026-06-09/portfolio_state.json"]
        assert "portfolio_value" not in state
        assert state["sim_book_value"] == pytest.approx(95565.86)
        assert state["book_role"] == paper_trader.SIM_BOOK_ROLE


class TestStateBoundaryRoundTrip:
    """The rename is a write-boundary transform: in-memory code keeps
    `portfolio_value`, published files carry `sim_book_value` + role markers,
    and the loader restores the internal shape (covers the midday writer,
    which routes through the same to_published_state)."""

    def test_to_published_state_renames_and_marks(self):
        published = paper_trader.to_published_state(dict(DEAD_BOOK_STATE))
        assert "portfolio_value" not in published
        assert published["sim_book_value"] == pytest.approx(95565.86)
        assert published["book_role"] == paper_trader.SIM_BOOK_ROLE
        # Legacy broker-era keys carried by pre-purge states are stripped.
        assert "broker_total_value" not in published
        assert "broker_reconciled" not in published
        # Untouched internals survive.
        assert published["cash"] == pytest.approx(1200.50)

    def test_restore_inverts_publish(self):
        published = paper_trader.to_published_state(dict(DEAD_BOOK_STATE))
        restored = paper_trader._restore_internal_keys(published)
        assert restored["portfolio_value"] == pytest.approx(95565.86)
        assert "sim_book_value" not in restored
        assert "book_role" not in restored
        assert "book_note" not in restored

    def test_restore_passes_through_historical_shape(self):
        historical = {"portfolio_value": 103025.53, "cash": 53000.0, "holdings": []}
        restored = paper_trader._restore_internal_keys(dict(historical))
        assert restored == historical


class TestEmailFidelity:
    """Required work 4: emails report the canon book — and say so."""

    def test_morning_email_reports_canon_book(self):
        body = format_morning_summary("2026-06-09", CANON_TOTAL, [], [], 12.0)
        assert "Portfolio (canon line): $114,256.00" in body
        assert "95,565" not in body

    def test_morning_email_honest_when_canon_unavailable(self):
        body = format_morning_summary("2026-06-09", None, [], [], 12.0)
        assert "unavailable" in body
        assert "$" not in body.split("\n")[3]  # no substituted value

    def test_night_email_reports_canon_book(self):
        body = format_night_summary(
            "2026-06-09", "risk_on_trend", CANON_TOTAL, [], "OK", 60.0
        )
        assert "Portfolio (canon line): $114,256.00" in body
