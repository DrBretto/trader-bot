"""Single-book invariant (PKT-TB-001 regression lock) — clean-core edition.

There is exactly ONE displayed book: the canon line, now the STORED equity
ledger (lines/ledger.py), surfaced as dashboard.json
metrics.total_value / equity_curve. The dead paper-broker account survives only
as an internal intent-sizing simulation (``sim_book_value`` + role markers) whose
value must NEVER be published under a live-book name or become the displayed line.

These tests run the REAL publish paths against a capture stub backed by a seeded
fake ledger, and fail if any freshly produced daily artifact carries a top-level
portfolio-value-shaped field, or if the displayed total ever comes from the dead
book instead of the ledger.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import pytest

from chassis.steps import paper_trader, publish_artifacts
from lines.ledger import CACHE_KEY, MANIFEST_KEY


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
RUN_DATE = "2026-06-09"

DEAD_BOOK_STATE = {
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


class _FakeRawS3:
    """Minimal raw-boto3 stand-in serving a seeded ledger (cache + manifest)."""

    def __init__(self, seeded: bool):
        self.store: Dict[str, bytes] = {}
        if seeded:
            row = {
                "date": RUN_DATE, "value": CANON_TOTAL, "benchmark": 104210.11,
                "comparison": None, "segment": "new_brain", "model_id": "FREEZE_ORB1@test",
            }
            self.store[CACHE_KEY] = (json.dumps(row) + "\n").encode()
            self.store[MANIFEST_KEY] = json.dumps({
                "schema": "equity_manifest.v1",
                "entries": [{"date": RUN_DATE, "content_sha": "deadbeef", "key": "k"}],
                "frontier": {"date": RUN_DATE, "content_sha": "deadbeef"},
            }).encode()

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise Exception("NoSuchKey")
        body = self.store[Key]

        class _B:
            def read(self_inner):
                return body
        return {"Body": _B()}

    def put_object(self, **kw):
        return {}

    def list_objects_v2(self, **kw):
        return {"Contents": [], "CommonPrefixes": [], "IsTruncated": False}


class CaptureS3:
    """Capture stub recording every JSON/JSONL/parquet write."""

    def __init__(self, bucket: str = "test-bucket", seeded_ledger: bool = True):
        self.bucket = bucket
        self.s3 = _FakeRawS3(seeded_ledger)  # raw client backing the ledger reads
        self.json_writes: Dict[str, Any] = {}
        self.jsonl_writes: Dict[str, List[Dict[str, Any]]] = {}
        self.parquet_writes: Dict[str, pd.DataFrame] = {}

    def write_json(self, obj: Any, key: str) -> bool:
        self.json_writes[key] = obj
        return True

    def read_json(self, key: str):
        return self.json_writes.get(key)

    def read_json_strict(self, key: str):
        return self.read_json(key)

    def append_jsonl(self, obj: Dict[str, Any], key: str) -> bool:
        self.jsonl_writes.setdefault(key, []).append(obj)
        return True

    def read_jsonl(self, key: str):
        return self.jsonl_writes.get(key, [])

    def read_jsonl_strict(self, key: str):
        return self.read_jsonl(key)

    def write_parquet(self, df: pd.DataFrame, key: str) -> bool:
        self.parquet_writes[key] = df
        return True

    def read_parquet(self, key: str) -> pd.DataFrame:
        return self.parquet_writes.get(key, pd.DataFrame())

    def list_daily_dates(self, max_days: int = 365):
        return []


def _run_night(monkeypatch, s3: CaptureS3) -> Dict[str, Any]:
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    return publish_artifacts.run(
        bucket="test-bucket",
        run_date=RUN_DATE,
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


def _run_morning(monkeypatch, s3: CaptureS3) -> Dict[str, Any]:
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    return publish_artifacts.publish_morning_artifacts(
        bucket="test-bucket",
        run_date=RUN_DATE,
        portfolio_state=dict(DEAD_BOOK_STATE),
        trades=[],
        morning_execution={"run_date": RUN_DATE, "trades_executed": 0},
        night_inference={"regime": {"label": "risk_on_trend", "probs": {}}},
        night_decisions={},
        night_weather={},
        expert_signals=MINIMAL_EXPERT_SIGNALS,
        morning_prices=pd.DataFrame(),
    )


def _assert_no_second_book(s3: CaptureS3):
    for key, obj in s3.json_writes.items():
        if not (key.startswith("daily/") and key.endswith(".json")):
            continue
        if not isinstance(obj, dict):
            continue
        leaked = PORTFOLIO_VALUE_SHAPED & set(obj.keys())
        assert not leaked, (
            f"{key} carries portfolio-value-shaped top-level field(s) {leaked}"
        )
    for key in ("dashboard/timeseries.json", "dashboard/data/timeseries.json"):
        rows = s3.json_writes.get(key) or []
        for row in rows:
            assert "portfolio_value" not in row, f"{key} row {row.get('date')} carries portfolio_value"
    for key, df in s3.parquet_writes.items():
        if key.endswith("signals.parquet") or key.endswith("timeseries.parquet"):
            assert "portfolio_value" not in df.columns, f"{key} carries a portfolio_value column"


class TestNightPublish:
    def test_no_second_book_in_daily_artifacts(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_night(monkeypatch, s3)
        assert result["success"] is True
        _assert_no_second_book(s3)

    def test_displayed_total_is_the_ledger_not_the_dead_book(self, monkeypatch):
        s3 = CaptureS3()
        _run_night(monkeypatch, s3)
        dash = s3.json_writes["dashboard/dashboard.json"]
        assert dash["metrics"]["total_value"] == pytest.approx(CANON_TOTAL)
        assert dash["metrics"]["total_value"] != pytest.approx(DEAD_BOOK_STATE["portfolio_value"])
        assert dash["metrics"]["canon_source"] == "ledger"

    def test_latest_json_is_pointer_only(self, monkeypatch):
        s3 = CaptureS3()
        _run_night(monkeypatch, s3)
        latest = s3.json_writes["daily/latest.json"]
        assert "portfolio_value" not in latest
        assert "positions_count" not in latest
        assert latest["date"] == RUN_DATE

    def test_state_file_is_marked_internal_sim(self, monkeypatch):
        s3 = CaptureS3()
        _run_night(monkeypatch, s3)
        state = s3.json_writes[f"daily/{RUN_DATE}/portfolio_state.json"]
        assert "portfolio_value" not in state
        assert state["sim_book_value"] == pytest.approx(95565.86)
        assert state["book_role"] == paper_trader.SIM_BOOK_ROLE
        assert "NOT a portfolio" in state["book_note"]

    def test_canon_value_returned_for_email(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_night(monkeypatch, s3)
        assert result["canon_total_value"] == pytest.approx(CANON_TOTAL)

    def test_empty_ledger_holds_and_yields_no_canon_value(self, monkeypatch):
        """If the ledger is unreadable the parity gate HOLDS; the email says
        'unavailable' — the dead book is NEVER substituted as the line."""
        s3 = CaptureS3(seeded_ledger=False)
        result = _run_night(monkeypatch, s3)
        assert result["canon_total_value"] is None
        assert "daily/latest.json" not in s3.json_writes
        _assert_no_second_book(s3)


class TestMorningPublish:
    def test_no_second_book_in_daily_artifacts(self, monkeypatch):
        s3 = CaptureS3()
        result = _run_morning(monkeypatch, s3)
        assert result["success"] is True
        _assert_no_second_book(s3)
        assert result["canon_total_value"] == pytest.approx(CANON_TOTAL)

    def test_legacy_latest_fields_are_scrubbed(self, monkeypatch):
        s3 = CaptureS3()
        s3.json_writes["daily/latest.json"] = {
            "date": "2026-06-08",
            "intents_date": RUN_DATE,
            "portfolio_value": 95565.86,
            "positions_count": 5,
        }
        _run_morning(monkeypatch, s3)
        latest = s3.json_writes["daily/latest.json"]
        assert "portfolio_value" not in latest
        assert "positions_count" not in latest
        _assert_no_second_book(s3)

    def test_state_file_is_marked_internal_sim(self, monkeypatch):
        s3 = CaptureS3()
        _run_morning(monkeypatch, s3)
        state = s3.json_writes[f"daily/{RUN_DATE}/portfolio_state.json"]
        assert "portfolio_value" not in state
        assert state["sim_book_value"] == pytest.approx(95565.86)
        assert state["book_role"] == paper_trader.SIM_BOOK_ROLE


class TestStateBoundaryRoundTrip:
    def test_to_published_state_renames_and_marks(self):
        published = paper_trader.to_published_state(dict(DEAD_BOOK_STATE))
        assert "portfolio_value" not in published
        assert published["sim_book_value"] == pytest.approx(95565.86)
        assert published["book_role"] == paper_trader.SIM_BOOK_ROLE
        assert "broker_total_value" not in published
        assert "broker_reconciled" not in published
        assert published["cash"] == pytest.approx(1200.50)
