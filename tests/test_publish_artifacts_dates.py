"""Tests for portfolio_state date stamping during artifact publication."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

import json as _json

from chassis.steps import publish_artifacts
from lines.ledger import CACHE_KEY, MANIFEST_KEY


MINIMAL_EXPERT_SIGNALS = {
    "macro_credit": {},
    "vol_uncertainty": {},
    "fragility": {},
    "entropy_shift": {},
}

_LEDGER_TERMINAL = {"date": "2026-03-12", "value": 100000.0, "benchmark": 100000.0}
# The stored-ledger line the repointed publish path renders + the parity gate checks.
_LEDGER_LINE = {
    "equity_curve": [dict(_LEDGER_TERMINAL, cumulative_external_cashflow=0.0,
                          optimized_value=100000.0, new_brain_value=100000.0,
                          champion_frozen_value=None, incumbent_value=None)],
    "drawdowns": [{"date": "2026-03-12", "drawdown": 0.0}],
}


class _LedgerRawS3:
    """Raw-boto3 stand-in serving a one-point seeded ledger (cache + manifest)."""

    def __init__(self):
        row = {"date": "2026-03-12", "value": 100000.0, "benchmark": 100000.0,
               "comparison": None, "segment": "new_brain", "model_id": "FREEZE_ORB1@test"}
        self.store = {
            CACHE_KEY: (_json.dumps(row) + "\n").encode(),
            MANIFEST_KEY: _json.dumps({"frontier": {"date": "2026-03-12", "content_sha": "x"},
                                       "entries": []}).encode(),
        }

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


class FakeS3:
    """Minimal capture stub for publish_artifacts module."""

    instances: List["FakeS3"] = []

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = _LedgerRawS3()
        self.json_writes: Dict[str, Any] = {}
        self.jsonl_writes: Dict[str, List[Dict[str, Any]]] = {}
        FakeS3.instances.append(self)

    def write_json(self, obj: Any, key: str) -> bool:
        self.json_writes[key] = obj
        return True

    def read_json(self, key: str):
        if key == "daily/latest.json":
            return self.json_writes.get(key, {})
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
        return True

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.DataFrame()

    def list_daily_dates(self, max_days: int = 365):
        return []


def test_night_publish_stamps_portfolio_state_date(monkeypatch):
    FakeS3.instances.clear()
    monkeypatch.setattr(publish_artifacts, "S3Client", FakeS3)
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *args, **kwargs: {"metrics": {}, **_LEDGER_LINE},
    )

    run_date = "2026-03-12"
    portfolio_state = {"portfolio_value": 100000.0, "cash": 100000.0, "holdings": []}

    result = publish_artifacts.run(
        bucket="test-bucket",
        run_date=run_date,
        prices_df=pd.DataFrame(),
        context_df=pd.DataFrame(),
        features_df=pd.DataFrame(),
        inference_output={"regime": {"label": "risk_off_trend"}},
        llm_risks={},
        decisions={},
        portfolio_state=portfolio_state,
        trades=[],
        weather={},
        validation={},
        expert_signals=MINIMAL_EXPERT_SIGNALS,
    )

    assert result["success"] is True
    s3 = FakeS3.instances[-1]
    written = s3.json_writes[f"daily/{run_date}/portfolio_state.json"]
    assert written["date"] == run_date


def test_night_publish_advances_with_optional_signals_missing(monkeypatch):
    FakeS3.instances.clear()
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *args, **kwargs: {"metrics": {}, **_LEDGER_LINE},
    )

    run_date = "2026-03-12"
    portfolio_state = {"portfolio_value": 100000.0, "cash": 100000.0, "holdings": []}
    s3 = FakeS3("test-bucket")
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    s3.json_writes["daily/latest.json"] = {
        "date": "2026-03-11",
        "snapshot_id": "prev",
    }

    result = publish_artifacts.run(
        bucket="test-bucket",
        run_date=run_date,
        prices_df=pd.DataFrame(),
        context_df=pd.DataFrame(),
        features_df=pd.DataFrame(),
        inference_output={"regime": {"label": "risk_off_trend"}},
        llm_risks={},
        decisions={},
        portfolio_state=portfolio_state,
        trades=[],
        weather={},
        validation={},
        expert_signals=None,
    )

    assert result["success"] is True
    latest = s3.json_writes["daily/latest.json"]
    assert latest["date"] == run_date
    assert "dashboard/dashboard.json" in s3.json_writes


def test_morning_publish_stamps_portfolio_state_date(monkeypatch):
    FakeS3.instances.clear()
    monkeypatch.setattr(publish_artifacts, "S3Client", FakeS3)
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *args, **kwargs: {"metrics": {}, **_LEDGER_LINE},
    )

    run_date = "2026-03-12"
    portfolio_state = {"portfolio_value": 100000.0, "cash": 100000.0, "holdings": []}

    result = publish_artifacts.publish_morning_artifacts(
        bucket="test-bucket",
        run_date=run_date,
        portfolio_state=portfolio_state,
        trades=[],
        morning_execution={},
        night_inference={"regime": {"label": "risk_off_trend"}},
        night_decisions={},
        night_weather={},
        expert_signals=MINIMAL_EXPERT_SIGNALS,
    )

    assert result["success"] is True
    s3 = FakeS3.instances[-1]
    written = s3.json_writes[f"daily/{run_date}/portfolio_state.json"]
    assert written["date"] == run_date


def test_morning_publish_advances_with_optional_signals_missing(monkeypatch):
    FakeS3.instances.clear()
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *args, **kwargs: {"metrics": {}, **_LEDGER_LINE},
    )

    run_date = "2026-03-12"
    portfolio_state = {"portfolio_value": 100000.0, "cash": 100000.0, "holdings": []}
    s3 = FakeS3("test-bucket")
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda bucket: s3)
    s3.json_writes["daily/latest.json"] = {
        "date": "2026-03-11",
        "snapshot_id": "prev",
    }

    result = publish_artifacts.publish_morning_artifacts(
        bucket="test-bucket",
        run_date=run_date,
        portfolio_state=portfolio_state,
        trades=[],
        morning_execution={},
        night_inference={"regime": {"label": "risk_off_trend"}},
        night_decisions={},
        night_weather={},
        expert_signals=None,
    )

    assert result["success"] is True
    latest = s3.json_writes["daily/latest.json"]
    assert latest["date"] == run_date
    assert latest["morning_executed"] is True
    assert "dashboard/dashboard.json" in s3.json_writes
