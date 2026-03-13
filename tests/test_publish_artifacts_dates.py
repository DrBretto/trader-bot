"""Tests for portfolio_state date stamping during artifact publication."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.steps import publish_artifacts


class FakeS3:
    """Minimal capture stub for publish_artifacts module."""

    instances: List["FakeS3"] = []

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.json_writes: Dict[str, Any] = {}
        self.jsonl_writes: Dict[str, List[Dict[str, Any]]] = {}
        FakeS3.instances.append(self)

    def write_json(self, obj: Any, key: str) -> bool:
        self.json_writes[key] = obj
        return True

    def read_json(self, key: str):
        if key == "daily/latest.json":
            return {}
        return self.json_writes.get(key)

    def append_jsonl(self, obj: Dict[str, Any], key: str) -> bool:
        self.jsonl_writes.setdefault(key, []).append(obj)
        return True

    def read_jsonl(self, key: str):
        return self.jsonl_writes.get(key, [])

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
        lambda *args, **kwargs: {"metrics": {}, "equity_curve": [], "drawdowns": []},
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
        expert_signals=None,
    )

    assert result["success"] is True
    s3 = FakeS3.instances[-1]
    written = s3.json_writes[f"daily/{run_date}/portfolio_state.json"]
    assert written["date"] == run_date


def test_morning_publish_stamps_portfolio_state_date(monkeypatch):
    FakeS3.instances.clear()
    monkeypatch.setattr(publish_artifacts, "S3Client", FakeS3)
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *args, **kwargs: {"metrics": {}, "equity_curve": [], "drawdowns": []},
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
        expert_signals=None,
    )

    assert result["success"] is True
    s3 = FakeS3.instances[-1]
    written = s3.json_writes[f"daily/{run_date}/portfolio_state.json"]
    assert written["date"] == run_date
