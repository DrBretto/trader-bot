"""Forward reliability locks for checkpoints, settled dates, and page freshness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

CORE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CORE_ROOT.parent
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CORE_ROOT / "tests"))

from _reality import LiveS3Reader, CANON_CACHE_KEY  # noqa: E402
from app.night import _merge_night_pointer, _persist_forecast_record  # noqa: E402
from app.morning import (  # noqa: E402
    _build_checkpoint,
    _repair_completed_pointer,
    _result_from_checkpoint,
)
from chassis.steps.midday_checker import (  # noqa: E402
    CHECKPOINT_SCHEMA as MIDDAY_CHECKPOINT_SCHEMA,
    _persist_prepared_checkpoint,
)
from chassis.steps.paper_trader import to_published_state  # noqa: E402
from chassis.steps.publish_artifacts import _trade_execution_fingerprint  # noqa: E402
from monitors import watchdog as W  # noqa: E402
from lines.ledger import CACHE_KEY as PROMOTED_LEDGER_CACHE_KEY  # noqa: E402

pytestmark = pytest.mark.unit

DATE = "2026-07-14"
LEDGER_ROW = {
    "date": DATE,
    "value": 114983.92,
    "benchmark": 108605.49,
    "comparison": 114310.90,
}


class FakeHealthS3:
    def __init__(self, snapshot_date=DATE, phase="morning", shadow_a=True, receipt=True):
        self.docs = {
            W.LATEST_KEY: {
                "date": DATE,
                "intents_date": DATE,
                "morning_executed": True,
            },
            W.PUBLISHED_DASHBOARD_KEY: {
                "snapshot": {"date": snapshot_date, "phase": phase},
                "equity_curve": [dict(LEDGER_ROW)],
            },
            W.SHADOW_KEY: {
                "as_of": "2026-07-15T04:34:00+00:00",
                "shadow_A": [[DATE, LEDGER_ROW["comparison"]]] if shadow_a else [],
                "shadow_F": [[DATE, 999999.0]],
            },
        }
        if receipt:
            self.docs[f"daily/{DATE}/morning_execution.json"] = {"run_date": DATE}
        for date, symbols in (
            ("2026-07-10", ["ARKK", "GLD", "QQQ"]),
            ("2026-07-13", ["ARKK", "IWM", "SMH"]),
            (DATE, ["GLD", "SMH", "XBI"]),
        ):
            self.docs[f"daily/{date}/trade_intents.json"] = {
                "expert_metrics": {"held_symbols": symbols}
            }

    def read_json(self, key):
        return self.docs.get(key)

    def read_jsonl(self, key):
        return [dict(LEDGER_ROW)] if key == W.LEDGER_CACHE_KEY else []

    def list_daily_dates(self, max_days=365):
        return ["2026-07-10", "2026-07-13", DATE]


class FakeWriteS3:
    def __init__(self):
        self.docs = {}

    def write_json(self, payload, key):
        self.docs[key] = payload
        return True

    def read_json_strict(self, key):
        return self.docs.get(key)


def test_morning_checkpoint_round_trip_preserves_exact_execution():
    result = {
        "portfolio_state": {
            "portfolio_value": 101234.5,
            "cash": 50000.0,
            "holdings": [],
        },
        "trades": [
            {
                "symbol": "SPY",
                "action": "BUY",
                "shares": 10,
                "market_price": 620.0,
                "price": 620.12,
                "reason": "BUY_SIGNAL",
            }
        ],
        "morning_prices": pd.DataFrame([{"symbol": "SPY", "price": 620.0}]),
        "validation_log": ["ok"],
        "intents_found": True,
    }

    first = _build_checkpoint(DATE, "2026-07-13", result)
    second = _build_checkpoint(DATE, "2026-07-13", result)
    restored = _result_from_checkpoint(first)

    assert first["trades"][0]["execution_id"] == second["trades"][0]["execution_id"]
    assert restored["portfolio_state"]["portfolio_value"] == pytest.approx(101234.5)
    assert restored["trades"] == first["trades"]
    assert restored["morning_prices"].iloc[0]["symbol"] == "SPY"


def test_completed_morning_checkpoint_repairs_pointer_without_execution():
    store = FakeWriteS3()
    store.docs["daily/latest.json"] = {
        "date": DATE,
        "intents_date": DATE,
        "phase": "night",
        "morning_executed": True,
        "snapshot_id": "2026-07-16:morning:original",
    }
    checkpoint = {
        "status": "completed",
        "run_date": "2026-07-16",
        "intents_date": DATE,
        "completed_at": "2026-07-16T14:00:00",
        "trades": [{"execution_id": "already-executed"}],
    }

    repaired = _repair_completed_pointer(store, checkpoint)

    latest = store.docs["daily/latest.json"]
    assert repaired is True
    assert latest["date"] == "2026-07-16"
    assert latest["phase"] == "morning"
    assert latest["morning_executed"] is True
    assert latest["trades_count"] == 1
    assert latest["snapshot_id"] == "2026-07-16:morning:original"


def test_legacy_trade_and_checkpoint_trade_share_a_dedupe_fingerprint():
    legacy = {
        "symbol": "SPY",
        "action": "BUY",
        "shares": 10,
        "market_price": 620.0,
        "price": 620.12,
        "reason": "BUY_SIGNAL",
    }
    checkpointed = {**legacy, "execution_id": "morning:2026-07-14:abc123"}

    assert _trade_execution_fingerprint(legacy) == _trade_execution_fingerprint(
        checkpointed
    )


def test_midday_prepared_checkpoint_resumes_without_reexecution():
    store = FakeWriteS3()
    checkpoint = {
        "schema": MIDDAY_CHECKPOINT_SCHEMA,
        "status": "prepared",
        "run_date": DATE,
        "portfolio_state": to_published_state({
            "cash": 90000.0,
            "holdings": [],
            "portfolio_value": 100100.0,
        }),
        "actions_taken": [{"symbol": "SPY", "action": "BUY"}],
        "check_log": ["computed once"],
        "circuit_breaker_active": False,
        "circuit_breaker_artifact": None,
        "midday_report": {"run_date": DATE, "total_actions": 1},
    }

    result = _persist_prepared_checkpoint(
        store,
        checkpoint,
        f"ops/midday_checkpoints/{DATE}.json",
        checkpoint_replayed=True,
    )

    assert result["checkpoint_replayed"] is True
    assert result["idempotent_replay"] is False
    assert result["portfolio_state"]["portfolio_value"] == 100100.0
    assert store.docs[f"ops/midday_checkpoints/{DATE}.json"]["status"] == "completed"
    assert store.docs[f"daily/{DATE}/midday_check_report.json"]["total_actions"] == 1


def test_health_requires_current_public_snapshot_and_morning_receipt():
    captured = []
    status = W.run_daily_health_check(
        FakeHealthS3(),
        today=DATE,
        require_morning=True,
        alert=lambda subject, body: captured.append((subject, body)),
    )

    assert status["ok"] is True
    assert status["dashboard"]["stale"] is False
    assert status["lines"]["TILT canon (solid blue)"]["at"] == DATE
    assert status["lines"]["two-stage comparison (dotted yellow)"]["at"] == DATE
    assert captured


def test_current_lines_cannot_hide_a_stale_page_snapshot():
    status = W.run_daily_health_check(
        FakeHealthS3(snapshot_date="2026-07-10"),
        today=DATE,
        alert=lambda *_: None,
    )

    assert status["ok"] is False
    assert status["dashboard"]["stale"] is True
    assert "snapshot" in status["dashboard"]["reason"]


def test_missing_yellow_line_cannot_fall_through_to_an_attribution_series():
    status = W.check_challenger_line(FakeHealthS3(shadow_a=False), DATE)

    assert status["stale"] is True
    assert status["populated"] is False


def test_post_morning_check_rejects_missing_receipt():
    status = W.run_daily_health_check(
        FakeHealthS3(receipt=False),
        today=DATE,
        require_morning=True,
        alert=lambda *_: None,
    )

    assert status["ok"] is False
    assert "receipt" in status["dashboard"]["reason"]


def test_substrate_monitor_uses_current_trade_intents_as_fallback():
    status = W.check_substrate_fresh(FakeHealthS3())

    assert status["observable"] is True
    assert status["stale"] is False
    assert status["days_checked"] == 3
    assert status["sources"] == [
        "trade_intents.expert_metrics.held_symbols",
        "trade_intents.expert_metrics.held_symbols",
        "trade_intents.expert_metrics.held_symbols",
    ]


def test_substrate_monitor_rejects_insufficient_evidence():
    store = FakeHealthS3()
    store.docs = {}
    store.list_daily_dates = lambda max_days=365: [DATE]

    status = W.run_daily_health_check(store, today=DATE, alert=lambda *_: None)

    assert status["ok"] is False
    assert status["substrate"]["observable"] is False


def test_lambda_runtime_packages_the_post_pipeline_canary_runner():
    requirements = (REPO_ROOT / "requirements-lambda.txt").read_text().splitlines()
    assert any(line.startswith("pytest==") for line in requirements)


def test_reality_canary_follows_the_promoted_ledger_pointer():
    assert CANON_CACHE_KEY == PROMOTED_LEDGER_CACHE_KEY


def test_forecast_fingerprint_ignores_date_and_timestamp_but_not_mu():
    docs = {
        "daily/2026-07-14/inference.json": json.dumps({
            "date": "2026-07-14", "recorded_at": "first",
            "mu": {"SPY": 0.1, "QQQ": -0.2},
        }).encode(),
        "daily/2026-07-15/inference.json": json.dumps({
            "date": "2026-07-15", "recorded_at": "second",
            "mu": {"QQQ": -0.2, "SPY": 0.1},
        }).encode(),
    }
    reader = LiveS3Reader(client=object())
    reader.get_bytes = lambda key: docs[key]

    prior, prior_vec = reader.inference_fingerprint("2026-07-14")
    current, current_vec = reader.inference_fingerprint("2026-07-15")
    assert current == prior
    assert current_vec == prior_vec

    docs["daily/2026-07-15/inference.json"] = json.dumps({
        "date": "2026-07-15", "recorded_at": "third",
        "mu": {"QQQ": -0.2, "SPY": 0.11},
    }).encode()
    changed, _ = reader.inference_fingerprint("2026-07-15")
    assert changed != prior


def test_night_persists_the_exact_successful_forecast_record():
    store = FakeWriteS3()
    record = {"date": DATE, "recorded_at": "now", "mu": {"SPY": 0.1}}

    persisted = _persist_forecast_record(store, DATE, record)

    assert persisted == record
    assert store.docs[f"daily/{DATE}/inference.json"] == record


def test_night_retry_cannot_regress_a_later_morning_pointer():
    morning = {
        "date": "2026-07-16",
        "intents_date": DATE,
        "phase": "morning",
        "timestamp": "morning-time",
        "snapshot_id": "2026-07-16:morning",
        "morning_executed": True,
    }

    merged = _merge_night_pointer(
        morning,
        DATE,
        {"regime": "risk_off_trend", "actions": [{}, {}]},
        "night-retry-time",
    )

    assert merged["date"] == "2026-07-16"
    assert merged["phase"] == "morning"
    assert merged["timestamp"] == "morning-time"
    assert merged["snapshot_id"] == "2026-07-16:morning"
    assert merged["morning_executed"] is True
    assert merged["intents_date"] == DATE
    assert merged["actions_count"] == 2


def test_night_pointer_advances_normally_before_morning():
    merged = _merge_night_pointer(
        {"date": "2026-07-14", "intents_date": "2026-07-14"},
        DATE,
        {"regime": "risk_off_trend", "actions": [{}]},
        "night-time",
    )

    assert merged == {
        "date": DATE,
        "intents_date": DATE,
        "regime": "risk_off_trend",
        "actions_count": 1,
        "phase": "night",
        "timestamp": "night-time",
    }


def test_cboe_fetch_uses_visible_fallback_and_browser_headers(
    monkeypatch, tmp_path
):
    from feeds import prices as price_feeds
    from forecast import data_layer as data_layer

    seen = {}

    class Forbidden:
        status_code = 403
        text = ""

    def reject(url, **kwargs):
        seen.update(kwargs)
        return Forbidden()

    fallback = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-15"), "symbol": "^VIX", "close": 16.7}
    ])
    monkeypatch.setattr(data_layer, "CBOE_INDICES", ["VIX"])
    monkeypatch.setattr(data_layer.requests, "get", reject)
    monkeypatch.setattr(price_feeds, "fetch_vol_index", lambda *a, **k: fallback)

    report = data_layer.fetch_cboe(tmp_path, force=True)

    assert seen["headers"] == data_layer.CBOE_HEADERS
    assert report["VIX"]["status"] == "OK"
    assert report["VIX"]["source"] == "yahoo_yfinance_fallback"
    assert report["VIX"]["last"] == "2026-07-15"
