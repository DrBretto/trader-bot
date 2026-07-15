"""Forward reliability locks for checkpoints, settled dates, and page freshness."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.morning import _build_checkpoint, _result_from_checkpoint  # noqa: E402
from chassis.steps.midday_checker import (  # noqa: E402
    CHECKPOINT_SCHEMA as MIDDAY_CHECKPOINT_SCHEMA,
    _persist_prepared_checkpoint,
)
from chassis.steps.paper_trader import to_published_state  # noqa: E402
from chassis.steps.publish_artifacts import _trade_execution_fingerprint  # noqa: E402
from monitors import watchdog as W  # noqa: E402

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
