"""Tests for canonical dashboard metrics and snapshot cohesion."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.publish_artifacts import build_dashboard_data
from src.utils.dashboard_metrics import compute_canonical_dashboard_metrics


class FakeS3:
    """Minimal in-memory S3 stub for dashboard metrics tests."""

    def __init__(
        self,
        states_by_date: Dict[str, Dict[str, Any]],
        trades_by_date: Dict[str, List[Dict[str, Any]]],
    ):
        self.states_by_date = states_by_date
        self.trades_by_date = trades_by_date

    def list_daily_dates(self, max_days: int = 365) -> List[str]:
        return sorted(self.states_by_date.keys())[-max_days:]

    def read_json(self, key: str):
        if key.startswith("daily/") and key.endswith("/portfolio_state.json"):
            date_str = key.split("/")[1]
            return self.states_by_date.get(date_str)
        return None

    def read_jsonl(self, key: str):
        if key.startswith("daily/") and key.endswith("/trades.jsonl"):
            date_str = key.split("/")[1]
            return self.trades_by_date.get(date_str, [])
        return []


class TestCanonicalDashboardMetrics:
    def test_metrics_reconciliation(self):
        states = {
            "2025-12-31": {
                "portfolio_value": 100000,
                "benchmark_value": 100000,
                "cash": 100000,
                "holdings": [],
            },
            "2026-01-02": {
                "portfolio_value": 102000,
                "benchmark_value": 101500,
                "cash": 90000,
                "holdings": [{"symbol": "AAA", "market_value": 12000}],
            },
            "2026-01-03": {
                "portfolio_value": 101000,
                "benchmark_value": 101000,
                "cash": 88000,
                "holdings": [{"symbol": "AAA", "market_value": 13000}],
            },
            "2026-02-01": {
                "portfolio_value": 103020,
                "benchmark_value": 102200,
                "cash": 91000,
                "holdings": [],
            },
        }
        trades = {
            "2026-01-02": [
                {
                    "timestamp": "2026-01-02T09:45:00",
                    "symbol": "AAA",
                    "action": "BUY",
                    "shares": 10,
                    "price": 100.0,
                    "market_price": 100.0,
                }
            ],
            "2026-01-03": [
                {
                    "timestamp": "2026-01-03T09:45:00",
                    "symbol": "BBB",
                    "action": "BUY",
                    "shares": 10,
                    "price": 50.0,
                    "market_price": 50.0,
                }
            ],
            "2026-02-01": [
                {
                    "timestamp": "2026-02-01T09:45:00",
                    "symbol": "AAA",
                    "action": "SELL",
                    "shares": 10,
                    "price": 110.0,
                    "market_price": 110.0,
                },
                {
                    "timestamp": "2026-02-01T09:46:00",
                    "symbol": "BBB",
                    "action": "SELL",
                    "shares": 10,
                    "price": 45.0,
                    "market_price": 45.0,
                },
            ],
        }
        s3 = FakeS3(states, trades)

        canonical = compute_canonical_dashboard_metrics(
            s3=s3,
            portfolio_state=states["2026-02-01"],
            snapshot_date="2026-02-01",
            current_state=states["2026-02-01"],
            max_days=730,
        )

        monthly = canonical["monthly_returns"]
        ytd_from_monthly = 1.0
        for row in monthly:
            if row["year"] == 2026:
                ytd_from_monthly *= 1 + row["return_pct"]
        ytd_from_monthly -= 1

        metrics = canonical["metrics"]
        assert metrics["ytd_return"] == pytest.approx(ytd_from_monthly, abs=1e-10)

        max_dd_from_series = min(point["drawdown"] for point in canonical["drawdowns"])
        assert metrics["max_drawdown"] == pytest.approx(max_dd_from_series, abs=1e-10)

        wins = metrics["wins"]
        losses = metrics["losses"]
        expected_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        assert metrics["win_rate"] == pytest.approx(expected_win_rate, abs=1e-12)
        assert metrics["total_trades"] == wins + losses

    def test_cashflow_exclusion(self):
        states = {
            "2026-01-01": {
                "portfolio_value": 100000,
                "benchmark_value": 100000,
                "cash": 100000,
                "holdings": [],
            },
            "2026-01-02": {
                "portfolio_value": 101000,
                "benchmark_value": 100500,
                "cash": 99000,
                "holdings": [{"symbol": "AAA", "market_value": 2000}],
                "external_cashflow": 0.0,
            },
            "2026-01-03": {
                "portfolio_value": 111000,
                "benchmark_value": 101000,
                "cash": 109000,
                "holdings": [{"symbol": "AAA", "market_value": 2000}],
                "external_cashflow": 10000.0,
            },
        }
        s3 = FakeS3(states, {})

        canonical = compute_canonical_dashboard_metrics(
            s3=s3,
            portfolio_state=states["2026-01-03"],
            snapshot_date="2026-01-03",
            current_state=states["2026-01-03"],
        )
        daily_returns = {
            row["date"]: row["daily_return"]
            for row in canonical["daily_returns"]
            if row["daily_return"] is not None
        }
        assert daily_returns["2026-01-02"] == pytest.approx(0.01, abs=1e-12)
        # Deposit should not appear as a return spike.
        assert daily_returns["2026-01-03"] == pytest.approx(0.0, abs=1e-12)

    def test_snapshot_cohesion(self):
        states = {
            "2026-02-16": {
                "portfolio_value": 100000,
                "benchmark_value": 100000,
                "cash": 90000,
                "holdings": [{"symbol": "SPY", "market_value": 10000}],
            },
            "2026-02-17": {
                "portfolio_value": 101000,
                "benchmark_value": 100800,
                "cash": 89000,
                "holdings": [{"symbol": "SPY", "market_value": 12000}],
                "last_updated": "2026-02-17T14:30:00",
            },
        }
        s3 = FakeS3(states, {})
        snapshot = {
            "id": "2026-02-17:morning:2026-02-17T14:30:00",
            "date": "2026-02-17",
            "phase": "morning",
            "timestamp": "2026-02-17T14:30:00",
        }

        dashboard = build_dashboard_data(
            portfolio_state=states["2026-02-17"],
            inference_output={"regime": {"label": "risk_on_trend", "probs": {}}},
            decisions={"expert_metrics": {"final_regime_label": "risk_on_trend"}},
            weather={"headline": "OK"},
            s3=s3,  # type: ignore[arg-type]
            snapshot_meta=snapshot,
        )

        panel_ids = dashboard["panel_snapshot_ids"]
        assert dashboard["snapshot"]["id"] == snapshot["id"]
        assert dashboard["metrics"]["snapshot_id"] == snapshot["id"]
        assert dashboard["metrics"]["timestamp"] == snapshot["timestamp"]
        assert all(value == snapshot["id"] for value in panel_ids.values())
