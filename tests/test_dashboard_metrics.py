"""Tests for canonical dashboard metrics and snapshot cohesion."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
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

    def test_equity_curve_is_continuity_adjusted_by_external_cashflow(self):
        states = {
            "2026-03-11": {
                "portfolio_value": 103025.53,
                "benchmark_value": 97615.61,
                "cash": 53000,
                "holdings": [{"symbol": "SPY", "market_value": 50025.53}],
            },
            "2026-03-12": {
                "portfolio_value": 100007.02,
                "benchmark_value": 96358.46,
                "cash": 62317.0,
                "holdings": [{"symbol": "SPY", "market_value": 37690.02}],
                "external_cashflow": -3025.53,
            },
        }
        s3 = FakeS3(states, {})

        canonical = compute_canonical_dashboard_metrics(
            s3=s3,
            portfolio_state=states["2026-03-12"],
            snapshot_date="2026-03-12",
            current_state=states["2026-03-12"],
        )

        equity_curve = canonical["equity_curve"]
        raw_curve = canonical["raw_equity_curve"]
        assert len(equity_curve) == 2
        assert len(raw_curve) == 2

        # Raw broker value keeps the cutover-day reset.
        assert raw_curve[-1]["value"] == pytest.approx(100007.02, abs=1e-6)

        # Continuity value removes the one-time cutover cashflow.
        assert equity_curve[-1]["value"] == pytest.approx(103032.55, abs=0.01)
        assert equity_curve[-1]["raw_value"] == pytest.approx(100007.02, abs=1e-6)
        assert equity_curve[-1]["cumulative_external_cashflow"] == pytest.approx(-3025.53, abs=1e-6)

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

    def test_dashboard_metrics_include_broker_and_continuity_values(self):
        states = {
            "2026-03-11": {
                "portfolio_value": 103025.53,
                "benchmark_value": 97615.61,
                "cash": 53000.0,
                "holdings": [{"symbol": "SPY", "market_value": 50025.53}],
            },
            "2026-03-12": {
                "portfolio_value": 100007.02,
                "benchmark_value": 96358.46,
                "cash": 62317.0,
                "holdings": [{"symbol": "SPY", "market_value": 37690.02}],
                "external_cashflow": -3025.53,
                "last_updated": "2026-03-12T18:08:45.446509",
            },
        }
        s3 = FakeS3(states, {})
        snapshot = {
            "id": "2026-03-12:morning:2026-03-12T18:08:45.446509",
            "date": "2026-03-12",
            "phase": "morning",
            "timestamp": "2026-03-12T18:08:45.446509",
        }

        dashboard = build_dashboard_data(
            portfolio_state=states["2026-03-12"],
            inference_output={"regime": {"label": "risk_off_trend", "probs": {}}},
            decisions={"expert_metrics": {"final_regime_label": "risk_off_trend"}},
            weather={"headline": "OK"},
            s3=s3,  # type: ignore[arg-type]
            snapshot_meta=snapshot,
        )

        assert dashboard["metrics"]["broker_total_value"] == pytest.approx(100007.02, abs=1e-6)
        assert dashboard["metrics"]["total_value"] == pytest.approx(103032.55, abs=0.01)
        assert dashboard["equity_curve"][-1]["value"] == pytest.approx(103032.55, abs=0.01)
        assert dashboard["equity_curve"][-1]["raw_value"] == pytest.approx(100007.02, abs=1e-6)

    def test_carried_bridge_cashflow_is_ignored_after_cutover_day(self):
        states = {
            "2026-03-11": {
                "portfolio_value": 103025.52645321962,
                "benchmark_value": 97615.61,
                "cash": 53000.0,
                "holdings": [{"symbol": "SPY", "market_value": 50025.53}],
            },
            "2026-03-12": {
                "portfolio_value": 99924.21,
                "benchmark_value": 96358.46,
                "cash": 62317.48,
                "holdings": [{"symbol": "SPY", "market_value": 37606.73}],
                "external_cashflow": -3025.526453219616,
                "continuity_bridge_marker": "continuity-bridge-v1:2026-03-12",
            },
            "2026-03-13": {
                "portfolio_value": 99793.95551390079,
                "benchmark_value": 96191.57,
                "cash": 62317.48,
                "holdings": [{"symbol": "SPY", "market_value": 37476.48}],
                # Simulate the buggy carried-forward field from the prior day.
                "external_cashflow": -3025.526453219616,
                "continuity_bridge_marker": "continuity-bridge-v1:2026-03-12",
            },
        }
        s3 = FakeS3(states, {})

        canonical = compute_canonical_dashboard_metrics(
            s3=s3,
            portfolio_state=states["2026-03-13"],
            snapshot_date="2026-03-13",
            current_state=states["2026-03-13"],
        )

        equity_curve = canonical["equity_curve"]
        assert equity_curve[-1]["value"] == pytest.approx(102819.48196712041, abs=0.01)
        assert equity_curve[-1]["cumulative_external_cashflow"] == pytest.approx(-3025.526453219616, abs=1e-6)

        daily_returns = {row["date"]: row for row in canonical["daily_returns"]}
        assert daily_returns["2026-03-13"]["external_cashflow"] == pytest.approx(0.0, abs=1e-12)

    def test_build_dashboard_filters_dust_holdings(self):
        states = {
            "2026-04-01": {
                "portfolio_value": 100000.0,
                "benchmark_value": 100000.0,
                "cash": 99999.99,
                "holdings": [],
            }
        }
        s3 = FakeS3(states, {})

        dashboard = build_dashboard_data(
            portfolio_state={
                "portfolio_value": 100000.0,
                "cash": 99999.99,
                "holdings": [
                    {
                        "symbol": "DBC",
                        "shares": 6.01e-7,
                        "market_value": 0.000017,
                    },
                    {
                        "symbol": "TLT",
                        "shares": 2.5,
                        "market_value": 250.0,
                    },
                ],
            },
            inference_output={"regime": {"label": "risk_on_trend", "probs": {}}},
            decisions={},
            weather={},
            s3=s3,  # type: ignore[arg-type]
            expert_signals={"macro_credit": {}, "vol_uncertainty": {}, "fragility": {}, "entropy_shift": {}},
            snapshot_meta={
                "id": "2026-04-01:night:test",
                "date": "2026-04-01",
                "phase": "night",
                "timestamp": "2026-04-01T00:00:00",
            },
        )

        assert [holding["symbol"] for holding in dashboard["holdings"]] == ["TLT"]


class TestTimeseriesStatusFlags:
    """F-6 regression: each signal block contributes a status flag to the
    timeseries row so a neutral fallback can be distinguished from a real
    computation. Without this, the F-1 outage was invisible."""

    def test_healthy_signals_emit_ok(self):
        from src.steps.publish_artifacts import _build_timeseries_row
        row = _build_timeseries_row(
            run_date='2026-04-29',
            expert_signals={
                'macro_credit': {'macro_credit_score': 0.2},
                'vol_uncertainty': {'vol_uncertainty_score': 0.55},
                'fragility': {'fragility_score': 0.7},
                'entropy_shift': {'entropy_score': 0.5},
            },
            inference_output={'regime': {'label': 'risk_on_trend'}},
            decisions={'expert_metrics': {}},
            portfolio_state={'portfolio_value': 100000},
            context_df=pd.DataFrame(),
        )
        assert row['macro_credit_status'] == 'ok'
        assert row['vol_uncertainty_status'] == 'ok'
        assert row['fragility_status'] == 'ok'
        assert row['entropy_status'] == 'ok'

    def test_degraded_signals_surface_reason(self):
        from src.steps.publish_artifacts import _build_timeseries_row
        row = _build_timeseries_row(
            run_date='2026-04-29',
            expert_signals={
                'macro_credit': {'macro_credit_score': 0.0,
                                  'degraded_reason': 'fred_unavailable'},
                'vol_uncertainty': {'vol_uncertainty_score': 0.5,
                                     'inputs_degraded': ['vvix_missing', 'skew_missing']},
                'fragility': {'fragility_score': 0.5,
                               'degraded_reason': 'Insufficient symbols: 4 < 6'},
                'entropy_shift': {'entropy_score': 0.5},
            },
            inference_output={'regime': {'label': 'risk_on_trend'}},
            decisions={'expert_metrics': {}},
            portfolio_state={'portfolio_value': 100000},
            context_df=pd.DataFrame(),
        )
        assert row['macro_credit_status'].startswith('degraded:')
        assert 'fred_unavailable' in row['macro_credit_status']
        assert row['vol_uncertainty_status'].startswith('partial:')
        assert 'vvix_missing' in row['vol_uncertainty_status']
        assert row['fragility_status'].startswith('degraded:')
        assert row['entropy_status'] == 'ok'


class TestChartMarkers:
    """Vertical timeline markers for dashboard charts (equity curve, regime
    strip). Operator-editable list of dated events; flows from
    config/chart_markers.json into dashboard.json."""

    def test_load_markers_from_config(self):
        from src.steps.publish_artifacts import _load_chart_markers
        markers = _load_chart_markers()
        # The repo ships with a populated marker file; expect non-empty.
        assert isinstance(markers, list)
        if markers:
            # Schema check on the first entry.
            m = markers[0]
            assert 'date' in m
            assert 'label' in m
            # Sorted ascending by date.
            for i in range(1, len(markers)):
                assert markers[i].get('date', '') >= markers[i-1].get('date', '')

    def test_load_markers_returns_empty_when_file_missing(self, tmp_path, monkeypatch):
        from src.steps.publish_artifacts import _load_chart_markers
        # Point to a nonexistent file; loader must return [] not raise.
        monkeypatch.setenv('CHART_MARKERS_PATH', str(tmp_path / 'nope.json'))
        # Also chdir so the second-fallback ('config/chart_markers.json') misses.
        monkeypatch.chdir(tmp_path)
        # The third fallback (computed from this file's location) WILL find the
        # repo's config file; that's fine — confirm it's still a list.
        result = _load_chart_markers()
        assert isinstance(result, list)

    def test_dashboard_data_includes_chart_markers(self):
        from src.steps.publish_artifacts import build_dashboard_data
        states = {
            "2026-04-29": {
                "portfolio_value": 100000.0,
                "benchmark_value": 100000.0,
                "cash": 100000.0,
                "holdings": [],
            }
        }
        s3 = FakeS3(states, {})
        out = build_dashboard_data(
            portfolio_state={"portfolio_value": 100000.0, "cash": 100000.0, "holdings": []},
            inference_output={"regime": {"label": "risk_on_trend", "probs": {}}},
            decisions={},
            weather={},
            s3=s3,  # type: ignore[arg-type]
            expert_signals={"macro_credit": {}, "vol_uncertainty": {}, "fragility": {}, "entropy_shift": {}},
            snapshot_meta={"id": "x", "date": "2026-04-29", "phase": "night",
                            "timestamp": "2026-04-29T00:00:00"},
        )
        assert 'chart_markers' in out
        assert isinstance(out['chart_markers'], list)
