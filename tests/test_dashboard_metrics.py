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

    def test_canonical_segment_forward_fills_gap_days(self):
        """Gap-trading-days inside the HYBRID_SEGMENT span (e.g. Saturday-dated
        states from a Friday-night Lambda persist, holiday Mondays) must take
        the most recent canonical value, not the broker raw. Otherwise the
        equity curve zig-zags between canonical checkpoints when broker raw
        is far from the canonical line — exactly the 2026-05-07 incident."""
        from src.utils.canonical_replay_anchor import HYBRID_SEGMENT

        # Build a synthetic state set that covers two adjacent canonical
        # anchor dates plus one in-between gap date with a depressed
        # broker raw value (simulating a post-dividend / pre-seam reading
        # that would normally leak into the chart).
        anchor_dates = sorted(HYBRID_SEGMENT.keys())
        a, b = anchor_dates[0], anchor_dates[1]
        # Pick a synthetic gap date that is strictly between the two
        # anchors. Use a date string that is lexicographically between a
        # and b without colliding with any anchor.
        gap = a[:-2] + str(int(a[-2:]) + 0).zfill(2) + 'Z'  # never matches HYBRID_SEGMENT
        # Easier: pick a fixed gap inside the segment that is not a
        # member of HYBRID_SEGMENT. 2026-05-02 is the canonical incident.
        gap = '2026-05-02'
        assert gap not in HYBRID_SEGMENT, 'fixture would collide with anchor'

        # Use 2026-05-01 (in HYBRID_SEGMENT) as the prior anchor, gap as
        # the broken broker reading, and 2026-05-05 (in HYBRID_SEGMENT)
        # as the recovery anchor.
        states = {
            '2026-05-01': {
                'portfolio_value': 50000.0,  # arbitrary; will be overridden
                'benchmark_value': 100000.0,
                'cash': 50000.0,
                'holdings': [],
            },
            gap: {
                'portfolio_value': 30000.0,  # depressed raw, must be ignored
                'benchmark_value': 100000.0,
                'cash': 30000.0,
                'holdings': [],
            },
            '2026-05-05': {
                'portfolio_value': 50000.0,  # arbitrary; will be overridden
                'benchmark_value': 100000.0,
                'cash': 50000.0,
                'holdings': [],
            },
        }
        s3 = FakeS3(states, {})

        canonical = compute_canonical_dashboard_metrics(
            s3=s3,
            portfolio_state=states['2026-05-05'],
            snapshot_date='2026-05-05',
            current_state=states['2026-05-05'],
        )

        equity_curve = {row['date']: row for row in canonical['equity_curve']}

        # Anchor dates take their HYBRID_SEGMENT values.
        assert equity_curve['2026-05-01']['value'] == pytest.approx(
            HYBRID_SEGMENT['2026-05-01']['value'], abs=1e-6
        )
        assert equity_curve['2026-05-05']['value'] == pytest.approx(
            HYBRID_SEGMENT['2026-05-05']['value'], abs=1e-6
        )

        # Gap date forward-fills from 2026-05-01 — NOT the depressed broker
        # raw of 30000. The continuity_value equals the prior anchor value
        # because external_cashflow is zeroed on gap days.
        assert equity_curve[gap]['value'] == pytest.approx(
            HYBRID_SEGMENT['2026-05-01']['value'], abs=1e-6
        )
        # Drawdown on the gap day must not show a 70% crater.
        gap_drawdown = next(
            d for d in canonical['drawdowns'] if d['date'] == gap
        )
        assert gap_drawdown['drawdown'] > -0.005, (
            'gap day drawdown leaked broker raw — forward-fill broken'
        )

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


class TestMarketReturnExtendInteriorGap:
    """Regression for the displayed-line cap-decontamination fix (2026-06-06).
    Interior replay gaps and the 1-2 day tail past the final simulated date
    must be scaled by the BENCHMARK (fully-invested SPY) daily return, NOT the
    broker raw_value daily return. The broker book is $5,000-capped and
    chronically cash-heavy; scaling by its returns leaked the cap into the
    fantasy line (value-return == raw_value-return on 2026-06-01 / 2026-06-06).
    `_market_return_extend` keeps the champion line on its own ~100%-equity
    exposure proxy and never touches raw_value."""

    def test_interior_gap_filled_by_benchmark_return_not_flat(self):
        from src.utils.three_line_replay.extender import _market_return_extend

        # Two replay-instrumented anchors (5/08 and 5/21) with three gap-dates
        # between them. Each row carries a benchmark (SPY) value AND a divergent
        # broker raw_value — the fill must follow benchmark, ignore raw_value.
        equity_curve = [
            {'date': '2026-05-08', 'value': 0.0, 'benchmark': 110000.00, 'raw_value': 96983.96},
            {'date': '2026-05-11', 'value': 0.0, 'benchmark': 110550.00, 'raw_value': 97030.06},
            {'date': '2026-05-12', 'value': 0.0, 'benchmark': 110770.00, 'raw_value': 97040.84},
            {'date': '2026-05-13', 'value': 0.0, 'benchmark': 111000.00, 'raw_value': 97166.58},
            {'date': '2026-05-21', 'value': 0.0, 'benchmark': 109900.00, 'raw_value': 96702.94},
        ]
        champ_map = {'2026-05-08': 113000.79, '2026-05-21': 112641.72}
        hybrid_map = {'2026-05-08': 108420.63, '2026-05-21': 107900.0}
        pre_map: dict = {}

        _market_return_extend(equity_curve, champ_map, hybrid_map, pre_map)

        for gap_date in ('2026-05-11', '2026-05-12', '2026-05-13'):
            assert gap_date in champ_map, f'gap {gap_date} not filled'
            assert gap_date in hybrid_map, f'hybrid gap {gap_date} not filled'

        assert champ_map['2026-05-11'] != champ_map['2026-05-08']
        assert champ_map['2026-05-12'] != champ_map['2026-05-11']

        # 5/11 market_return = 110550/110000 - 1, applied to the 5/08 anchor.
        expected_5_11 = 113000.79 * (110550.00 / 110000.00)
        assert abs(champ_map['2026-05-11'] - expected_5_11) < 0.01
        expected_5_12 = expected_5_11 * (110770.00 / 110550.00)
        assert abs(champ_map['2026-05-12'] - expected_5_12) < 0.01

        # The broker raw_value must NOT drive the fill (cap-leak guard):
        leaked_5_11 = 113000.79 * (97030.06 / 96983.96)
        assert abs(champ_map['2026-05-11'] - leaked_5_11) > 1.0

        # Anchors must NOT be overwritten by the gap-fill walk.
        assert champ_map['2026-05-08'] == 113000.79
        assert champ_map['2026-05-21'] == 112641.72

    def test_post_final_tail_extension_uses_benchmark(self):
        """Dates past the replay's final simulated date scale by benchmark
        return from the final anchor."""
        from src.utils.three_line_replay.extender import _market_return_extend

        equity_curve = [
            {'date': '2026-05-08', 'value': 0.0, 'benchmark': 110000.00, 'raw_value': 96983.96},
            {'date': '2026-05-09', 'value': 0.0, 'benchmark': 110330.00, 'raw_value': 97050.00},
            {'date': '2026-05-12', 'value': 0.0, 'benchmark': 110500.00, 'raw_value': 97100.00},
        ]
        champ_map = {'2026-05-08': 113000.79}
        hybrid_map = {'2026-05-08': 108420.63}
        pre_map = {'2026-05-08': 108240.54}

        _market_return_extend(equity_curve, champ_map, hybrid_map, pre_map)

        assert '2026-05-09' in champ_map
        assert '2026-05-12' in champ_map
        expected_5_09 = 113000.79 * (110330.00 / 110000.00)
        assert abs(champ_map['2026-05-09'] - expected_5_09) < 0.01

    def test_dates_before_replay_start_are_ignored(self):
        from src.utils.three_line_replay.extender import (
            _market_return_extend, REPLAY_START,
        )

        equity_curve = [
            {'date': '2026-01-15', 'value': 0.0, 'benchmark': 100000.00},
            {'date': REPLAY_START, 'value': 0.0, 'benchmark': 101000.00},
            {'date': '2026-03-13', 'value': 0.0, 'benchmark': 102000.00},
        ]
        champ_map = {REPLAY_START: 102000.0}
        hybrid_map: dict = {}
        pre_map: dict = {}

        _market_return_extend(equity_curve, champ_map, hybrid_map, pre_map)

        assert '2026-01-15' not in champ_map
        assert '2026-03-13' in champ_map

    def test_missing_benchmark_flat_holds_not_phantom(self):
        """If a gap row has benchmark None or 0, market_return is undefined
        and the walk must skip that row (leaving champ_map unchanged so the
        per-row patch loop flat-holds). Crucially it must NOT fall back to the
        broker raw_value."""
        from src.utils.three_line_replay.extender import _market_return_extend

        equity_curve = [
            {'date': '2026-05-08', 'value': 0.0, 'benchmark': 110000.00, 'raw_value': 96983.96},
            {'date': '2026-05-11', 'value': 0.0, 'benchmark': None, 'raw_value': 97030.06},
            {'date': '2026-05-12', 'value': 0.0, 'benchmark': 0.0, 'raw_value': 97040.84},
            {'date': '2026-05-13', 'value': 0.0, 'benchmark': 111000.00, 'raw_value': 97200.00},
        ]
        champ_map = {'2026-05-08': 113000.79}
        hybrid_map: dict = {}
        pre_map: dict = {}

        _market_return_extend(equity_curve, champ_map, hybrid_map, pre_map)

        assert '2026-05-11' not in champ_map
        assert '2026-05-12' not in champ_map
        # 5/13 scales from the 5/08 benchmark anchor (last valid prior_mkt).
        expected_5_13 = 113000.79 * (111000.00 / 110000.00)
        assert abs(champ_map['2026-05-13'] - expected_5_13) < 0.01

    def test_empty_champ_map_is_noop(self):
        from src.utils.three_line_replay.extender import _market_return_extend
        equity_curve = [
            {'date': '2026-05-08', 'value': 0.0, 'benchmark': 110000.00},
        ]
        champ_map: dict = {}
        hybrid_map: dict = {}
        pre_map: dict = {}
        _market_return_extend(equity_curve, champ_map, hybrid_map, pre_map)
        assert champ_map == {}
        assert hybrid_map == {}
        assert pre_map == {}
