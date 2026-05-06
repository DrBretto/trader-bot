"""Regression tests for the Alpaca-forward continuity repair invariants.

These tests lock the contract the dashboard's performance line must honor:

* Pre-2026-03-12 (Alpaca paper cutover) chart values are unchanged regardless
  of post-cutover cashflow events. The continuity adjustment subtracts
  cumulative external cashflow, which is zero before the first cashflow
  event, so pre-cutover values must collapse to raw values for free.

* The first continuity-bridge cashflow at 2026-03-12 (-$3,025.53) and the
  second event at 2026-04-22 (-$7,052.30, cumulative -$10,077.83) are
  absorbed without producing a cliff in the continuity-adjusted series:
  the day-over-day change in `continuity_value` reflects only the day's
  organic P&L, not the cashflow patch.

* The raw Alpaca/broker account value stays distinct from the corrected
  continuity-adjusted value. A consumer that wants broker truth can read
  `raw_value`; a consumer that wants performance truth can read `value`.
  The two must not be conflated. The gap is exactly cumulative external
  cashflow.

* The VUG split-correction prevents the May 5 SELL from producing a phantom
  realized loss (covered in test_historical_corrections.py); this file adds
  the dashboard-data-shape invariant tests, not duplicate FIFO assertions.

If any of these tests fail, the chart's continuity contract is broken and
the dashboard's performance line is misleading.
"""

import sys

sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.utils.dashboard_metrics import _build_continuity_rows


CUTOVER_DATE = "2026-03-12"
SECOND_EVENT_DATE = "2026-04-22"
FIRST_BRIDGE_CASHFLOW = -3025.53
SECOND_BRIDGE_CASHFLOW = -7052.30


def _row(date, value, benchmark, external_cashflow, daily_return=None):
    return {
        "date": date,
        "value": value,
        "benchmark": benchmark,
        "external_cashflow": external_cashflow,
        "daily_return": daily_return,
    }


class TestPreCutoverPreservation:
    """Pre-2026-03-12 values must not be altered by the continuity logic."""

    def test_pre_cutover_continuity_equals_raw_when_no_prior_cashflow(self):
        rows = [
            _row("2025-08-04", 100000.0, 100000.0, 0.0),
            _row("2026-01-31", 100000.0, 99701.75, 0.0),
            _row("2026-02-04", 103206.04, 100197.39, 0.0),
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
        ]
        out = _build_continuity_rows(rows)
        for original, computed in zip(rows, out):
            assert computed["continuity_value"] == original["value"], (
                f"pre-cutover {computed['date']} mutated: "
                f"raw={original['value']} continuity={computed['continuity_value']}"
            )
            assert computed["cumulative_external_cashflow"] == 0.0

    def test_pre_cutover_unaffected_by_post_cutover_cashflow(self):
        # Even with a large post-cutover cashflow event, pre-cutover continuity
        # values must stay equal to raw — cumulative cashflow is zero before
        # the cashflow row.
        rows = [
            _row("2026-03-10", 103265.67, 97934.71, 0.0),
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        # Pre-cutover rows
        assert out[0]["continuity_value"] == 103265.67
        assert out[0]["cumulative_external_cashflow"] == 0.0
        assert out[1]["continuity_value"] == 103025.53
        assert out[1]["cumulative_external_cashflow"] == 0.0
        # Cutover row absorbs the cashflow into the cumulative
        assert out[2]["cumulative_external_cashflow"] == FIRST_BRIDGE_CASHFLOW
        assert abs(out[2]["continuity_value"] - (99924.21 - FIRST_BRIDGE_CASHFLOW)) < 1e-6


class TestContinuityBridgeNoCliff:
    """The continuity-adjusted line must not show a cliff at a bridge event."""

    def test_first_bridge_2026_03_12_absorbs_cashflow(self):
        # Day-before vs day-of: the continuity-adjusted delta should equal
        # the organic P&L (a small move), NOT the broker_value delta which
        # includes the bridge cashflow.
        rows = [
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        cont_delta = out[1]["continuity_value"] - out[0]["continuity_value"]
        raw_delta = rows[1]["value"] - rows[0]["value"]
        # raw drops by ~$3,101, continuity drops by ~$76. The packet's
        # invariant: continuity delta ≠ raw delta when a bridge fires.
        assert abs(raw_delta) > 3000.0  # raw shows the full broker step-down
        assert abs(cont_delta) < 200.0  # continuity is just the organic P&L
        # The cashflow is fully absorbed: cont_delta + cashflow ≈ raw_delta
        assert abs(cont_delta + FIRST_BRIDGE_CASHFLOW - raw_delta) < 1e-6

    def test_second_event_2026_04_22_steps_cumulative_to_full_amount(self):
        # By the second event, cumulative_external_cashflow must equal the
        # sum of both bridge cashflows (-$10,077.83 in production data).
        rows = [
            _row("2026-04-21", 104069.06, 102186.89, 0.0),
            _row(SECOND_EVENT_DATE, 97335.82, 102788.16, SECOND_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        # In production, the row series carries the first cashflow forward
        # via the cumulative; here we test the local stepping from 0 to the
        # second event's cashflow value.
        assert out[0]["cumulative_external_cashflow"] == 0.0
        assert out[1]["cumulative_external_cashflow"] == SECOND_BRIDGE_CASHFLOW

    def test_bridge_cashflows_fully_compose(self):
        # Both bridge events composed produce cumulative -$10,077.83.
        rows = [
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
            _row("2026-04-21", 104069.06, 102186.89, 0.0),
            _row(SECOND_EVENT_DATE, 97335.82, 102788.16, SECOND_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        assert out[3]["cumulative_external_cashflow"] == (
            FIRST_BRIDGE_CASHFLOW + SECOND_BRIDGE_CASHFLOW
        )
        # Total cumulative (~-$10,077.83) matches the documented production
        # cumulative_external_cashflow steady-state value within rounding.
        assert abs(out[3]["cumulative_external_cashflow"] - (-10077.83)) < 0.01


class TestRawVsCorrectedContinuitySeparation:
    """Raw Alpaca account value must remain accessible separately."""

    def test_continuity_rows_preserve_raw_value(self):
        rows = [
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
            _row(SECOND_EVENT_DATE, 97335.82, 102788.16, SECOND_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        # Each output row preserves the original `value` (raw broker total).
        for original, computed in zip(rows, out):
            assert computed["value"] == original["value"], (
                "raw broker value must be preserved as `value` in the row; "
                "continuity-adjusted goes in `continuity_value`."
            )

    def test_gap_equals_cumulative_external_cashflow(self):
        rows = [
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
            _row(SECOND_EVENT_DATE, 97335.82, 102788.16, SECOND_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        for r in out:
            gap = r["continuity_value"] - r["value"]
            assert abs(gap - (-r["cumulative_external_cashflow"])) < 1e-6, (
                f"row {r['date']}: gap between continuity and raw must equal "
                f"-cumulative_external_cashflow. Got gap={gap}, "
                f"cum_cf={r['cumulative_external_cashflow']}"
            )

    def test_post_event_continuity_does_not_force_match_with_raw(self):
        # By design, after a cashflow event the continuity-adjusted value
        # diverges from raw by the cumulative cashflow. They MUST NOT be
        # forced to match.
        rows = [
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
            _row(SECOND_EVENT_DATE, 97335.82, 102788.16, SECOND_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        for r in out:
            assert r["continuity_value"] != r["value"], (
                "continuity_value must not collapse onto raw `value` once a "
                "bridge cashflow has fired"
            )


class TestProductionDataShape:
    """Validate the fields downstream consumers (publish_artifacts, frontend) read."""

    def test_continuity_rows_shape_includes_required_fields(self):
        rows = [_row("2025-08-04", 100000.0, 100000.0, 0.0)]
        out = _build_continuity_rows(rows)
        required = {"date", "value", "benchmark", "external_cashflow",
                    "continuity_value", "cumulative_external_cashflow"}
        assert required.issubset(out[0].keys()), (
            f"missing required fields: {required - set(out[0].keys())}"
        )

    def test_continuity_value_renders_as_chart_value(self):
        # publish_artifacts.build_dashboard_data emits each equity_curve row as:
        #   {date, value: continuity_value, raw_value: original value, benchmark, cum_cf}
        # The frontend's PerformanceChart reads `value` as the portfolio line.
        # This test locks the math that feeds that contract: the field
        # named `continuity_value` is what becomes `value` in equity_curve.
        rows = [
            _row("2026-03-11", 103025.53, 97615.61, 0.0),
            _row(CUTOVER_DATE, 99924.21, 96358.46, FIRST_BRIDGE_CASHFLOW),
        ]
        out = _build_continuity_rows(rows)
        # Pre-cutover: continuity_value == raw value (no cliff)
        assert out[0]["continuity_value"] == 103025.53
        # Cutover: continuity_value is raw + |cumulative cashflow added back|
        expected = 99924.21 - FIRST_BRIDGE_CASHFLOW  # subtracting a negative
        assert abs(out[1]["continuity_value"] - expected) < 1e-6


class TestSizingCorrectionRequiresEvidence:
    """A sizing correction (e.g. auto-nerf) cannot be applied without explicit
    date-range evidence. The 2026-04-30 ensemble double-application finding
    was a false positive (see docs/plans/2026-04-30-ensemble-double-fix-RETURN.md
    and the TestEnsembleMultiplierCallerPatterns suite). This test locks the
    policy: the v3 production caller defense at decision_engine.py must keep
    passing ensemble_multiplier=1.0 when expert_signals is non-None."""

    def test_v3_caller_defense_in_source(self):
        # Read the source file; if a refactor drops the `if expert_signals
        # is None else 1.0` defense, the test fails and the developer has
        # to either restore it or explicitly enable the gate.
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "src" / "steps" / "decision_engine.py").read_text()
        # Both v3 call sites must carry the defense.
        defenses = src.count("ensemble_multiplier if expert_signals is None else 1.0")
        assert defenses >= 2, (
            "v3 caller defense in src/steps/decision_engine.py must be "
            "preserved at both compute_position_size call sites. If it is "
            "intentionally removed, enable ensemble_multiplier_already_applied "
            "or surface a finding before merging."
        )
