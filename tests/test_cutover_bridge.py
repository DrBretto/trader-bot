"""Tests for src/utils/cutover_bridge.py — cutover continuity bridge logic."""

import pytest
from src.utils.cutover_bridge import (
    compute_bridge_cashflow,
    build_cutover_patch,
    apply_patch,
    describe_patch,
    CONTINUITY_MARKER_PREFIX,
)


class TestComputeBridgeCashflow:
    def test_value_decreased(self):
        # $103k -> $100k: cashflow = -$3k
        assert compute_bridge_cashflow(100000.0, 103000.0) == -3000.0

    def test_value_increased(self):
        # $100k -> $105k: cashflow = +$5k
        assert compute_bridge_cashflow(105000.0, 100000.0) == 5000.0

    def test_no_change(self):
        assert compute_bridge_cashflow(100000.0, 100000.0) == 0.0

    def test_precise_values(self):
        result = compute_bridge_cashflow(100000.0, 103025.53)
        assert abs(result - (-3025.53)) < 0.01


class TestBuildCutoverPatch:
    def test_basic_patch(self):
        cutover = {"portfolio_value": 100000.0}
        previous = {"portfolio_value": 103025.53}
        patch = build_cutover_patch(cutover, previous, "2026-03-12")

        assert abs(patch["external_cashflow"] - (-3025.53)) < 0.01
        assert patch["continuity_bridge_marker"] == f"{CONTINUITY_MARKER_PREFIX}:2026-03-12"

    def test_benchmark_carried_when_missing(self):
        cutover = {"portfolio_value": 100000.0}
        previous = {
            "portfolio_value": 103000.0,
            "benchmark_start_price": 694.04,
            "benchmark_shares": 144.389,
        }
        patch = build_cutover_patch(cutover, previous, "2026-03-12")

        assert patch["benchmark_start_price"] == 694.04
        assert patch["benchmark_shares"] == 144.389

    def test_benchmark_not_overwritten_when_present(self):
        cutover = {
            "portfolio_value": 100000.0,
            "benchmark_start_price": 700.0,
            "benchmark_shares": 142.0,
        }
        previous = {
            "portfolio_value": 103000.0,
            "benchmark_start_price": 694.04,
            "benchmark_shares": 144.389,
        }
        patch = build_cutover_patch(cutover, previous, "2026-03-12")

        # Should NOT include benchmark fields since cutover already has them
        assert "benchmark_start_price" not in patch
        assert "benchmark_shares" not in patch


class TestApplyPatch:
    def test_applies_patch(self):
        state = {"portfolio_value": 100000.0, "cash": 100000.0}
        patch = {
            "external_cashflow": -3025.53,
            "continuity_bridge_marker": f"{CONTINUITY_MARKER_PREFIX}:2026-03-12",
        }
        result = apply_patch(state, patch)

        assert result["external_cashflow"] == -3025.53
        assert result["continuity_bridge_marker"] == f"{CONTINUITY_MARKER_PREFIX}:2026-03-12"
        assert result["portfolio_value"] == 100000.0
        assert result["cash"] == 100000.0

    def test_does_not_mutate_input(self):
        state = {"portfolio_value": 100000.0}
        patch = {"external_cashflow": -3000.0, "continuity_bridge_marker": "test"}
        result = apply_patch(state, patch)

        assert "external_cashflow" not in state
        assert result is not state

    def test_idempotent_when_already_patched(self):
        marker = f"{CONTINUITY_MARKER_PREFIX}:2026-03-12"
        state = {
            "portfolio_value": 100000.0,
            "external_cashflow": -3025.53,
            "continuity_bridge_marker": marker,
        }
        patch = {
            "external_cashflow": -3025.53,
            "continuity_bridge_marker": marker,
        }
        result = apply_patch(state, patch)

        # Should be unchanged
        assert result == state

    def test_re_patches_with_different_marker(self):
        state = {
            "portfolio_value": 100000.0,
            "external_cashflow": -1000.0,
            "continuity_bridge_marker": f"{CONTINUITY_MARKER_PREFIX}:2026-03-10",
        }
        patch = {
            "external_cashflow": -3025.53,
            "continuity_bridge_marker": f"{CONTINUITY_MARKER_PREFIX}:2026-03-12",
        }
        result = apply_patch(state, patch)

        assert result["external_cashflow"] == -3025.53
        assert result["continuity_bridge_marker"] == f"{CONTINUITY_MARKER_PREFIX}:2026-03-12"


class TestDescribePatch:
    def test_summary_fields(self):
        previous = {"portfolio_value": 103025.53, "date": "2026-03-11"}
        cutover = {"portfolio_value": 100000.0}
        patch = {
            "external_cashflow": -3025.53,
            "continuity_bridge_marker": f"{CONTINUITY_MARKER_PREFIX}:2026-03-12",
            "benchmark_start_price": 694.04,
        }
        summary = describe_patch(previous, cutover, patch)

        assert summary["previous_date"] == "2026-03-11"
        assert summary["previous_value"] == 103025.53
        assert summary["cutover_value"] == 100000.0
        assert abs(summary["value_delta"] - (-3025.53)) < 0.01
        assert summary["external_cashflow"] == -3025.53
        assert summary["neutralized_return"] == 0.0
        assert "benchmark_start_price" in summary["benchmark_fields_carried"]
        assert not summary["already_patched"]

    def test_already_patched_flag(self):
        cutover = {
            "portfolio_value": 100000.0,
            "continuity_bridge_marker": "existing",
        }
        previous = {"portfolio_value": 103000.0}
        patch = {"external_cashflow": -3000.0}
        summary = describe_patch(previous, cutover, patch)

        assert summary["already_patched"] is True
