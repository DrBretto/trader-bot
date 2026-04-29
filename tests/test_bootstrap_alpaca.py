"""Tests for bootstrap_alpaca_from_sim_state.py — holdings bootstrap logic."""

import pytest
from scripts.bootstrap_alpaca_from_sim_state import (
    compute_source_weights,
    compute_bootstrap_orders,
    make_bootstrap_client_order_id,
)


class TestComputeSourceWeights:
    def test_basic_weights(self):
        holdings = [
            {"symbol": "SPY", "shares": 10, "current_price": 500.0},
            {"symbol": "GLD", "shares": 20, "current_price": 250.0},
        ]
        weights = compute_source_weights(holdings, 10000.0)

        assert len(weights) == 2
        # Both are $5000 market value, so 50% each
        assert weights[0]["weight"] == 0.5
        assert weights[1]["weight"] == 0.5

    def test_sorted_by_weight_descending(self):
        holdings = [
            {"symbol": "A", "shares": 1, "current_price": 100.0},
            {"symbol": "B", "shares": 10, "current_price": 100.0},
        ]
        weights = compute_source_weights(holdings, 1100.0)

        assert weights[0]["symbol"] == "B"
        assert weights[1]["symbol"] == "A"

    def test_uses_market_value_field(self):
        holdings = [
            {"symbol": "SPY", "shares": 10, "market_value": 5000.0},
        ]
        weights = compute_source_weights(holdings, 10000.0)
        assert weights[0]["source_market_value"] == 5000.0
        assert weights[0]["weight"] == 0.5

    def test_skips_empty_symbol(self):
        holdings = [
            {"symbol": "", "shares": 10, "current_price": 100.0},
            {"symbol": "SPY", "shares": 5, "current_price": 100.0},
        ]
        weights = compute_source_weights(holdings, 500.0)
        assert len(weights) == 1
        assert weights[0]["symbol"] == "SPY"

    def test_zero_portfolio_value(self):
        holdings = [{"symbol": "SPY", "shares": 10, "current_price": 100.0}]
        weights = compute_source_weights(holdings, 0.0)
        assert weights[0]["weight"] == 0.0


class TestComputeBootstrapOrders:
    def _make_weights(self, symbols_weights):
        """Helper: create source_weights from [(symbol, weight), ...]."""
        return [
            {
                "symbol": sym,
                "source_shares": 0,
                "source_market_value": w * 100000,
                "weight": w,
            }
            for sym, w in symbols_weights
        ]

    def test_basic_scaled_orders(self):
        weights = self._make_weights([("SPY", 0.4), ("GLD", 0.3), ("XLE", 0.15)])
        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=[],
        )

        assert len(orders) == 3
        assert orders[0]["symbol"] == "SPY"
        assert orders[0]["dollars"] == 5000.0  # capped at max_per_order
        assert orders[1]["symbol"] == "GLD"
        assert orders[1]["dollars"] == 5000.0  # capped at max_per_order
        assert orders[2]["symbol"] == "XLE"
        assert orders[2]["dollars"] == 5000.0  # capped at max_per_order

    def test_small_weight_not_capped(self):
        weights = self._make_weights([("SPY", 0.02)])
        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=[],
            max_per_order=5000.0,
        )

        assert len(orders) == 1
        assert orders[0]["dollars"] == 2000.0  # 2% of 100k

    def test_below_minimum_skipped(self):
        weights = self._make_weights([("SPY", 0.000005)])
        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=[],
        )

        assert len(orders) == 0
        assert len(skipped) == 1
        assert "minimum" in skipped[0]["reason"].lower() or "below" in skipped[0]["reason"].lower()

    def test_already_positioned_skipped(self):
        weights = self._make_weights([("SPY", 0.4)])
        existing = [{"symbol": "SPY", "market_value": 40000.0}]

        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=existing,
        )

        assert len(orders) == 0
        assert len(skipped) == 1

    def test_partially_positioned_tops_up(self):
        weights = self._make_weights([("SPY", 0.4)])
        existing = [{"symbol": "SPY", "market_value": 20000.0}]

        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=existing,
            max_per_order=50000.0,
        )

        assert len(orders) == 1
        # Target is $40k, existing is $20k, so buy $20k
        assert orders[0]["dollars"] == 20000.0

    def test_max_total_cap(self):
        weights = self._make_weights([
            ("A", 0.4), ("B", 0.3), ("C", 0.2), ("D", 0.1)
        ])
        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=[],
            max_per_order=50000.0,
            max_total=50000.0,
        )

        total = sum(o["dollars"] for o in orders)
        assert total <= 50000.0

    def test_allowlist_filters(self):
        weights = self._make_weights([("SPY", 0.4), ("GLD", 0.3)])
        orders, skipped = compute_bootstrap_orders(
            source_weights=weights,
            target_equity=100000.0,
            existing_positions=[],
            symbol_allowlist=["SPY"],
        )

        assert len(orders) == 1
        assert orders[0]["symbol"] == "SPY"
        assert len(skipped) == 1
        assert skipped[0]["symbol"] == "GLD"
        assert "allowlist" in skipped[0]["reason"].lower()


class TestMakeBootstrapClientOrderId:
    def test_deterministic(self):
        id1 = make_bootstrap_client_order_id("SPY", "2026-03-12")
        id2 = make_bootstrap_client_order_id("SPY", "2026-03-12")
        assert id1 == id2

    def test_different_symbols_different_ids(self):
        id1 = make_bootstrap_client_order_id("SPY", "2026-03-12")
        id2 = make_bootstrap_client_order_id("GLD", "2026-03-12")
        assert id1 != id2

    def test_prefix(self):
        oid = make_bootstrap_client_order_id("SPY", "2026-03-12")
        assert oid.startswith("tb-boot-")
