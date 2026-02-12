"""Tests for transaction cost simulation."""

import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.utils.transaction_costs import (
    apply_transaction_costs,
    get_half_spread_bps,
)


class TestSpreadLookup:
    """Test spread tier lookups."""

    def test_broad_equity_has_lowest_spread(self):
        assert get_half_spread_bps('broad', 'equity') == 1.0

    def test_volatility_has_highest_spread(self):
        assert get_half_spread_bps('volatility', 'vol') == 8.0

    def test_unknown_sector_uses_asset_class_default(self):
        assert get_half_spread_bps('nonexistent_sector', 'bond') == 2.5

    def test_unknown_everything_uses_fallback(self):
        assert get_half_spread_bps('unknown', 'unknown') == 3.0


class TestApplyTransactionCosts:
    """Test cost application to trade prices."""

    def test_buy_fills_above_market(self):
        fill, cost_bps = apply_transaction_costs(100.0, 'BUY', 'broad', 'equity')
        assert fill > 100.0
        assert cost_bps > 0

    def test_sell_fills_near_or_below_market(self):
        """On average, sell fills should be below market price."""
        fills = [apply_transaction_costs(100.0, 'SELL', 'broad', 'equity')[0] for _ in range(100)]
        avg_fill = sum(fills) / len(fills)
        assert avg_fill < 100.0  # Average sell fill is below market
        # Individual fills can be slightly above due to favorable slippage

    def test_cost_scales_with_spread_tier(self):
        """Higher-spread assets should have larger price deviations on average."""
        buy_fills_broad = [
            apply_transaction_costs(100.0, 'BUY', 'broad', 'equity')[1]
            for _ in range(100)
        ]
        buy_fills_vol = [
            apply_transaction_costs(100.0, 'BUY', 'volatility', 'vol')[1]
            for _ in range(100)
        ]
        assert sum(buy_fills_vol) / 100 > sum(buy_fills_broad) / 100

    def test_reduce_treated_as_sell(self):
        """REDUCE action should behave like SELL (fill near/below market on average)."""
        fills = [apply_transaction_costs(100.0, 'REDUCE', 'broad', 'equity')[0] for _ in range(100)]
        avg_fill = sum(fills) / len(fills)
        assert avg_fill < 100.0

    def test_cost_is_small_fraction(self):
        """Transaction costs should be a tiny fraction of price."""
        fill, cost_bps = apply_transaction_costs(500.0, 'BUY', 'broad', 'equity')
        assert cost_bps < 10  # Less than 10 bps for any trade
        assert abs(fill - 500.0) < 0.50  # Less than $0.50 on a $500 stock
