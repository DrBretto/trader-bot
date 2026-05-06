"""Tests for src/utils/historical_corrections.py.

The corrections layer is data-driven (STOCK_SPLITS list). These tests lock the
shape of the layer's behavior — adding new entries to STOCK_SPLITS doesn't
require changing tests; changing the entries' effect on adjusted basis or
the FIFO matching outcome would.
"""

import sys

sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.utils.historical_corrections import (
    STOCK_SPLITS,
    SplitEvent,
    apply_split_corrections_to_fills,
    split_adjust_basis,
    split_adjust_holding,
)
from src.utils.dashboard_metrics import _build_trade_summary


class TestSplitAdjustBasis:
    """The core split-adjustment math."""

    def test_pre_split_buy_gets_scaled_up(self):
        # VUG 6:1 at 2026-04-22; an Apr 7 BUY of 17.127 sh @ $441.05 should
        # become 102.766 sh @ $73.508 post-adjustment.
        result = split_adjust_basis(
            symbol="VUG", fill_date="2026-04-07",
            shares=17.127628206, price=441.05,
        )
        assert abs(result["shares"] - 102.765769236) < 1e-6
        assert abs(result["price"] - (441.05 / 6.0)) < 1e-6
        assert result["ratio_applied"] == 6.0

    def test_post_split_sell_passes_through(self):
        # A sell on May 5 (post-split) is recorded at $84.38 already-post-split;
        # no further adjustment needed.
        result = split_adjust_basis(
            symbol="VUG", fill_date="2026-05-05",
            shares=56, price=84.38,
        )
        assert result["shares"] == 56
        assert result["price"] == 84.38
        assert result["ratio_applied"] == 1.0

    def test_unknown_symbol_passes_through(self):
        # No split entry for SPY → no adjustment.
        result = split_adjust_basis(
            symbol="SPY", fill_date="2025-01-01",
            shares=100, price=450,
        )
        assert result["shares"] == 100
        assert result["price"] == 450
        assert result["ratio_applied"] == 1.0

    def test_dollar_basis_preserved_across_adjustment(self):
        # 17.127 × $441.05 = $7,553.40
        # post-adjusted: 102.766 × $73.508 = $7,553.40 (within rounding)
        pre_dollars = 17.127628206 * 441.05
        result = split_adjust_basis(
            symbol="VUG", fill_date="2026-04-07",
            shares=17.127628206, price=441.05,
        )
        post_dollars = result["shares"] * result["price"]
        assert abs(pre_dollars - post_dollars) < 0.01


class TestApplySplitCorrectionsToFills:
    """Fills round-trip through the corrections function."""

    def test_pre_split_vug_buy_adjusted(self):
        fills = [{
            "symbol": "VUG", "action": "BUY",
            "shares": 17.127628206, "price": 441.05,
            "market_price": 441.05,
            "dollars": 7553.40, "_trade_date": "2026-04-07",
        }]
        adjusted = apply_split_corrections_to_fills(fills)
        assert len(adjusted) == 1
        assert abs(adjusted[0]["shares"] - 102.765769236) < 1e-6
        assert abs(adjusted[0]["price"] - 73.5083333) < 1e-4
        # market_price must scale with price; otherwise _fill_cost_dollars
        # computes a phantom spread of (441.05 - 73.51) × shares ≈ $37k.
        assert abs(adjusted[0]["market_price"] - 73.5083333) < 1e-4
        assert adjusted[0]["_split_ratio_applied"] == 6.0
        assert adjusted[0]["_pre_split_shares"] == 17.127628206

    def test_market_price_adjusted_to_match_price(self):
        """Regression: apply_split_corrections_to_fills must not leave
        market_price on the pre-split scale while adjusting price. The
        spread/slippage cost calculation depends on |price - market_price|."""
        from src.utils.dashboard_metrics import _fill_cost_dollars
        fills = [{
            "symbol": "VUG", "action": "BUY",
            "shares": 17.127628206, "price": 441.05,
            "market_price": 441.05,
            "dollars": 7553.40, "_trade_date": "2026-04-07",
        }]
        adjusted = apply_split_corrections_to_fills(fills)
        cost = _fill_cost_dollars(adjusted[0])
        # Pre-fix: ≈ $37,765 phantom spread. Post-fix: $0 (price == market_price).
        assert cost == 0.0

    def test_post_split_vug_sell_unchanged(self):
        fills = [{
            "symbol": "VUG", "action": "SELL",
            "shares": 56, "price": 84.38,
            "dollars": 4723.6, "_trade_date": "2026-05-05",
        }]
        adjusted = apply_split_corrections_to_fills(fills)
        assert adjusted[0]["shares"] == 56
        assert adjusted[0]["price"] == 84.38
        assert "_split_ratio_applied" not in adjusted[0]

    def test_other_symbols_pass_through(self):
        fills = [
            {"symbol": "FXI", "action": "BUY", "shares": 213, "price": 35.39,
             "_trade_date": "2026-04-07"},
            {"symbol": "XLF", "action": "BUY", "shares": 152, "price": 49.67,
             "_trade_date": "2026-04-07"},
        ]
        adjusted = apply_split_corrections_to_fills(fills)
        assert adjusted == [dict(f) for f in fills]

    def test_does_not_mutate_input(self):
        fills = [{
            "symbol": "VUG", "action": "BUY",
            "shares": 17.127628206, "price": 441.05,
            "_trade_date": "2026-04-07",
        }]
        original = dict(fills[0])
        apply_split_corrections_to_fills(fills)
        assert fills[0] == original


class TestSplitAdjustHolding:
    def test_pre_split_holding_adjusted(self):
        result = split_adjust_holding(
            symbol="VUG", as_of_date="2026-05-06",
            shares=17.127628206, entry_price=441.05,
            entry_date="2026-04-07",
        )
        assert abs(result["shares"] - 102.765769236) < 1e-6
        assert abs(result["entry_price"] - 73.5083333) < 1e-4

    def test_post_split_holding_unchanged(self):
        result = split_adjust_holding(
            symbol="VUG", as_of_date="2026-05-06",
            shares=102.765769236, entry_price=73.48,
            entry_date="2026-04-22",
        )
        assert result["shares"] == 102.765769236
        assert result["entry_price"] == 73.48
        assert result["ratio_applied"] == 1.0


class TestRoundTripPnLWithSplit:
    """End-to-end: the May 5 VUG SELL no longer produces a phantom loss
    when matched against the Apr 7 BUY, because the corrections layer
    adjusts the BUY's basis to post-split before FIFO matching. This is
    the operator-visible bug the corrections module was built to fix."""

    def _vug_round_trip_pnl(self):
        fills = [
            {
                "symbol": "VUG", "action": "BUY",
                "shares": 17.127628206, "price": 441.05,
                "dollars": 7553.40,
                "timestamp": "2026-04-07T13:45:43.178424",
                "_trade_date": "2026-04-07",
            },
            {
                "symbol": "VUG", "action": "SELL",
                "shares": 56.0, "price": 84.38,
                "dollars": 4723.6,
                "timestamp": "2026-05-05T13:45:43.652459",
                "_trade_date": "2026-05-05",
            },
        ]
        summary = _build_trade_summary(fills)
        vug_rts = [rt for rt in summary["round_trips"] if rt["symbol"] == "VUG"]
        return summary, vug_rts

    def test_corrected_realized_pnl_is_a_gain_not_a_loss(self):
        summary, vug_rts = self._vug_round_trip_pnl()
        assert len(vug_rts) == 1
        rt = vug_rts[0]
        # Pre-fix: -$6,063 phantom loss. Post-fix: ~+$609 real gain.
        assert rt["realized_pnl"] > 0
        # 56 sh × ($84.38 - $73.508) ≈ $608.84
        assert 600 < rt["realized_pnl"] < 620

    def test_corrected_round_trip_uses_post_split_entry_price(self):
        _, vug_rts = self._vug_round_trip_pnl()
        rt = vug_rts[0]
        # post-split entry price = $441.05 / 6 = $73.508
        assert abs(rt["entry_price"] - 441.05 / 6.0) < 0.01
        assert rt["exit_price"] == 84.38

    def test_no_unmatched_sell_shares_after_correction(self):
        """Pre-fix: 17 BUY sh int-truncated, then 56 - 17 = 39 unmatched
        SELL sh accumulated to trade_summary.unmatched_closing_shares.
        Post-fix: 102 BUY sh covers the 56 SELL fully."""
        summary, _ = self._vug_round_trip_pnl()
        assert summary["unmatched_closing_shares"] == 0


class TestStockSplitsRegistry:
    """Regression: STOCK_SPLITS contains the VUG entry the audit relied on."""

    def test_vug_entry_present(self):
        vug = [s for s in STOCK_SPLITS if s.symbol == "VUG"]
        assert len(vug) == 1
        assert vug[0].ex_date == "2026-04-22"
        assert vug[0].ratio == 6.0

    def test_all_entries_have_dated_iso_format(self):
        for entry in STOCK_SPLITS:
            assert isinstance(entry, SplitEvent)
            # YYYY-MM-DD shape
            assert len(entry.ex_date) == 10
            assert entry.ex_date[4] == "-" and entry.ex_date[7] == "-"
            assert entry.ratio > 0
