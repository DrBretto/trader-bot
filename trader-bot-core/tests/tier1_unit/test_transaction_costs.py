"""Tier 1 unit-fixture — SALVAGE #3 (demoted to unit).

``src.utils.transaction_costs`` is pure deterministic lookups + a real cost
model — no mock, no planted-input theater. Kept verbatim as a unit test (never
"proof the system works"). Reads the real spread-tier table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# conftest puts the repo root on sys.path; the cost model lives in the parent repo.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from chassis.utils.transaction_costs import (  # noqa: E402
    apply_transaction_costs,
    get_half_spread_bps,
)

pytestmark = pytest.mark.unit


class TestSpreadLookup:
    def test_broad_equity_has_lowest_spread(self):
        assert get_half_spread_bps('broad', 'equity') == 1.0

    def test_volatility_has_highest_spread(self):
        assert get_half_spread_bps('volatility', 'vol') == 8.0

    def test_unknown_sector_uses_asset_class_default(self):
        assert get_half_spread_bps('nonexistent_sector', 'bond') == 2.5

    def test_unknown_everything_uses_fallback(self):
        assert get_half_spread_bps('unknown', 'unknown') == 3.0


class TestApplyTransactionCosts:
    def test_buy_fills_above_market(self):
        # Slippage is random per draw, so a SINGLE buy fill can land below market
        # ~25% of the time; assert on the batch mean (matching the SELL test's
        # 100-sample pattern) so the every-commit gate is not flaky.
        results = [apply_transaction_costs(100.0, 'BUY', 'broad', 'equity') for _ in range(100)]
        fills = [fill for fill, _ in results]
        cost_bps = [c for _, c in results]
        assert sum(fills) / len(fills) > 100.0
        assert sum(cost_bps) / len(cost_bps) > 0

    def test_sell_fills_near_or_below_market(self):
        fills = [apply_transaction_costs(100.0, 'SELL', 'broad', 'equity')[0] for _ in range(100)]
        assert sum(fills) / len(fills) < 100.0

    def test_cost_scales_with_spread_tier(self):
        broad = [apply_transaction_costs(100.0, 'BUY', 'broad', 'equity')[1] for _ in range(100)]
        vol = [apply_transaction_costs(100.0, 'BUY', 'volatility', 'vol')[1] for _ in range(100)]
        assert sum(vol) / 100 > sum(broad) / 100

    def test_reduce_treated_as_sell(self):
        fills = [apply_transaction_costs(100.0, 'REDUCE', 'broad', 'equity')[0] for _ in range(100)]
        assert sum(fills) / len(fills) < 100.0

    def test_cost_is_small_fraction(self):
        fill, cost_bps = apply_transaction_costs(500.0, 'BUY', 'broad', 'equity')
        assert cost_bps < 10
        assert abs(fill - 500.0) < 0.50
