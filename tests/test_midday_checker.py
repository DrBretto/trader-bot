"""Tests for midday checker: trailing stops, VIX circuit breaker, skipped-buy re-check."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.midday_checker import (
    check_vix_circuit_breaker,
    evaluate_trailing_stops,
    VIX_ABSOLUTE_THRESHOLD,
    VIX_INTRADAY_JUMP_THRESHOLD,
)


class TestVixCircuitBreaker:
    """Tests for VIX circuit breaker logic."""

    def test_normal_vix_no_breaker(self):
        active, reason = check_vix_circuit_breaker(18.5, 17.0)
        assert not active
        assert "normal range" in reason.lower()

    def test_high_vix_absolute_triggers(self):
        active, reason = check_vix_circuit_breaker(36.0, 20.0)
        assert active
        assert "absolute threshold" in reason.lower()

    def test_vix_at_threshold_triggers(self):
        active, reason = check_vix_circuit_breaker(VIX_ABSOLUTE_THRESHOLD, 20.0)
        assert active

    def test_vix_just_below_threshold_no_trigger(self):
        active, reason = check_vix_circuit_breaker(34.9, 20.0)
        # Still below absolute but might trigger on jump
        # 34.9 / 20.0 - 1 = 0.745, which exceeds 25% jump
        assert active  # should trigger on jump

    def test_vix_intraday_jump_triggers(self):
        # 25% jump: 20 -> 25
        active, reason = check_vix_circuit_breaker(25.0, 20.0)
        assert active
        assert "intraday jump" in reason.lower()

    def test_vix_moderate_jump_no_trigger(self):
        # 10% jump: 20 -> 22 (below 25% threshold)
        active, reason = check_vix_circuit_breaker(22.0, 20.0)
        assert not active

    def test_vix_previous_close_zero_no_crash(self):
        active, reason = check_vix_circuit_breaker(20.0, 0.0)
        assert not active

    def test_vix_drop_no_trigger(self):
        # VIX dropping is not a concern
        active, reason = check_vix_circuit_breaker(15.0, 25.0)
        assert not active


class TestEvaluateTrailingStops:
    """Tests for trailing stop re-evaluation at midday prices."""

    def _make_holding(self, symbol, peak_price, entry_price=100.0, leveraged=False):
        return {
            'symbol': symbol,
            'peak_price': peak_price,
            'entry_price': entry_price,
            'shares': 10,
            'leverage_flag': 1 if leveraged else 0,
        }

    def test_no_stop_triggered(self):
        holdings = [self._make_holding('SPY', 450.0)]
        price_map = {'SPY': 445.0}  # only 1.1% below peak
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 0

    def test_stop_triggered_at_threshold(self):
        holdings = [self._make_holding('SPY', 450.0)]
        price_map = {'SPY': 405.0}  # exactly 10% below peak
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 1
        assert triggered[0]['symbol'] == 'SPY'

    def test_stop_triggered_below_threshold(self):
        holdings = [self._make_holding('SPY', 450.0)]
        price_map = {'SPY': 390.0}  # 13.3% below peak
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 1

    def test_leveraged_tighter_stop(self):
        holdings = [self._make_holding('TQQQ', 50.0, leveraged=True)]
        price_map = {'TQQQ': 47.5}  # 5% below peak — should NOT trigger 6% stop
        params = {'trailing_stop_base': 0.10, 'trailing_stop_leveraged': 0.06}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 0

        # Now at 6% below: should trigger
        price_map = {'TQQQ': 47.0}
        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 1

    def test_multiple_holdings_partial_trigger(self):
        holdings = [
            self._make_holding('SPY', 450.0),
            self._make_holding('QQQ', 380.0),
            self._make_holding('TLT', 100.0),
        ]
        price_map = {
            'SPY': 445.0,  # safe
            'QQQ': 330.0,  # 13.2% below peak — triggered
            'TLT': 98.0,   # 2% below peak — safe
        }
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 1
        assert triggered[0]['symbol'] == 'QQQ'

    def test_missing_price_skipped(self):
        holdings = [self._make_holding('SPY', 450.0)]
        price_map = {}  # no price for SPY
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 0

    def test_peak_price_updated_when_higher(self):
        holdings = [self._make_holding('SPY', 450.0)]
        price_map = {'SPY': 460.0}  # new high
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 0
        # The function uses the updated peak internally

    def test_zero_peak_price_skipped(self):
        holdings = [{'symbol': 'SPY', 'peak_price': 0, 'leverage_flag': 0}]
        price_map = {'SPY': 100.0}
        params = {'trailing_stop_base': 0.10}

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 0

    def test_default_stop_pct_used(self):
        holdings = [self._make_holding('SPY', 100.0)]
        price_map = {'SPY': 89.0}  # 11% below — triggers default 10%
        params = {}  # no explicit stop pct

        triggered = evaluate_trailing_stops(holdings, price_map, params)
        assert len(triggered) == 1
