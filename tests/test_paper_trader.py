"""Tests for the paper trader."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.paper_trader import (
    execute_trade,
    update_portfolio_values
)


class TestExecuteTrade:
    """Tests for trade execution."""

    @patch('src.steps.paper_trader.apply_transaction_costs')
    def test_buy_trade_deducts_cash(self, mock_costs):
        """Test that buy trades deduct cash correctly with transaction costs."""
        # Mock deterministic costs: fill at 450.045 (1 bps spread on broad equity)
        mock_costs.return_value = (450.045, 1.0)

        portfolio = {
            'cash': 100000,
            'holdings': []
        }

        action = {
            'action': 'BUY',
            'symbol': 'SPY',
            'shares': 100,
            'price': 450,
            'health': 0.75
        }

        universe_df = pd.DataFrame({
            'symbol': ['SPY'],
            'asset_class': ['equity'],
            'sector': ['broad']
        })

        trade = execute_trade(portfolio, action, 'risk_on_trend', universe_df)

        assert portfolio['cash'] == pytest.approx(100000 - 100 * 450.045, abs=0.01)
        assert len(portfolio['holdings']) == 1
        assert portfolio['holdings'][0]['symbol'] == 'SPY'
        assert portfolio['holdings'][0]['shares'] == 100
        assert trade['action'] == 'BUY'
        assert trade['market_price'] == 450
        assert trade['transaction_cost_bps'] == 1.0
        assert portfolio['cumulative_transaction_costs'] == pytest.approx(4.5, abs=0.01)

    @patch('src.steps.paper_trader.apply_transaction_costs')
    def test_sell_trade_adds_cash(self, mock_costs):
        """Test that sell trades add cash correctly with transaction costs."""
        # Sell fill slightly below market (spread cost)
        mock_costs.return_value = (449.955, 1.0)

        portfolio = {
            'cash': 50000,
            'holdings': [{
                'symbol': 'SPY',
                'shares': 100,
                'entry_price': 440,
                'entry_date': '2025-01-01'
            }]
        }

        action = {
            'action': 'SELL',
            'symbol': 'SPY',
            'shares': 100,
            'price': 450,
            'reason': 'STOP_HIT'
        }

        trade = execute_trade(portfolio, action, 'risk_on_trend', pd.DataFrame())

        assert portfolio['cash'] == pytest.approx(50000 + 100 * 449.955, abs=0.01)
        assert len(portfolio['holdings']) == 0
        # P&L uses fill price vs entry price
        assert trade['pnl'] == pytest.approx((449.955 - 440) * 100, abs=0.01)
        assert trade['market_price'] == 450
        assert trade['transaction_cost_bps'] == 1.0

    @patch('src.steps.paper_trader.apply_transaction_costs')
    def test_sell_trade_calculates_pnl(self, mock_costs):
        """Test that P&L is calculated correctly with costs."""
        mock_costs.return_value = (449.955, 1.0)

        portfolio = {
            'cash': 0,
            'holdings': [{
                'symbol': 'SPY',
                'shares': 50,
                'entry_price': 400,
                'entry_date': '2025-01-01'
            }]
        }

        action = {
            'action': 'SELL',
            'symbol': 'SPY',
            'shares': 50,
            'price': 450,
            'reason': 'PROFIT_TAKE'
        }

        trade = execute_trade(portfolio, action, 'risk_on_trend', pd.DataFrame())

        assert trade['pnl'] == pytest.approx((449.955 - 400) * 50, abs=0.1)
        assert trade['pnl_pct'] == pytest.approx(449.955 / 400 - 1, rel=0.01)

    def test_trade_records_cost_fields(self):
        """Test that trade records include transaction cost metadata."""
        portfolio = {
            'cash': 100000,
            'holdings': []
        }

        action = {
            'action': 'BUY',
            'symbol': 'SPY',
            'shares': 10,
            'price': 500,
            'health': 0.8
        }

        universe_df = pd.DataFrame({
            'symbol': ['SPY'],
            'asset_class': ['equity'],
            'sector': ['broad']
        })

        trade = execute_trade(portfolio, action, 'calm_uptrend', universe_df)

        # Should have cost fields
        assert 'market_price' in trade
        assert 'transaction_cost_bps' in trade
        assert trade['market_price'] == 500
        assert trade['transaction_cost_bps'] >= 0
        # Cumulative costs tracked in portfolio
        assert 'cumulative_transaction_costs' in portfolio
        assert portfolio['cumulative_transaction_costs'] > 0


class TestUpdatePortfolioValues:
    """Tests for portfolio value updates."""

    def test_updates_market_values(self):
        """Test that holdings get updated market values."""
        portfolio = {
            'cash': 50000,
            'holdings': [
                {
                    'symbol': 'SPY',
                    'shares': 100,
                    'entry_price': 440,
                    'peak_price': 445
                },
                {
                    'symbol': 'QQQ',
                    'shares': 50,
                    'entry_price': 380,
                    'peak_price': 390
                }
            ]
        }

        prices_df = pd.DataFrame({
            'symbol': ['SPY', 'QQQ'],
            'date': [pd.Timestamp.now(), pd.Timestamp.now()],
            'close': [450, 400]
        })

        result = update_portfolio_values(portfolio, prices_df)

        # SPY: 100 * 450 = 45000
        # QQQ: 50 * 400 = 20000
        # Total holdings: 65000
        # Total portfolio: 115000

        assert result['holdings_value'] == 65000
        assert result['portfolio_value'] == 115000
        assert result['holdings'][0]['market_value'] == 45000
        assert result['holdings'][1]['market_value'] == 20000

    def test_updates_unrealized_pnl(self):
        """Test that unrealized P&L is calculated."""
        portfolio = {
            'cash': 0,
            'holdings': [{
                'symbol': 'SPY',
                'shares': 100,
                'entry_price': 440,
                'peak_price': 445
            }]
        }

        prices_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': [pd.Timestamp.now()],
            'close': [450]
        })

        result = update_portfolio_values(portfolio, prices_df)

        holding = result['holdings'][0]
        assert holding['unrealized_pnl'] == 1000  # (450 - 440) * 100
        assert holding['unrealized_pnl_pct'] == pytest.approx(0.0227, rel=0.01)

    def test_updates_peak_price(self):
        """Test that peak price is updated when price rises."""
        portfolio = {
            'cash': 0,
            'holdings': [{
                'symbol': 'SPY',
                'shares': 100,
                'entry_price': 440,
                'peak_price': 445
            }]
        }

        prices_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': [pd.Timestamp.now()],
            'close': [460]  # New high
        })

        result = update_portfolio_values(portfolio, prices_df)

        assert result['holdings'][0]['peak_price'] == 460

    def test_benchmark_dividend_adjusted(self):
        """Test that SPY benchmark uses dividend-adjusted total return."""
        portfolio = {
            'cash': 100000,
            'holdings': [],
            'benchmark_start_price': 450,
            'benchmark_shares': 100000 / 450,
        }

        prices_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': [pd.Timestamp.now()],
            'close': [450]  # Same price — but dividends should increase value
        })

        result = update_portfolio_values(portfolio, prices_df)

        # Benchmark should be slightly above 100000 due to dividend reinvestment
        assert result['benchmark_value'] > 100000
        assert result['benchmark_shares'] > 100000 / 450
