"""Tests for the paper trader."""

import pytest
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.paper_trader import (
    execute_trade,
    update_portfolio_values
)


class TestExecuteTrade:
    """Tests for trade execution."""

    def test_buy_trade_deducts_cash(self):
        """Test that buy trades deduct cash correctly."""
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

        assert portfolio['cash'] == 55000  # 100000 - (100 * 450)
        assert len(portfolio['holdings']) == 1
        assert portfolio['holdings'][0]['symbol'] == 'SPY'
        assert portfolio['holdings'][0]['shares'] == 100
        assert trade['action'] == 'BUY'
        assert trade['dollars'] == 45000

    def test_sell_trade_adds_cash(self):
        """Test that sell trades add cash correctly."""
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

        assert portfolio['cash'] == 95000  # 50000 + (100 * 450)
        assert len(portfolio['holdings']) == 0
        assert trade['pnl'] == 1000  # (450 - 440) * 100
        assert trade['pnl_pct'] == pytest.approx(0.0227, rel=0.01)

    def test_sell_trade_calculates_pnl(self):
        """Test that P&L is calculated correctly."""
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
            'price': 450,  # 12.5% gain
            'reason': 'PROFIT_TAKE'
        }

        trade = execute_trade(portfolio, action, 'risk_on_trend', pd.DataFrame())

        assert trade['pnl'] == 2500  # (450 - 400) * 50
        assert trade['pnl_pct'] == pytest.approx(0.125, rel=0.01)


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
