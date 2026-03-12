"""Tests for morning executor with broker adapter integration."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.morning_executor import run, _execute_via_broker, _reconcile_portfolio_from_broker
from src.brokers.base import BaseBroker
from src.brokers.router import SimulatedBroker


class MockBroker(BaseBroker):
    """Test broker that records calls."""

    def __init__(self, tradable=True):
        self._tradable = tradable
        self.submitted_orders = []
        self._positions = []

    @property
    def mode_label(self):
        return 'mock_broker'

    def check_account(self):
        return {
            'status': 'ACTIVE',
            'cash': 50000.0,
            'buying_power': 100000.0,
            'equity': 75000.0,
            'tradable': self._tradable,
            'account_number': 'MOCK-123',
            'raw': {},
        }

    def submit_order(self, symbol='', side='', dollars=None, qty=None,
                     order_type='market', time_in_force='day',
                     client_order_id=None):
        order = {
            'order_id': f'mock-{len(self.submitted_orders)}',
            'client_order_id': client_order_id or 'mock-coid',
            'symbol': symbol,
            'side': side,
            'qty': str(qty) if qty else None,
            'notional': str(dollars) if dollars else None,
            'status': 'accepted',
            'submitted_at': datetime.now().isoformat(),
            'raw': {},
        }
        self.submitted_orders.append(order)
        return order

    def get_order(self, order_id):
        return {'id': order_id, 'status': 'filled'}

    def list_positions(self):
        return self._positions

    def close_position(self, symbol):
        return {'status': 'closed', 'symbol': symbol}

    def list_orders(self, status='all', limit=50, after=None):
        return self.submitted_orders


class TestExecuteViaBroker:
    """Tests for the broker execution helper."""

    def test_buy_uses_notional(self):
        broker = MockBroker()
        intent = {
            'symbol': 'SPY',
            'action': 'BUY',
            'dollars': 500.0,
            'price': 490.0,
            'shares': 1,
        }
        trade = _execute_via_broker(broker, intent, 495.0, '2026-03-12')

        assert len(broker.submitted_orders) == 1
        assert broker.submitted_orders[0]['side'] == 'buy'
        assert broker.submitted_orders[0]['notional'] == '500.0'
        assert trade['execution_mode'] == 'mock_broker'
        assert trade['broker_status'] == 'accepted'

    def test_sell_uses_qty(self):
        broker = MockBroker()
        intent = {
            'symbol': 'SPY',
            'action': 'SELL',
            'shares': 10,
            'price': 490.0,
        }
        trade = _execute_via_broker(broker, intent, 495.0, '2026-03-12')

        assert len(broker.submitted_orders) == 1
        assert broker.submitted_orders[0]['side'] == 'sell'
        assert broker.submitted_orders[0]['qty'] == '10.0'


class TestReconcilePortfolio:
    """Tests for broker reconciliation."""

    def test_reconciles_from_broker(self):
        broker = MockBroker()
        broker._positions = [
            {
                'symbol': 'SPY',
                'qty': 5.5,
                'market_value': 2750.0,
                'avg_entry_price': 490.0,
                'current_price': 500.0,
                'unrealized_pl': 55.0,
                'side': 'long',
            }
        ]

        portfolio = {
            'cash': 90000,
            'holdings': [],
            'portfolio_value': 90000,
        }

        result = _reconcile_portfolio_from_broker(broker, portfolio)

        assert result['cash'] == 50000.0  # from broker account
        assert len(result['holdings']) == 1
        assert result['holdings'][0]['symbol'] == 'SPY'
        assert result['holdings'][0]['shares'] == 5.5
        assert result['broker_reconciled'] is True


class TestMorningRunWithBroker:
    """Integration test for morning run with broker adapter."""

    @patch('src.steps.morning_executor.ingest_prices')
    @patch('src.steps.morning_executor.paper_trader')
    def test_simulated_broker_uses_paper_trader(self, mock_pt, mock_ip):
        """SimulatedBroker should use paper_trader path (backward compat)."""
        mock_pt.load_portfolio_state.return_value = {
            'cash': 100000,
            'holdings': [],
            'portfolio_value': 100000,
        }
        mock_pt.compute_portfolio_stats.side_effect = lambda p, s: p

        # No intents
        s3_mock = MagicMock()
        s3_mock.read_json.return_value = None

        with patch('src.steps.morning_executor.S3Client', return_value=s3_mock):
            result = run('test-bucket', {}, broker=SimulatedBroker())

        assert result['intents_found'] is False
        assert result['trades'] == []

    @patch('src.steps.morning_executor.ingest_prices')
    @patch('src.steps.morning_executor.paper_trader')
    def test_broker_mode_submits_orders(self, mock_pt, mock_ip):
        """Real broker adapter should submit orders."""
        mock_pt.load_portfolio_state.return_value = {
            'cash': 100000,
            'holdings': [],
            'portfolio_value': 100000,
        }
        mock_pt.compute_portfolio_stats.side_effect = lambda p, s: p

        morning_quotes = pd.DataFrame({
            'symbol': ['SPY'],
            'price': [500.0],
            'timestamp': [datetime.now().isoformat()],
        })
        mock_ip.fetch_morning_quotes.return_value = morning_quotes

        intents = {
            'generated_date': datetime.now().strftime('%Y-%m-%d'),
            'regime': 'risk_on_trend',
            'actions': [{
                'symbol': 'SPY',
                'action': 'BUY',
                'shares': 10,
                'price': 498.0,
                'dollars': 4980.0,
                'reason': 'SCORE',
            }],
        }

        s3_mock = MagicMock()
        s3_mock.read_json.side_effect = [
            {'intents_date': datetime.now().strftime('%Y-%m-%d')},  # latest.json
            intents,  # trade_intents.json
        ]
        s3_mock.read_csv.return_value = pd.DataFrame({
            'symbol': ['SPY'],
            'asset_class': ['equity'],
            'sector': ['broad'],
            'eligible': [1],
            'leverage_flag': [0],
        })

        broker = MockBroker()
        broker._positions = [{
            'symbol': 'SPY',
            'qty': 9.96,
            'market_value': 4980.0,
            'avg_entry_price': 500.0,
            'current_price': 500.0,
            'unrealized_pl': 0.0,
            'side': 'long',
        }]

        with patch('src.steps.morning_executor.S3Client', return_value=s3_mock):
            result = run('test-bucket', {'decision_params': {}}, broker=broker)

        assert result['intents_executed'] == 1
        assert result['execution_mode'] == 'mock_broker'
        assert len(broker.submitted_orders) == 1
        assert broker.submitted_orders[0]['side'] == 'buy'

    @patch('src.steps.morning_executor.ingest_prices')
    @patch('src.steps.morning_executor.paper_trader')
    def test_untradable_account_falls_back(self, mock_pt, mock_ip):
        """If account is not tradable, should abort broker mode."""
        mock_pt.load_portfolio_state.return_value = {
            'cash': 100000,
            'holdings': [],
            'portfolio_value': 100000,
        }
        mock_pt.compute_portfolio_stats.side_effect = lambda p, s: p

        morning_quotes = pd.DataFrame({
            'symbol': ['SPY'],
            'price': [500.0],
            'timestamp': [datetime.now().isoformat()],
        })
        mock_ip.fetch_morning_quotes.return_value = morning_quotes

        intents = {
            'generated_date': datetime.now().strftime('%Y-%m-%d'),
            'regime': 'risk_on_trend',
            'actions': [{
                'symbol': 'SPY',
                'action': 'BUY',
                'shares': 10,
                'price': 498.0,
                'dollars': 4980.0,
            }],
        }

        s3_mock = MagicMock()
        s3_mock.read_json.side_effect = [
            {'intents_date': datetime.now().strftime('%Y-%m-%d')},
            intents,
        ]
        s3_mock.read_csv.return_value = pd.DataFrame({
            'symbol': ['SPY'],
            'asset_class': ['equity'],
            'sector': ['broad'],
            'eligible': [1],
            'leverage_flag': [0],
        })

        broker = MockBroker(tradable=False)

        with patch('src.steps.morning_executor.S3Client', return_value=s3_mock):
            result = run('test-bucket', {'decision_params': {}}, broker=broker)

        # Should fall back to paper_trader (simulated), not submit broker orders
        assert len(broker.submitted_orders) == 0
        assert any('not tradable' in msg for msg in result['validation_log'])
