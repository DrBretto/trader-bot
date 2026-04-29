"""Tests for Alpaca broker adapter with mocked HTTP."""

import json
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.brokers.alpaca import AlpacaBroker, PAPER_BASE_URL, LIVE_BASE_URL


def _mock_urlopen(response_data, status=200):
    """Create a mock urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestAlpacaBrokerInit:
    """Tests for broker initialization."""

    def test_paper_mode_uses_paper_url(self):
        broker = AlpacaBroker('key', 'secret', paper=True)
        assert broker._base_url == PAPER_BASE_URL
        assert broker.mode_label == 'alpaca_paper'

    def test_live_mode_uses_live_url(self):
        broker = AlpacaBroker('key', 'secret', paper=False)
        assert broker._base_url == LIVE_BASE_URL
        assert broker.mode_label == 'alpaca_live'

    def test_missing_credentials_raises(self):
        with pytest.raises(ValueError, match="required"):
            AlpacaBroker('', 'secret')
        with pytest.raises(ValueError, match="required"):
            AlpacaBroker('key', '')


class TestAlpacaBrokerClientOrderId:
    """Tests for deterministic order ID generation."""

    def test_deterministic(self):
        id1 = AlpacaBroker.make_client_order_id('SPY', 'buy', '2026-03-12')
        id2 = AlpacaBroker.make_client_order_id('SPY', 'buy', '2026-03-12')
        assert id1 == id2
        assert id1.startswith('tb-')

    def test_different_inputs_different_ids(self):
        id1 = AlpacaBroker.make_client_order_id('SPY', 'buy', '2026-03-12')
        id2 = AlpacaBroker.make_client_order_id('QQQ', 'buy', '2026-03-12')
        id3 = AlpacaBroker.make_client_order_id('SPY', 'sell', '2026-03-12')
        assert id1 != id2
        assert id1 != id3


class TestAlpacaBrokerAllowlist:
    """Tests for symbol allowlist enforcement."""

    def test_allowlist_blocks_disallowed(self):
        broker = AlpacaBroker('key', 'secret', symbol_allowlist=['SPY', 'QQQ'])
        with pytest.raises(ValueError, match="not in allowlist"):
            broker._enforce_allowlist('TLT')

    def test_allowlist_permits_allowed(self):
        broker = AlpacaBroker('key', 'secret', symbol_allowlist=['SPY', 'QQQ'])
        broker._enforce_allowlist('SPY')  # should not raise

    def test_no_allowlist_permits_all(self):
        broker = AlpacaBroker('key', 'secret')
        broker._enforce_allowlist('ANY_SYMBOL')  # should not raise


class TestAlpacaBrokerAccountCheck:
    """Tests for account health check."""

    @patch('src.brokers.alpaca.urlopen')
    def test_account_check_success(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({
            'status': 'ACTIVE',
            'cash': '50000.00',
            'buying_power': '100000.00',
            'equity': '75000.00',
            'trading_blocked': False,
            'account_blocked': False,
            'account_number': '123456',
        })

        broker = AlpacaBroker('key', 'secret')
        acct = broker.check_account()

        assert acct['status'] == 'ACTIVE'
        assert acct['cash'] == 50000.0
        assert acct['buying_power'] == 100000.0
        assert acct['tradable'] is True

    @patch('src.brokers.alpaca.urlopen')
    def test_account_blocked_not_tradable(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({
            'status': 'ACTIVE',
            'cash': '50000.00',
            'buying_power': '100000.00',
            'equity': '75000.00',
            'trading_blocked': True,
            'account_blocked': False,
            'account_number': '123456',
        })

        broker = AlpacaBroker('key', 'secret')
        acct = broker.check_account()
        assert acct['tradable'] is False


class TestAlpacaBrokerSubmitOrder:
    """Tests for order submission."""

    @patch('src.brokers.alpaca.urlopen')
    def test_notional_buy_order(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({
            'id': 'order-123',
            'client_order_id': 'tb-abc',
            'symbol': 'SPY',
            'side': 'buy',
            'qty': None,
            'notional': '500.00',
            'status': 'accepted',
            'submitted_at': '2026-03-12T10:00:00Z',
        })

        broker = AlpacaBroker('key', 'secret', max_order_notional=1000)
        result = broker.submit_order(
            symbol='SPY', side='buy', dollars=500.0,
            client_order_id='tb-abc'
        )

        assert result['order_id'] == 'order-123'
        assert result['status'] == 'accepted'
        assert result['notional'] == '500.00'

    @patch('src.brokers.alpaca.urlopen')
    def test_qty_sell_order(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({
            'id': 'order-456',
            'client_order_id': 'tb-def',
            'symbol': 'SPY',
            'side': 'sell',
            'qty': '10',
            'notional': None,
            'status': 'accepted',
            'submitted_at': '2026-03-12T10:00:00Z',
        })

        broker = AlpacaBroker('key', 'secret')
        result = broker.submit_order(
            symbol='SPY', side='sell', qty=10.0,
            client_order_id='tb-def'
        )

        assert result['order_id'] == 'order-456'
        assert result['qty'] == '10'

    def test_max_notional_cap_enforced(self):
        broker = AlpacaBroker('key', 'secret', max_order_notional=1000)
        with pytest.raises(ValueError, match="exceeds cap"):
            broker.submit_order(symbol='SPY', side='buy', dollars=1500.0)

    def test_min_notional_enforced(self):
        broker = AlpacaBroker('key', 'secret')
        with pytest.raises(ValueError, match="below minimum"):
            broker.submit_order(symbol='SPY', side='buy', dollars=0.50)

    def test_no_dollars_or_qty_raises(self):
        broker = AlpacaBroker('key', 'secret')
        with pytest.raises(ValueError, match="Either dollars"):
            broker.submit_order(symbol='SPY', side='buy')

    def test_allowlist_enforced_on_submit(self):
        broker = AlpacaBroker('key', 'secret', symbol_allowlist=['QQQ'])
        with pytest.raises(ValueError, match="not in allowlist"):
            broker.submit_order(symbol='SPY', side='buy', dollars=100)


class TestAlpacaBrokerPositions:
    """Tests for position listing."""

    @patch('src.brokers.alpaca.urlopen')
    def test_list_positions(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen([
            {
                'symbol': 'SPY',
                'qty': '10.5',
                'market_value': '5250.00',
                'avg_entry_price': '490.00',
                'current_price': '500.00',
                'unrealized_pl': '105.00',
                'side': 'long',
            }
        ])

        broker = AlpacaBroker('key', 'secret')
        positions = broker.list_positions()

        assert len(positions) == 1
        assert positions[0]['symbol'] == 'SPY'
        assert positions[0]['qty'] == 10.5
        assert positions[0]['market_value'] == 5250.0

    @patch('src.brokers.alpaca.urlopen')
    def test_empty_positions(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen([])
        broker = AlpacaBroker('key', 'secret')
        assert broker.list_positions() == []
