"""Tests for morning executor with broker adapter integration."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.morning_executor import (
    run,
    _execute_via_broker,
    _reconcile_portfolio_from_broker,
    _await_broker_order_update,
    _wait_for_orders_to_settle,
    BROKER_OPEN_ORDER_STATUSES,
    BROKER_TERMINAL_ORDER_STATUSES,
)
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


class DelayedFillBroker(MockBroker):
    """Broker that fills shortly after submit and exposes positions on retry."""

    def __init__(self):
        super().__init__(tradable=True)
        self._get_order_calls = 0
        self._list_positions_calls = 0

    def submit_order(self, symbol='', side='', dollars=None, qty=None,
                     order_type='market', time_in_force='day',
                     client_order_id=None):
        order = super().submit_order(
            symbol=symbol,
            side=side,
            dollars=dollars,
            qty=qty,
            order_type=order_type,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        order['status'] = 'pending_new'
        return order

    def get_order(self, order_id):
        self._get_order_calls += 1
        return {
            'id': order_id,
            'status': 'filled',
            'filled_qty': '9.96',
            'filled_avg_price': '500.0',
            'notional': '4980.0',
        }

    def list_positions(self):
        self._list_positions_calls += 1
        if self._list_positions_calls == 1:
            return []
        return [{
            'symbol': 'SPY',
            'qty': 9.96,
            'market_value': 4980.0,
            'avg_entry_price': 500.0,
            'current_price': 500.0,
            'unrealized_pl': 0.0,
            'side': 'long',
        }]


class SlowFillReportBroker(MockBroker):
    """Broker where order polling never detects the fill, but position appears.

    Simulates the April 6, 2026 failure: order stayed 'pending_new' in the
    polling loop, but broker.list_positions() eventually returned the filled
    position.  Before the fix, expected_buy_symbols excluded 'pending_new'
    buys, so reconciliation never retried and published holdings=[].
    """

    def __init__(self):
        super().__init__(tradable=True)
        self._list_positions_calls = 0

    def submit_order(self, symbol='', side='', dollars=None, qty=None,
                     order_type='market', time_in_force='day',
                     client_order_id=None):
        order = super().submit_order(
            symbol=symbol, side=side, dollars=dollars, qty=qty,
            order_type=order_type, time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        order['status'] = 'pending_new'
        return order

    def get_order(self, order_id):
        # Order polling never sees the fill (simulates API lag / network issue)
        return {'id': order_id, 'status': 'pending_new'}

    def list_positions(self):
        self._list_positions_calls += 1
        if self._list_positions_calls == 1:
            return []
        return [{
            'symbol': 'ARKK',
            'qty': 109.775263,
            'market_value': 7614.91,
            'avg_entry_price': 69.36,
            'current_price': 69.36,
            'unrealized_pl': 0.0,
            'side': 'long',
        }]


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

    def test_sell_floor_truncates_fractional_qty(self):
        """Sell qty must floor-truncate (never round up past available)."""
        broker = MockBroker()
        # Alpaca available: 31.465540746 — round() gives 31.465541 which is OVER
        intent = {
            'symbol': 'SLV',
            'action': 'SELL',
            'shares': 31.465540746,
            'price': 30.0,
        }
        trade = _execute_via_broker(broker, intent, 30.5, '2026-03-19')

        assert len(broker.submitted_orders) == 1
        submitted_qty = float(broker.submitted_orders[0]['qty'])
        # Must be <= available, not rounded up
        assert submitted_qty <= 31.465540746
        assert submitted_qty == 31.46554  # floor to 6 decimals

    def test_sell_dust_position_raises(self):
        """Dust shares (< 0.001 after truncation) should raise ValueError."""
        broker = MockBroker()
        intent = {
            'symbol': 'GLD',
            'action': 'SELL',
            'shares': 4.76e-07,
            'price': 200.0,
        }
        with pytest.raises(ValueError, match="non-positive sell qty"):
            _execute_via_broker(broker, intent, 200.0, '2026-03-19')

    def test_reduce_floor_truncates(self):
        """REDUCE halves qty then floor-truncates."""
        broker = MockBroker()
        intent = {
            'symbol': 'XLU',
            'action': 'REDUCE',
            'shares': 101.491025641,
            'price': 45.0,
        }
        trade = _execute_via_broker(broker, intent, 46.0, '2026-03-19')

        submitted_qty = float(broker.submitted_orders[0]['qty'])
        # 101.491025641 * 0.5 = 50.7455128205; floor to 6 = 50.745512
        assert submitted_qty <= 50.7455128205
        assert submitted_qty == 50.745512

    @patch('src.steps.morning_executor.time.sleep', return_value=None)
    def test_buy_refreshes_broker_status_before_return(self, _mock_sleep):
        broker = DelayedFillBroker()
        intent = {
            'symbol': 'SPY',
            'action': 'BUY',
            'dollars': 4980.0,
            'price': 498.0,
            'shares': 10,
        }

        trade = _execute_via_broker(broker, intent, 500.0, '2026-03-12')

        assert trade['broker_status'] == 'filled'
        assert trade['shares'] == 9.96
        assert trade['dollars'] == 4980.0


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

    def test_pnl_dollar_and_pct_share_one_cost_basis(self):
        """F-4 / F-5 regression: when the broker's avg_entry_price drifts from
        the local entry_price (the VUG post-cutover failure mode), the published
        unrealized_pnl ($) and unrealized_pnl_pct (%) MUST agree on sign and
        on cost basis. The fix uses the local entry_price for both fields."""
        broker = MockBroker()
        # Broker thinks avg_entry is $142.11 (legacy lot from cutover).
        # Local thinks entry is $73.48 (the bot-tracked cost).
        # Current price is $82.77.
        broker._positions = [
            {
                'symbol': 'VUG',
                'qty': 102.766,
                'market_value': 102.766 * 82.77,
                'avg_entry_price': 142.11,
                'current_price': 82.77,
                'unrealized_pl': (82.77 - 142.11) * 102.766,  # broker: -$6098
                'side': 'long',
            }
        ]
        portfolio = {
            'cash': 0,
            'holdings': [
                {'symbol': 'VUG', 'shares': 102.766, 'entry_price': 73.48,
                 'entry_date': '2026-03-01T00:00:00', 'peak_price': 84.0}
            ],
            'portfolio_value': 0,
        }
        result = _reconcile_portfolio_from_broker(broker, portfolio)
        h = result['holdings'][0]
        # Both fields must reflect the local entry_price = 73.48.
        expected_pnl = (82.77 - 73.48) * 102.766
        expected_pct = 82.77 / 73.48 - 1
        assert abs(h['unrealized_pnl'] - expected_pnl) < 0.01
        assert abs(h['unrealized_pnl_pct'] - expected_pct) < 1e-6
        # Sign coherence: $ and % must agree. The pre-fix code published
        # pnl<0 alongside pct>0 in this exact scenario.
        assert (h['unrealized_pnl'] >= 0) == (h['unrealized_pnl_pct'] >= 0)


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

    @patch('src.steps.morning_executor.time.sleep', return_value=None)
    @patch('src.steps.morning_executor.ingest_prices')
    @patch('src.steps.morning_executor.paper_trader')
    def test_broker_mode_retries_reconcile_until_fill_visible(
        self, mock_pt, mock_ip, _mock_sleep
    ):
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

        broker = DelayedFillBroker()

        with patch('src.steps.morning_executor.S3Client', return_value=s3_mock):
            result = run('test-bucket', {'decision_params': {}}, broker=broker)

        assert result['trades'][0]['broker_status'] == 'filled'
        assert len(result['portfolio_state']['holdings']) == 1
        assert result['portfolio_state']['holdings'][0]['symbol'] == 'SPY'

    @patch('src.steps.morning_executor.time.sleep', return_value=None)
    @patch('src.steps.morning_executor.ingest_prices')
    @patch('src.steps.morning_executor.paper_trader')
    def test_pending_new_buy_still_reconciles_position(
        self, mock_pt, mock_ip, _mock_sleep
    ):
        """Regression: order polling returns pending_new, but position appears.

        Before the fix, expected_buy_symbols required broker_status='filled',
        so a pending_new buy bypassed the reconciliation retry.  The single
        list_positions() call returned empty, publishing holdings=[].
        """
        mock_pt.load_portfolio_state.return_value = {
            'cash': 100000,
            'holdings': [],
            'portfolio_value': 100000,
        }
        mock_pt.compute_portfolio_stats.side_effect = lambda p, s: p

        morning_quotes = pd.DataFrame({
            'symbol': ['ARKK'],
            'price': [69.36],
            'timestamp': [datetime.now().isoformat()],
        })
        mock_ip.fetch_morning_quotes.return_value = morning_quotes

        intents = {
            'generated_date': datetime.now().strftime('%Y-%m-%d'),
            'regime': 'risk_on_trend',
            'actions': [{
                'symbol': 'ARKK',
                'action': 'BUY',
                'shares': 109,
                'price': 69.0,
                'dollars': 7614.91,
                'reason': 'SCORE',
            }],
        }

        s3_mock = MagicMock()
        s3_mock.read_json.side_effect = [
            {'intents_date': datetime.now().strftime('%Y-%m-%d')},
            intents,
        ]
        s3_mock.read_csv.return_value = pd.DataFrame({
            'symbol': ['ARKK'],
            'asset_class': ['equity'],
            'sector': ['broad'],
            'eligible': [1],
            'leverage_flag': [0],
        })

        broker = SlowFillReportBroker()

        with patch('src.steps.morning_executor.S3Client', return_value=s3_mock):
            result = run('test-bucket', {'decision_params': {}}, broker=broker)

        # Trade record shows pending_new (order polling never saw the fill)
        assert result['trades'][0]['broker_status'] == 'pending_new'
        # But portfolio MUST have the holding from reconciliation retry
        assert len(result['portfolio_state']['holdings']) == 1
        assert result['portfolio_state']['holdings'][0]['symbol'] == 'ARKK'
        assert result['portfolio_state']['broker_reconciled'] is True


class TestPartialFillHandling:
    """Alpaca `partially_filled` is NOT terminal — the order stays active.

    The May 5 morning execution sold VUG and XRT via SELLs that came back
    `partially_filled` and were treated as terminal. The trader-bot snapshot
    saw the broker mid-fill and locked VUG=28.77 / XRT=73.09 into S3 while
    Alpaca went on to clear the rest to dust. These tests lock the fix:
    `partially_filled` belongs in OPEN, not TERMINAL.
    """

    def test_partially_filled_classified_as_open(self):
        assert 'partially_filled' in BROKER_OPEN_ORDER_STATUSES
        assert 'partially_filled' not in BROKER_TERMINAL_ORDER_STATUSES

    def test_filled_classified_as_terminal(self):
        assert 'filled' in BROKER_TERMINAL_ORDER_STATUSES
        assert 'filled' not in BROKER_OPEN_ORDER_STATUSES

    def test_await_continues_polling_through_partial_fill(self):
        """The await loop must keep polling while status=partially_filled and
        only return when the order reaches a real terminal state."""

        class PartialThenFilledBroker(MockBroker):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def get_order(self, order_id):
                self._calls += 1
                if self._calls < 3:
                    return {'id': order_id, 'status': 'partially_filled',
                            'filled_qty': '5.0', 'qty': '10.0'}
                return {'id': order_id, 'status': 'filled',
                        'filled_qty': '10.0', 'qty': '10.0',
                        'filled_avg_price': '50.0', 'notional': '500.0'}

        broker = PartialThenFilledBroker()
        order = {
            'order_id': 'test-1', 'status': 'pending_new',
            'qty': '10.0', 'notional': '500.0', 'raw': {},
        }

        with patch('src.steps.morning_executor.time.sleep', return_value=None):
            result = _await_broker_order_update(
                broker, order, timeout_sec=2.0, poll_interval_sec=0.001
            )

        assert result['status'] == 'filled'
        assert result['filled_qty'] == '10.0'
        assert broker._calls >= 3

    def test_wait_for_orders_to_settle_blocks_until_clear(self):
        """`_wait_for_orders_to_settle` must keep polling until every submitted
        order leaves the OPEN set. Without it, the post-execution reconcile
        snapshots positions while orders are still working."""

        class WorkingThenSettledBroker(MockBroker):
            def __init__(self):
                super().__init__()
                self._calls = {}

            def get_order(self, order_id):
                self._calls[order_id] = self._calls.get(order_id, 0) + 1
                # First two polls: still working. Third+: filled.
                if self._calls[order_id] < 3:
                    return {'id': order_id, 'status': 'partially_filled'}
                return {'id': order_id, 'status': 'filled'}

        broker = WorkingThenSettledBroker()
        with patch('src.steps.morning_executor.time.sleep', return_value=None):
            settled = _wait_for_orders_to_settle(
                broker, ['o-1', 'o-2'], timeout_sec=2.0, poll_interval_sec=0.001
            )

        assert settled == {'o-1': 'filled', 'o-2': 'filled'}
        assert broker._calls['o-1'] >= 3
        assert broker._calls['o-2'] >= 3

    def test_wait_for_orders_returns_open_status_after_deadline(self):
        """If the deadline elapses with the order still working, return the
        last observed status so the validation_log can flag it. The reconcile
        proceeds on best-effort truth rather than hanging forever."""

        class ForeverWorkingBroker(MockBroker):
            def get_order(self, order_id):
                return {'id': order_id, 'status': 'partially_filled'}

        broker = ForeverWorkingBroker()
        with patch('src.steps.morning_executor.time.sleep', return_value=None):
            settled = _wait_for_orders_to_settle(
                broker, ['o-1'], timeout_sec=0.05, poll_interval_sec=0.001
            )

        assert settled.get('o-1') == 'partially_filled'

    def test_empty_order_list_short_circuits(self):
        broker = MockBroker()
        result = _wait_for_orders_to_settle(broker, [], timeout_sec=10.0)
        assert result == {}


class TestReconcilePreservesHealthCounter:
    """Broker reconciliation must preserve the per-holding
    `consecutive_below_health_days` counter so the sell_health_days persistence
    gate survives across the night → morning → next-night cycle. If the counter
    were dropped on every reconcile, the gate could never accumulate."""

    def test_counter_carried_through_reconcile(self):
        broker = MockBroker()
        broker._positions = [
            {
                'symbol': 'XLK', 'qty': 33.4, 'market_value': 5500.0,
                'avg_entry_price': 135.87, 'current_price': 165.0,
                'unrealized_pl': 970.0, 'side': 'long',
            }
        ]
        portfolio = {
            'cash': 50000,
            'holdings': [{
                'symbol': 'XLK', 'shares': 33.4, 'entry_price': 135.87,
                'entry_date': '2026-04-07', 'peak_price': 162.89,
                'consecutive_below_health_days': 2,
            }],
            'portfolio_value': 0,
        }
        result = _reconcile_portfolio_from_broker(broker, portfolio)
        assert len(result['holdings']) == 1
        assert result['holdings'][0]['consecutive_below_health_days'] == 2

    def test_counter_defaults_to_zero_for_new_position(self):
        """A position the local state hasn't seen yet (e.g. dust filtered out
        previously, broker carried it) starts the counter at 0."""
        broker = MockBroker()
        broker._positions = [
            {
                'symbol': 'NEW', 'qty': 10.0, 'market_value': 1000.0,
                'avg_entry_price': 100.0, 'current_price': 100.0,
                'unrealized_pl': 0.0, 'side': 'long',
            }
        ]
        portfolio = {'cash': 50000, 'holdings': [], 'portfolio_value': 0}
        result = _reconcile_portfolio_from_broker(broker, portfolio)
        assert result['holdings'][0]['consecutive_below_health_days'] == 0
