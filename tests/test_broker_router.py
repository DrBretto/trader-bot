"""Tests for broker routing and mode resolution."""

import os
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.brokers.router import (
    resolve_broker_mode,
    get_broker,
    is_trading_enabled,
    BrokerMode,
    SimulatedBroker,
)


class TestResolveBrokerMode:
    """Tests for broker mode resolution."""

    def test_defaults_to_simulated(self):
        assert resolve_broker_mode() == BrokerMode.SIMULATED

    def test_config_takes_priority(self):
        config = {'broker_mode': 'alpaca_paper'}
        assert resolve_broker_mode(config) == BrokerMode.ALPACA_PAPER

    def test_env_var_used_when_no_config(self, monkeypatch):
        monkeypatch.setenv('BROKER_MODE', 'alpaca_live')
        assert resolve_broker_mode() == BrokerMode.ALPACA_LIVE

    def test_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv('BROKER_MODE', 'alpaca_live')
        config = {'broker_mode': 'simulated'}
        assert resolve_broker_mode(config) == BrokerMode.SIMULATED

    def test_unknown_mode_falls_back_to_simulated(self):
        config = {'broker_mode': 'unknown_broker'}
        assert resolve_broker_mode(config) == BrokerMode.SIMULATED

    def test_case_insensitive(self):
        config = {'broker_mode': 'ALPACA_PAPER'}
        assert resolve_broker_mode(config) == BrokerMode.ALPACA_PAPER


class TestIsTradingEnabled:
    """Tests for kill switch."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv('BROKER_TRADING_ENABLED', raising=False)
        assert is_trading_enabled() is False

    def test_disabled_when_false(self, monkeypatch):
        monkeypatch.setenv('BROKER_TRADING_ENABLED', 'false')
        assert is_trading_enabled() is False

    def test_enabled_when_true(self, monkeypatch):
        monkeypatch.setenv('BROKER_TRADING_ENABLED', 'true')
        assert is_trading_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv('BROKER_TRADING_ENABLED', 'TRUE')
        assert is_trading_enabled() is True


class TestGetBroker:
    """Tests for broker factory."""

    def test_simulated_by_default(self):
        broker = get_broker()
        assert isinstance(broker, SimulatedBroker)
        assert broker.mode_label == 'simulated'

    def test_simulated_account_check(self):
        broker = SimulatedBroker()
        acct = broker.check_account()
        assert acct['tradable'] is True
        assert acct['status'] == 'SIMULATED'

    def test_alpaca_without_kill_switch_falls_back(self, monkeypatch):
        monkeypatch.delenv('BROKER_TRADING_ENABLED', raising=False)
        config = {'broker_mode': 'alpaca_paper'}
        broker = get_broker(config, 'key', 'secret')
        # Should fall back to simulated because kill switch is off
        assert isinstance(broker, SimulatedBroker)

    def test_alpaca_with_kill_switch_creates_alpaca(self, monkeypatch):
        monkeypatch.setenv('BROKER_TRADING_ENABLED', 'true')
        config = {'broker_mode': 'alpaca_paper'}
        broker = get_broker(config, 'test-key', 'test-secret')
        assert broker.mode_label == 'alpaca_paper'

    def test_alpaca_live_with_kill_switch(self, monkeypatch):
        monkeypatch.setenv('BROKER_TRADING_ENABLED', 'true')
        config = {'broker_mode': 'alpaca_live'}
        broker = get_broker(config, 'live-key', 'live-secret')
        assert broker.mode_label == 'alpaca_live'

    def test_simulated_broker_submit_raises(self):
        broker = SimulatedBroker()
        with pytest.raises(NotImplementedError):
            broker.submit_order(symbol='SPY', side='buy', dollars=100)
