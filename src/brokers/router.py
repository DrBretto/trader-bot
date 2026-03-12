"""Broker routing: selects the correct broker adapter based on config."""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from src.brokers.base import BaseBroker

logger = logging.getLogger(__name__)


class BrokerMode(str, Enum):
    SIMULATED = 'simulated'
    ALPACA_PAPER = 'alpaca_paper'
    ALPACA_LIVE = 'alpaca_live'


class SimulatedBroker(BaseBroker):
    """No-op broker that signals the caller to use paper_trader logic."""

    @property
    def mode_label(self) -> str:
        return 'simulated'

    def check_account(self) -> Dict[str, Any]:
        return {
            'status': 'SIMULATED',
            'cash': 0,
            'buying_power': 0,
            'equity': 0,
            'tradable': True,
            'account_number': 'SIMULATED',
            'raw': {},
        }

    def submit_order(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("SimulatedBroker does not submit orders")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        raise NotImplementedError("SimulatedBroker does not track orders")

    def list_positions(self) -> List[Dict[str, Any]]:
        return []

    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError("SimulatedBroker does not manage positions")

    def list_orders(self, **kwargs) -> List[Dict[str, Any]]:
        return []


def resolve_broker_mode(config: Optional[Dict[str, Any]] = None) -> BrokerMode:
    """Determine broker mode from config dict, env var, or default.

    Priority:
    1. config['broker_mode'] if provided
    2. BROKER_MODE env var
    3. Default: simulated
    """
    mode_str = 'simulated'

    if config and config.get('broker_mode'):
        mode_str = config['broker_mode']
    elif os.environ.get('BROKER_MODE'):
        mode_str = os.environ['BROKER_MODE']

    try:
        return BrokerMode(mode_str.lower())
    except ValueError:
        logger.warning("Unknown broker mode '%s', falling back to simulated", mode_str)
        return BrokerMode.SIMULATED


def is_trading_enabled() -> bool:
    """Check the global kill switch. Trading is disabled unless explicitly enabled."""
    return os.environ.get('BROKER_TRADING_ENABLED', 'false').lower() == 'true'


def get_broker(
    config: Optional[Dict[str, Any]] = None,
    alpaca_key_id: str = '',
    alpaca_secret_key: str = '',
) -> BaseBroker:
    """Factory: return the appropriate broker adapter.

    Args:
        config: Pipeline config dict (may contain broker_mode, broker settings)
        alpaca_key_id: Alpaca API key ID
        alpaca_secret_key: Alpaca API secret key

    Returns:
        A BaseBroker implementation
    """
    mode = resolve_broker_mode(config)

    if mode == BrokerMode.SIMULATED:
        logger.info("Broker mode: simulated (paper_trader path)")
        return SimulatedBroker()

    # For Alpaca modes, enforce kill switch
    if not is_trading_enabled():
        logger.warning(
            "Broker mode is %s but BROKER_TRADING_ENABLED is not 'true'. "
            "Falling back to simulated mode.",
            mode.value,
        )
        return SimulatedBroker()

    broker_config = (config or {}).get('broker', {})
    max_order_notional = broker_config.get('max_order_notional', 5000.0)
    symbol_allowlist = broker_config.get('symbol_allowlist')

    if mode == BrokerMode.ALPACA_PAPER:
        from src.brokers.alpaca import AlpacaBroker
        logger.info("Broker mode: alpaca_paper")
        return AlpacaBroker(
            key_id=alpaca_key_id,
            secret_key=alpaca_secret_key,
            paper=True,
            max_order_notional=max_order_notional,
            symbol_allowlist=symbol_allowlist,
        )

    if mode == BrokerMode.ALPACA_LIVE:
        from src.brokers.alpaca import AlpacaBroker
        logger.info("Broker mode: alpaca_live")
        return AlpacaBroker(
            key_id=alpaca_key_id,
            secret_key=alpaca_secret_key,
            paper=False,
            max_order_notional=max_order_notional,
            symbol_allowlist=symbol_allowlist,
        )

    return SimulatedBroker()
