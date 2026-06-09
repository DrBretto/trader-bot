"""Simulated broker: the only execution path (pure trading simulation)."""

import logging
from typing import Any, Dict, List

from src.brokers.base import BaseBroker

logger = logging.getLogger(__name__)


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
