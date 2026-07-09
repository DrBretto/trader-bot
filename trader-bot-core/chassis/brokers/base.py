"""Abstract base class for broker adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseBroker(ABC):
    """Interface that all broker adapters must implement."""

    @abstractmethod
    def check_account(self) -> Dict[str, Any]:
        """Return account status, cash, buying power, and whether trading is allowed.

        Returns:
            Dict with keys: status, cash, buying_power, equity, tradable (bool),
                            account_number, raw (full API response)
        """

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        dollars: Optional[float] = None,
        qty: Optional[float] = None,
        order_type: str = 'market',
        time_in_force: str = 'day',
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a single order.

        For buys, prefer ``dollars`` (notional) for fractional support.
        For sells, prefer ``qty`` (shares held).

        Returns:
            Dict with keys: order_id, client_order_id, symbol, side, qty, notional,
                            status, submitted_at, raw
        """

    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch order status by broker order ID."""

    @abstractmethod
    def list_positions(self) -> List[Dict[str, Any]]:
        """Return current positions.

        Each position dict has: symbol, qty, market_value, avg_entry_price,
                                current_price, unrealized_pl, side
        """

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close entire position for a symbol. Returns order dict."""

    @abstractmethod
    def list_orders(
        self,
        status: str = 'all',
        limit: int = 50,
        after: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List recent orders for reconciliation."""

    @property
    @abstractmethod
    def mode_label(self) -> str:
        """Human-readable mode label (e.g. 'simulated')."""

    def get_snapshots(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch latest price snapshots for the given symbols.

        Returns a list of dicts with: symbol, price, open, high, low, volume, timestamp.
        Default implementation returns empty list (subclasses override).
        """
        return []
