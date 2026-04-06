"""Alpaca broker adapter — supports both paper and live endpoints."""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from src.brokers.base import BaseBroker

logger = logging.getLogger(__name__)

PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
LIVE_BASE_URL = 'https://api.alpaca.markets'
DATA_BASE_URL = 'https://data.alpaca.markets'

# Safety defaults
DEFAULT_MAX_ORDER_NOTIONAL = 5000.0  # per-order cap in dollars
DEFAULT_MIN_ORDER_NOTIONAL = 1.0


class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker adapter.

    Args:
        key_id: APCA-API-KEY-ID
        secret_key: APCA-API-SECRET-KEY
        paper: If True use paper endpoint, else live endpoint
        max_order_notional: Per-order dollar cap (safety rail)
        symbol_allowlist: If set, only these symbols may be traded
    """

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        paper: bool = True,
        max_order_notional: float = DEFAULT_MAX_ORDER_NOTIONAL,
        symbol_allowlist: Optional[List[str]] = None,
    ):
        if not key_id or not secret_key:
            raise ValueError("Alpaca key_id and secret_key are required")
        self._key_id = key_id
        self._secret_key = secret_key
        self._base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self._paper = paper
        self._max_order_notional = max_order_notional
        self._symbol_allowlist = set(symbol_allowlist) if symbol_allowlist else None

    # -- internal helpers --

    def _headers(self) -> Dict[str, str]:
        return {
            'APCA-API-KEY-ID': self._key_id,
            'APCA-API-SECRET-KEY': self._secret_key,
            'Content-Type': 'application/json',
        }

    def _request(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> Dict[str, Any]:
        url = f'{self._base_url}{path}'
        data = json.dumps(body).encode() if body else None
        req = Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            error_body = exc.read().decode() if exc.fp else ''
            logger.error(
                "Alpaca API %s %s → %s: %s", method, path, exc.code, error_body
            )
            raise RuntimeError(
                f"Alpaca API error {exc.code} on {method} {path}: {error_body}"
            ) from exc
        except URLError as exc:
            logger.error("Alpaca API connection error on %s %s: %s", method, path, exc)
            raise RuntimeError(f"Alpaca connection error: {exc}") from exc

    def _enforce_allowlist(self, symbol: str) -> None:
        if self._symbol_allowlist is not None and symbol not in self._symbol_allowlist:
            raise ValueError(
                f"Symbol {symbol} not in allowlist: {sorted(self._symbol_allowlist)}"
            )

    @staticmethod
    def make_client_order_id(symbol: str, side: str, run_date: str) -> str:
        """Deterministic client_order_id to prevent duplicate submissions."""
        raw = f"{run_date}|{symbol}|{side}"
        return f"tb-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    # -- BaseBroker interface --

    @property
    def mode_label(self) -> str:
        return 'alpaca_paper' if self._paper else 'alpaca_live'

    def check_account(self) -> Dict[str, Any]:
        raw = self._request('GET', '/v2/account')
        tradable = (
            raw.get('status') == 'ACTIVE'
            and not raw.get('trading_blocked', True)
            and not raw.get('account_blocked', True)
        )
        return {
            'status': raw.get('status'),
            'cash': float(raw.get('cash', 0)),
            'buying_power': float(raw.get('buying_power', 0)),
            'equity': float(raw.get('equity', 0)),
            'tradable': tradable,
            'account_number': raw.get('account_number'),
            'raw': raw,
        }

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
        self._enforce_allowlist(symbol)

        # Determine notional for cap check
        notional_value = dollars or 0
        if qty and not dollars:
            # Can't easily know notional without price; skip cap for qty-based sells
            notional_value = 0

        if notional_value > self._max_order_notional:
            raise ValueError(
                f"Order notional ${notional_value:.2f} exceeds cap "
                f"${self._max_order_notional:.2f}"
            )

        if notional_value > 0 and notional_value < DEFAULT_MIN_ORDER_NOTIONAL:
            raise ValueError(
                f"Order notional ${notional_value:.2f} below minimum "
                f"${DEFAULT_MIN_ORDER_NOTIONAL:.2f}"
            )

        payload: Dict[str, Any] = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force,
        }

        if dollars is not None:
            payload['notional'] = str(round(dollars, 2))
        elif qty is not None:
            payload['qty'] = str(qty)
        else:
            raise ValueError("Either dollars (notional) or qty must be specified")

        if client_order_id:
            payload['client_order_id'] = client_order_id

        logger.info("Submitting order: %s", {k: v for k, v in payload.items()})
        raw = self._request('POST', '/v2/orders', payload)

        return {
            'order_id': raw.get('id'),
            'client_order_id': raw.get('client_order_id'),
            'symbol': raw.get('symbol'),
            'side': raw.get('side'),
            'qty': raw.get('qty'),
            'notional': raw.get('notional'),
            'status': raw.get('status'),
            'submitted_at': raw.get('submitted_at'),
            'raw': raw,
        }

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request('GET', f'/v2/orders/{order_id}')

    def list_positions(self) -> List[Dict[str, Any]]:
        raw_list = self._request('GET', '/v2/positions')
        positions = []
        for p in raw_list:
            positions.append({
                'symbol': p.get('symbol'),
                'qty': float(p.get('qty', 0)),
                'market_value': float(p.get('market_value', 0)),
                'avg_entry_price': float(p.get('avg_entry_price', 0)),
                'current_price': float(p.get('current_price', 0)),
                'unrealized_pl': float(p.get('unrealized_pl', 0)),
                'side': p.get('side', 'long'),
            })
        return positions

    def close_position(self, symbol: str) -> Dict[str, Any]:
        return self._request('DELETE', f'/v2/positions/{symbol}')

    def list_orders(
        self,
        status: str = 'all',
        limit: int = 50,
        after: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = f'?status={status}&limit={limit}'
        if after:
            params += f'&after={after}'
        return self._request('GET', f'/v2/orders{params}')

    def get_snapshots(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch latest stock snapshots from the Alpaca data API.

        Returns a list of dicts with: symbol, price, open, high, low, volume, timestamp.
        Symbols that fail are silently omitted.

        Tries IEX feed first (free tier for paper accounts), then retries
        without a feed parameter to let Alpaca use the account default.
        Falls back to latest-bars endpoint if snapshots return nothing.
        """
        if not symbols:
            return []

        results = self._fetch_snapshots_with_feed(symbols, feed='iex')
        if results:
            return results

        # Retry without explicit feed — lets Alpaca use the account default
        logger.info("IEX snapshots empty, retrying without feed parameter")
        results = self._fetch_snapshots_with_feed(symbols, feed=None)
        if results:
            return results

        # Final fallback: latest bars endpoint
        logger.info("Snapshots empty, falling back to latest bars")
        return self._fetch_latest_bars(symbols)

    def _fetch_snapshots_with_feed(
        self, symbols: List[str], feed: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch snapshots, optionally with a specific feed."""
        sym_param = ','.join(symbols)
        url = f'{DATA_BASE_URL}/v2/stocks/snapshots?symbols={sym_param}'
        if feed:
            url += f'&feed={feed}'
        req = Request(url, headers=self._headers(), method='GET')

        try:
            with urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
        except (HTTPError, URLError) as exc:
            logger.error("Alpaca snapshot error (feed=%s): %s", feed, exc)
            return []

        logger.info(
            "Alpaca snapshots (feed=%s): %d symbols returned for %d requested",
            feed, len(raw), len(symbols),
        )

        results = []
        for symbol, snap in raw.items():
            try:
                daily = snap.get('dailyBar') or {}
                trade = snap.get('latestTrade') or {}
                # Prefer latest trade price, fall back to daily close
                price = float(trade.get('p', 0)) or float(daily.get('c', 0))
                if price <= 0:
                    continue
                results.append({
                    'symbol': symbol,
                    'price': price,
                    'open': float(daily.get('o', price)),
                    'high': float(daily.get('h', price)),
                    'low': float(daily.get('l', price)),
                    'volume': int(daily.get('v', 0)),
                    'timestamp': trade.get('t', daily.get('t', '')),
                })
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Snapshot parse error for %s: %s", symbol, exc)
                continue

        return results

    def _fetch_latest_bars(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fallback: fetch latest daily bars for each symbol."""
        sym_param = ','.join(symbols)
        url = f'{DATA_BASE_URL}/v2/stocks/bars/latest?symbols={sym_param}'
        req = Request(url, headers=self._headers(), method='GET')

        try:
            with urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
        except (HTTPError, URLError) as exc:
            logger.error("Alpaca latest-bars error: %s", exc)
            return []

        bars = raw.get('bars', {})
        logger.info("Alpaca latest-bars: %d symbols returned", len(bars))

        results = []
        for symbol, bar in bars.items():
            try:
                price = float(bar.get('c', 0))
                if price <= 0:
                    continue
                results.append({
                    'symbol': symbol,
                    'price': price,
                    'open': float(bar.get('o', price)),
                    'high': float(bar.get('h', price)),
                    'low': float(bar.get('l', price)),
                    'volume': int(bar.get('v', 0)),
                    'timestamp': bar.get('t', ''),
                })
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Latest-bar parse error for %s: %s", symbol, exc)
                continue

        return results
