"""Reconcile post-cutoff fills against Alpaca-truth orders.

Background
----------
Two distinct bugs corrupt the local trades.jsonl ledger and therefore poison
realized-P&L round-trip accounting and the dashboard's trade log:

1. Partial-fill stale snapshot. The morning executor wrote a trade row when
   the broker returned ``status=partially_filled`` and treated that as
   terminal; the order then continued working and the final filled_qty (much
   larger) was never written back. Examples on 2026-04-07 (XLF 47 vs 151.66,
   XRT 12 vs 93.09), 2026-05-04 (XLK 22 vs 55.37, XBI 55 vs 58.94, ARKK 87
   vs 109.78, SLV 112 vs 136.45), 2026-05-05 (VUG 56 vs 102.77, XRT 20 vs
   93.09).

2. Orphan bot-decided fills. A small number of bot-issued orders (e.g.
   2026-04-28 SLV SELL 81.107334 @ $66.34) were filled at Alpaca but the
   write to trades.jsonl was lost. The position vanished from the broker
   while the local ledger still believes it is held.

Fix
---
For every fill loaded from local trades.jsonl, look up the matching Alpaca
order by ``broker_order_id``. If Alpaca shows a different filled qty/price
or a more advanced status, replace the local fields with Alpaca-truth values.
Then inject any post-cutoff Alpaca order that has no local counterpart and
is bot-decided (``client_order_id`` starts with ``tb-`` and is not a
``tb-boot-`` bootstrap or smoke-test order, and is not a synthetic split
BUY).

The cached Alpaca orders live at ``s3://<bucket>/dashboard/alpaca_orders.json``
so the hot path (``compute_canonical_dashboard_metrics`` invoked by Lambda)
does not have to call the Alpaca REST API. ``refresh_alpaca_orders_cache``
fetches and writes the cache; it is invoked once at the top of
``publish_artifacts.run``.

Pre-cutoff fills (paper_trader simulated trades with no ``broker_order_id``)
pass through unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CUTOFF_DATE = "2026-03-12"
ORDERS_CACHE_KEY = "dashboard/alpaca_orders.json"

# Synthetic-split / bootstrap / smoke-test markers we deliberately ignore.
_BOOT_PREFIXES = ("tb-boot-", "smoke-test-")


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_bot_decided(client_order_id: Optional[str]) -> bool:
    """A bot-decided client_order_id is ``tb-<16hex>``.

    The bootstrap script and the smoke test use ``tb-boot-*`` and
    ``smoke-test-*`` respectively; both must be excluded from the canonical
    fill chain because the positions they re-create were already represented
    by pre-cutoff simulated trades.
    """
    if not client_order_id:
        return False
    if any(client_order_id.startswith(p) for p in _BOOT_PREFIXES):
        return False
    return client_order_id.startswith("tb-")


def _is_synthetic_split(order: Dict[str, Any], orders_by_symbol: Dict[str, List[dict]]) -> bool:
    """Identify Alpaca's qty-based synthetic BUY that materializes a split.

    Pattern: ``side=buy``, ``notional`` is null, ``qty`` is exactly an integer
    multiple ``N`` of a prior BUY's ``filled_qty`` for the same symbol, and
    a later qty-based SELL exists whose qty matches ``prior_qty * (N+1)``.
    """
    if order.get("side") != "buy" or order.get("notional"):
        return False
    qty = _coerce_float(order.get("qty"))
    price = _coerce_float(order.get("filled_avg_price"))
    if not qty or not price:
        return False
    sym = order.get("symbol")
    if not sym:
        return False
    same_symbol = orders_by_symbol.get(sym, [])
    submitted = order.get("submitted_at") or ""
    for prior in same_symbol:
        if (prior.get("submitted_at") or "") >= submitted:
            continue
        if prior.get("side") != "buy":
            continue
        p_qty = _coerce_float(prior.get("filled_qty"))
        p_price = _coerce_float(prior.get("filled_avg_price"))
        if not p_qty or not p_price:
            continue
        ratio = qty / p_qty
        for n in range(2, 21):
            if abs(ratio - n) < 0.001:
                # Look for a later SELL of total_post_split shares
                post_split_total = p_qty * (n + 1)
                for later in same_symbol:
                    if (later.get("submitted_at") or "") <= submitted:
                        continue
                    if later.get("side") != "sell":
                        continue
                    l_qty = _coerce_float(later.get("filled_qty"))
                    if l_qty is None:
                        continue
                    if abs(l_qty - post_split_total) < 0.01:
                        return True
                break
    return False


def _index_alpaca_orders(orders: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """By ``id``."""
    return {o["id"]: o for o in orders if o.get("id")}


def _orders_by_symbol(orders: Iterable[Dict[str, Any]]) -> Dict[str, List[dict]]:
    by_sym: Dict[str, List[dict]] = {}
    for o in sorted(
        orders,
        key=lambda x: x.get("filled_at") or x.get("submitted_at") or "",
    ):
        sym = o.get("symbol")
        if not sym:
            continue
        by_sym.setdefault(sym, []).append(o)
    return by_sym


def apply_alpaca_truth_to_fills(
    fills: List[Dict[str, Any]],
    alpaca_orders: List[Dict[str, Any]],
    cutoff_date: str = CUTOFF_DATE,
) -> List[Dict[str, Any]]:
    """Reconcile fills against Alpaca truth.

    Behaviour:

    * Fills WITHOUT ``broker_order_id`` (e.g. paper_trader simulated trades) pass
      through untouched.
    * Fills WITH ``broker_order_id`` matching an Alpaca order: the row's
      ``shares``, ``price``, ``market_price``, ``dollars`` and ``broker_status``
      are replaced with Alpaca-truth values whenever Alpaca's ``filled_qty``
      differs from local ``shares`` by more than 0.5% or the local status is
      not already ``filled``. ``_alpaca_truth_applied`` marker is added.
    * Alpaca orders that have status ``filled``, are post-cutoff, are
      bot-decided (``tb-`` prefix, not ``tb-boot-`` / ``smoke-test``), are not
      synthetic split BUYs, and are not present in the local fills list — are
      injected as new fills with ``_alpaca_only_injection`` marker so FIFO
      round-trip accounting and the trade log reflect them.
    """
    if not alpaca_orders:
        return list(fills)

    by_id = _index_alpaca_orders(alpaca_orders)
    by_symbol = _orders_by_symbol(alpaca_orders)

    out: List[Dict[str, Any]] = []
    seen_order_ids: set[str] = set()

    for fill in fills:
        order_id = fill.get("broker_order_id")
        if not order_id or order_id not in by_id:
            out.append(dict(fill))
            continue
        order = by_id[order_id]
        seen_order_ids.add(order_id)
        if order.get("status") != "filled":
            out.append(dict(fill))
            continue

        a_qty = _coerce_float(order.get("filled_qty")) or 0.0
        a_avg = _coerce_float(order.get("filled_avg_price")) or 0.0
        l_qty = _coerce_float(fill.get("shares")) or 0.0
        l_status = (fill.get("broker_status") or "").lower()
        qty_diff_rel = abs(a_qty - l_qty) / max(l_qty, 1e-9)
        needs_patch = (qty_diff_rel > 0.005) or (l_status not in ("filled", "filled_simulated"))
        if not needs_patch:
            out.append(dict(fill))
            continue

        patched = dict(fill)
        patched["shares"] = a_qty
        patched["price"] = a_avg
        # market_price tracks raw market reference for cost calculation; set it
        # to the filled_avg_price so spread/slippage math stays bps-scale.
        patched["market_price"] = a_avg
        patched["dollars"] = round(a_qty * a_avg, 4)
        patched["broker_status"] = "filled"
        patched["_alpaca_truth_applied"] = True
        patched["_alpaca_truth_local_shares"] = l_qty
        patched["_alpaca_truth_local_price"] = _coerce_float(fill.get("price"))
        out.append(patched)

    # Inject orphan bot-decided fills.
    for order in alpaca_orders:
        oid = order.get("id")
        if not oid or oid in seen_order_ids:
            continue
        if order.get("status") != "filled":
            continue
        filled_at = order.get("filled_at") or order.get("submitted_at") or ""
        if filled_at[:10] < cutoff_date:
            continue
        if not _is_bot_decided(order.get("client_order_id")):
            continue
        if _is_synthetic_split(order, by_symbol):
            continue
        # Build a fill row in the same shape as a local trade.
        a_qty = _coerce_float(order.get("filled_qty")) or 0.0
        a_avg = _coerce_float(order.get("filled_avg_price")) or 0.0
        out.append({
            "timestamp": filled_at,
            "symbol": order.get("symbol", ""),
            "action": (order.get("side") or "").upper(),
            "shares": a_qty,
            "price": a_avg,
            "market_price": a_avg,
            "dollars": round(a_qty * a_avg, 4),
            "reason": "ALPACA_TRUTH_INJECTED",
            "broker_order_id": oid,
            "broker_client_order_id": order.get("client_order_id"),
            "broker_status": "filled",
            "execution_mode": "alpaca_paper",
            "_alpaca_only_injection": True,
        })

    return out


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def load_alpaca_orders_cache(s3) -> List[Dict[str, Any]]:
    """Read cached Alpaca orders from S3. Returns empty list when missing."""
    try:
        cached = s3.read_json(ORDERS_CACHE_KEY)
    except Exception as exc:
        logger.info("alpaca orders cache unavailable: %s", exc)
        return []
    if isinstance(cached, list):
        return cached
    if isinstance(cached, dict) and "orders" in cached:
        return cached["orders"]
    return []


def refresh_alpaca_orders_cache(s3, broker, cutoff_date: str = CUTOFF_DATE) -> List[Dict[str, Any]]:
    """Pull every Alpaca order >= ``cutoff_date`` and write to S3 cache.

    Pagination matches the public Alpaca contract: ``status=all``, descending
    by ``submitted_at``, ``until`` walks backwards until the cutoff is
    reached. ``broker`` must expose ``list_orders(status, limit, after)`` —
    the public ``AlpacaBroker`` class satisfies this.
    """
    if broker is None:
        return load_alpaca_orders_cache(s3)

    seen: Dict[str, Dict[str, Any]] = {}
    until: Optional[str] = None
    page_size = 500
    cutoff_ts = f"{cutoff_date}T00:00:00Z"
    for _ in range(60):  # safety bound; 60 pages * 500 = 30k orders
        params = {
            "status": "all",
            "limit": page_size,
            "direction": "desc",
            "nested": "true",
        }
        if until:
            params["until"] = until
        try:
            page = broker._request("GET", "/v2/orders" + _qs(params))
        except Exception as exc:
            logger.warning("alpaca orders page fetch failed (%s); using existing cache", exc)
            return load_alpaca_orders_cache(s3)
        if not isinstance(page, list) or not page:
            break
        new_in_page = 0
        earliest = until
        for o in page:
            oid = o.get("id")
            if not oid or oid in seen:
                continue
            seen[oid] = o
            new_in_page += 1
            sub = o.get("submitted_at")
            if sub and (earliest is None or sub < earliest):
                earliest = sub
        # Stop when the entire page is older than cutoff
        if all((o.get("submitted_at") or "") < cutoff_ts for o in page):
            break
        if len(page) < page_size or new_in_page == 0:
            break
        until = earliest

    orders = sorted(
        seen.values(),
        key=lambda x: x.get("filled_at") or x.get("submitted_at") or "",
    )
    try:
        s3.write_json(orders, ORDERS_CACHE_KEY)
        logger.info("Refreshed alpaca orders cache: %d orders → %s", len(orders), ORDERS_CACHE_KEY)
    except Exception as exc:
        logger.warning("Failed to write alpaca orders cache: %s", exc)
    return orders


def _qs(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    from urllib.parse import urlencode
    return "?" + urlencode({k: v for k, v in params.items() if v is not None})
