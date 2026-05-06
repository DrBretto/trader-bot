"""Reproducible historical corrections for accounting + replay paths.

This module holds the data-driven knowledge needed to undo bugs and
real-world events that contaminate historical artifacts (trade records,
portfolio_state files) so downstream computations like realized-P&L
round-trip accounting and the dashboard equity curve render the truth
of what the strategy actually accomplished.

The corrections live as data, not code: each entry is auditable, dated,
and a future engineer can append more without changing computation
shape. Applied uniformly through `apply_split_corrections_to_fills` and
`split_adjust_holding`.

Currently encoded:

* VUG 6:1 effective basis adjustment at 2026-04-22. Mechanism:
  Alpaca's paper-trading account did not cleanly apply VUG's stock
  split, so the simulated lot from 2026-04-07 carried the pre-split
  price ($441.05, 17.127628206 sh, $7,553 basis) until 2026-04-22,
  when a portfolio-state reconciliation event re-stated the holding
  at the broker-truth post-split basis ($73.48, 102.765769236 sh,
  $7,551 basis). The ratio is exactly 6.0 and the dollar basis is
  preserved across the adjustment. Without this correction, the
  May 5 56-share SELL at $84.38 matches against the pre-split lot
  and produces a phantom -$6,063 realized loss instead of the real
  ~+$609 gain.

  Note on dates: 2026-04-22 is when the position-state correction
  shows up in our data, NOT when Alpaca paper was first cut over.
  The Alpaca paper cutover was 2026-03-12 (see config/chart_markers.json
  and the 2026-03-12 first continuity-bridge cashflow of -$3,025.53).
  2026-04-22 is the SECOND continuity event — the cumulative external
  cashflow stepped from -$3,025.53 to -$10,077.83 and VUG was the
  largest contributor.

Anchor evidence:

* daily/2026-04-08/portfolio_state.json: VUG shares=17.127628206
  entry_price=440.89
* daily/2026-04-22/portfolio_state.json: VUG shares=102.765769236
  entry_price=73.48 (ratio 6.0001)
* config/chart_markers.json: 2026-03-12 = Alpaca paper cutover;
  2026-04-22 = second continuity cashflow event (cumulative external
  cashflow stepped from -$3,025.53 to -$10,077.83).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SplitEvent:
    """A symbol's effective basis adjustment.

    `ratio` follows the convention shares_post = shares_pre * ratio
    and price_post = price_pre / ratio. So a 2-for-1 split is ratio=2.0;
    a reverse 1-for-2 is ratio=0.5.
    """

    symbol: str
    ex_date: str  # YYYY-MM-DD; events on or after this date use post-split basis
    ratio: float
    note: str = ""


# Append-only list. Order does not matter — corrections are looked up by symbol.
STOCK_SPLITS: List[SplitEvent] = [
    SplitEvent(
        symbol="VUG",
        ex_date="2026-04-22",
        ratio=6.0,
        note=(
            "Alpaca paper account did not apply VUG's stock split cleanly. "
            "Apr 7 paper buy recorded at $441.05 (17.127 sh) until Apr 22, "
            "when a portfolio-state reconciliation re-stated the lot at "
            "post-split basis (102.766 sh @ $73.48). Ratio exactly 6.0; "
            "dollar basis preserved. Round-trip P&L must use the post-split "
            "basis. (Note: Apr 22 is the second continuity-cashflow event "
            "date, not the cutover; cutover was Mar 12.)"
        ),
    ),
]


def _splits_for(symbol: str) -> List[SplitEvent]:
    return [s for s in STOCK_SPLITS if s.symbol == symbol]


def split_adjust_basis(
    symbol: str,
    fill_date: Optional[str],
    shares: float,
    price: float,
) -> Dict[str, float]:
    """Apply every split that occurred AFTER fill_date to (shares, price).

    Returns a dict of (shares, price, ratio_applied) reflecting the post-split
    basis a sell on or after the latest split date should be matched against.

    Pre-split fills get scaled up; post-split fills pass through unchanged.
    Sells that happened BEFORE a later split also get adjusted (their basis
    is the contemporary basis, but we adjust here to match against post-split
    sells in FIFO).
    """
    if shares == 0 or price == 0:
        return {"shares": shares, "price": price, "ratio_applied": 1.0}

    cumulative_ratio = 1.0
    for event in _splits_for(symbol):
        if not fill_date:
            cumulative_ratio *= event.ratio
            continue
        if fill_date < event.ex_date:
            cumulative_ratio *= event.ratio

    return {
        "shares": shares * cumulative_ratio,
        "price": price / cumulative_ratio if cumulative_ratio != 0 else price,
        "ratio_applied": cumulative_ratio,
    }


def apply_split_corrections_to_fills(
    fills: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a new fill list with split-adjusted shares/price for every fill.

    The original fill dicts are not mutated. A `_split_ratio_applied` key is
    added to each adjusted fill for audit traceability — downstream code can
    detect that a correction was applied and surface it if needed.

    Both BUYs and SELLs are adjusted by the cumulative ratio of splits that
    occurred AFTER the fill date. This keeps round-trip FIFO matching
    coherent: a pre-split BUY of N shares becomes N*ratio post-adjusted
    shares matched against a post-split SELL of M shares (also at the
    post-split basis) without the matching itself needing to be split-aware.
    """
    out: List[Dict[str, Any]] = []
    for fill in fills:
        symbol = str(fill.get("symbol", ""))
        if not symbol or not _splits_for(symbol):
            out.append(dict(fill))
            continue

        try:
            shares = float(fill.get("shares", 0) or 0)
        except (TypeError, ValueError):
            shares = 0.0
        try:
            price = float(fill.get("price", 0) or 0)
        except (TypeError, ValueError):
            price = 0.0

        fill_date = (
            fill.get("_trade_date")
            or (fill.get("timestamp", "")[:10] if fill.get("timestamp") else None)
        )

        adj = split_adjust_basis(symbol, fill_date, shares, price)
        ratio = adj["ratio_applied"]
        if ratio == 1.0:
            out.append(dict(fill))
            continue

        adjusted = dict(fill)
        adjusted["shares"] = adj["shares"]
        adjusted["price"] = adj["price"]
        # `market_price` rides on the same basis as `price` for split adjustment.
        # _fill_cost_dollars computes spread/slippage as
        # abs(fill_price - market_price) * shares — leaving market_price at the
        # pre-split scale while price gets adjusted produces a phantom spread
        # equal to (price_pre - price_post) * shares_post (e.g. $37k of "spread"
        # on a VUG fill). Adjust both with the same ratio so the spread math
        # stays bps-scale.
        market_price_raw = fill.get("market_price")
        if market_price_raw is not None:
            try:
                mp = float(market_price_raw)
                if ratio != 0:
                    adjusted["market_price"] = mp / ratio
            except (TypeError, ValueError):
                pass
        # Preserve dollar amount within rounding (sanity check downstream).
        try:
            original_dollars = float(fill.get("dollars", 0) or 0)
        except (TypeError, ValueError):
            original_dollars = 0.0
        if original_dollars > 0:
            adjusted["dollars"] = adj["shares"] * adj["price"]
        adjusted["_split_ratio_applied"] = ratio
        adjusted["_pre_split_shares"] = shares
        adjusted["_pre_split_price"] = price
        out.append(adjusted)

    return out


def split_adjust_holding(
    symbol: str,
    as_of_date: str,
    shares: float,
    entry_price: float,
    entry_date: Optional[str] = None,
) -> Dict[str, float]:
    """Convenience for adjusting a holding's recorded basis to as_of_date.

    Used by historical equity-curve reconstruction to translate a pre-split
    holding snapshot into its post-split basis when comparing against later
    market values.
    """
    if entry_date is None:
        entry_date = "1900-01-01"

    cumulative_ratio = 1.0
    for event in _splits_for(symbol):
        if entry_date < event.ex_date <= as_of_date:
            cumulative_ratio *= event.ratio

    return {
        "shares": shares * cumulative_ratio,
        "entry_price": entry_price / cumulative_ratio if cumulative_ratio != 0 else entry_price,
        "ratio_applied": cumulative_ratio,
    }
