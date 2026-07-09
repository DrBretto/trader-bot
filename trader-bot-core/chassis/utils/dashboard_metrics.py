"""Canonical dashboard metrics computed from a single daily snapshot series."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, stdev
from typing import Any, Deque, Dict, List, Optional, Tuple

from chassis.utils.historical_corrections import apply_split_corrections_to_fills


def sanitize_nan_for_json(obj):
    """Recursively replace NaN/inf floats with None so the dashboard always
    serializes (json.dumps with allow_nan=False rejects them: "Out of range float
    values are not JSON compliant"). PKT-TB-012: NaN can enter from incomplete
    bars / degraded chassis data; the surface must still publish rather than crash
    the whole dashboard. Returns a sanitized copy."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_nan_for_json(v) for v in obj]
    return obj


def attach_new_brain_surface(
    dashboard_data: Dict[str, Any],
    shadow_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Carry the standing exposure-stripped FORECAST-RUNG rent + CI onto the
    live primary-line surface (PKT-TB-012; Skeptic closing condition 1 / Attack 1).

    The live New Brain line must not read as "a market-driven equity curve": a
    viewer has to see the brain's selection contribution — the multi-factor
    exposure-stripped forecast-rung rent (F-R), a zero-straddling band — right on
    the surface. Also surfaces the certified IC skill receipt, the U-E universe
    rung, the three-valued verdict + BH-FDR state, and the ``forward_confirmed:
    false`` tag. Reads the PKT-TB-011 rent ladder from the shadow payload; never
    computes rent here. Idempotent + defensive: a missing/!=v2 payload leaves a
    'pending' block so the surface still carries the standing-fact framing.

    None of this asserts a dollar edge: the stripped residual is measured at zero.
    """
    metrics = dashboard_data.setdefault("metrics", {})
    block: Dict[str, Any] = {
        "forward_confirmed": False,
        "forward_confirmed_note": (
            "Every New Brain go-live universe name is forward_confirmed:false — a "
            "strong in-sample prior to falsify, tilted live before any forward "
            "fold confirms it (DESIGN_DOSSIER Attack 2)."),
        "non_assertion": (
            "Dollar conversion is measured forward, pre-registered, never an "
            "in-sample replay. The exposure-stripped selection residual is "
            "currently measured at zero (LIVE_PREREG.md). No surface implies the "
            "live brain has a dollar edge."),
        "live_prereg": "LIVE_PREREG.md (the forward read contract)",
    }

    stats = (shadow_payload or {}).get("stats", {}) if shadow_payload else {}
    organ_ledger = (
        (shadow_payload or {}).get("organ_ledger")
        or stats.get("organ_ledger")
        or []
    )
    forecast_rung = next(
        (r for r in organ_ledger if r.get("component") == "forecast"), None)
    universe_rung = next(
        (r for r in organ_ledger if r.get("component") == "universe"), None)
    forecast_leg = (
        (shadow_payload or {}).get("forecast_leg")
        or stats.get("forecast_leg")
        or {})

    if forecast_rung is not None:
        block["forecast_rung_rent"] = {
            "stripped_bp_day": forecast_rung.get("stripped_bp_day"),
            "stripped_ci": forecast_rung.get("stripped_ci"),
            "gross_bp_day": forecast_rung.get("gross_bp_day"),
            "t": forecast_rung.get("t"),
            "n_days": forecast_rung.get("n_days"),
            "verdict": forecast_rung.get("verdict"),
            "fdr_survivor": forecast_rung.get("fdr_survivor"),
            "book_pair": forecast_rung.get("book_pair"),
            "caveat": forecast_rung.get("caveat"),
            "label": "Forecast-rung rent (exposure-stripped, F-R) bp/day",
        }
    else:
        block["forecast_rung_rent"] = {
            "status": "accruing",
            "label": "Forecast-rung rent (exposure-stripped, F-R) bp/day",
            "note": "shadow rent ladder not yet available (armed; accruing).",
        }

    if universe_rung is not None:
        block["universe_rung"] = {
            "stripped_bp_day": universe_rung.get("stripped_bp_day"),
            "stripped_ci": universe_rung.get("stripped_ci"),
            "verdict": universe_rung.get("verdict"),
            "caveat": universe_rung.get("caveat"),
            "book_pair": universe_rung.get("book_pair"),
            "label": "Universe-choice rung (U-E) bp/day",
        }

    if forecast_leg:
        block["forecast_skill"] = {
            "mean_ic": forecast_leg.get("mean_ic"),
            "ic_t": forecast_leg.get("ic_t"),
            "n_weeks": forecast_leg.get("n_weeks"),
            "certified": forecast_leg.get("certified", False),
            "note": "Certified skill receipt (realized weekly rank-IC); NEVER "
                    "multiplied by a notional. Conversion is the forecast rung above.",
        }

    if stats.get("materiality_bp") is not None:
        block["materiality_bp"] = stats.get("materiality_bp")
    if stats.get("fdr") is not None:
        block["fdr"] = stats.get("fdr")

    metrics["new_brain"] = block
    return dashboard_data


def _parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD strings safely for sorting/grouping."""
    return datetime.strptime(value, "%Y-%m-%d")


def _parse_timestamp(value: str) -> Optional[datetime]:
    """Parse ISO timestamps defensively."""
    if not value:
        return None
    try:
        # Handle common trailing Z form without requiring dateutil.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _load_fills(s3, dates: List[str]) -> List[Dict[str, Any]]:
    """Load fills from the local trades.jsonl for the specified dates.

    Fills flow straight from each day's `daily/<date>/trades.jsonl`
    (paper_trader simulated fills). This is a pure simulation — there is no
    broker reconcile; the simulated fill is authoritative.
    """
    fills: List[Dict[str, Any]] = []
    for date_str in dates:
        day_fills = s3.read_jsonl(f"daily/{date_str}/trades.jsonl")
        for idx, fill in enumerate(day_fills):
            # Keep date+index fallback so ordering stays deterministic if timestamp missing.
            normalized = dict(fill)
            normalized["_trade_date"] = date_str
            normalized["_trade_index"] = idx
            fills.append(normalized)

    def _sort_key(fill: Dict[str, Any]) -> Tuple[datetime, int]:
        ts = _parse_timestamp(str(fill.get("timestamp", "")))
        if ts is not None:
            return ts, int(fill.get("_trade_index", 0))
        day = _parse_date(str(fill.get("_trade_date", "1970-01-01")))
        return day, int(fill.get("_trade_index", 0))

    # Set _trade_date on injected fills so the sort key has a stable fallback.
    for fill in fills:
        if "_trade_date" not in fill:
            ts = str(fill.get("timestamp", ""))
            fill["_trade_date"] = ts[:10] if ts else "1970-01-01"
            fill["_trade_index"] = 9999  # sort injected fills last within their day

    fills.sort(key=_sort_key)
    return fills


def _fill_cost_dollars(fill: Dict[str, Any]) -> float:
    """Derive transaction cost dollars from fill metadata."""
    try:
        shares = float(fill.get("shares", 0) or 0)
    except (TypeError, ValueError):
        shares = 0.0
    if shares <= 0:
        return 0.0

    try:
        fill_price = float(fill.get("price", 0) or 0)
    except (TypeError, ValueError):
        fill_price = 0.0

    market_price_raw = fill.get("market_price")
    try:
        market_price = float(market_price_raw) if market_price_raw is not None else None
    except (TypeError, ValueError):
        market_price = None

    spread_slippage = abs(fill_price - market_price) * shares if market_price is not None else 0.0
    try:
        commission = float(fill.get("commission", 0.0) or 0.0)
    except (TypeError, ValueError):
        commission = 0.0
    return spread_slippage + commission


def _build_trade_summary(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct deterministic FIFO round-trip accounting from fill records.

    Fills are split-adjusted before FIFO matching via
    `apply_split_corrections_to_fills`. Without that adjustment, a pre-split
    BUY (e.g. VUG 17.127 sh @ $441.05) gets matched against a post-split SELL
    (56 sh @ $84.38), producing a phantom realized loss of -$6,063 driven by
    cost-basis mismatch rather than actual strategy P&L. The corrections
    layer translates the pre-split lot to its post-split basis (102.77 sh @
    $73.51) so the same dollar exposure matches the same dollar basis.
    """
    fills = apply_split_corrections_to_fills(fills)

    open_lots: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
    round_trips: List[Dict[str, Any]] = []
    unmatched_closing_shares = 0
    cumulative_costs = 0.0

    for idx, fill in enumerate(fills):
        action = str(fill.get("action", "")).upper()
        symbol = str(fill.get("symbol", ""))
        if not symbol:
            continue

        try:
            shares = int(float(fill.get("shares", 0) or 0))
        except (TypeError, ValueError):
            shares = 0
        if shares <= 0:
            continue

        try:
            price = float(fill.get("price", 0.0) or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            continue

        cumulative_costs += _fill_cost_dollars(fill)
        fill_id = str(fill.get("fill_id") or f"{fill.get('_trade_date', 'unknown')}:{idx}")
        ts = _parse_timestamp(str(fill.get("timestamp", "")))

        if action == "BUY":
            open_lots[symbol].append(
                {
                    "fill_id": fill_id,
                    "symbol": symbol,
                    "entry_price": price,
                    "entry_timestamp": ts,
                    "entry_date": fill.get("_trade_date"),
                    "shares_remaining": shares,
                }
            )
            continue

        if action not in ("SELL", "REDUCE"):
            continue

        shares_to_close = shares
        while shares_to_close > 0 and open_lots[symbol]:
            lot = open_lots[symbol][0]
            matched = min(shares_to_close, int(lot["shares_remaining"]))
            entry_price = float(lot["entry_price"])
            pnl = (price - entry_price) * matched
            pnl_pct = (price / entry_price - 1.0) if entry_price > 0 else 0.0
            days_held: Optional[int] = None
            if lot.get("entry_timestamp") is not None and ts is not None:
                days_held = max(0, (ts - lot["entry_timestamp"]).days)

            round_trips.append(
                {
                    "round_trip_id": f"{lot['fill_id']}->{fill_id}:{matched}",
                    "symbol": symbol,
                    "entry_fill_id": lot["fill_id"],
                    "exit_fill_id": fill_id,
                    "entry_date": lot.get("entry_date"),
                    "exit_date": fill.get("_trade_date"),
                    "entry_price": entry_price,
                    "exit_price": price,
                    "shares": matched,
                    "realized_pnl": pnl,
                    "realized_pnl_pct": pnl_pct,
                    "days_held": days_held,
                }
            )

            lot["shares_remaining"] -= matched
            shares_to_close -= matched
            if lot["shares_remaining"] <= 0:
                open_lots[symbol].popleft()

        unmatched_closing_shares += max(shares_to_close, 0)

    wins = sum(1 for rt in round_trips if rt["realized_pnl"] > 0)
    losses = sum(1 for rt in round_trips if rt["realized_pnl"] < 0)
    breakeven = len(round_trips) - wins - losses
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    return {
        "fills_total": len(fills),
        "realized_round_trips": len(round_trips),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "unmatched_closing_shares": unmatched_closing_shares,
        "cumulative_transaction_costs": cumulative_costs,
        "round_trips": round_trips,
    }


def _exposure_metrics(
    portfolio_state: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Compute lightweight exposure/concentration/beta transparency metrics."""
    portfolio_value = float(portfolio_state.get("portfolio_value", 0.0) or 0.0)
    cash = float(portfolio_state.get("cash", 0.0) or 0.0)
    holdings = portfolio_state.get("holdings", [])

    if portfolio_value <= 0:
        return {
            "cash_pct": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "top_position_pct": 0.0,
            "beta_proxy": None,
        }

    abs_values: List[float] = []
    signed_values: List[float] = []
    beta_numerator = 0.0
    beta_weight_denom = 0.0

    for holding in holdings:
        if "market_value" in holding:
            mv = float(holding.get("market_value", 0.0) or 0.0)
        else:
            shares = float(holding.get("shares", 0.0) or 0.0)
            price = float(
                holding.get("current_price", holding.get("entry_price", 0.0)) or 0.0
            )
            mv = shares * price

        abs_mv = abs(mv)
        abs_values.append(abs_mv)
        signed_values.append(mv)

        beta = holding.get("beta")
        if beta is not None:
            try:
                beta_f = float(beta)
                beta_numerator += abs_mv * beta_f
                beta_weight_denom += abs_mv
            except (TypeError, ValueError):
                pass

    gross_exposure = sum(abs_values) / portfolio_value
    net_exposure = sum(signed_values) / portfolio_value
    top_position_pct = (max(abs_values) / portfolio_value) if abs_values else 0.0
    beta_proxy = (beta_numerator / beta_weight_denom) if beta_weight_denom > 0 else None

    return {
        "cash_pct": cash / portfolio_value,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "top_position_pct": top_position_pct,
        "beta_proxy": beta_proxy,
    }


def compute_trade_and_exposure_metrics(
    s3,
    portfolio_state: Dict[str, Any],
    current_state: Optional[Dict[str, Any]] = None,
    max_days: int = 730,
) -> Dict[str, Any]:
    """Trade-history + current-exposure metrics — the NON-LINE half of the dashboard.

    Clean-core split (FP-08-3/-4): the displayed equity LINE (equity_curve,
    drawdowns, monthly returns, ytd/mtd/sharpe/max-dd) now comes from the stored
    ledger (``src/canon/equity_line.py``), NOT from a nightly recompute of the
    per-day sim book. This function keeps only what is genuinely derived elsewhere:

      * trade stats from the FIFO round-trip accounting over ``trades.jsonl``
        (win_rate / wins / losses / costs) — never the line;
      * current-posture exposure ratios from the live ``portfolio_state`` holdings
        (cash_pct / gross / net / top_position / beta_proxy).

    It does NOT read ``sim_book_value`` into any displayed line and never calls the
    deleted recompute pipeline (``_load_daily_states`` & friends).
    """
    effective_state = current_state or portfolio_state
    dates = sorted(s3.list_daily_dates(max_days=max_days))
    fills = _load_fills(s3, dates)
    trade_summary = _build_trade_summary(fills)
    exposure = _exposure_metrics(effective_state)

    metrics = {
        "win_rate": trade_summary["win_rate"],
        "total_trades": trade_summary["wins"] + trade_summary["losses"],
        "wins": trade_summary["wins"],
        "losses": trade_summary["losses"],
        "breakeven_trades": trade_summary["breakeven"],
        "realized_round_trips": trade_summary["realized_round_trips"],
        "total_fills": trade_summary["fills_total"],
        "cumulative_transaction_costs": trade_summary["cumulative_transaction_costs"],
        "cash_pct": exposure["cash_pct"],
        "gross_exposure": exposure["gross_exposure"],
        "net_exposure": exposure["net_exposure"],
        "top_position_pct": exposure["top_position_pct"],
        "beta_proxy": exposure["beta_proxy"],
    }
    return {
        "fills": fills,
        "trade_summary": {k: v for k, v in trade_summary.items() if k != "round_trips"},
        "round_trips": trade_summary["round_trips"],
        "metrics": metrics,
    }


