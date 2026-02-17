"""Canonical dashboard metrics computed from a single daily snapshot series."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, stdev
from typing import Any, Deque, Dict, List, Optional, Tuple


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


def extract_external_cashflow(state: Dict[str, Any]) -> float:
    """Extract net external cashflow for a state row.

    Positive values are deposits/transfers in, negative values are withdrawals.
    """
    direct_keys = (
        "external_cashflow",
        "external_cashflow_t",
        "net_external_cashflow",
        "net_cashflow",
        "cashflow",
        "cash_flow",
    )
    for key in direct_keys:
        if key in state and state.get(key) is not None:
            try:
                return float(state.get(key, 0.0))
            except (TypeError, ValueError):
                continue

    def _num(key: str) -> float:
        try:
            return float(state.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    deposits = (
        _num("deposit")
        + _num("deposits")
        + _num("cash_deposit")
        + _num("external_deposit")
        + _num("transfer_in")
        + _num("inflow")
    )
    withdrawals = (
        _num("withdrawal")
        + _num("withdrawals")
        + _num("cash_withdrawal")
        + _num("external_withdrawal")
        + _num("transfer_out")
        + _num("outflow")
    )
    return deposits - withdrawals


def _extract_state_reset_marker(state: Dict[str, Any]) -> Optional[str]:
    """Read explicit reset marker if available."""
    for key in ("metrics_reset_id", "reset_id", "portfolio_reset_id"):
        value = state.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _load_daily_states(
    s3,
    max_days: int,
    snapshot_date: Optional[str],
    current_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load ordered daily state rows and optionally inject current in-memory state."""
    dates = sorted(s3.list_daily_dates(max_days=max_days))
    rows: List[Dict[str, Any]] = []

    for date_str in dates:
        state = s3.read_json(f"daily/{date_str}/portfolio_state.json")
        if not state:
            continue
        try:
            value = float(state.get("portfolio_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "date": date_str,
                "value": value,
                "benchmark": float(state.get("benchmark_value", value) or value),
                "cash": float(state.get("cash", 0.0) or 0.0),
                "holdings_count": len(state.get("holdings", [])),
                "external_cashflow": extract_external_cashflow(state),
                "state_timestamp": state.get("last_updated"),
                "reset_marker": _extract_state_reset_marker(state),
            }
        )

    # Inject or replace today's state to avoid one-day lag when stats are computed
    # before daily/{date}/portfolio_state.json is persisted.
    if current_state is not None:
        as_of_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
        current_row = {
            "date": as_of_date,
            "value": float(current_state.get("portfolio_value", 0.0) or 0.0),
            "benchmark": float(
                current_state.get("benchmark_value", current_state.get("portfolio_value", 0.0))
                or 0.0
            ),
            "cash": float(current_state.get("cash", 0.0) or 0.0),
            "holdings_count": len(current_state.get("holdings", [])),
            "external_cashflow": extract_external_cashflow(current_state),
            "state_timestamp": current_state.get("last_updated"),
            "reset_marker": _extract_state_reset_marker(current_state),
        }

        replaced = False
        for idx, existing in enumerate(rows):
            if existing["date"] == as_of_date:
                rows[idx] = current_row
                replaced = True
                break
        if not replaced:
            rows.append(current_row)

    rows.sort(key=lambda row: row["date"])
    return rows


def _select_active_segment(
    rows: List[Dict[str, Any]],
    initial_value: float,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Select active segment and infer reset boundary if present."""
    if not rows:
        return [], None

    # Prefer explicit reset markers when available.
    latest_marker = None
    for row in reversed(rows):
        if row.get("reset_marker"):
            latest_marker = row["reset_marker"]
            break
    if latest_marker is not None:
        active = [row for row in rows if row.get("reset_marker") == latest_marker]
        if active:
            return active, {
                "method": "explicit_marker",
                "marker": latest_marker,
                "start_date": active[0]["date"],
            }

    # Heuristic fallback: large discontinuity into near-initial mostly-cash state.
    boundary_idx = 0
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        prev_value = prev["value"]
        curr_value = curr["value"]
        if prev_value <= 0:
            continue

        jump = (curr_value - prev_value) / prev_value
        near_initial = abs(curr_value - initial_value) <= initial_value * 0.05
        mostly_cash = curr["cash"] >= curr_value * 0.9
        large_discontinuity = abs(jump) >= 0.30

        if near_initial and mostly_cash and large_discontinuity:
            boundary_idx = idx

    if boundary_idx > 0:
        active = rows[boundary_idx:]
        return active, {
            "method": "heuristic_discontinuity",
            "start_date": active[0]["date"],
            "details": (
                "Detected reset-like discontinuity and restarted return aggregation "
                "from this boundary."
            ),
        }

    return rows, None


def _build_return_rows(active_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build canonical daily return series adjusted for external cashflows."""
    results: List[Dict[str, Any]] = []
    prev_value: Optional[float] = None

    for row in active_rows:
        if prev_value is None or prev_value <= 0:
            daily_return = None
        else:
            daily_return = (row["value"] - prev_value - row["external_cashflow"]) / prev_value

        results.append(
            {
                "date": row["date"],
                "value": row["value"],
                "benchmark": row["benchmark"],
                "external_cashflow": row["external_cashflow"],
                "daily_return": daily_return,
            }
        )
        prev_value = row["value"]
    return results


def _compound(returns: List[float]) -> float:
    """Compound a list of arithmetic returns."""
    if not returns:
        return 0.0
    total = 1.0
    for ret in returns:
        total *= 1.0 + ret
    return total - 1.0


def _period_return(
    return_rows: List[Dict[str, Any]],
    start_date: str,
) -> float:
    """Compute compounded return from start_date through snapshot."""
    values = [
        row["daily_return"]
        for row in return_rows
        if row["date"] >= start_date and row["daily_return"] is not None
    ]
    return _compound([float(x) for x in values])


def _monthly_returns(return_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build monthly compounded returns from canonical daily returns."""
    monthly_map: Dict[str, List[float]] = defaultdict(list)
    for row in return_rows:
        if row["daily_return"] is None:
            continue
        ym = row["date"][:7]
        monthly_map[ym].append(float(row["daily_return"]))

    result: List[Dict[str, Any]] = []
    for ym in sorted(monthly_map.keys()):
        year, month = ym.split("-")
        result.append(
            {
                "year": int(year),
                "month": int(month),
                "return_pct": _compound(monthly_map[ym]),
                "observations": len(monthly_map[ym]),
            }
        )
    return result


def _drawdown_series(active_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build drawdown series from canonical equity curve."""
    if not active_rows:
        return []

    peak = active_rows[0]["value"]
    drawdowns: List[Dict[str, Any]] = []
    for row in active_rows:
        peak = max(peak, row["value"])
        drawdown = (row["value"] - peak) / peak if peak > 0 else 0.0
        drawdowns.append({"date": row["date"], "drawdown": drawdown})
    return drawdowns


def _load_fills(s3, dates: List[str]) -> List[Dict[str, Any]]:
    """Load fills from trades.jsonl for the specified dates."""
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
    """Construct deterministic FIFO round-trip accounting from fill records."""
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


def compute_canonical_dashboard_metrics(
    s3,
    portfolio_state: Dict[str, Any],
    snapshot_date: Optional[str] = None,
    current_state: Optional[Dict[str, Any]] = None,
    max_days: int = 730,
    initial_value: float = 100000.0,
    risk_free_rate_annual: float = 0.0,
    min_sharpe_observations: int = 60,
) -> Dict[str, Any]:
    """Compute canonical dashboard metrics and series from daily artifacts."""
    as_of_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    effective_state = current_state or portfolio_state

    rows = _load_daily_states(
        s3=s3,
        max_days=max_days,
        snapshot_date=as_of_date,
        current_state=effective_state,
    )
    active_rows, reset_boundary = _select_active_segment(rows, initial_value=initial_value)
    return_rows = _build_return_rows(active_rows)
    drawdowns = _drawdown_series(active_rows)
    monthly_returns = _monthly_returns(return_rows)

    start_of_year = f"{as_of_date[:4]}-01-01"
    start_of_month = f"{as_of_date[:7]}-01"
    ytd_return = _period_return(return_rows, start_of_year)
    mtd_return = _period_return(return_rows, start_of_month)

    daily_returns = [row["daily_return"] for row in return_rows if row["daily_return"] is not None]
    sharpe_ratio: Optional[float]
    if len(daily_returns) < min_sharpe_observations:
        sharpe_ratio = None
    else:
        rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / 252.0) - 1.0
        excess = [ret - rf_daily for ret in daily_returns]
        if len(excess) < 2:
            sharpe_ratio = None
        else:
            sigma = stdev(excess)
            sharpe_ratio = (mean(excess) / sigma * sqrt(252.0)) if sigma > 0 else None

    max_drawdown = min((point["drawdown"] for point in drawdowns), default=0.0)
    current_drawdown = drawdowns[-1]["drawdown"] if drawdowns else 0.0

    dates = [row["date"] for row in active_rows]
    fills = _load_fills(s3, dates)
    trade_summary = _build_trade_summary(fills)
    exposure = _exposure_metrics(effective_state)

    metrics = {
        "ytd_return": ytd_return,
        "mtd_return": mtd_return,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_observations": len(daily_returns),
        "sharpe_min_observations": min_sharpe_observations,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "win_rate": trade_summary["win_rate"],
        # "Total trades" is counted-trade basis (wins + losses) to keep
        # denominator semantics consistent with win_rate = wins / (wins + losses).
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
        "snapshot_date": as_of_date,
        "reset_boundary": reset_boundary,
        "active_start_date": active_rows[0]["date"] if active_rows else None,
        "equity_curve": [
            {"date": row["date"], "value": row["value"], "benchmark": row["benchmark"]}
            for row in active_rows
        ],
        "drawdowns": drawdowns,
        "monthly_returns": monthly_returns,
        "daily_returns": [
            {
                "date": row["date"],
                "daily_return": row["daily_return"],
                "external_cashflow": row["external_cashflow"],
            }
            for row in return_rows
        ],
        "fills": fills,
        "trade_summary": {
            key: value
            for key, value in trade_summary.items()
            if key != "round_trips"
        },
        "round_trips": trade_summary["round_trips"],
        "metrics": metrics,
    }
