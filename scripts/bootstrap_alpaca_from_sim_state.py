#!/usr/bin/env python3
"""Bootstrap Alpaca paper positions from simulated portfolio state.

Reads the last simulated portfolio, computes target notional buys scaled
to current Alpaca equity, and submits fractional/notional orders.

Usage:
    # Dry-run (default)
    python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11

    # Apply
    python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11 --apply
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Allow running from repo root
sys.path.insert(0, ".")

from src.utils.s3_client import S3Client
from src.brokers.alpaca import AlpacaBroker


DEFAULT_MAX_PER_ORDER = 5000.0
DEFAULT_MAX_TOTAL = 95000.0
MIN_ORDER_NOTIONAL = 1.0


def compute_source_weights(
    holdings: List[Dict[str, Any]],
    portfolio_value: float,
) -> List[Dict[str, Any]]:
    """Compute portfolio weight for each holding.

    Args:
        holdings: List of holding dicts from portfolio_state.
        portfolio_value: Total portfolio value.

    Returns:
        List of dicts with symbol, shares, market_value, weight.
    """
    results = []
    for h in holdings:
        symbol = h.get("symbol", "")
        if not symbol:
            continue
        shares = float(h.get("shares", 0) or 0)
        price = float(
            h.get("current_price", h.get("entry_price", 0)) or 0
        )
        market_value = float(h.get("market_value", shares * price) or shares * price)
        weight = market_value / portfolio_value if portfolio_value > 0 else 0.0

        results.append({
            "symbol": symbol,
            "source_shares": shares,
            "source_market_value": market_value,
            "weight": weight,
        })

    return sorted(results, key=lambda x: x["weight"], reverse=True)


def compute_bootstrap_orders(
    source_weights: List[Dict[str, Any]],
    target_equity: float,
    existing_positions: List[Dict[str, Any]],
    max_per_order: float = DEFAULT_MAX_PER_ORDER,
    max_total: float = DEFAULT_MAX_TOTAL,
    symbol_allowlist: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute bootstrap buy orders from source weights and target equity.

    Args:
        source_weights: Output of compute_source_weights().
        target_equity: Current Alpaca paper equity to scale to.
        existing_positions: Current Alpaca positions (for idempotent skip).
        max_per_order: Per-order notional cap.
        max_total: Total notional cap for the bootstrap run.
        symbol_allowlist: Optional list of allowed symbols.

    Returns:
        (orders, skipped) — orders to submit and skipped entries with reasons.
    """
    # Build lookup of existing position market values
    existing_by_symbol = {}
    for pos in existing_positions:
        sym = pos.get("symbol", "")
        existing_by_symbol[sym] = float(pos.get("market_value", 0) or 0)

    allowlist_set = set(symbol_allowlist) if symbol_allowlist else None

    orders = []
    skipped = []
    running_total = 0.0

    for entry in source_weights:
        symbol = entry["symbol"]
        weight = entry["weight"]

        # Allowlist check
        if allowlist_set and symbol not in allowlist_set:
            skipped.append({
                **entry,
                "reason": f"Not in allowlist",
            })
            continue

        target_dollars = weight * target_equity
        existing_value = existing_by_symbol.get(symbol, 0.0)
        needed_dollars = target_dollars - existing_value

        if needed_dollars < MIN_ORDER_NOTIONAL:
            skipped.append({
                **entry,
                "target_dollars": target_dollars,
                "existing_value": existing_value,
                "needed_dollars": needed_dollars,
                "reason": "Below minimum or already positioned",
            })
            continue

        # Cap per-order
        order_dollars = min(needed_dollars, max_per_order)

        # Cap running total
        if running_total + order_dollars > max_total:
            remaining = max_total - running_total
            if remaining < MIN_ORDER_NOTIONAL:
                skipped.append({
                    **entry,
                    "target_dollars": target_dollars,
                    "needed_dollars": needed_dollars,
                    "reason": "Max total notional cap reached",
                })
                continue
            order_dollars = remaining

        orders.append({
            "symbol": symbol,
            "side": "buy",
            "dollars": round(order_dollars, 2),
            "target_dollars": round(target_dollars, 2),
            "weight": weight,
            "existing_value": existing_value,
        })
        running_total += order_dollars

    return orders, skipped


def make_bootstrap_client_order_id(symbol: str, run_date: str) -> str:
    """Deterministic client_order_id for bootstrap orders."""
    raw = f"bootstrap|{run_date}|{symbol}|buy"
    return f"tb-boot-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Alpaca paper positions from simulated state"
    )
    parser.add_argument(
        "--bucket", default="investment-system-data", help="S3 bucket"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="AWS region"
    )
    parser.add_argument(
        "--source-date",
        help="Source simulated date (YYYY-MM-DD). Default: latest pre-cutover date."
    )
    parser.add_argument(
        "--mode", choices=["scaled", "exact"], default="scaled",
        help=(
            "Bootstrap mode: scaled (match weights to current equity) "
            "or exact (match source-dollar exposure, not exact share counts)"
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Print plan without placing orders (default)"
    )
    parser.add_argument(
        "--apply", action="store_true", dest="apply_mode",
        help="Submit bootstrap orders to Alpaca"
    )
    parser.add_argument(
        "--max-per-order", type=float, default=DEFAULT_MAX_PER_ORDER,
        help=f"Max notional per order (default ${DEFAULT_MAX_PER_ORDER})"
    )
    parser.add_argument(
        "--max-total", type=float, default=DEFAULT_MAX_TOTAL,
        help=f"Max total notional for bootstrap run (default ${DEFAULT_MAX_TOTAL})"
    )
    parser.add_argument(
        "--symbol-allowlist", nargs="+",
        help="Optional symbol allowlist (space-separated)"
    )
    parser.add_argument(
        "--profile", default=None,
        help="AWS profile to use"
    )
    args = parser.parse_args()

    if args.profile:
        import boto3
        from botocore.exceptions import ProfileNotFound

        try:
            boto3.setup_default_session(profile_name=args.profile)
        except ProfileNotFound:
            print(
                f"ERROR: AWS profile '{args.profile}' not found. "
                "Pass a valid --profile or omit it to use environment/instance credentials."
            )
            sys.exit(2)

    s3 = S3Client(bucket=args.bucket, region=args.region)
    today = datetime.now().strftime("%Y-%m-%d")

    # Find source date
    source_date = args.source_date
    if not source_date:
        dates = s3.list_daily_dates(max_days=30)
        for d in sorted(dates, reverse=True):
            state = s3.read_json(f"daily/{d}/portfolio_state.json")
            if state and not state.get("broker_reconciled"):
                source_date = d
                break
        if not source_date:
            print("ERROR: Could not find a pre-cutover simulated date")
            sys.exit(1)

    print(f"Source date: {source_date}")
    source_state = s3.read_json(f"daily/{source_date}/portfolio_state.json")
    if not source_state:
        print(f"ERROR: No portfolio_state.json for {source_date}")
        sys.exit(1)

    holdings = source_state.get("holdings", [])
    portfolio_value = float(source_state.get("portfolio_value", 0.0) or 0.0)
    print(f"Source portfolio: ${portfolio_value:,.2f} with {len(holdings)} holdings")

    if not holdings:
        print("No holdings to bootstrap. Exiting.")
        return

    # Compute source weights
    source_weights = compute_source_weights(holdings, portfolio_value)

    # Get Alpaca credentials
    key_id = os.environ.get("ALPACA_PAPER_KEY_ID", "")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY", "")

    if not key_id or not secret_key:
        # Try Secrets Manager (secrets stored as JSON {"api_key": "..."})
        try:
            import boto3
            sm = boto3.client("secretsmanager", region_name=args.region)

            def _extract_secret(secret_id):
                raw = sm.get_secret_value(SecretId=secret_id)["SecretString"]
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed.get("api_key", parsed.get("key", str(parsed)))
                    return parsed
                except (json.JSONDecodeError, TypeError):
                    return raw

            key_id = _extract_secret("investment-system/alpaca-paper-key-id")
            secret_key = _extract_secret("investment-system/alpaca-paper-secret-key")
        except Exception as e:
            print(f"ERROR: Cannot load Alpaca credentials: {e}")
            print("Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY env vars,")
            print("or ensure Secrets Manager has the keys.")
            sys.exit(1)

    broker = AlpacaBroker(
        key_id=key_id,
        secret_key=secret_key,
        paper=True,
        max_order_notional=args.max_per_order,
    )

    # Get current Alpaca state
    account = broker.check_account()
    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    existing_positions = broker.list_positions()

    print(f"\nAlpaca paper account:")
    print(f"  Equity: ${equity:,.2f}")
    print(f"  Cash:   ${cash:,.2f}")
    print(f"  Positions: {len(existing_positions)}")

    # Compute orders
    if args.mode == "scaled":
        target_equity = equity
    else:
        # "exact" mode here means matching source-dollar exposure.
        # It does not attempt exact share-count recreation.
        target_equity = portfolio_value

    orders, skipped = compute_bootstrap_orders(
        source_weights=source_weights,
        target_equity=target_equity,
        existing_positions=existing_positions,
        max_per_order=args.max_per_order,
        max_total=args.max_total,
        symbol_allowlist=args.symbol_allowlist,
    )

    total_notional = sum(o["dollars"] for o in orders)

    print(f"\n=== Bootstrap Plan ({args.mode} mode) ===")
    print(f"Target equity:    ${target_equity:,.2f}")
    print(f"Orders to place:  {len(orders)}")
    print(f"Total notional:   ${total_notional:,.2f}")
    print(f"Skipped symbols:  {len(skipped)}")

    if orders:
        print(f"\nOrders:")
        for o in orders:
            print(
                f"  {o['side'].upper():4s} {o['symbol']:6s}  "
                f"${o['dollars']:>9,.2f}  "
                f"(weight {o['weight']:.2%}, target ${o['target_dollars']:,.2f})"
            )

    if skipped:
        print(f"\nSkipped:")
        for s in skipped:
            print(f"  {s['symbol']:6s}  {s['reason']}")

    if args.apply_mode:
        if not orders:
            print("\nNo orders to submit.")
            return

        if cash < total_notional:
            print(
                f"\nWARNING: Available cash ${cash:,.2f} < total notional "
                f"${total_notional:,.2f}. Some orders may be rejected."
            )

        print(f"\nSubmitting {len(orders)} orders...")
        results = []
        for order in orders:
            client_order_id = make_bootstrap_client_order_id(
                order["symbol"], today
            )
            try:
                result = broker.submit_order(
                    symbol=order["symbol"],
                    side="buy",
                    dollars=order["dollars"],
                    client_order_id=client_order_id,
                )
                status = result.get("status", "unknown")
                print(f"  {order['symbol']:6s}  ${order['dollars']:>9,.2f}  -> {status}")
                results.append({
                    **order,
                    "client_order_id": client_order_id,
                    "order_id": result.get("order_id"),
                    "status": status,
                    "error": None,
                })
            except Exception as e:
                print(f"  {order['symbol']:6s}  ${order['dollars']:>9,.2f}  -> ERROR: {e}")
                results.append({
                    **order,
                    "client_order_id": client_order_id,
                    "order_id": None,
                    "status": "error",
                    "error": str(e),
                })

        submitted = sum(1 for r in results if r["status"] != "error")
        errored = sum(1 for r in results if r["status"] == "error")
        print(f"\nSubmitted: {submitted}, Errors: {errored}")

        # Write result artifact
        result_artifact = {
            "action": "apply",
            "mode": args.mode,
            "source_date": source_date,
            "target_equity": target_equity,
            "total_notional": total_notional,
            "orders": results,
            "skipped": skipped,
            "submitted_count": submitted,
            "error_count": errored,
            "timestamp": datetime.now().isoformat(),
        }
        result_key = f"daily/{today}/alpaca_bootstrap_result.json"
        s3.write_json(result_artifact, result_key)
        print(f"Result artifact: s3://{args.bucket}/{result_key}")
    else:
        # Dry-run: write plan artifact
        plan = {
            "action": "dry_run",
            "mode": args.mode,
            "source_date": source_date,
            "target_equity": target_equity,
            "total_notional": total_notional,
            "orders": orders,
            "skipped": skipped,
            "source_weights": source_weights,
            "timestamp": datetime.now().isoformat(),
        }
        plan_key = f"daily/{today}/alpaca_bootstrap_plan.json"
        s3.write_json(plan, plan_key)
        print(f"\nDry-run plan: s3://{args.bucket}/{plan_key}")
        print("Re-run with --apply to submit orders.")


if __name__ == "__main__":
    main()
