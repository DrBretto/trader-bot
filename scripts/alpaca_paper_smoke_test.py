#!/usr/bin/env python3
"""Alpaca paper trading smoke test.

Validates connectivity, optionally places a tiny test order, and optionally closes it.

Usage:
    python scripts/alpaca_paper_smoke_test.py --help
    python scripts/alpaca_paper_smoke_test.py --account-check
    python scripts/alpaca_paper_smoke_test.py --place-order
    python scripts/alpaca_paper_smoke_test.py --place-order --close-after
"""

import argparse
import json
import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brokers.alpaca import AlpacaBroker


def get_credentials() -> tuple:
    """Get Alpaca paper credentials from env or Secrets Manager."""
    key_id = os.environ.get('ALPACA_PAPER_KEY_ID', '')
    secret_key = os.environ.get('ALPACA_PAPER_SECRET_KEY', '')

    if not key_id or not secret_key:
        # Try Secrets Manager
        try:
            from src.handler import get_secret
            key_id = get_secret('investment-system/alpaca-paper-key-id')
            secret_key = get_secret('investment-system/alpaca-paper-secret-key')
        except Exception:
            pass

    return key_id, secret_key


def run_account_check(broker: AlpacaBroker) -> bool:
    """Check account health and print summary."""
    print("\n--- Account Check ---")
    try:
        acct = broker.check_account()
        print(f"  Status:         {acct['status']}")
        print(f"  Account:        {acct['account_number']}")
        print(f"  Cash:           ${acct['cash']:,.2f}")
        print(f"  Buying Power:   ${acct['buying_power']:,.2f}")
        print(f"  Equity:         ${acct['equity']:,.2f}")
        print(f"  Tradable:       {acct['tradable']}")

        if not acct['tradable']:
            print("  WARN: Account is NOT tradable!")
            return False

        print("  PASS: Account check OK")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def run_positions_check(broker: AlpacaBroker) -> bool:
    """List current positions."""
    print("\n--- Positions ---")
    try:
        positions = broker.list_positions()
        if not positions:
            print("  No open positions")
        else:
            for p in positions:
                print(
                    f"  {p['symbol']}: {p['qty']} shares, "
                    f"${p['market_value']:,.2f} "
                    f"(P&L: ${p['unrealized_pl']:,.2f})"
                )
        print("  PASS: Positions check OK")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def run_place_order(
    broker: AlpacaBroker,
    symbol: str = 'SPY',
    notional: float = 1.0,
) -> dict:
    """Place a tiny notional market order."""
    print(f"\n--- Place Order: ${notional} {symbol} ---")
    try:
        result = broker.submit_order(
            symbol=symbol,
            side='buy',
            dollars=notional,
            client_order_id=f"smoke-test-{int(time.time())}",
        )
        print(f"  Order ID:       {result['order_id']}")
        print(f"  Status:         {result['status']}")
        print(f"  Symbol:         {result['symbol']}")
        print(f"  Notional:       {result['notional']}")
        print("  PASS: Order submitted")
        return result
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return {}


def run_close_position(broker: AlpacaBroker, symbol: str = 'SPY') -> bool:
    """Close the test position."""
    print(f"\n--- Close Position: {symbol} ---")
    # Wait a moment for order to fill
    print("  Waiting 3s for fill...")
    time.sleep(3)

    try:
        result = broker.close_position(symbol)
        print(f"  Close result: {json.dumps(result, indent=2, default=str)[:200]}")
        print("  PASS: Position closed")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Alpaca paper trading smoke test')
    parser.add_argument(
        '--account-check', action='store_true',
        help='Check account status and connectivity'
    )
    parser.add_argument(
        '--place-order', action='store_true',
        help='Place a $1 notional SPY paper order'
    )
    parser.add_argument(
        '--close-after', action='store_true',
        help='Close the test position after placing order'
    )
    parser.add_argument(
        '--symbol', default='SPY',
        help='Symbol to use for test order (default: SPY)'
    )
    parser.add_argument(
        '--notional', type=float, default=1.0,
        help='Dollar amount for test order (default: $1)'
    )
    parser.add_argument(
        '--positions', action='store_true',
        help='List current positions'
    )
    args = parser.parse_args()

    # Default to --account-check if nothing specified
    if not any([args.account_check, args.place_order, args.positions]):
        args.account_check = True

    key_id, secret_key = get_credentials()
    if not key_id or not secret_key:
        print("ERROR: Alpaca paper credentials not found.")
        print("Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY env vars,")
        print("or run infrastructure/secrets_setup.sh to store in Secrets Manager.")
        sys.exit(1)

    broker = AlpacaBroker(
        key_id=key_id,
        secret_key=secret_key,
        paper=True,
        max_order_notional=100.0,  # Low cap for smoke tests
    )

    results = []

    if args.account_check:
        results.append(run_account_check(broker))

    if args.positions:
        results.append(run_positions_check(broker))

    if args.place_order:
        order = run_place_order(broker, args.symbol, args.notional)
        results.append(bool(order))

        if args.close_after and order:
            results.append(run_close_position(broker, args.symbol))

    # Summary
    print("\n--- Summary ---")
    passed = all(results) if results else False
    print(f"  Tests run: {len(results)}")
    print(f"  All passed: {passed}")

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
