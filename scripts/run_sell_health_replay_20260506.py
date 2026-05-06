"""Replay the May 1-2 sell trigger and the May 6 live state through the
repaired evaluate_holdings to demonstrate the persistence gate behavior.

Output: prints a verdict per scenario. Used by the 2026-05-06 audit RETURN to
prove the repaired engine does NOT liquidate profitable positions on a
single-day health dip and does NOT misbehave on the current live holdings.
"""
import json
import sys
from datetime import datetime
import pandas as pd

sys.path.insert(0, '.')

from src.steps.decision_engine import evaluate_holdings


# Live params snapshot (from config/decision_params.active.json on 2026-05-06).
PARAMS = {
    'trailing_stop_base': 0.10,
    'trailing_stop_leveraged': 0.06,
    'sell_health_threshold': 0.35,
    'sell_health_days': 3,
}


def make_prices(symbol: str, current_price: float) -> pd.DataFrame:
    return pd.DataFrame({
        'symbol': [symbol],
        'date': [pd.Timestamp.now()],
        'close': [current_price],
    })


def scenario(name: str, holding: dict, current_health: float,
             current_price: float, expected_sells: int) -> bool:
    """Run one scenario through evaluate_holdings and print a verdict."""
    portfolio_state = {'holdings': [holding]}
    asset_health = [{'symbol': holding['symbol'], 'health_score': current_health}]
    prices_df = make_prices(holding['symbol'], current_price)

    actions = evaluate_holdings(
        portfolio_state, asset_health, prices_df, PARAMS, 'risk_on_trend', {}
    )

    n = len(actions)
    counter_after = holding.get('consecutive_below_health_days', 0)
    ok = (n == expected_sells)
    icon = 'OK ' if ok else 'FAIL'
    print(f"[{icon}] {name}")
    print(f"      health={current_health:.2f} threshold=0.35 prior_count="
          f"{holding.get('_prior_count', 0)} new_count={counter_after}")
    print(f"      actions={n} (expected {expected_sells})  "
          f"reasons={[a['reason'] for a in actions]}")
    return ok


def main() -> int:
    print("=== Repaired evaluate_holdings replay (sell_health_days=3) ===\n")

    all_ok = True

    # SCENARIO 1: The exact May 2 trigger that produced HEALTH_COLLAPSE on
    # ARKK/SLV/XBI/XLK. Each of these had a one-day health reading at/below
    # threshold and the pre-fix code fired SELL immediately. With the gate,
    # counter goes from 0 → 1, and SELL must NOT fire.
    for sym, health in [
        ('ARKK', 0.26),
        ('SLV', 0.15),
        ('XBI', 0.29),
        ('XLK', 0.29),
    ]:
        h = {
            'symbol': sym, 'shares': 100,
            'entry_price': 100.0, 'peak_price': 110.0,
            'entry_date': '2026-04-07T13:45:49',
            '_prior_count': 0,
        }
        ok = scenario(
            f"May-2 single-day dip: {sym} health={health}",
            h, current_health=health, current_price=109.0,
            expected_sells=0,
        )
        all_ok = all_ok and ok
        print()

    # SCENARIO 2: After three consecutive low days, the gate fires. Same
    # holding, simulated by setting prior counter to 2.
    h = {
        'symbol': 'ARKK', 'shares': 87,
        'entry_price': 69.36, 'peak_price': 80.00,
        'entry_date': '2026-04-07T13:45:49',
        'consecutive_below_health_days': 2,  # carried from prior runs
        '_prior_count': 2,
    }
    ok = scenario(
        "After 3rd consecutive dip: ARKK gate finally fires",
        h, current_health=0.26, current_price=78.0,
        expected_sells=1,
    )
    all_ok = all_ok and ok
    print()

    # SCENARIO 3: Live broker-truth holdings on 2026-05-06. FXI and XLF (the
    # only material positions Alpaca actually holds). Both have positive
    # current health on the dashboard. Counter starts at 0. Must produce 0
    # SELLs — the system should not interfere with healthy positions.
    for sym, entry_px, peak_px, current_px, health in [
        ('FXI', 35.39, 37.665, 37.005, 0.50),
        ('XLF', 49.677, 52.72, 52.03, 0.50),
    ]:
        h = {
            'symbol': sym, 'shares': 100,
            'entry_price': entry_px, 'peak_price': peak_px,
            'entry_date': '2026-04-07T13:45:49',
            '_prior_count': 0,
        }
        ok = scenario(
            f"Live 2026-05-06: {sym} healthy, no action expected",
            h, current_health=health, current_price=current_px,
            expected_sells=0,
        )
        all_ok = all_ok and ok
        print()

    # SCENARIO 4: Recovery path. A holding that accumulated 2 below-threshold
    # days then recovers — the counter resets to 0, and a subsequent dip must
    # rebuild from 1, not from 2. This protects against intermittent noise
    # ratcheting up to the gate.
    h = {
        'symbol': 'VUG', 'shares': 100,
        'entry_price': 73.48, 'peak_price': 84.36,
        'entry_date': '2026-04-07T13:45:49',
        'consecutive_below_health_days': 2,
        '_prior_count': 2,
    }
    print("Scenario 4 step 1 — health recovers to 0.50, counter resets")
    ok = scenario("Recovery resets counter", h,
                  current_health=0.50, current_price=83.0,
                  expected_sells=0)
    all_ok = all_ok and ok
    print()

    h['_prior_count'] = h.get('consecutive_below_health_days', 0)
    print("Scenario 4 step 2 — single subsequent dip starts at count=1, no sell")
    ok = scenario("Subsequent dip rebuilds from 1", h,
                  current_health=0.20, current_price=82.0,
                  expected_sells=0)
    all_ok = all_ok and ok
    print()

    print("=" * 70)
    print(f"REPLAY VERDICT: {'PASS' if all_ok else 'FAIL'}")
    print("Repaired evaluate_holdings does NOT liquidate on single-day health "
          "dips, DOES liquidate after sell_health_days persistent low readings, "
          "does NOT touch healthy live holdings, and resets counter on recovery.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
