"""Midday check phase: trailing stop re-evaluation, VIX circuit breaker, skipped-buy re-check.

A single lightweight Lambda invocation at 1:00 PM ET that improves execution
responsiveness without changing the core daily strategy. All logic reuses
existing calibrated parameters — no new model inference, features, or signals.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from chassis.steps import ingest_prices, paper_trader, morning_executor
from chassis.utils.s3_client import S3Client

logger = logging.getLogger(__name__)

# VIX circuit breaker thresholds (conservative — fires only during genuine panic)
VIX_ABSOLUTE_THRESHOLD = 35.0
VIX_INTRADAY_JUMP_THRESHOLD = 0.25  # 25% intraday jump from previous close


def check_vix_circuit_breaker(
    vix_current: float,
    vix_previous_close: float,
) -> Tuple[bool, str]:
    """Check whether VIX conditions warrant a circuit breaker.

    Returns (breaker_active, reason).
    """
    if vix_current >= VIX_ABSOLUTE_THRESHOLD:
        return True, (
            f"VIX at {vix_current:.1f} exceeds absolute threshold "
            f"({VIX_ABSOLUTE_THRESHOLD})"
        )

    if vix_previous_close > 0:
        jump_pct = (vix_current - vix_previous_close) / vix_previous_close
        if jump_pct >= VIX_INTRADAY_JUMP_THRESHOLD:
            return True, (
                f"VIX intraday jump {jump_pct:.1%} exceeds threshold "
                f"({VIX_INTRADAY_JUMP_THRESHOLD:.0%}): "
                f"{vix_previous_close:.1f} -> {vix_current:.1f}"
            )

    return False, "VIX within normal range"


def evaluate_trailing_stops(
    holdings: List[Dict[str, Any]],
    price_map: Dict[str, float],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Re-evaluate trailing stops for held positions at midday prices.

    Returns list of holdings that have hit their trailing stop.
    """
    triggered = []
    for holding in holdings:
        symbol = holding['symbol']
        midday_price = price_map.get(symbol)
        if midday_price is None:
            continue

        peak_price = holding.get('peak_price', 0)
        if peak_price <= 0:
            continue

        # Update peak if midday price is higher
        if midday_price > peak_price:
            peak_price = midday_price

        is_leveraged = holding.get('leverage_flag', 0) == 1
        stop_pct = params.get(
            'trailing_stop_leveraged' if is_leveraged else 'trailing_stop_base',
            0.10
        )
        trailing_stop_price = peak_price * (1 - stop_pct)

        if midday_price <= trailing_stop_price:
            triggered.append({
                'symbol': symbol,
                'midday_price': midday_price,
                'peak_price': peak_price,
                'stop_price': trailing_stop_price,
                'stop_pct': stop_pct,
                'is_leveraged': is_leveraged,
                'holding': holding,
            })

    return triggered


def find_skipped_buys(s3: S3Client) -> List[Dict[str, Any]]:
    """Load buy intents that were skipped during morning execution due to price gap.

    Parses the morning execution validation_log for 'SKIP BUY' entries caused
    by price gap, then recovers the original intent data from trade_intents.json
    so the midday re-check has the intent price and dollar target.
    """
    latest = s3.read_json('daily/latest.json')
    if not latest:
        return []

    run_date = latest.get('date', '')
    morning_report = s3.read_json(f'daily/{run_date}/morning_execution.json')
    if not morning_report:
        return []

    # Find gap-skipped symbols from validation log
    validation_log = morning_report.get('validation_log', [])
    gap_skipped_symbols = set()
    for entry in validation_log:
        if 'SKIP BUY' in entry and 'Price gap' in entry:
            parts = entry.split('SKIP BUY ')
            if len(parts) > 1:
                sym = parts[1].split(':')[0].strip()
                gap_skipped_symbols.add(sym)

    if not gap_skipped_symbols:
        return []

    # Load original trade intents to recover prices/dollars
    intents_date = latest.get('intents_date', run_date)
    trade_intents = s3.read_json(f'daily/{intents_date}/trade_intents.json')
    if not trade_intents:
        return []

    skipped = []
    for intent in trade_intents.get('actions', []):
        if intent.get('action') == 'BUY' and intent.get('symbol') in gap_skipped_symbols:
            skipped.append({
                'symbol': intent['symbol'],
                'intent_price': intent.get('price', 0),
                'dollars': intent.get('dollars', intent.get('shares', 0) * intent.get('price', 0)),
                'shares': intent.get('shares', 0),
            })

    return skipped


def run(
    bucket: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the midday check: trailing stops, VIX breaker, skipped-buy re-check.

    Args:
        bucket: S3 bucket name
        config: Pipeline config (decision_params, portfolio_state)

    Returns:
        Dict with actions_taken, check_log, circuit_breaker_active
    """
    s3 = S3Client(bucket)
    params = config.get('decision_params', {})
    transaction_cost_config = config.get('transaction_cost_overrides')
    check_log: List[str] = []
    actions_taken: List[Dict[str, Any]] = []
    circuit_breaker_active = False
    run_date = datetime.now().strftime('%Y-%m-%d')

    # Load portfolio state
    portfolio = paper_trader.load_portfolio_state(s3)
    holdings = portfolio.get('holdings', [])
    held_symbols = [h['symbol'] for h in holdings]

    # Collect symbols we need midday quotes for
    skipped_buys = find_skipped_buys(s3)
    skipped_symbols = [sb['symbol'] for sb in skipped_buys]
    all_symbols = list(set(held_symbols + skipped_symbols + ['^VIX', 'SPY']))

    # Fetch midday quotes
    check_log.append(f"Fetching midday quotes for {len(all_symbols)} symbols")
    midday_quotes = ingest_prices.fetch_morning_quotes(all_symbols)

    if len(midday_quotes) == 0:
        check_log.append("ABORT: No midday quotes fetched")
        return {
            'actions_taken': [],
            'check_log': check_log,
            'circuit_breaker_active': False,
            'portfolio_state': portfolio,
        }

    price_map = {row['symbol']: row['price'] for _, row in midday_quotes.iterrows()}
    check_log.append(f"Fetched midday prices for {len(price_map)} symbols")

    # --- Step 1: VIX Circuit Breaker ---
    vix_current = price_map.get('^VIX') or price_map.get('VIX')

    # Load previous VIX close from overnight context
    latest = s3.read_json('daily/latest.json') or {}
    night_date = latest.get('date', run_date)
    night_context = s3.read_json(f'daily/{night_date}/context.json')
    vix_previous_close = 0.0
    if night_context:
        vix_previous_close = float(night_context.get('vix_close', 0) or 0)

    if vix_current is not None:
        breaker_active, breaker_reason = check_vix_circuit_breaker(
            vix_current, vix_previous_close
        )
        if breaker_active:
            circuit_breaker_active = True
            check_log.append(f"CIRCUIT BREAKER ACTIVE: {breaker_reason}")
            s3.write_json(
                {
                    'active': True,
                    'reason': breaker_reason,
                    'vix_current': vix_current,
                    'vix_previous_close': vix_previous_close,
                    'timestamp': datetime.now().isoformat(),
                },
                f'daily/{run_date}/circuit_breaker.json'
            )
        else:
            check_log.append(
                f"VIX check OK: {vix_current:.1f} "
                f"(prev close: {vix_previous_close:.1f})"
            )
    else:
        check_log.append("VIX quote unavailable — skipping circuit breaker check")

    # --- Step 2: Trailing Stop Re-evaluation ---
    if holdings:
        triggered = evaluate_trailing_stops(holdings, price_map, params)

        if triggered:
            check_log.append(
                f"TRAILING STOPS TRIGGERED: {len(triggered)} positions"
            )

            universe_df = config.get('universe', pd.DataFrame())
            if len(universe_df) == 0:
                universe_df = s3.read_csv('config/universe.csv')

            for stop_info in triggered:
                symbol = stop_info['symbol']
                holding = stop_info['holding']
                midday_price = stop_info['midday_price']

                check_log.append(
                    f"  STOP HIT {symbol}: ${midday_price:.2f} <= "
                    f"stop ${stop_info['stop_price']:.2f} "
                    f"(peak ${stop_info['peak_price']:.2f}, "
                    f"stop {stop_info['stop_pct']:.0%})"
                )

                sell_intent = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': holding.get('shares', 0),
                    'price': midday_price,
                    'reason': 'MIDDAY_STOP_HIT',
                    'regime': portfolio.get('current_regime', 'unknown'),
                }

                trade = paper_trader.execute_trade(
                    portfolio,
                    sell_intent,
                    sell_intent['regime'],
                    universe_df,
                    transaction_cost_config=transaction_cost_config,
                )
                actions_taken.append(trade)
                check_log.append(
                    f"  EXECUTED SELL {symbol} (simulated) "
                    f"@ ${midday_price:.2f}"
                )
        else:
            check_log.append(
                f"Trailing stops OK: {len(holdings)} positions checked, "
                f"none triggered"
            )
    else:
        check_log.append("No holdings to check trailing stops")

    # --- Step 3: Skipped Buy Re-check ---
    if skipped_buys and not circuit_breaker_active:
        check_log.append(
            f"Re-checking {len(skipped_buys)} morning-skipped buy intents"
        )

        universe_df = config.get('universe', pd.DataFrame())
        if len(universe_df) == 0:
            universe_df = s3.read_csv('config/universe.csv')

        for skipped in skipped_buys:
            symbol = skipped['symbol']
            midday_price = price_map.get(symbol)
            if midday_price is None:
                check_log.append(f"  SKIP re-check {symbol}: no midday price")
                continue

            intent_price = skipped.get('intent_price', 0)
            if intent_price <= 0:
                check_log.append(f"  SKIP re-check {symbol}: no intent price")
                continue

            valid, msg = morning_executor.validate_buy_intent(
                {'price': intent_price}, midday_price
            )
            if valid:
                target_dollars = skipped.get('dollars', 0)
                min_order = params.get('min_order_dollars', 250)
                if target_dollars < min_order:
                    check_log.append(
                        f"  SKIP re-check {symbol}: "
                        f"${target_dollars:.2f} below min order"
                    )
                    continue

                buy_intent = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': midday_price,
                    'dollars': target_dollars,
                    'shares': skipped.get('shares', 0),
                    'reason': 'MIDDAY_REATTEMPT',
                }

                buy_intent['shares'] = int(target_dollars / midday_price)
                if buy_intent['shares'] <= 0:
                    check_log.append(
                        f"  SKIP re-check {symbol}: 0 shares at midday price"
                    )
                    continue
                if buy_intent['shares'] * midday_price > portfolio['cash']:
                    buy_intent['shares'] = int(
                        portfolio['cash'] / midday_price
                    )
                if buy_intent['shares'] > 0:
                    trade = paper_trader.execute_trade(
                        portfolio,
                        buy_intent,
                        portfolio.get('current_regime', 'unknown'),
                        universe_df,
                        transaction_cost_config=transaction_cost_config,
                    )
                    actions_taken.append(trade)
                    gap_pct = (midday_price / intent_price - 1) * 100
                    check_log.append(
                        f"  RE-EXECUTED BUY {symbol} "
                        f"@ ${midday_price:.2f} "
                        f"(gap now {gap_pct:+.1f}%) simulated"
                    )
            else:
                check_log.append(f"  Gap still too wide for {symbol}: {msg}")
    elif skipped_buys and circuit_breaker_active:
        check_log.append(
            f"Skipping {len(skipped_buys)} buy re-checks: "
            f"circuit breaker active"
        )
    else:
        check_log.append("No skipped buys to re-check")

    # --- Post-check: update portfolio state ---
    # Update valuations with midday prices (whether or not trades occurred).
    portfolio = morning_executor._update_valuations_from_quotes(
        portfolio, midday_quotes
    )

    # Update peak prices for all holdings
    for holding in portfolio.get('holdings', []):
        midday_price = price_map.get(holding['symbol'])
        if midday_price and midday_price > holding.get('peak_price', 0):
            holding['peak_price'] = midday_price

    # Save updated portfolio state to S3 (single-book invariant: published under
    # the role-marked sim-book shape, never a raw live-portfolio shape).
    portfolio['date'] = run_date
    s3.write_json(
        paper_trader.to_published_state(portfolio),
        f'daily/{run_date}/portfolio_state.json',
    )

    # Save midday check report
    midday_report = {
        'run_date': run_date,
        'timestamp': datetime.now().isoformat(),
        'phase': 'midday-check',
        'circuit_breaker_active': circuit_breaker_active,
        'stops_triggered': len([a for a in actions_taken if a.get('reason') == 'MIDDAY_STOP_HIT']),
        'buys_reattempted': len([a for a in actions_taken if a.get('reason') == 'MIDDAY_REATTEMPT']),
        'total_actions': len(actions_taken),
        'check_log': check_log,
    }
    s3.write_json(midday_report, f'daily/{run_date}/midday_check_report.json')

    check_log.append(
        f"Midday check complete: {len(actions_taken)} actions taken"
    )

    return {
        'actions_taken': actions_taken,
        'check_log': check_log,
        'circuit_breaker_active': circuit_breaker_active,
        'portfolio_state': portfolio,
    }
