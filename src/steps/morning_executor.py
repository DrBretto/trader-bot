"""Morning execution phase: validate overnight trade intents and execute at market prices.

Supports three execution modes:
- simulated (default): existing paper_trader logic
- alpaca_paper: Alpaca paper trading via broker adapter
- alpaca_live: Alpaca live trading via broker adapter
"""

import logging
import hashlib
import math
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from src.brokers.base import BaseBroker
from src.brokers.router import SimulatedBroker
from src.steps import ingest_prices, paper_trader
from src.utils.s3_client import S3Client

logger = logging.getLogger(__name__)

DUST_SHARE_EPSILON = 0.001
DUST_VALUE_EPSILON = 0.01


# Maximum age for trade intents (calendar days).
# Friday night → Monday morning = 3 days, so 3 is the minimum safe value.
MAX_INTENT_AGE_DAYS = 3

# Maximum price gap allowed for BUY intents.
# If morning price differs from intent price by more than this, skip the buy.
BUY_PRICE_GAP_THRESHOLD = 0.05

BROKER_TERMINAL_ORDER_STATUSES = {
    'filled',
    'partially_filled',
    'canceled',
    'cancelled',
    'expired',
    'rejected',
    'suspended',
}


def load_trade_intents(s3: S3Client) -> Optional[Dict[str, Any]]:
    """Load the most recent trade intents from S3."""
    latest = s3.read_json('daily/latest.json')
    if not latest:
        return None

    intents_date = latest.get('intents_date')
    if not intents_date:
        return None

    return s3.read_json(f'daily/{intents_date}/trade_intents.json')


def validate_intent_freshness(intents: Dict[str, Any]) -> bool:
    """Check that intents are not stale (within MAX_INTENT_AGE_DAYS)."""
    generated_date = intents.get('generated_date')
    if not generated_date:
        return False

    gen_dt = datetime.strptime(generated_date, '%Y-%m-%d')
    age_days = (datetime.now() - gen_dt).days
    return age_days <= MAX_INTENT_AGE_DAYS


def validate_buy_intent(intent: Dict, morning_price: float) -> Tuple[bool, str]:
    """Validate a BUY intent against morning price."""
    intent_price = intent.get('price', 0)
    if intent_price <= 0:
        return False, "Invalid intent price"

    gap_pct = abs(morning_price - intent_price) / intent_price
    if gap_pct > BUY_PRICE_GAP_THRESHOLD:
        return False, (
            f"Price gap {gap_pct:.1%} exceeds "
            f"{BUY_PRICE_GAP_THRESHOLD:.0%} threshold "
            f"(intent ${intent_price:.2f} → morning ${morning_price:.2f})"
        )

    return True, "OK"


def validate_sell_intent(
    intent: Dict,
    morning_price: float,
    holding: Dict,
    params: Dict
) -> Tuple[bool, str]:
    """Re-evaluate a SELL intent against morning price.

    Health collapse, regime panic, LLM veto, and leverage cap: always execute.
    Trailing stop: re-check with morning price (cancel if recovered).
    """
    reason = intent.get('reason', '')

    # Non-negotiable sells: execute regardless of morning price
    if reason in ('HEALTH_COLLAPSE', 'REGIME_PANIC', 'LLM_VETO', 'LEVERAGE_HOLD_CAP'):
        return True, f"Executing {reason} sell at morning price"

    # Trailing stop: re-check with morning price
    if reason == 'STOP_HIT':
        peak_price = holding.get('peak_price', intent.get('price', 0))
        is_leveraged = holding.get('leverage_flag', 0) == 1
        stop_pct = params.get(
            'trailing_stop_leveraged' if is_leveraged else 'trailing_stop_base',
            0.10
        )
        trailing_stop_price = peak_price * (1 - stop_pct)

        if morning_price <= trailing_stop_price:
            return True, (
                f"Stop still hit: ${morning_price:.2f} <= "
                f"stop ${trailing_stop_price:.2f} (peak ${peak_price:.2f})"
            )
        else:
            return False, (
                f"Price recovered: ${morning_price:.2f} > "
                f"stop ${trailing_stop_price:.2f}"
            )

    # Unknown reason: execute conservatively
    return True, f"Executing sell ({reason}) at morning price"


def _update_valuations_from_quotes(
    portfolio: Dict[str, Any],
    quotes_df: pd.DataFrame
) -> Dict[str, Any]:
    """Update portfolio holdings valuations from morning quotes."""
    if len(quotes_df) == 0:
        return portfolio

    price_map = {row['symbol']: row['price'] for _, row in quotes_df.iterrows()}

    holdings_value = 0.0
    for holding in portfolio.get('holdings', []):
        symbol = holding['symbol']
        price = price_map.get(
            symbol,
            holding.get('current_price', holding.get('entry_price', 0))
        )

        if price > holding.get('peak_price', 0):
            holding['peak_price'] = price

        holding['current_price'] = price
        holding['market_value'] = holding['shares'] * price
        holding['unrealized_pnl'] = (price - holding['entry_price']) * holding['shares']
        holding['unrealized_pnl_pct'] = (
            price / holding['entry_price'] - 1
        ) if holding['entry_price'] > 0 else 0

        # Update days held
        entry_date = holding.get('entry_date')
        if entry_date:
            holding['days_held'] = (datetime.now() - pd.to_datetime(entry_date)).days

        holdings_value += holding['market_value']

    portfolio['holdings_value'] = holdings_value
    portfolio['invested'] = holdings_value
    portfolio['portfolio_value'] = portfolio['cash'] + holdings_value
    portfolio['last_updated'] = datetime.now().isoformat()

    # Update SPY benchmark (dividend-adjusted total return)
    spy_price = price_map.get('SPY')
    if spy_price and portfolio.get('benchmark_start_price'):
        # Migrate legacy portfolios that lack benchmark_shares
        if portfolio.get('benchmark_shares') is None:
            start_price = portfolio['benchmark_start_price']
            if start_price > 0:
                portfolio['benchmark_shares'] = 100000 / start_price

        # Reinvest estimated daily dividends (~1.3% annual yield)
        shares = portfolio.get('benchmark_shares', 0)
        if shares > 0:
            daily_div_per_share = spy_price * (0.013 / 252)
            div_cash = shares * daily_div_per_share
            portfolio['benchmark_shares'] = shares + (div_cash / spy_price)
            portfolio['benchmark_value'] = portfolio['benchmark_shares'] * spy_price

    return portfolio


def _execute_via_broker(
    broker: BaseBroker,
    intent: Dict[str, Any],
    morning_price: float,
    run_date: str,
) -> Dict[str, Any]:
    """Execute a single trade via the broker adapter.

    Returns a trade record dict compatible with paper_trader records.
    """
    symbol = intent['symbol']
    action_type = intent['action']
    side = 'buy' if action_type == 'BUY' else 'sell'

    # Include action_type so SELL and REDUCE on the same symbol/day never collide.
    raw_id = f"{run_date}|{symbol}|{action_type}|{side}"
    client_order_id = f"tb-{hashlib.sha256(raw_id.encode()).hexdigest()[:16]}"

    order_result: Dict[str, Any]
    submitted_qty: float = 0.0
    submitted_notional: float = 0.0
    if action_type == 'BUY':
        target_dollars = intent.get(
            'dollars', intent.get('shares', 0) * intent.get('price', 0)
        )
        submitted_notional = float(target_dollars or 0)
        order_result = broker.submit_order(
            symbol=symbol,
            side='buy',
            dollars=target_dollars,
            client_order_id=client_order_id,
        )
    else:
        # SELL / REDUCE: use qty from holding
        qty = float(intent.get('shares', 0) or 0)
        if action_type == 'REDUCE':
            qty *= 0.5
        # Floor-truncate to 6 decimals: never request more than available
        qty = math.floor(qty * 1e6) / 1e6
        if qty <= 0:
            raise ValueError(
                f"Computed non-positive sell qty for {action_type} {symbol}: {qty}"
            )
        submitted_qty = qty
        order_result = broker.submit_order(
            symbol=symbol,
            side='sell',
            qty=qty,
            client_order_id=client_order_id,
        )

    order_result = _await_broker_order_update(broker, order_result)

    try:
        share_value = (
            order_result.get('filled_qty')
            if order_result.get('filled_qty') is not None
            else order_result.get('qty')
        )
        executed_shares = float(share_value) if share_value is not None else 0.0
    except (TypeError, ValueError):
        executed_shares = 0.0
    if executed_shares <= 0:
        executed_shares = submitted_qty if submitted_qty > 0 else float(intent.get('shares', 0) or 0)

    try:
        executed_notional = float(order_result.get('notional')) if order_result.get('notional') is not None else 0.0
    except (TypeError, ValueError):
        executed_notional = 0.0
    if executed_notional <= 0:
        try:
            filled_qty = float(order_result.get('filled_qty') or 0.0)
            filled_avg_price = float(order_result.get('filled_avg_price') or 0.0)
            if filled_qty > 0 and filled_avg_price > 0:
                executed_notional = filled_qty * filled_avg_price
        except (TypeError, ValueError):
            pass
    if executed_notional <= 0:
        executed_notional = submitted_notional if submitted_notional > 0 else float(intent.get('dollars', 0) or 0)

    # Build trade record compatible with paper_trader format
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action_type,
        'shares': executed_shares,
        'price': morning_price,
        'market_price': morning_price,
        'dollars': round(executed_notional, 2),
        'reason': intent.get('reason', ''),
        'regime': intent.get('regime', ''),
        'broker_order_id': order_result.get('order_id'),
        'broker_client_order_id': order_result.get('client_order_id'),
        'broker_status': order_result.get('status'),
        'execution_mode': broker.mode_label,
    }


def _await_broker_order_update(
    broker: BaseBroker,
    order_result: Dict[str, Any],
    timeout_sec: float = 8.0,
    poll_interval_sec: float = 0.5,
) -> Dict[str, Any]:
    """Poll broker order status briefly so reconciliation sees near-immediate fills."""
    order_id = order_result.get('order_id')
    status = str(order_result.get('status') or '').lower()

    if not order_id or status in BROKER_TERMINAL_ORDER_STATUSES:
        return order_result

    latest_raw = order_result.get('raw') or {}
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        time.sleep(poll_interval_sec)
        try:
            latest_raw = broker.get_order(order_id) or latest_raw
        except Exception as exc:
            logger.warning("Broker order refresh failed for %s: %s", order_id, exc)
            break

        status = str(latest_raw.get('status') or status).lower()
        if status in BROKER_TERMINAL_ORDER_STATUSES:
            break

    merged = dict(order_result)
    merged['status'] = latest_raw.get('status', order_result.get('status'))
    merged['qty'] = latest_raw.get('qty', order_result.get('qty'))
    merged['filled_qty'] = latest_raw.get('filled_qty')
    merged['notional'] = latest_raw.get('notional', order_result.get('notional'))
    merged['filled_avg_price'] = latest_raw.get('filled_avg_price')
    merged['raw'] = latest_raw
    return merged


def _reconcile_portfolio_from_broker_with_retry(
    broker: BaseBroker,
    portfolio: Dict[str, Any],
    expected_buy_symbols: Optional[List[str]] = None,
    timeout_sec: float = 8.0,
    poll_interval_sec: float = 0.5,
) -> Dict[str, Any]:
    """Retry reconciliation briefly so fresh fills show up in published holdings."""
    expected = set(expected_buy_symbols or [])
    deadline = time.time() + timeout_sec
    latest = _reconcile_portfolio_from_broker(broker, portfolio)

    if not expected:
        return latest

    while time.time() < deadline:
        current_symbols = {h.get('symbol') for h in latest.get('holdings', [])}
        if expected.issubset(current_symbols):
            return latest
        time.sleep(poll_interval_sec)
        latest = _reconcile_portfolio_from_broker(broker, portfolio)

    return latest


def _reconcile_portfolio_from_broker(
    broker: BaseBroker,
    portfolio: Dict[str, Any],
) -> Dict[str, Any]:
    """Sync local portfolio state with broker positions and account."""
    try:
        acct = broker.check_account()
        positions = broker.list_positions()
    except Exception as exc:
        logger.warning("Broker reconciliation failed: %s", exc)
        return portfolio

    # Update cash from broker
    portfolio['cash'] = acct['cash']

    # Rebuild holdings from broker positions
    broker_holdings = []
    holdings_value = 0.0
    existing_map = {h['symbol']: h for h in portfolio.get('holdings', [])}

    for pos in positions:
        qty = float(pos['qty'] or 0)
        market_value = float(pos['market_value'] or 0)
        if abs(qty) < DUST_SHARE_EPSILON or abs(market_value) < DUST_VALUE_EPSILON:
            continue

        symbol = pos['symbol']
        existing = existing_map.get(symbol, {})
        entry_price = existing.get('entry_price', pos['avg_entry_price'])
        current_price = pos['current_price']
        # F-4 / F-5: unrealized_pnl and unrealized_pnl_pct must use the SAME
        # cost basis. The prior code used pos['unrealized_pl'] (broker basis)
        # for $ and existing['entry_price'] (local basis) for %. When the
        # broker's avg_entry_price drifts from local entry_price (e.g. VUG
        # post-cutover) the two fields disagree on sign and magnitude. The
        # local-basis path preserves continuity-bridge intent (broker_total_value
        # is exposed separately for broker-truth reconciliation).
        if entry_price > 0:
            unrealized_pnl = (current_price - entry_price) * qty
            unrealized_pnl_pct = current_price / entry_price - 1
        else:
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0
        holding = {
            'symbol': symbol,
            'shares': qty,
            'entry_price': entry_price,
            'entry_date': existing.get('entry_date', datetime.now().isoformat()),
            'peak_price': max(
                existing.get('peak_price', 0), current_price
            ),
            'current_price': current_price,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'entry_regime': existing.get('entry_regime', 'unknown'),
            'entry_health': existing.get('entry_health', 0.5),
            'peak_health': existing.get('peak_health', 0.5),
            'asset_class': existing.get('asset_class', 'equity'),
            'sector': existing.get('sector', 'broad'),
            'leverage_flag': existing.get('leverage_flag', 0),
        }
        broker_holdings.append(holding)
        holdings_value += market_value

    portfolio['holdings'] = broker_holdings
    portfolio['holdings_value'] = holdings_value
    portfolio['invested'] = holdings_value
    portfolio['portfolio_value'] = acct['equity']
    portfolio['last_updated'] = datetime.now().isoformat()
    portfolio['broker_reconciled'] = True

    return portfolio


def run(bucket: str, config: Dict[str, Any], broker: Optional[BaseBroker] = None) -> Dict[str, Any]:
    """
    Execute morning phase: validate intents, fetch prices, execute trades.

    Args:
        bucket: S3 bucket name
        config: Pipeline config (decision_params, portfolio_state, universe)
        broker: Optional broker adapter. If None or SimulatedBroker, use paper_trader.

    Returns:
        Dict with portfolio_state, trades, validation_log, morning_prices,
        intents_found, intents_executed
    """
    s3 = S3Client(bucket)
    params = config.get('decision_params', {})
    transaction_cost_config = config.get('transaction_cost_overrides')
    validation_log: List[str] = []

    # Load trade intents
    intents = load_trade_intents(s3)

    if intents is None:
        validation_log.append("No trade intents found")
        portfolio = paper_trader.load_portfolio_state(s3)
        held_symbols = [h['symbol'] for h in portfolio.get('holdings', [])]
        if held_symbols:
            morning_quotes = ingest_prices.fetch_morning_quotes(
                list(set(held_symbols + ['SPY'])), broker=broker
            )
            portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
        return {
            'portfolio_state': portfolio,
            'trades': [],
            'morning_prices': pd.DataFrame(),
            'validation_log': validation_log,
            'intents_found': False,
            'intents_executed': 0
        }

    # Check freshness
    if not validate_intent_freshness(intents):
        validation_log.append(
            f"Intents stale (generated {intents.get('generated_date')}, "
            f"max age {MAX_INTENT_AGE_DAYS} days)"
        )
        portfolio = paper_trader.load_portfolio_state(s3)
        held_symbols = [h['symbol'] for h in portfolio.get('holdings', [])]
        if held_symbols:
            morning_quotes = ingest_prices.fetch_morning_quotes(
                list(set(held_symbols + ['SPY'])), broker=broker
            )
            portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
        return {
            'portfolio_state': portfolio,
            'trades': [],
            'morning_prices': pd.DataFrame(),
            'validation_log': validation_log,
            'intents_found': True,
            'intents_stale': True,
            'intents_executed': 0
        }

    validation_log.append(
        f"Loaded intents from {intents.get('generated_date')} "
        f"({len(intents.get('actions', []))} actions)"
    )

    # Collect all symbols we need morning prices for
    intent_actions = intents.get('actions', [])
    intent_symbols = list(set(a['symbol'] for a in intent_actions))

    portfolio = paper_trader.load_portfolio_state(s3)
    held_symbols = [h['symbol'] for h in portfolio.get('holdings', [])]
    all_symbols = list(set(intent_symbols + held_symbols + ['SPY']))

    # Fetch morning prices (broker snapshots preferred, yfinance/Stooq fallback)
    print(f"Fetching morning quotes for {len(all_symbols)} symbols...")
    morning_quotes = ingest_prices.fetch_morning_quotes(all_symbols, broker=broker)

    if len(morning_quotes) == 0:
        validation_log.append("CRITICAL: No morning quotes fetched, skipping all trades")
        return {
            'portfolio_state': portfolio,
            'trades': [],
            'morning_prices': morning_quotes,
            'validation_log': validation_log,
            'intents_found': True,
            'intents_executed': 0
        }

    morning_price_map = {
        row['symbol']: row['price'] for _, row in morning_quotes.iterrows()
    }
    validation_log.append(f"Fetched morning prices for {len(morning_price_map)} symbols")

    # Load universe for execute_trade
    universe_df = config.get('universe', pd.DataFrame())
    if len(universe_df) == 0:
        universe_df = s3.read_csv('config/universe.csv')

    regime_label = intents.get('regime', 'risk_on_trend')
    holding_map = {h['symbol']: h for h in portfolio.get('holdings', [])}
    run_date = datetime.now().strftime('%Y-%m-%d')

    # Determine execution mode
    use_broker = broker is not None and not isinstance(broker, SimulatedBroker)

    if use_broker:
        # Pre-trade account health check
        try:
            acct = broker.check_account()
            if not acct['tradable']:
                validation_log.append(
                    f"ABORT: broker account not tradable (status={acct['status']})"
                )
                use_broker = False
            else:
                validation_log.append(
                    f"Broker account OK: ${acct['buying_power']:,.2f} buying power, "
                    f"mode={broker.mode_label}"
                )
        except Exception as exc:
            validation_log.append(f"ABORT broker: account check failed: {exc}")
            use_broker = False

    # Process each intent
    trades: List[Dict[str, Any]] = []
    skipped_buys: List[Dict[str, Any]] = []
    for intent in intent_actions:
        symbol = intent['symbol']
        morning_price = morning_price_map.get(symbol)

        if morning_price is None:
            validation_log.append(f"SKIP {intent['action']} {symbol}: no morning price")
            continue

        if intent['action'] == 'BUY':
            valid, msg = validate_buy_intent(intent, morning_price)
            if not valid:
                validation_log.append(f"SKIP BUY {symbol}: {msg}")
                skipped_buys.append({
                    'symbol': symbol,
                    'intent_price': intent.get('price', 0),
                    'morning_price': morning_price,
                    'dollars': intent.get('dollars', intent.get('shares', 0) * intent.get('price', 0)),
                    'shares': intent.get('shares', 0),
                    'reason': msg,
                    'skip_type': 'price_gap',
                })
                continue

            # Recompute target dollars at morning price (same dollar amount as intent)
            target_dollars = intent.get('dollars', intent.get('shares', 0) * intent.get('price', 0))
            min_order = params.get('min_order_dollars', 250)

            if use_broker:
                # Broker mode: use notional dollars directly (fractional support)
                if target_dollars < min_order:
                    validation_log.append(
                        f"SKIP BUY {symbol}: ${target_dollars:.2f} "
                        f"below min order ${min_order}"
                    )
                    continue

                try:
                    broker_intent = {
                        **intent,
                        'price': morning_price,
                        'dollars': target_dollars,
                    }
                    trade = _execute_via_broker(
                        broker, broker_intent, morning_price, run_date
                    )
                    trade['regime'] = regime_label
                    trades.append(trade)
                    gap_pct = (morning_price / intent['price'] - 1) * 100
                    validation_log.append(
                        f"BUY ${target_dollars:.2f} {symbol} @ ${morning_price:.2f} "
                        f"(intent: ${intent['price']:.2f}, gap: {gap_pct:+.1f}%) "
                        f"[{broker.mode_label}]"
                    )
                except Exception as exc:
                    validation_log.append(
                        f"FAIL BUY {symbol}: broker error: {exc}"
                    )
            else:
                # Simulated mode: whole-share floor
                shares = int(target_dollars / morning_price)
                if shares <= 0 or shares * morning_price < min_order:
                    validation_log.append(
                        f"SKIP BUY {symbol}: {shares} shares @ ${morning_price:.2f} "
                        f"below min order ${min_order}"
                    )
                    continue

                # Check cash
                if shares * morning_price > portfolio['cash']:
                    shares = int(portfolio['cash'] / morning_price)
                    if shares <= 0:
                        validation_log.append(f"SKIP BUY {symbol}: insufficient cash")
                        continue

                adjusted_intent = {
                    **intent,
                    'price': morning_price,
                    'shares': shares,
                    'dollars': shares * morning_price
                }
                trade = paper_trader.execute_trade(
                    portfolio,
                    adjusted_intent,
                    regime_label,
                    universe_df,
                    transaction_cost_config=transaction_cost_config,
                )
                trades.append(trade)
                gap_pct = (morning_price / intent['price'] - 1) * 100
                validation_log.append(
                    f"BUY {shares} {symbol} @ ${morning_price:.2f} "
                    f"(intent: ${intent['price']:.2f}, gap: {gap_pct:+.1f}%)"
                )

        elif intent['action'] in ('SELL', 'REDUCE'):
            holding = holding_map.get(symbol)
            if holding is None:
                validation_log.append(f"SKIP {intent['action']} {symbol}: not in holdings")
                continue

            valid, msg = validate_sell_intent(intent, morning_price, holding, params)
            if not valid:
                validation_log.append(f"CANCEL {intent['action']} {symbol}: {msg}")
                continue

            if use_broker:
                holding_shares = float(holding.get('shares', intent.get('shares', 0)) or 0)
                # Skip dust positions (sub-penny value, untradeable on any broker)
                if holding_shares < 0.001:
                    validation_log.append(
                        f"SKIP {intent['action']} {symbol}: dust position "
                        f"({holding_shares:.9g} shares)"
                    )
                    continue

                try:
                    broker_intent = {
                        **intent,
                        'price': morning_price,
                        'shares': holding_shares,
                    }
                    trade = _execute_via_broker(
                        broker, broker_intent, morning_price, run_date
                    )
                    trade['regime'] = regime_label
                    trades.append(trade)
                    validation_log.append(
                        f"{intent['action']} {symbol} @ ${morning_price:.2f}: "
                        f"{msg} [{broker.mode_label}]"
                    )
                except Exception as exc:
                    validation_log.append(
                        f"FAIL {intent['action']} {symbol}: broker error: {exc}"
                    )
            else:
                adjusted_intent = {**intent, 'price': morning_price}
                trade = paper_trader.execute_trade(
                    portfolio,
                    adjusted_intent,
                    regime_label,
                    universe_df,
                    transaction_cost_config=transaction_cost_config,
                )
                trades.append(trade)
                validation_log.append(
                    f"{intent['action']} {intent.get('shares', '?')} {symbol} "
                    f"@ ${morning_price:.2f}: {msg}"
                )

    # Post-execution: reconcile broker truth when broker mode is active.
    # Include ALL submitted buys (not just confirmed fills) in the expected set
    # so that the retry loop waits for fast fills to settle in the positions API.
    # Only exclude buys that were definitively rejected or canceled.
    if use_broker:
        _REJECTED_ORDER_STATUSES = {
            'rejected', 'canceled', 'cancelled', 'expired', 'suspended',
        }
        expected_buy_symbols = [
            trade['symbol']
            for trade in trades
            if trade.get('action') == 'BUY'
            and str(trade.get('broker_status') or '').lower()
                not in _REJECTED_ORDER_STATUSES
        ]
        reconciled = _reconcile_portfolio_from_broker_with_retry(
            broker, portfolio, expected_buy_symbols=expected_buy_symbols
        )
        if reconciled.get('broker_reconciled'):
            portfolio = reconciled
        else:
            portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
    else:
        portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)

    # Compute stats
    try:
        portfolio = paper_trader.compute_portfolio_stats(portfolio, s3)
    except Exception as e:
        validation_log.append(f"Stats computation warning: {e}")

    portfolio['trades_today'] = trades

    execution_mode = broker.mode_label if use_broker else 'simulated'
    print(f"Morning execution ({execution_mode}): {len(trades)} trades, "
          f"portfolio ${portfolio['portfolio_value']:,.2f}")

    return {
        'portfolio_state': portfolio,
        'trades': trades,
        'morning_prices': morning_quotes,
        'validation_log': validation_log,
        'intents_found': True,
        'intents_executed': len(trades),
        'execution_mode': execution_mode,
        'skipped_buys': skipped_buys,
    }
