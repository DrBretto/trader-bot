"""Morning execution phase: validate overnight trade intents and execute at market prices.

Pure trading simulation: executes via paper_trader logic.
"""

import logging
import pandas as pd
from datetime import date, datetime
from typing import Dict, Any, List, Tuple, Optional

from chassis.steps import ingest_prices, paper_trader
from chassis.utils.market_calendar import (
    is_trading_session,
    ny_today,
    trading_sessions_between,
)
from chassis.utils.s3_client import S3Client

logger = logging.getLogger(__name__)

DUST_SHARE_EPSILON = 0.001
DUST_VALUE_EPSILON = 0.01


# Overnight intents are valid for the next real market session only. Counting NYSE
# sessions makes Friday->Monday and a pre-holiday night->next open valid without
# making a several-session-old instruction executable.
MAX_INTENT_AGE_SESSIONS = 1

# Maximum price gap allowed for BUY intents.
# If morning price differs from intent price by more than this, skip the buy.
BUY_PRICE_GAP_THRESHOLD = 0.05


def load_trade_intents(s3: S3Client) -> Optional[Dict[str, Any]]:
    """Load the most recent trade intents from S3."""
    latest = s3.read_json('daily/latest.json')
    if not latest:
        return None

    intents_date = latest.get('intents_date')
    if not intents_date:
        return None

    return s3.read_json(f'daily/{intents_date}/trade_intents.json')


def validate_intent_freshness(
    intents: Dict[str, Any],
    as_of_date: Optional[str] = None,
) -> bool:
    """Check that intents belong to this or the immediately next NYSE session."""
    generated_date = intents.get('generated_date')
    if not generated_date:
        return False
    try:
        generated = date.fromisoformat(str(generated_date)[:10])
        current = date.fromisoformat(as_of_date) if as_of_date else ny_today()
    except (TypeError, ValueError):
        return False
    if not is_trading_session(generated) or not is_trading_session(current):
        return False
    age_sessions = trading_sessions_between(generated, current)
    return 0 <= age_sessions <= MAX_INTENT_AGE_SESSIONS


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


def run(
    bucket: str,
    config: Dict[str, Any],
    run_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute morning phase: validate intents, fetch prices, execute trades.

    Args:
        bucket: S3 bucket name
        config: Pipeline config (decision_params, portfolio_state, universe)

    Returns:
        Dict with portfolio_state, trades, validation_log, morning_prices,
        intents_found, intents_executed
    """
    s3 = S3Client(bucket)
    params = config.get('decision_params', {})
    transaction_cost_config = config.get('transaction_cost_overrides')
    validation_log: List[str] = []

    # Symbols the displayed CHAMPION replay line may hold — a DIFFERENT set than
    # the live paper account. Quote these too so the single-date provisional
    # morning_prices.parquet covers the champion's holdings and today's dot moves
    # on real prices instead of flat-holding at last close. Union of the trading
    # universe and the champion's current holdings (from the last-published
    # dashboard). Best-effort; failures degrade to the live-account symbols only.
    coverage_symbols = set()
    try:
        coverage_symbols.update(s3.read_csv('config/universe.csv')['symbol'].tolist())
    except Exception as exc:
        validation_log.append(f"coverage: universe load failed (non-fatal): {exc}")
    try:
        _dash = s3.read_json('dashboard/dashboard.json') or {}
        coverage_symbols.update(
            h['symbol'] for h in _dash.get('holdings', []) if h.get('symbol')
        )
    except Exception as exc:
        validation_log.append(f"coverage: dashboard holdings load failed (non-fatal): {exc}")

    # Load trade intents
    intents = load_trade_intents(s3)

    if intents is None:
        validation_log.append("No trade intents found")
        portfolio = paper_trader.load_portfolio_state(s3)
        held_symbols = [h['symbol'] for h in portfolio.get('holdings', [])]
        morning_quotes = pd.DataFrame()
        fetch_set = set(held_symbols) | {'SPY'} | coverage_symbols
        if fetch_set:
            morning_quotes = ingest_prices.fetch_morning_quotes(list(fetch_set))
            portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
        return {
            'portfolio_state': portfolio,
            'trades': [],
            'morning_prices': morning_quotes,
            'validation_log': validation_log,
            'intents_found': False,
            'intents_executed': 0
        }

    execution_date = run_date or ny_today().isoformat()

    # Check freshness in real market sessions, not calendar days.
    if not validate_intent_freshness(intents, as_of_date=execution_date):
        validation_log.append(
            f"Intents stale (generated {intents.get('generated_date')}, "
            f"max age {MAX_INTENT_AGE_SESSIONS} trading session)"
        )
        portfolio = paper_trader.load_portfolio_state(s3)
        held_symbols = [h['symbol'] for h in portfolio.get('holdings', [])]
        morning_quotes = pd.DataFrame()
        fetch_set = set(held_symbols) | {'SPY'} | coverage_symbols
        if fetch_set:
            morning_quotes = ingest_prices.fetch_morning_quotes(list(fetch_set))
            portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
        return {
            'portfolio_state': portfolio,
            'trades': [],
            'morning_prices': morning_quotes,
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
    all_symbols = list(set(intent_symbols + held_symbols + ['SPY']) | coverage_symbols)

    # Fetch morning prices (yfinance/Stooq)
    print(f"Fetching morning quotes for {len(all_symbols)} symbols...")
    morning_quotes = ingest_prices.fetch_morning_quotes(all_symbols)

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
    run_date = execution_date

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

            # Simulated execution: whole-share floor
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

    # Post-execution: update valuations from morning quotes.
    portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)

    # Compute stats
    try:
        portfolio = paper_trader.compute_portfolio_stats(portfolio, s3)
    except Exception as e:
        validation_log.append(f"Stats computation warning: {e}")

    portfolio['trades_today'] = trades

    execution_mode = 'simulated'
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
