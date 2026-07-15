"""Paper trader for tracking portfolio state and executing simulated trades."""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

from chassis.utils.s3_client import S3Client
from chassis.utils.dashboard_metrics import compute_trade_and_exposure_metrics
from chassis.utils.transaction_costs import apply_transaction_costs


TRANSIENT_ROLLOVER_KEYS = (
    'external_cashflow',
    'external_cashflow_t',
    'net_external_cashflow',
    'net_cashflow',
    'cashflow',
    'cash_flow',
    'continuity_bridge_marker',
)


# PKT-TB-012 single-book invariant: the internal intent-sizing simulation keeps
# its value under `portfolio_value` in memory, but the PUBLISHED daily artifact
# carries it under `sim_book_value` plus role markers so the file can never be
# read back (by a person or by code) as a second live portfolio. The only
# published portfolio value is the canon line in dashboard.json metrics.
SIM_BOOK_ROLE = "internal_sim_book"

_SIM_BOOK_NOTE = (
    "Internal intent-sizing simulation — NOT a portfolio. The only live book is "
    "the canon line in dashboard.json (metrics.total_value)."
)

# Portfolio-value-shaped top-level keys that must never appear in a published
# daily artifact (mirrors PORTFOLIO_VALUE_SHAPED in the regression lock). The
# internal value is re-keyed to `sim_book_value`; every other variant + the
# legacy broker-era reconciliation flag is stripped at the write boundary.
_PUBLISH_STRIP_KEYS = (
    'portfolio_value',
    'account_value',
    'broker_total_value',
    'broker_reconciled',
    'portfolio_total',
    'book_value',
    'equity',
    'nav',
    'total_value',
)


def to_published_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the internal sim book into its published, role-marked form.

    Re-keys `portfolio_value` -> `sim_book_value`, strips every other
    portfolio-value-shaped / legacy broker field, and stamps role markers.
    Inverse of `_restore_internal_keys`. All other internals (cash, holdings,
    benchmark_* tracking, etc.) pass through untouched.
    """
    value = state.get('portfolio_value')
    published = {k: v for k, v in state.items() if k not in _PUBLISH_STRIP_KEYS}
    if value is not None:
        published['sim_book_value'] = value
    published['book_role'] = SIM_BOOK_ROLE
    published['book_note'] = _SIM_BOOK_NOTE
    return published


def _restore_internal_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of `to_published_state`: published sim-book shape -> internal shape.

    Historical states already keyed `portfolio_value` (pre-PKT-TB-012 raw writes)
    pass through unchanged.
    """
    restored = dict(state)
    if 'sim_book_value' in restored:
        restored['portfolio_value'] = restored.pop('sim_book_value')
    restored.pop('book_role', None)
    restored.pop('book_note', None)
    return restored


def from_published_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Restore a role-marked sim state for private execution/checkpoint use."""
    return _restore_internal_keys(state)


def _normalize_loaded_portfolio_state(
    state: Dict[str, Any],
    state_date: Optional[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """Drop one-day accounting fields when rolling state into a new date."""
    normalized = dict(state)
    if state_date and state_date != as_of_date:
        for key in TRANSIENT_ROLLOVER_KEYS:
            normalized.pop(key, None)
    return normalized


def load_portfolio_state(s3: S3Client) -> Dict[str, Any]:
    """
    Load current portfolio state from S3.

    Returns:
        Portfolio state dict or default if not found
    """
    # Try to load latest portfolio state
    latest = s3.read_json('daily/latest.json')

    if latest is None:
        # Return initial portfolio state
        return {
            'cash': 100000,
            'holdings': [],
            'portfolio_value': 100000,
            'benchmark_value': 100000,
            'benchmark_start_price': None,  # Will be set on first run
            'trades_today': [],
            'last_updated': datetime.now().isoformat()
        }

    # Load portfolio state from the latest date
    latest_date = latest.get('date')
    if latest_date:
        state = s3.read_json(f'daily/{latest_date}/portfolio_state.json')
        if state:
            as_of_date = datetime.now().strftime('%Y-%m-%d')
            state = _restore_internal_keys(state)
            return _normalize_loaded_portfolio_state(state, latest_date, as_of_date)

    return {
        'cash': 100000,
        'holdings': [],
        'portfolio_value': 100000,
        'benchmark_value': 100000,
        'benchmark_start_price': None,
        'trades_today': [],
        'last_updated': datetime.now().isoformat()
    }


def execute_trade(
    portfolio: Dict[str, Any],
    action: Dict[str, Any],
    regime_label: str,
    universe_df: pd.DataFrame,
    timestamp: Optional[datetime] = None,
    rng: Optional[random.Random] = None,
    transaction_cost_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a single trade and return trade record.

    Args:
        portfolio: Current portfolio state
        action: Action dict with symbol, shares, price, action type
        regime_label: Current market regime
        universe_df: Universe DataFrame with asset metadata

    Returns:
        Trade record dict
    """
    symbol = action['symbol']
    shares = action['shares']
    market_price = action['price']
    action_type = action['action']

    # Look up asset metadata for transaction cost tier
    asset_class = 'equity'
    sector = 'broad'
    if len(universe_df) > 0:
        symbol_row = universe_df[universe_df['symbol'] == symbol]
        if len(symbol_row) > 0:
            asset_class = symbol_row.iloc[0].get('asset_class', 'equity')
            sector = symbol_row.iloc[0].get('sector', 'broad')

    # Apply bid-ask spread + slippage
    fill_price, cost_bps = apply_transaction_costs(
        market_price,
        action_type,
        sector=sector,
        asset_class=asset_class,
        rng=rng,
        cost_config=transaction_cost_config,
    )

    # Use fill_price for all cash/P&L math
    price = fill_price

    trade_time = timestamp or datetime.now()

    trade_record = {
        'timestamp': trade_time.isoformat(),
        'symbol': symbol,
        'action': action_type,
        'shares': shares,
        'price': round(fill_price, 4),
        'market_price': round(market_price, 4),
        'dollars': round(shares * fill_price, 2),
        'transaction_cost_bps': round(cost_bps, 2),
        'reason': action.get('reason', ''),
        'regime': regime_label
    }

    # Track cumulative transaction costs
    cost_dollars = abs(fill_price - market_price) * shares
    portfolio.setdefault('cumulative_transaction_costs', 0.0)
    portfolio['cumulative_transaction_costs'] += cost_dollars

    if action_type == 'BUY':
        # Deduct cash
        portfolio['cash'] -= shares * price

        # Get leverage flag from universe
        leverage_flag = 0
        if len(universe_df) > 0:
            symbol_row = universe_df[universe_df['symbol'] == symbol]
            if len(symbol_row) > 0:
                leverage_flag = symbol_row.iloc[0].get('leverage_flag', 0)

        # Add to holdings
        portfolio['holdings'].append({
            'symbol': symbol,
            'shares': shares,
            'entry_price': price,
            'entry_date': trade_time.isoformat(),
            'peak_price': price,
            'entry_regime': regime_label,
            'entry_health': action.get('health', 0.5),
            'peak_health': action.get('health', 0.5),
            'asset_class': asset_class,
            'sector': sector,
            'leverage_flag': leverage_flag
        })

        trade_record['entry_price'] = price

    elif action_type == 'SELL':
        # Find the holding
        holding_idx = None
        for i, h in enumerate(portfolio['holdings']):
            if h['symbol'] == symbol:
                holding_idx = i
                break

        if holding_idx is not None:
            holding = portfolio['holdings'][holding_idx]

            # Calculate P&L
            entry_price = holding['entry_price']
            pnl = (price - entry_price) * shares
            pnl_pct = (price / entry_price - 1) if entry_price > 0 else 0

            trade_record['entry_price'] = entry_price
            trade_record['pnl'] = pnl
            trade_record['pnl_pct'] = pnl_pct
            # entry_date may be absent on holdings written by a resim/reconcile
            # (those carry entry_price + days_held but not entry_date). Fall back
            # to the holding's own days_held rather than KeyError out of the whole
            # morning run (the SELL still executes; only the derived metric differs).
            entry_date = holding.get('entry_date')
            if entry_date:
                trade_record['days_held'] = (
                    trade_time - pd.to_datetime(entry_date)
                ).days
            else:
                trade_record['days_held'] = int(holding.get('days_held', 0) or 0)

            # Add cash
            portfolio['cash'] += shares * price

            # Remove from holdings
            portfolio['holdings'].pop(holding_idx)

    elif action_type == 'REDUCE':
        # Find the holding
        for h in portfolio['holdings']:
            if h['symbol'] == symbol:
                # Exact-share trims (PKT-TB-004 exposure_trim candidate)
                # carry 'reduce_shares'; legacy REDUCE intents halve the
                # position (behavior unchanged when the field is absent).
                reduce_shares = action.get('reduce_shares')
                if reduce_shares is None:
                    reduce_shares = shares // 2
                reduce_shares = min(reduce_shares, h['shares'])
                if reduce_shares > 0:
                    portfolio['cash'] += reduce_shares * price
                    h['shares'] -= reduce_shares

                    trade_record['shares'] = reduce_shares
                    trade_record['dollars'] = reduce_shares * price
                break

    return trade_record


def update_portfolio_values(
    portfolio: Dict[str, Any],
    prices_df: pd.DataFrame,
    current_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Update portfolio values based on current prices.

    Args:
        portfolio: Current portfolio state
        prices_df: Current price data

    Returns:
        Updated portfolio state
    """
    valuation_time = current_time or datetime.now()

    holdings_value = 0

    for holding in portfolio['holdings']:
        symbol = holding['symbol']

        # Get current price
        symbol_prices = prices_df[prices_df['symbol'] == symbol]
        if len(symbol_prices) > 0:
            current_price = symbol_prices.sort_values('date')['close'].iloc[-1]
        else:
            current_price = holding.get('entry_price', 0)

        # Update peak price
        if current_price > holding.get('peak_price', 0):
            holding['peak_price'] = current_price

        holding['current_price'] = current_price
        holding['market_value'] = holding['shares'] * current_price
        holding['unrealized_pnl'] = (current_price - holding['entry_price']) * holding['shares']
        holding['unrealized_pnl_pct'] = (
            current_price / holding['entry_price'] - 1
        ) if holding['entry_price'] > 0 else 0

        # Compute days held
        entry_date = holding.get('entry_date')
        if entry_date:
            holding['days_held'] = (valuation_time - pd.to_datetime(entry_date)).days
        else:
            holding['days_held'] = 0

        holdings_value += holding['market_value']

    portfolio['holdings_value'] = holdings_value
    portfolio['invested'] = holdings_value
    portfolio['portfolio_value'] = portfolio['cash'] + holdings_value
    portfolio['last_updated'] = valuation_time.isoformat()

    # Update SPY buy-and-hold benchmark (dividend-adjusted total return)
    spy_prices = prices_df[prices_df['symbol'] == 'SPY']
    if len(spy_prices) > 0:
        current_spy_price = spy_prices.sort_values('date')['close'].iloc[-1]

        # Initialize benchmark on first run
        if portfolio.get('benchmark_start_price') is None:
            portfolio['benchmark_start_price'] = current_spy_price
            portfolio['benchmark_shares'] = 100000 / current_spy_price
            portfolio['benchmark_value'] = 100000
        else:
            # Migrate legacy portfolios that lack benchmark_shares
            if portfolio.get('benchmark_shares') is None:
                start_price = portfolio['benchmark_start_price']
                if start_price > 0:
                    portfolio['benchmark_shares'] = 100000 / start_price

            # Reinvest estimated daily dividends (~1.3% annual yield)
            shares = portfolio.get('benchmark_shares', 0)
            if shares > 0:
                daily_div_per_share = current_spy_price * (0.013 / 252)
                div_cash = shares * daily_div_per_share
                portfolio['benchmark_shares'] = shares + (div_cash / current_spy_price)
                portfolio['benchmark_value'] = portfolio['benchmark_shares'] * current_spy_price

    return portfolio


def compute_portfolio_stats(
    portfolio: Dict[str, Any],
    s3: S3Client
) -> Dict[str, Any]:
    """Compute lifecycle stats: trade/exposure from fills, LINE stats from the ledger.

    Clean core: the displayed equity LINE (ytd/mtd/sharpe/max-dd) is the STORED
    ledger, not a recompute of this internal sim book. Trade + exposure stats come
    from the FIFO round-trip accounting over fills + the current posture.
    """
    trade_exp = compute_trade_and_exposure_metrics(
        s3=s3, portfolio_state=portfolio, current_state=portfolio, max_days=730,
    )
    metrics = trade_exp['metrics']

    # LINE stats from the stored ledger (the displayed, anchored series).
    try:
        from lines.line import load_line_view
        lm = load_line_view(s3.s3).get('line_metrics', {}) or {}
    except Exception:
        lm = {}
    portfolio['ytd_return'] = lm.get('ytd_return')
    portfolio['mtd_return'] = lm.get('mtd_return')
    portfolio['sharpe_ratio'] = lm.get('sharpe_ratio')
    portfolio['sharpe_observations'] = lm.get('sharpe_observations')
    portfolio['max_drawdown'] = lm.get('max_drawdown')
    portfolio['current_drawdown'] = lm.get('current_drawdown')
    portfolio['win_rate'] = metrics['win_rate']
    portfolio['total_trades'] = metrics['total_trades']
    portfolio['wins'] = metrics['wins']
    portfolio['losses'] = metrics['losses']
    portfolio['breakeven_trades'] = metrics['breakeven_trades']
    portfolio['realized_round_trips'] = metrics['realized_round_trips']
    portfolio['total_fills'] = metrics['total_fills']
    # Keep this derivable from fills. Falls back to existing value only if necessary.
    portfolio['cumulative_transaction_costs'] = metrics.get(
        'cumulative_transaction_costs',
        portfolio.get('cumulative_transaction_costs', 0.0),
    )
    portfolio['cash_pct'] = metrics['cash_pct']
    portfolio['gross_exposure'] = metrics['gross_exposure']
    portfolio['net_exposure'] = metrics['net_exposure']
    portfolio['top_position_pct'] = metrics['top_position_pct']
    portfolio['beta_proxy'] = metrics['beta_proxy']
    portfolio['metrics_reset_boundary'] = None  # no recompute segment (clean core)

    return portfolio


def run(
    decisions: Dict[str, Any],
    prices_df: pd.DataFrame,
    bucket: str,
    transaction_cost_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Execute paper trades based on decisions.

    Args:
        decisions: Decisions from decision engine
        prices_df: Current price data
        bucket: S3 bucket name

    Returns:
        Tuple of (updated portfolio state, list of trade records)
    """
    print("Executing paper trades...")

    s3 = S3Client(bucket)

    # Load current portfolio state
    portfolio = load_portfolio_state(s3)
    regime_label = decisions.get('regime', 'risk_on_trend')

    # Load universe for asset metadata
    universe_df = s3.read_csv('config/universe.csv')
    if len(universe_df) == 0:
        universe_df = pd.DataFrame()

    # Execute each action
    trades = []
    for action in decisions.get('actions', []):
        trade = execute_trade(
            portfolio,
            action,
            regime_label,
            universe_df,
            transaction_cost_config=transaction_cost_config,
        )
        trades.append(trade)
        print(f"  {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

    # Update portfolio values
    portfolio = update_portfolio_values(portfolio, prices_df)

    # Compute performance stats (ytd, sharpe, etc.)
    try:
        portfolio = compute_portfolio_stats(portfolio, s3)
    except Exception as e:
        print(f"  Warning: stats computation failed: {e}")

    # Store trades for today
    portfolio['trades_today'] = trades

    print(f"  Portfolio value: ${portfolio['portfolio_value']:,.2f}")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"  Holdings: {len(portfolio['holdings'])}")

    return portfolio, trades
