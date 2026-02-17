"""Paper trader for tracking portfolio state and executing simulated trades."""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

from src.utils.s3_client import S3Client
from src.utils.dashboard_metrics import compute_canonical_dashboard_metrics
from src.utils.transaction_costs import apply_transaction_costs


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
            return state

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
            trade_record['days_held'] = (
                trade_time - pd.to_datetime(holding['entry_date'])
            ).days

            # Add cash
            portfolio['cash'] += shares * price

            # Remove from holdings
            portfolio['holdings'].pop(holding_idx)

    elif action_type == 'REDUCE':
        # Find the holding
        for h in portfolio['holdings']:
            if h['symbol'] == symbol:
                # Reduce by 50%
                reduce_shares = shares // 2
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
    """Compute canonical performance + lifecycle stats from one coherent series."""
    snapshot_date = datetime.now().strftime('%Y-%m-%d')
    canonical = compute_canonical_dashboard_metrics(
        s3=s3,
        portfolio_state=portfolio,
        snapshot_date=snapshot_date,
        current_state=portfolio,
        max_days=730,
        initial_value=100000.0,
        risk_free_rate_annual=0.0,
        min_sharpe_observations=60,
    )
    metrics = canonical['metrics']

    portfolio['ytd_return'] = metrics['ytd_return']
    portfolio['mtd_return'] = metrics['mtd_return']
    portfolio['sharpe_ratio'] = metrics['sharpe_ratio']
    portfolio['sharpe_observations'] = metrics['sharpe_observations']
    portfolio['max_drawdown'] = metrics['max_drawdown']
    portfolio['current_drawdown'] = metrics['current_drawdown']
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
    portfolio['metrics_reset_boundary'] = canonical.get('reset_boundary')

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
