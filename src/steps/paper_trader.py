"""Paper trader for tracking portfolio state and executing simulated trades."""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.utils.s3_client import S3Client
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
    universe_df: pd.DataFrame
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
        market_price, action_type, sector=sector, asset_class=asset_class
    )

    # Use fill_price for all cash/P&L math
    price = fill_price

    trade_record = {
        'timestamp': datetime.now().isoformat(),
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
            'entry_date': datetime.now().isoformat(),
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
                datetime.now() - pd.to_datetime(holding['entry_date'])
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
    prices_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Update portfolio values based on current prices.

    Args:
        portfolio: Current portfolio state
        prices_df: Current price data

    Returns:
        Updated portfolio state
    """
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
            holding['days_held'] = (datetime.now() - pd.to_datetime(entry_date)).days
        else:
            holding['days_held'] = 0

        holdings_value += holding['market_value']

    portfolio['holdings_value'] = holdings_value
    portfolio['invested'] = holdings_value
    portfolio['portfolio_value'] = portfolio['cash'] + holdings_value
    portfolio['last_updated'] = datetime.now().isoformat()

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
    """Compute ytd_return, mtd_return, sharpe_ratio, max_drawdown, win_rate, total_trades."""
    import numpy as np

    pv = portfolio['portfolio_value']
    initial = 100000

    # Load trade history to compute win_rate and total_trades
    dates = s3.list_daily_dates(max_days=365)
    dates = sorted(dates)

    total_trades = 0
    winning_trades = 0
    daily_values = []

    # Find YTD and MTD start values
    now = datetime.now()
    ytd_start_value = initial
    mtd_start_value = initial

    for date_str in dates:
        state = s3.read_json(f'daily/{date_str}/portfolio_state.json')
        if state is None:
            continue

        val = state.get('portfolio_value', initial)
        daily_values.append(val)

        # Count trades from trades.jsonl (resilient to portfolio_state overwrites)
        day_trades = s3.read_jsonl(f'daily/{date_str}/trades.jsonl')
        for t in day_trades:
            if t.get('action') == 'SELL' and 'pnl' in t:
                total_trades += 1
                if t['pnl'] > 0:
                    winning_trades += 1

        # Track YTD start (first trading day of current year)
        if date_str[:4] == str(now.year) and ytd_start_value == initial:
            # Use the value from the day before, or initial
            idx = dates.index(date_str)
            if idx > 0:
                prev_state = s3.read_json(f'daily/{dates[idx-1]}/portfolio_state.json')
                if prev_state:
                    ytd_start_value = prev_state.get('portfolio_value', initial)

        # Track MTD start (first trading day of current month)
        if date_str[:7] == f"{now.year}-{now.month:02d}" and mtd_start_value == initial:
            idx = dates.index(date_str)
            if idx > 0:
                prev_state = s3.read_json(f'daily/{dates[idx-1]}/portfolio_state.json')
                if prev_state:
                    mtd_start_value = prev_state.get('portfolio_value', initial)

    # Compute returns
    ytd_return = (pv / ytd_start_value - 1) if ytd_start_value > 0 else 0
    mtd_return = (pv / mtd_start_value - 1) if mtd_start_value > 0 else 0

    # Compute Sharpe ratio from daily returns
    sharpe_ratio = 0.0
    if len(daily_values) >= 20:
        values = np.array(daily_values)
        daily_returns = np.diff(values) / values[:-1]
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

    # Compute max drawdown
    max_drawdown = 0.0
    if len(daily_values) >= 2:
        values = np.array(daily_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        max_drawdown = float(np.min(drawdowns))

    # Current drawdown
    current_drawdown = 0.0
    if len(daily_values) >= 2:
        peak_value = max(daily_values)
        if peak_value > 0:
            current_drawdown = (pv - peak_value) / peak_value

    # Win rate
    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0

    portfolio['ytd_return'] = ytd_return
    portfolio['mtd_return'] = mtd_return
    portfolio['sharpe_ratio'] = sharpe_ratio
    portfolio['max_drawdown'] = max_drawdown
    portfolio['current_drawdown'] = current_drawdown
    portfolio['win_rate'] = win_rate
    portfolio['total_trades'] = total_trades

    return portfolio


def run(
    decisions: Dict[str, Any],
    prices_df: pd.DataFrame,
    bucket: str
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
        trade = execute_trade(portfolio, action, regime_label, universe_df)
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
