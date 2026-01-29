"""Paper trader for tracking portfolio state and executing simulated trades."""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.utils.s3_client import S3Client


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
    price = action['price']
    action_type = action['action']

    trade_record = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action_type,
        'shares': shares,
        'price': price,
        'dollars': shares * price,
        'reason': action.get('reason', ''),
        'regime': regime_label
    }

    if action_type == 'BUY':
        # Deduct cash
        portfolio['cash'] -= shares * price

        # Get asset metadata from universe
        asset_info = {}
        if len(universe_df) > 0:
            symbol_row = universe_df[universe_df['symbol'] == symbol]
            if len(symbol_row) > 0:
                asset_info = symbol_row.iloc[0].to_dict()

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
            'asset_class': asset_info.get('asset_class', 'equity'),
            'sector': asset_info.get('sector', 'broad'),
            'leverage_flag': asset_info.get('leverage_flag', 0)
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

        holdings_value += holding['market_value']

    portfolio['holdings_value'] = holdings_value
    portfolio['portfolio_value'] = portfolio['cash'] + holdings_value
    portfolio['last_updated'] = datetime.now().isoformat()

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

    # Store trades for today
    portfolio['trades_today'] = trades

    print(f"  Portfolio value: ${portfolio['portfolio_value']:,.2f}")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"  Holdings: {len(portfolio['holdings'])}")

    return portfolio, trades
