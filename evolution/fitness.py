"""
Fitness evaluation for PolicyGenome using backtesting.

Evaluates genomes by simulating trading decisions on historical data
and computing risk-adjusted performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from evolution.genome import PolicyGenome


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    profit_factor: float
    calmar_ratio: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    trades: List[Dict]


class FitnessEvaluator:
    """
    Evaluates fitness of PolicyGenome instances through backtesting.

    Fitness is a weighted combination of:
    - Sharpe ratio (risk-adjusted returns)
    - Calmar ratio (return / max drawdown)
    - Win rate
    - Trade frequency penalty (too few or too many trades)
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        context_df: pd.DataFrame,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.04,
        fitness_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize evaluator with historical data.

        Args:
            prices_df: Historical prices with columns [date, symbol, close]
            features_df: Asset features with columns [date, symbol, health_score, vol_bucket, ...]
            context_df: Market context with columns [date, regime, ...]
            initial_capital: Starting portfolio value
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            fitness_weights: Weights for fitness components
        """
        self.prices_df = prices_df.copy()
        self.features_df = features_df.copy()
        self.context_df = context_df.copy()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.fitness_weights = fitness_weights or {
            'sharpe': 0.35,
            'calmar': 0.25,
            'win_rate': 0.15,
            'return': 0.15,
            'consistency': 0.10
        }

        # Preprocess data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and index data for fast lookups."""
        # Ensure date columns are datetime
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        self.features_df['date'] = pd.to_datetime(self.features_df['date'])
        self.context_df['date'] = pd.to_datetime(self.context_df['date'])

        # Get unique dates
        self.dates = sorted(self.prices_df['date'].unique())

        # Create price lookup: (date, symbol) -> close
        self.price_lookup = {}
        for _, row in self.prices_df.iterrows():
            self.price_lookup[(row['date'], row['symbol'])] = row['close']

        # Create feature lookup: (date, symbol) -> features dict
        self.feature_lookup = {}
        for _, row in self.features_df.iterrows():
            self.feature_lookup[(row['date'], row['symbol'])] = row.to_dict()

        # Create context lookup: date -> context dict
        self.context_lookup = {}
        for _, row in self.context_df.iterrows():
            self.context_lookup[row['date']] = row.to_dict()

        # Get all symbols
        self.symbols = list(self.prices_df['symbol'].unique())

    def evaluate(self, genome: PolicyGenome) -> float:
        """
        Evaluate fitness of a genome.

        Args:
            genome: PolicyGenome to evaluate

        Returns:
            Fitness score (higher is better)
        """
        result = self.backtest(genome)

        # Compute fitness components
        components = {}

        # Sharpe ratio (capped at reasonable values)
        components['sharpe'] = np.clip(result.sharpe_ratio, -2, 3)

        # Calmar ratio (return / max drawdown)
        components['calmar'] = np.clip(result.calmar_ratio, -2, 5)

        # Win rate
        components['win_rate'] = result.win_rate

        # Total return (annualized, capped)
        components['return'] = np.clip(result.annualized_return, -0.5, 1.0)

        # Consistency (penalize high volatility of returns)
        if result.volatility > 0:
            components['consistency'] = 1.0 / (1.0 + result.volatility)
        else:
            components['consistency'] = 0.5

        # Trade frequency penalty
        days = len(self.dates)
        expected_trades = days / 20  # Roughly one trade per month
        trade_ratio = result.total_trades / max(expected_trades, 1)
        if trade_ratio < 0.2:  # Too few trades
            components['trade_penalty'] = -0.3
        elif trade_ratio > 5.0:  # Too many trades
            components['trade_penalty'] = -0.2
        else:
            components['trade_penalty'] = 0.0

        # Compute weighted fitness
        fitness = 0.0
        for component, weight in self.fitness_weights.items():
            if component in components:
                fitness += weight * components[component]

        fitness += components.get('trade_penalty', 0)

        # Store components in genome
        genome.fitness = fitness
        genome.fitness_components = components

        return fitness

    def backtest(self, genome: PolicyGenome) -> BacktestResult:
        """
        Run backtest simulation for a genome.

        Args:
            genome: PolicyGenome with trading parameters

        Returns:
            BacktestResult with performance metrics
        """
        params = genome.to_decision_params()
        regime_compat = genome.to_regime_compatibility()

        # Initialize portfolio
        cash = self.initial_capital
        holdings: Dict[str, Dict] = {}  # symbol -> {shares, entry_price, entry_date}
        equity_curve = []
        trades = []

        for date in self.dates:
            context = self.context_lookup.get(date, {})
            regime = context.get('regime', 'choppy')

            # Get current prices and features
            current_prices = {}
            current_features = {}
            for symbol in self.symbols:
                price = self.price_lookup.get((date, symbol))
                if price:
                    current_prices[symbol] = price
                features = self.feature_lookup.get((date, symbol))
                if features:
                    current_features[symbol] = features

            # Calculate portfolio value
            portfolio_value = cash
            for symbol, holding in holdings.items():
                price = current_prices.get(symbol, holding['entry_price'])
                portfolio_value += holding['shares'] * price

            equity_curve.append(portfolio_value)

            # Check exits
            symbols_to_sell = []
            for symbol, holding in holdings.items():
                price = current_prices.get(symbol)
                if price is None:
                    continue

                features = current_features.get(symbol, {})
                health = features.get('health_score', 0.5)

                # Calculate return since entry
                entry_price = holding['entry_price']
                pct_return = (price - entry_price) / entry_price

                # Days held
                days_held = (date - holding['entry_date']).days

                # Exit conditions
                should_sell = False
                reason = ''

                # Trailing stop
                peak_price = holding.get('peak_price', entry_price)
                if price > peak_price:
                    holding['peak_price'] = price
                    peak_price = price
                drawdown_from_peak = (peak_price - price) / peak_price
                if drawdown_from_peak > params['trailing_stop_base']:
                    should_sell = True
                    reason = 'trailing_stop'

                # Health collapse
                if health < params['health_collapse_threshold']:
                    should_sell = True
                    reason = 'health_collapse'

                # Profit taking
                if pct_return > params['profit_take_threshold']:
                    should_sell = True
                    reason = 'profit_take'

                # Max holding period
                if days_held > params['max_holding_days']:
                    should_sell = True
                    reason = 'max_holding'

                if should_sell:
                    symbols_to_sell.append((symbol, reason))

            # Execute sells
            for symbol, reason in symbols_to_sell:
                holding = holdings.pop(symbol)
                price = current_prices[symbol]
                proceeds = holding['shares'] * price
                cash += proceeds

                pnl = proceeds - (holding['shares'] * holding['entry_price'])
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': holding['shares'],
                    'price': price,
                    'pnl': pnl,
                    'reason': reason
                })

            # Check entries (if we have capacity)
            if len(holdings) < params['max_positions']:
                # Score candidates
                candidates = []
                for symbol, features in current_features.items():
                    if symbol in holdings:
                        continue

                    health = features.get('health_score', 0)
                    if health < params['min_health_buy']:
                        continue

                    vol_bucket = features.get('vol_bucket', 'med')
                    return_21d = features.get('return_21d', 0)
                    return_63d = features.get('return_63d', 0)

                    # Calculate score
                    momentum_score = (return_21d + return_63d) / 2
                    mean_rev_score = -return_21d if return_21d < -0.05 else 0

                    score = (
                        health * 0.4 +
                        momentum_score * params['momentum_weight'] * 0.3 +
                        mean_rev_score * params['mean_reversion_weight'] * 0.3
                    )

                    # Apply regime multiplier
                    regime_mult = regime_compat.get(regime, {}).get('position_multiplier', 1.0)
                    score *= regime_mult

                    # Apply vol multiplier
                    vol_mult = regime_compat.get(regime, {}).get(f'vol_{vol_bucket}', 1.0)
                    score *= vol_mult

                    if score >= params['buy_score_threshold']:
                        candidates.append({
                            'symbol': symbol,
                            'score': score,
                            'price': current_prices.get(symbol),
                            'vol_bucket': vol_bucket
                        })

                # Sort by score and buy top candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)

                for candidate in candidates:
                    if len(holdings) >= params['max_positions']:
                        break

                    symbol = candidate['symbol']
                    price = candidate['price']
                    if price is None:
                        continue

                    # Calculate position size
                    max_position_value = portfolio_value * params['max_position_weight']
                    position_value = min(max_position_value, cash * 0.95)

                    if position_value < params['min_order_dollars']:
                        continue

                    shares = int(position_value / price)
                    if shares <= 0:
                        continue

                    cost = shares * price
                    if cost > cash:
                        continue

                    # Execute buy
                    cash -= cost
                    holdings[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date,
                        'peak_price': price
                    }

                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'pnl': 0,
                        'reason': 'entry'
                    })

        # Calculate final metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        days = len(self.dates)
        annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1

        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        winning_trades = [t for t in trades if t['action'] == 'SELL' and t['pnl'] > 0]
        losing_trades = [t for t in trades if t['action'] == 'SELL' and t['pnl'] <= 0]
        total_sells = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_sells if total_sells > 0 else 0

        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average holding period
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        if buy_trades:
            # Simplified - just count total trades / 2
            avg_holding_days = days / max(len(buy_trades), 1)
        else:
            avg_holding_days = 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_holding_days=avg_holding_days,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            equity_curve=equity_curve.tolist(),
            drawdown_curve=drawdown.tolist(),
            trades=trades
        )

    def evaluate_population(self, population: List[PolicyGenome]) -> List[PolicyGenome]:
        """
        Evaluate fitness for all genomes in a population.

        Args:
            population: List of PolicyGenome instances

        Returns:
            Same list with fitness scores populated
        """
        for genome in population:
            if genome.fitness is None:
                self.evaluate(genome)
        return population
