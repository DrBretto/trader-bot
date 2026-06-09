"""Smoke test for the simulated morning execution path (only path post-Alpaca-removal)."""

import sys
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps import morning_executor


def _portfolio():
    return {
        'cash': 100000.0,
        'holdings': [],
        'portfolio_value': 100000.0,
        'benchmark_value': 100000.0,
    }


def _intents():
    return {
        'generated_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'regime': 'risk_on_trend',
        'actions': [
            {
                'symbol': 'SPY',
                'action': 'BUY',
                'price': 600.0,
                'shares': 16,
                'dollars': 9600.0,
                'reason': 'BUY_SIGNAL',
            }
        ],
    }


def _quotes():
    return pd.DataFrame([
        {'symbol': 'SPY', 'price': 600.0, 'open': 599.0, 'high': 601.0,
         'low': 598.0, 'volume': 1000000, 'timestamp': '2026-06-08T10:00:00'},
    ])


def test_simulated_morning_path_executes_and_reports_simulated_mode():
    config = {
        'decision_params': {'min_order_dollars': 250},
        'universe': pd.DataFrame([{'symbol': 'SPY', 'asset_class': 'equity', 'sector': 'broad'}]),
    }

    with patch.object(morning_executor, 'load_trade_intents', return_value=_intents()), \
         patch.object(morning_executor.paper_trader, 'load_portfolio_state', return_value=_portfolio()), \
         patch.object(morning_executor.ingest_prices, 'fetch_morning_quotes', return_value=_quotes()), \
         patch.object(morning_executor.paper_trader, 'compute_portfolio_stats', side_effect=lambda p, s3: p), \
         patch('src.steps.morning_executor.S3Client'):
        result = morning_executor.run('test-bucket', config)

    assert result['execution_mode'] == 'simulated'
    assert result['intents_found'] is True
    assert len(result['trades']) == 1
    assert result['trades'][0]['symbol'] == 'SPY'
    assert result['trades'][0]['action'] == 'BUY'
    # paper_trader.execute_trade ran: 16 shares bought
    assert result['trades'][0]['shares'] == 16
    assert 'skipped_buys' in result
