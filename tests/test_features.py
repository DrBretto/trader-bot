"""Tests for feature engineering utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.utils.feature_utils import (
    compute_asset_features,
    compute_relative_strength,
    compute_context_features
)


class TestComputeAssetFeatures:
    """Tests for compute_asset_features function."""

    def test_returns_calculation(self):
        """Test that returns are calculated correctly."""
        # Create 100 days of linear price growth
        dates = pd.date_range('2025-01-01', periods=100)
        prices = [100 + i for i in range(100)]

        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        })

        result = compute_asset_features(df)

        # Check 1-day return (1/199 ≈ 0.005)
        assert 'return_1d' in result.columns
        assert result.iloc[-1]['return_1d'] == pytest.approx(1/199, rel=0.01)

        # Check 21-day return
        assert 'return_21d' in result.columns
        expected_21d = (199 - 178) / 178
        assert result.iloc[-1]['return_21d'] == pytest.approx(expected_21d, rel=0.05)

    def test_volatility_calculation(self):
        """Test that volatility is annualized correctly."""
        # Create data with known volatility
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100)
        returns = np.random.normal(0, 0.01, 100)  # 1% daily std
        prices = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000000] * 100
        })

        result = compute_asset_features(df)

        # 21-day vol should be approximately 1% * sqrt(252) ≈ 15.87%
        assert 'vol_21d' in result.columns
        # Allow for some variance due to random data
        assert 0.10 < result.iloc[-1]['vol_21d'] < 0.25

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Create data with a drawdown
        dates = pd.date_range('2025-01-01', periods=100)
        prices = list(range(100, 150)) + list(range(149, 99, -1))  # Up then down

        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000000] * 100
        })

        result = compute_asset_features(df)

        assert 'drawdown_63d' in result.columns
        # At the end, price is 100, peak was 149
        # Drawdown should be (100 - 149) / 149 ≈ -0.329
        assert result.iloc[-1]['drawdown_63d'] == pytest.approx(-0.329, rel=0.05)

    def test_handles_empty_dataframe(self):
        """Test that empty DataFrame returns empty DataFrame with required columns."""
        df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        # Should not raise an error
        result = compute_asset_features(df)
        # Result should be empty
        assert len(result) == 0


class TestComputeRelativeStrength:
    """Tests for compute_relative_strength function."""

    def test_relative_strength_vs_spy(self):
        """Test relative strength calculation."""
        dates = pd.date_range('2025-01-01', periods=100)

        # Asset with 25% 21d return - include all required columns
        asset_df = pd.DataFrame({
            'date': dates,
            'open': [100] * 79 + [125] * 21,
            'high': [101] * 79 + [126] * 21,
            'low': [99] * 79 + [124] * 21,
            'close': [100] * 79 + [125] * 21,  # Jump to 125
            'volume': [1000000] * 100
        })
        asset_df = compute_asset_features(asset_df)

        # SPY with 10% 21d return
        spy_df = pd.DataFrame({
            'date': dates,
            'open': [100] * 79 + [110] * 21,
            'high': [101] * 79 + [111] * 21,
            'low': [99] * 79 + [109] * 21,
            'close': [100] * 79 + [110] * 21,  # Jump to 110
            'volume': [5000000] * 100
        })
        spy_df = compute_asset_features(spy_df)

        result = compute_relative_strength(asset_df, spy_df)

        assert 'rel_strength_21d' in result.columns


class TestComputeContextFeatures:
    """Tests for compute_context_features function."""

    def test_context_features_structure(self):
        """Test that context features have expected columns."""
        dates = pd.date_range('2025-01-01', periods=100)
        base_df = pd.DataFrame({
            'date': dates,
            'close': [100] * 100,
            'return_1d': [0.001] * 100,
            'return_21d': [0.02] * 100,
            'vol_21d': [0.15] * 100
        })

        fred_data = {'DGS2': 4.5, 'DGS10': 4.0}
        gdelt_data = {'gdelt_doc_count': 1000, 'gdelt_avg_tone': 0.5}

        result = compute_context_features(
            spy_df=base_df,
            tlt_df=base_df,
            hyg_df=base_df,
            ief_df=base_df,
            vixy_df=base_df,
            fred_data=fred_data,
            gdelt_data=gdelt_data
        )

        assert 'date' in result.columns
        assert 'spy_return_1d' in result.columns
        assert 'spy_return_21d' in result.columns
        assert 'yield_slope' in result.columns
        assert 'credit_spread_proxy' in result.columns
        assert 'gdelt_doc_count' in result.columns

    def test_yield_slope_calculation(self):
        """Test yield slope is 10Y - 2Y."""
        dates = pd.date_range('2025-01-01', periods=10)
        base_df = pd.DataFrame({
            'date': dates,
            'close': [100] * 10,
            'return_21d': [0.01] * 10,
            'vol_21d': [0.15] * 10
        })

        fred_data = {'DGS2': 4.5, 'DGS10': 4.0}
        gdelt_data = {}

        result = compute_context_features(
            spy_df=base_df,
            tlt_df=base_df,
            hyg_df=base_df,
            ief_df=base_df,
            vixy_df=base_df,
            fred_data=fred_data,
            gdelt_data=gdelt_data
        )

        # Yield slope should be 4.0 - 4.5 = -0.5
        assert result.iloc[0]['yield_slope'] == pytest.approx(-0.5, rel=0.01)
