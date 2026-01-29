"""Tests for baseline models."""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.models.baseline_regime import (
    baseline_regime_model,
    REGIMES,
    get_regime_description,
    get_regime_risk_level
)
from src.models.baseline_health import (
    baseline_health_model,
    get_health_tier,
    get_top_candidates
)


class TestBaselineRegimeModel:
    """Tests for the baseline regime model."""

    def test_returns_valid_regime(self):
        """Test that model returns a valid regime label."""
        context = pd.Series({
            'spy_return_21d': 0.05,
            'spy_vol_21d': 0.15,
            'credit_spread_proxy': 0,
            'vixy_return_21d': -0.05
        })

        result = baseline_regime_model(context)

        assert 'regime_label' in result
        assert result['regime_label'] in REGIMES
        assert 'regime_probs' in result
        assert sum(result['regime_probs'].values()) == pytest.approx(1.0)

    def test_high_vol_panic_detection(self):
        """Test that high vol panic is detected correctly."""
        context = pd.Series({
            'spy_return_21d': -0.10,  # Strong negative returns
            'spy_vol_21d': 0.30,       # High volatility
            'credit_spread_proxy': -0.05,
            'vixy_return_21d': 0.25    # VIX spike
        })

        result = baseline_regime_model(context)

        assert result['regime_label'] == 'high_vol_panic'

    def test_calm_uptrend_detection(self):
        """Test that calm uptrend is detected correctly."""
        context = pd.Series({
            'spy_return_21d': 0.08,   # Strong positive returns
            'spy_vol_21d': 0.12,      # Low volatility
            'credit_spread_proxy': 0.02,
            'vixy_return_21d': -0.10
        })

        result = baseline_regime_model(context)

        assert result['regime_label'] == 'calm_uptrend'

    def test_risk_off_detection(self):
        """Test that risk-off is detected correctly."""
        context = pd.Series({
            'spy_return_21d': -0.06,  # Negative returns
            'spy_vol_21d': 0.18,
            'credit_spread_proxy': -0.04,  # Credit stress
            'vixy_return_21d': 0.10
        })

        result = baseline_regime_model(context)

        assert result['regime_label'] == 'risk_off_trend'

    def test_choppy_detection(self):
        """Test that choppy market is detected correctly."""
        context = pd.Series({
            'spy_return_21d': 0.01,   # Flat returns
            'spy_vol_21d': 0.20,      # Elevated volatility
            'credit_spread_proxy': 0,
            'vixy_return_21d': 0
        })

        result = baseline_regime_model(context)

        assert result['regime_label'] == 'choppy'

    def test_default_to_risk_on(self):
        """Test that default regime is risk-on."""
        context = pd.Series({
            'spy_return_21d': 0.03,   # Moderate positive
            'spy_vol_21d': 0.16,      # Normal vol
            'credit_spread_proxy': 0.01,
            'vixy_return_21d': -0.02
        })

        result = baseline_regime_model(context)

        assert result['regime_label'] == 'risk_on_trend'

    def test_regime_descriptions_exist(self):
        """Test that all regimes have descriptions."""
        for regime in REGIMES:
            desc = get_regime_description(regime)
            assert desc is not None
            assert len(desc) > 0

    def test_regime_risk_levels(self):
        """Test that risk levels are in valid range."""
        for regime in REGIMES:
            level = get_regime_risk_level(regime)
            assert 1 <= level <= 5


class TestBaselineHealthModel:
    """Tests for the baseline health model."""

    def test_returns_health_scores(self):
        """Test that model returns health scores for all symbols."""
        dates = pd.date_range('2025-01-01', periods=100)

        features_df = pd.DataFrame({
            'date': [dates[-1]] * 5,
            'symbol': ['SPY', 'QQQ', 'TLT', 'GLD', 'XLK'],
            'return_21d': [0.05, 0.08, -0.02, 0.03, 0.10],
            'return_63d': [0.12, 0.15, 0.01, 0.08, 0.20],
            'vol_21d': [0.15, 0.20, 0.10, 0.12, 0.25],
            'vol_63d': [0.16, 0.22, 0.11, 0.13, 0.26],
            'drawdown_63d': [-0.03, -0.05, -0.02, -0.04, -0.08],
            'rel_strength_21d': [0, 0.03, -0.07, -0.02, 0.05],
            'rel_strength_63d': [0, 0.03, -0.11, -0.04, 0.08]
        })

        result = baseline_health_model(features_df, dates[-1])

        assert len(result) == 5
        assert 'symbol' in result.columns
        assert 'health_score' in result.columns
        assert 'vol_bucket' in result.columns
        assert 'behavior' in result.columns

    def test_health_scores_in_range(self):
        """Test that health scores are between 0 and 1."""
        dates = pd.date_range('2025-01-01', periods=100)

        features_df = pd.DataFrame({
            'date': [dates[-1]] * 10,
            'symbol': [f'SYM{i}' for i in range(10)],
            'return_21d': np.random.uniform(-0.2, 0.2, 10),
            'return_63d': np.random.uniform(-0.3, 0.3, 10),
            'vol_21d': np.random.uniform(0.1, 0.4, 10),
            'vol_63d': np.random.uniform(0.1, 0.4, 10),
            'drawdown_63d': np.random.uniform(-0.3, 0, 10),
            'rel_strength_21d': np.random.uniform(-0.1, 0.1, 10),
            'rel_strength_63d': np.random.uniform(-0.15, 0.15, 10)
        })

        result = baseline_health_model(features_df, dates[-1])

        assert all(0 <= score <= 1 for score in result['health_score'])

    def test_vol_bucket_assignment(self):
        """Test that vol buckets are assigned correctly."""
        dates = pd.date_range('2025-01-01', periods=100)

        features_df = pd.DataFrame({
            'date': [dates[-1]] * 3,
            'symbol': ['LOW_VOL', 'MED_VOL', 'HIGH_VOL'],
            'return_21d': [0.05, 0.05, 0.05],
            'return_63d': [0.10, 0.10, 0.10],
            'vol_21d': [0.10, 0.20, 0.35],
            'vol_63d': [0.10, 0.20, 0.35],
            'drawdown_63d': [-0.02, -0.02, -0.02],
            'rel_strength_21d': [0, 0, 0],
            'rel_strength_63d': [0, 0, 0]
        })

        result = baseline_health_model(features_df, dates[-1])

        vol_buckets = set(result['vol_bucket'].unique())
        assert vol_buckets.issubset({'low', 'med', 'high'})

    def test_health_tier_function(self):
        """Test health tier classification."""
        assert get_health_tier(0.80) == 'excellent'
        assert get_health_tier(0.65) == 'good'
        assert get_health_tier(0.50) == 'fair'
        assert get_health_tier(0.40) == 'poor'
        assert get_health_tier(0.20) == 'critical'

    def test_get_top_candidates(self):
        """Test top candidates selection."""
        health_df = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'health_score': [0.90, 0.75, 0.60, 0.45, 0.30],
            'vol_bucket': ['low', 'med', 'med', 'high', 'high'],
            'behavior': ['momentum'] * 5
        })

        result = get_top_candidates(health_df, n=3, min_health=0.50)

        assert len(result) == 3
        assert list(result['symbol']) == ['A', 'B', 'C']
