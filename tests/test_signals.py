"""Tests for expert signal modules and regime fusion v3."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# --- Macro Credit ---

class TestMacroCredit:
    def test_positive_score_healthy_slope(self):
        from src.signals.macro_credit import compute_macro_credit
        result = compute_macro_credit(rate_10y=4.0, rate_3m=2.0)
        assert result['macro_credit_score'] > 0
        assert result['yield_slope_10y_3m'] == 2.0

    def test_negative_score_inverted_curve(self):
        from src.signals.macro_credit import compute_macro_credit
        result = compute_macro_credit(rate_10y=3.0, rate_3m=5.0)
        assert result['macro_credit_score'] < 0
        assert result['yield_slope_10y_3m'] == -2.0

    def test_score_bounded(self):
        from src.signals.macro_credit import compute_macro_credit
        result = compute_macro_credit(rate_10y=10.0, rate_3m=0.0)
        assert -1.0 <= result['macro_credit_score'] <= 1.0

    def test_fallback_to_2y_rate(self):
        from src.signals.macro_credit import compute_macro_credit
        result = compute_macro_credit(rate_10y=4.0, rate_3m=0, rate_2y=3.0)
        assert result['yield_slope_10y_3m'] == 1.0

    def test_with_hy_spread(self):
        from src.signals.macro_credit import compute_macro_credit
        dates = pd.date_range('2024-01-01', periods=30)
        hyg = pd.Series(np.linspace(80, 82, 30), index=dates)
        ief = pd.Series(np.linspace(100, 100.5, 30), index=dates)
        result = compute_macro_credit(rate_10y=4.0, rate_3m=2.0,
                                       hyg_prices=hyg, ief_prices=ief)
        assert 'hy_spread_proxy' in result
        assert result['hy_spread_proxy'] != 0.0


# --- Vol Uncertainty ---

class TestVolUncertainty:
    def test_calm_regime(self):
        from src.signals.vol_uncertainty import compute_vol_uncertainty
        result = compute_vol_uncertainty(vix=15.0, vvix=80.0, skew=120.0)
        assert result['vol_regime_label'] == 'calm'
        assert 0.0 <= result['vol_uncertainty_score'] <= 1.0

    def test_panic_regime(self):
        from src.signals.vol_uncertainty import compute_vol_uncertainty
        # Both VIX and VVIX in high percentiles
        vix_history = pd.Series(np.random.normal(17, 3, 252))
        vvix_history = pd.Series(np.random.normal(85, 10, 252))
        result = compute_vol_uncertainty(
            vix=35.0, vvix=130.0, skew=150.0,
            vix_history=vix_history, vvix_history=vvix_history
        )
        assert result['vol_regime_label'] == 'panic'

    def test_unstable_calm(self):
        from src.signals.vol_uncertainty import compute_vol_uncertainty
        # VVIX high, VIX low
        vix_history = pd.Series(np.random.normal(17, 3, 252))
        vvix_history = pd.Series(np.random.normal(85, 10, 252))
        result = compute_vol_uncertainty(
            vix=14.0, vvix=130.0,
            vix_history=vix_history, vvix_history=vvix_history
        )
        assert result['vol_regime_label'] == 'unstable_calm'

    def test_graceful_degradation_no_vvix(self):
        from src.signals.vol_uncertainty import compute_vol_uncertainty
        result = compute_vol_uncertainty(vix=20.0, vvix=None, skew=None)
        assert result['vol_regime_label'] in ('calm', 'unstable_calm', 'panic')
        assert result['vvix_percentile'] == 0.5  # neutral

    def test_score_bounded(self):
        from src.signals.vol_uncertainty import compute_vol_uncertainty
        result = compute_vol_uncertainty(vix=50.0, vvix=150.0, skew=170.0)
        assert 0.0 <= result['vol_uncertainty_score'] <= 1.0


# --- Fragility ---

class TestFragility:
    def _make_prices_df(self, n_days=80, n_symbols=8, correlated=False):
        """Create synthetic price data."""
        dates = pd.date_range('2024-01-01', periods=n_days)
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'EFA', 'EEM'][:n_symbols]
        rows = []
        base_returns = np.random.normal(0, 0.01, n_days)
        for symbol in symbols:
            if correlated:
                returns = base_returns + np.random.normal(0, 0.002, n_days)
            else:
                returns = np.random.normal(0, 0.01, n_days)
            prices = 100 * np.exp(np.cumsum(returns))
            for i, date in enumerate(dates):
                rows.append({'date': date, 'symbol': symbol, 'close': prices[i]})
        return pd.DataFrame(rows)

    def test_normal_fragility(self):
        from src.signals.fragility import compute_fragility
        df = self._make_prices_df(correlated=False)
        result = compute_fragility(df)
        assert 0.0 <= result['fragility_score'] <= 1.0
        assert result['symbols_used'] == 8

    def test_high_fragility_correlated(self):
        from src.signals.fragility import compute_fragility
        df = self._make_prices_df(correlated=True)
        result = compute_fragility(df)
        # Highly correlated assets should produce higher fragility
        assert result['avg_correlation'] > 0.5

    def test_insufficient_symbols(self):
        from src.signals.fragility import compute_fragility
        df = self._make_prices_df(n_symbols=3)
        result = compute_fragility(df)
        assert result['fragility_score'] == 0.5
        assert 'degraded_reason' in result

    def test_insufficient_history(self):
        from src.signals.fragility import compute_fragility
        df = self._make_prices_df(n_days=20)
        result = compute_fragility(df)
        assert result['fragility_score'] == 0.5


# --- Entropy Shift ---

class TestEntropyShift:
    def test_normal_entropy(self):
        from src.signals.entropy_shift import compute_entropy_shift
        returns = pd.Series(np.random.normal(0, 0.01, 120))
        result = compute_entropy_shift(returns)
        assert 0.0 <= result['entropy_score'] <= 1.0
        assert not result['entropy_shift_flag']

    def test_insufficient_data(self):
        from src.signals.entropy_shift import compute_entropy_shift
        returns = pd.Series(np.random.normal(0, 0.01, 10))
        result = compute_entropy_shift(returns)
        assert result['entropy_score'] == 0.5
        assert 'degraded_reason' in result

    def test_consecutive_days_counter(self):
        from src.signals.entropy_shift import compute_entropy_shift
        # Create returns with extreme z-score
        normal = np.random.normal(0, 0.01, 200)
        # Last 60 days have very different distribution
        extreme = np.random.normal(0.05, 0.04, 60)
        returns = pd.Series(np.concatenate([normal, extreme]))
        result = compute_entropy_shift(
            returns,
            prev_consecutive_days=2,
            prev_above_threshold=True
        )
        # If above threshold, counter should be >= 3 → flag triggered
        if result['entropy_above_threshold']:
            assert result['entropy_consecutive_days'] >= 3

    def test_counter_resets(self):
        from src.signals.entropy_shift import compute_entropy_shift
        returns = pd.Series(np.random.normal(0, 0.01, 120))
        result = compute_entropy_shift(
            returns,
            prev_consecutive_days=5,
            prev_above_threshold=True
        )
        if not result['entropy_above_threshold']:
            assert result['entropy_consecutive_days'] == 0


# --- Regime Fusion v3 ---

class TestRegimeFusion:
    def test_panic_override_high_prob(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='risk_on_trend',
            trend_risk_on_prob=0.8,
            panic_prob=0.8,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=0.5,
            vol_uncertainty_score=0.9,
            vol_regime_label='panic',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'high_vol_panic'
        assert result['position_size_modifier'] == 0.25
        assert result['risk_throttle_factor'] == 1.0

    def test_panic_override_vol_regime(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='calm_uptrend',
            trend_risk_on_prob=0.7,
            panic_prob=0.1,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=0.5,
            vol_uncertainty_score=0.9,
            vol_regime_label='panic',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'high_vol_panic'

    def test_unstable_calm_override(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='risk_on_trend',
            trend_risk_on_prob=0.6,
            panic_prob=0.4,
            ensemble_disagreement=0.1,
            ensemble_multiplier=0.9,
            macro_credit_score=0.2,
            vol_uncertainty_score=0.7,
            vol_regime_label='unstable_calm',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'risk_off_trend'
        assert result['position_size_modifier'] <= 0.50

    def test_macro_downgrade(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='risk_on_trend',
            trend_risk_on_prob=0.7,
            panic_prob=0.1,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=-0.7,
            vol_uncertainty_score=0.3,
            vol_regime_label='calm',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'choppy'

    def test_macro_upgrade(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='choppy',
            trend_risk_on_prob=0.5,
            panic_prob=0.1,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=0.7,
            vol_uncertainty_score=0.3,
            vol_regime_label='calm',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'risk_on_trend'

    def test_fragility_gate(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='risk_on_trend',
            trend_risk_on_prob=0.7,
            panic_prob=0.1,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=0.3,
            vol_uncertainty_score=0.3,
            vol_regime_label='calm',
            fragility_score=0.85,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['position_size_modifier'] <= 0.60
        assert result['risk_throttle_factor'] >= 0.2

    def test_entropy_shift_gate(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='risk_on_trend',
            trend_risk_on_prob=0.7,
            panic_prob=0.1,
            ensemble_disagreement=0.0,
            ensemble_multiplier=1.0,
            macro_credit_score=0.3,
            vol_uncertainty_score=0.3,
            vol_regime_label='calm',
            fragility_score=0.3,
            entropy_score=0.8,
            entropy_shift_flag=True,
        )
        assert result['position_size_modifier'] < 1.0
        assert result['risk_throttle_factor'] >= 0.15

    def test_all_calm(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='calm_uptrend',
            trend_risk_on_prob=0.7,
            panic_prob=0.05,
            ensemble_disagreement=0.05,
            ensemble_multiplier=1.0,
            macro_credit_score=0.5,
            vol_uncertainty_score=0.2,
            vol_regime_label='calm',
            fragility_score=0.3,
            entropy_score=0.3,
            entropy_shift_flag=False,
        )
        assert result['final_regime_label'] == 'calm_uptrend'
        assert result['position_size_modifier'] == 1.0
        assert result['risk_throttle_factor'] == 0.0

    def test_output_clamps(self):
        from src.signals.regime_fusion import decide_regime_v3
        result = decide_regime_v3(
            ensemble_regime_label='high_vol_panic',
            trend_risk_on_prob=0.0,
            panic_prob=0.95,
            ensemble_disagreement=0.8,
            ensemble_multiplier=0.5,
            macro_credit_score=-1.0,
            vol_uncertainty_score=1.0,
            vol_regime_label='panic',
            fragility_score=1.0,
            entropy_score=1.0,
            entropy_shift_flag=True,
        )
        assert 0.25 <= result['position_size_modifier'] <= 1.0
        assert 0.0 <= result['risk_throttle_factor'] <= 1.0
        assert 0.0 <= result['regime_confidence'] <= 1.0


# --- Compute Signals Orchestrator ---

class TestComputeSignals:
    def _make_test_data(self):
        """Create minimal test data for the orchestrator."""
        dates = pd.date_range('2024-01-01', periods=120)
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'EFA', 'EEM', 'IEF']
        rows = []
        for symbol in symbols:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 120)))
            for i, date in enumerate(dates):
                rows.append({
                    'date': date, 'symbol': symbol,
                    'open': prices[i], 'high': prices[i] * 1.01,
                    'low': prices[i] * 0.99, 'close': prices[i],
                    'volume': 1000000
                })
        prices_df = pd.DataFrame(rows)

        fred_rows = []
        for series_id, base in [('DGS10', 4.0), ('DGS3MO', 2.0), ('DGS2', 3.0), ('VIXCLS', 17.0)]:
            for date in dates:
                fred_rows.append({
                    'date': date, 'series_id': series_id,
                    'value': base + np.random.normal(0, 0.1)
                })
        fred_df = pd.DataFrame(fred_rows)

        context_df = pd.DataFrame([{
            'date': dates[-1], 'spy_return_1d': 0.005,
            'spy_return_21d': 0.02, 'spy_vol_21d': 0.12,
            'rate_2y': 3.0, 'rate_3m': 2.0, 'rate_10y': 4.0,
            'yield_slope': 1.0, 'yield_slope_10y_3m': 2.0,
            'credit_spread_proxy': 0.01, 'risk_off_proxy': 0.01,
            'vixy_return_21d': -0.02, 'vvix_value': 85,
            'skew_value': 125, 'gdelt_doc_count': 100,
            'gdelt_avg_tone': 0, 'gdelt_tone_std': 0,
            'gdelt_neg_tone_share': 0,
        }])

        return prices_df, fred_df, context_df

    def test_orchestrator_runs(self):
        from src.signals.compute_signals import run
        prices_df, fred_df, context_df = self._make_test_data()
        result = run(prices_df, fred_df, context_df)
        assert 'macro_credit' in result
        assert 'vol_uncertainty' in result
        assert 'fragility' in result
        assert 'entropy_shift' in result
        assert 'computed_at' in result

    def test_all_scores_present(self):
        from src.signals.compute_signals import run
        prices_df, fred_df, context_df = self._make_test_data()
        result = run(prices_df, fred_df, context_df)
        assert 'macro_credit_score' in result['macro_credit']
        assert 'vol_uncertainty_score' in result['vol_uncertainty']
        assert 'fragility_score' in result['fragility']
        assert 'entropy_score' in result['entropy_shift']

    def test_graceful_with_empty_data(self):
        from src.signals.compute_signals import run
        result = run(
            pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame([{'date': '2024-01-01'}])
        )
        # Should not crash; all modules return neutral/fallback
        assert result['macro_credit']['macro_credit_score'] == 0.0
        assert result['fragility']['fragility_score'] == 0.5
