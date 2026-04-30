"""Tests for the decision engine."""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.decision_engine import (
    score_candidates,
    filter_buy_candidates,
    evaluate_holdings,
    compute_position_size
)


class TestScoreCandidates:
    """Tests for candidate scoring."""

    def test_scores_all_eligible_symbols(self):
        """Test that all eligible symbols get scores."""
        asset_health = [
            {'symbol': 'SPY', 'health_score': 0.75, 'vol_bucket': 'low', 'behavior': 'momentum'},
            {'symbol': 'QQQ', 'health_score': 0.80, 'vol_bucket': 'med', 'behavior': 'momentum'},
            {'symbol': 'TLT', 'health_score': 0.60, 'vol_bucket': 'low', 'behavior': 'mixed'}
        ]

        features_df = pd.DataFrame()

        universe_df = pd.DataFrame({
            'symbol': ['SPY', 'QQQ', 'TLT'],
            'asset_class': ['equity', 'equity', 'bond'],
            'sector': ['broad', 'broad', 'treas_long'],
            'eligible': [1, 1, 1],
            'leverage_flag': [0, 0, 0]
        })

        regime_compat = {
            'risk_on_trend': {'equity': 1.10, 'bond': 0.95}
        }

        result = score_candidates(
            asset_health,
            features_df,
            universe_df,
            'risk_on_trend',
            regime_compat
        )

        assert len(result) == 3
        assert 'final_score' in result.columns
        assert all(0 <= score <= 1 for score in result['final_score'])

    def test_applies_regime_multiplier(self):
        """Test that regime compatibility affects scores."""
        asset_health = [
            {'symbol': 'EQUITY', 'health_score': 0.70, 'vol_bucket': 'med', 'behavior': 'momentum'},
            {'symbol': 'BOND', 'health_score': 0.70, 'vol_bucket': 'low', 'behavior': 'mixed'}
        ]

        universe_df = pd.DataFrame({
            'symbol': ['EQUITY', 'BOND'],
            'asset_class': ['equity', 'bond'],
            'sector': ['broad', 'aggregate'],
            'eligible': [1, 1],
            'leverage_flag': [0, 0]
        })

        regime_compat = {
            'risk_on_trend': {'equity': 1.20, 'bond': 0.80}
        }

        result = score_candidates(
            asset_health,
            pd.DataFrame(),
            universe_df,
            'risk_on_trend',
            regime_compat
        )

        equity_score = result[result['symbol'] == 'EQUITY']['final_score'].iloc[0]
        bond_score = result[result['symbol'] == 'BOND']['final_score'].iloc[0]

        # Equity should have higher score due to regime boost
        assert equity_score > bond_score


class TestFilterBuyCandidates:
    """Tests for buy candidate filtering."""

    def test_excludes_current_holdings(self):
        """Test that current holdings are excluded."""
        scored_df = pd.DataFrame({
            'symbol': ['SPY', 'QQQ', 'TLT'],
            'final_score': [0.80, 0.75, 0.70],
            'health_score': [0.80, 0.75, 0.70],
            'vol_bucket': ['low', 'med', 'low'],
            'asset_class': ['equity', 'equity', 'bond']
        })

        params = {
            'buy_score_threshold': 0.65,
            'min_health_buy': 0.60
        }

        result = filter_buy_candidates(
            scored_df,
            current_holdings=['SPY'],  # SPY already held
            params=params,
            regime_label='risk_on_trend',
            llm_risk_flags={}
        )

        assert 'SPY' not in result['symbol'].values
        assert 'QQQ' in result['symbol'].values
        assert 'TLT' in result['symbol'].values

    def test_applies_score_threshold(self):
        """Test that low scores are filtered out."""
        scored_df = pd.DataFrame({
            'symbol': ['HIGH', 'LOW'],
            'final_score': [0.80, 0.50],  # LOW below threshold
            'health_score': [0.80, 0.70],
            'vol_bucket': ['med', 'med'],
            'asset_class': ['equity', 'equity']
        })

        params = {
            'buy_score_threshold': 0.65,
            'min_health_buy': 0.60
        }

        result = filter_buy_candidates(
            scored_df,
            current_holdings=[],
            params=params,
            regime_label='risk_on_trend',
            llm_risk_flags={}
        )

        assert 'HIGH' in result['symbol'].values
        assert 'LOW' not in result['symbol'].values

    def test_llm_veto_applied(self):
        """Test that LLM vetoes are respected."""
        scored_df = pd.DataFrame({
            'symbol': ['GOOD', 'VETOED'],
            'final_score': [0.80, 0.80],
            'health_score': [0.80, 0.80],
            'vol_bucket': ['med', 'med'],
            'asset_class': ['equity', 'equity']
        })

        params = {
            'buy_score_threshold': 0.65,
            'min_health_buy': 0.60
        }

        llm_risk_flags = {
            'VETOED': {'structural_risk_veto': True}
        }

        result = filter_buy_candidates(
            scored_df,
            current_holdings=[],
            params=params,
            regime_label='risk_on_trend',
            llm_risk_flags=llm_risk_flags
        )

        assert 'GOOD' in result['symbol'].values
        assert 'VETOED' not in result['symbol'].values


class TestEvaluateHoldings:
    """Tests for holding evaluation."""

    def test_stop_hit_triggers_sell(self):
        """Test that stop hit triggers a sell."""
        portfolio_state = {
            'holdings': [{
                'symbol': 'SPY',
                'shares': 100,
                'entry_price': 450,
                'peak_price': 460,  # Trailing stop at 414 (10% below peak)
                'entry_date': '2025-01-01'
            }]
        }

        asset_health = [{'symbol': 'SPY', 'health_score': 0.70}]

        # Current price below stop
        prices_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': [pd.Timestamp.now()],
            'close': [410]  # Below 414 stop
        })

        params = {
            'trailing_stop_base': 0.10,
            'sell_health_threshold': 0.35
        }

        result = evaluate_holdings(
            portfolio_state,
            asset_health,
            prices_df,
            params,
            'risk_on_trend',
            {}
        )

        assert len(result) == 1
        assert result[0]['action'] == 'SELL'
        assert result[0]['reason'] == 'STOP_HIT'

    def test_health_collapse_triggers_sell(self):
        """Test that health collapse triggers a sell."""
        portfolio_state = {
            'holdings': [{
                'symbol': 'SPY',
                'shares': 100,
                'entry_price': 450,
                'peak_price': 460,
                'entry_date': '2025-01-01'
            }]
        }

        asset_health = [{'symbol': 'SPY', 'health_score': 0.30}]  # Below threshold

        prices_df = pd.DataFrame({
            'symbol': ['SPY'],
            'date': [pd.Timestamp.now()],
            'close': [455]  # Above stop, so stop won't trigger
        })

        params = {
            'trailing_stop_base': 0.10,
            'sell_health_threshold': 0.35
        }

        result = evaluate_holdings(
            portfolio_state,
            asset_health,
            prices_df,
            params,
            'risk_on_trend',
            {}
        )

        assert len(result) == 1
        assert result[0]['action'] == 'SELL'
        assert result[0]['reason'] == 'HEALTH_COLLAPSE'


class TestComputePositionSize:
    """Tests for position sizing."""

    def test_basic_position_size(self):
        """Test basic position size calculation."""
        result = compute_position_size(
            symbol='SPY',
            portfolio_value=100000,
            current_price=450,
            vol_bucket='med',
            regime_label='risk_on_trend',
            params={'max_position_weight': 0.15, 'min_order_dollars': 250}
        )

        assert result['shares'] > 0
        assert result['dollars'] > 0
        assert 0 < result['final_weight'] <= 0.20

    def test_vol_bucket_adjustment(self):
        """Test that high vol bucket reduces position size."""
        low_vol = compute_position_size(
            symbol='SPY',
            portfolio_value=100000,
            current_price=450,
            vol_bucket='low',
            regime_label='risk_on_trend',
            params={'max_position_weight': 0.15, 'min_order_dollars': 250}
        )

        high_vol = compute_position_size(
            symbol='SPY',
            portfolio_value=100000,
            current_price=450,
            vol_bucket='high',
            regime_label='risk_on_trend',
            params={'max_position_weight': 0.15, 'min_order_dollars': 250}
        )

        assert low_vol['dollars'] > high_vol['dollars']

    def test_minimum_order_enforced(self):
        """Test that orders below minimum are rejected."""
        result = compute_position_size(
            symbol='SPY',
            portfolio_value=1000,  # Small portfolio
            current_price=450,
            vol_bucket='high',  # Further reduction
            regime_label='high_vol_panic',  # Panic reduces even more
            params={'max_position_weight': 0.05, 'min_order_dollars': 500}
        )

        # Should return 0 because order would be below minimum
        assert result['shares'] == 0
        assert result['dollars'] == 0


class TestEnsembleMultiplierCallerPatterns:
    """Tests that lock the actual production caller patterns for
    `compute_position_size`, plus the defense-in-depth gate.

    The 2026-04-30 fragility audit's F-2030-A finding ("ensemble_multiplier
    is double-applied") was a false positive. The v3 production callers at
    `decision_engine.py:798` and `:828` pass `ensemble_multiplier=1.0`
    whenever `expert_signals is not None`; the v2 callers pass
    `position_size_modifier=1.0` (the default). Either way, the multiplier
    is applied exactly once.

    These tests lock both caller patterns so a future regression that breaks
    the defense (e.g., a refactor that drops the `if expert_signals is None
    else 1.0` guard at line 798) is caught at the unit level. They also
    verify the `ensemble_multiplier_already_applied` gate forces correct
    math even if a caller passes both arguments non-neutral.
    See: docs/plans/2026-04-30-ensemble-double-fix-RETURN.md
    """

    BASE_KWARGS = dict(
        symbol='SPY',
        portfolio_value=100000.0,
        current_price=100.0,
        vol_bucket='med',
        regime_label='risk_on_trend',
        params={'max_position_weight': 0.20, 'min_order_dollars': 250},
        llm_confidence_adj=0.0,
        risk_throttle_factor=0.0,
    )

    def test_v3_caller_pattern_single_application(self):
        # Production v3 path: caller passes ensemble_multiplier=1.0 (because
        # `expert_signals is not None`, see decision_engine.py:798) and a
        # position_size_modifier that already includes the ensemble multiplier
        # via regime_fusion line 262. Net: ensemble applied exactly once
        # via position_size_modifier.
        # Chain: $20k * regime_adj(1.10) * llm(1.0) * ens_adj(1.0) *
        #        expert_adj(0.85) * throttle(1.0) = $18,700 → 187 shares
        result = compute_position_size(
            ensemble_multiplier=1.0,        # v3 defense at caller line 798
            position_size_modifier=0.85,    # 0.85 from regime_fusion (1.0 base * 0.85 ensemble)
            **self.BASE_KWARGS,
        )
        assert result['shares'] == 187
        assert result['dollars'] == 18700.0

    def test_v2_caller_pattern_single_application(self):
        # Production v2 / pre-hybrid path: caller passes ensemble_multiplier
        # directly (because `expert_signals is None`) and
        # position_size_modifier=1.0 (the parameter default). Net: ensemble
        # applied exactly once via ensemble_adj.
        # Chain: $20k * regime_adj(1.10) * llm(1.0) * ens_adj(0.85) *
        #        expert_adj(1.0) * throttle(1.0) = $18,700 → 187 shares
        result = compute_position_size(
            ensemble_multiplier=0.85,       # v2: caller passes ensemble directly
            position_size_modifier=1.0,     # v2: default (no fusion modifier)
            **self.BASE_KWARGS,
        )
        assert result['shares'] == 187
        assert result['dollars'] == 18700.0

    def test_unsafe_caller_pattern_double_applies_without_gate(self):
        # If a future caller forgets the v3 defense and passes BOTH
        # arguments non-neutral, the function double-applies. This test
        # documents the susceptibility (current production never invokes
        # this pattern; the gate below protects against future regression).
        # Chain: $20k * 1.10 * 0.85 * 0.85 * 1.0 = $15,895 → 158 shares
        result = compute_position_size(
            ensemble_multiplier=0.85,
            position_size_modifier=0.85,
            **self.BASE_KWARGS,
        )
        assert result['shares'] == 158
        assert result['dollars'] == 15800.0

    def test_gate_on_neutralizes_unsafe_caller(self):
        # The defense-in-depth gate forces ensemble_adj=1.0, so even an
        # unsafe caller (both non-neutral) gets correct single-application
        # math. Chain: $20k * 1.10 * 1.00 * 0.85 * 1.0 = $18,700 → 187 shares
        result = compute_position_size(
            ensemble_multiplier=0.85,
            position_size_modifier=0.85,
            decision_engine_overrides={
                'position_size': {'ensemble_multiplier_already_applied': True},
            },
            **self.BASE_KWARGS,
        )
        assert result['shares'] == 187
        assert result['dollars'] == 18700.0

    def test_gate_on_with_v3_caller_is_no_op(self):
        # Critical: when the v3 caller is correct (ensemble=1.0), the gate
        # has no effect. This proves the gate is safe to ship default-on:
        # under the actual production path it changes nothing.
        without = compute_position_size(
            ensemble_multiplier=1.0,
            position_size_modifier=0.85,
            **self.BASE_KWARGS,
        )
        with_gate = compute_position_size(
            ensemble_multiplier=1.0,
            position_size_modifier=0.85,
            decision_engine_overrides={
                'position_size': {'ensemble_multiplier_already_applied': True},
            },
            **self.BASE_KWARGS,
        )
        assert without == with_gate

    def test_gate_default_off_explicit_false_no_op(self):
        # Setting the override to False explicitly must match the no-override
        # default — no operator-error room.
        without = compute_position_size(
            ensemble_multiplier=0.85,
            position_size_modifier=0.85,
            **self.BASE_KWARGS,
        )
        explicit = compute_position_size(
            ensemble_multiplier=0.85,
            position_size_modifier=0.85,
            decision_engine_overrides={
                'position_size': {'ensemble_multiplier_already_applied': False},
            },
            **self.BASE_KWARGS,
        )
        assert without == explicit
