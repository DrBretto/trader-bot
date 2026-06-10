"""Tests for the PKT-TB-004 replay-ablation flags.

Every flag defaults OFF and must produce byte-identical behavior when absent.
Each test exercises one toggle: flag off = production behavior, flag on =
the layer is neutralized.
"""

import pandas as pd

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.decision_engine import (
    filter_buy_candidates,
    evaluate_holdings,
)
from src.signals.regime_fusion import decide_regime_v3


def _scored_df(rows):
    return pd.DataFrame(rows)


def _fusion_kwargs(**over):
    base = dict(
        ensemble_regime_label='risk_on_trend',
        trend_risk_on_prob=0.6,
        panic_prob=0.05,
        ensemble_disagreement=0.4,
        ensemble_multiplier=0.7,
        macro_credit_score=0.0,
        vol_uncertainty_score=0.3,
        vol_regime_label='calm',
        fragility_score=0.3,
        entropy_score=0.5,
        entropy_shift_flag=False,
    )
    base.update(over)
    return base


class TestFusionAblationFlags:

    def test_neutralize_ensemble_multiplier(self):
        on = decide_regime_v3(**_fusion_kwargs(),
                              params={'neutralize_ensemble_multiplier': True})
        off = decide_regime_v3(**_fusion_kwargs())
        assert off['position_size_modifier'] == 0.7
        assert on['position_size_modifier'] == 1.0

    def test_force_raw_ensemble_label_suppresses_steering(self):
        # panic override flips the label; the flag reverts the label but
        # keeps the override's sizing effects.
        kwargs = _fusion_kwargs(panic_prob=0.9, ensemble_multiplier=1.0)
        steered = decide_regime_v3(**kwargs)
        forced = decide_regime_v3(**kwargs, params={'force_raw_ensemble_label': True})
        assert steered['final_regime_label'] == 'high_vol_panic'
        assert forced['final_regime_label'] == 'risk_on_trend'
        assert forced['position_size_modifier'] == steered['position_size_modifier']
        assert forced['risk_throttle_factor'] == steered['risk_throttle_factor']
        assert 'ablation_force_raw_label' in (forced['override_reason'] or '')

    def test_defaults_unchanged(self):
        a = decide_regime_v3(**_fusion_kwargs())
        b = decide_regime_v3(**_fusion_kwargs(), params={})
        assert a == b


class TestBuyFilterAblationFlags:

    def _candidates(self):
        return _scored_df([
            {'symbol': 'HIVOL', 'final_score': 0.70, 'health_score': 0.70,
             'vol_bucket': 'high', 'reason_code': 'SCORED',
             'asset_class': 'equity', 'sector': 'broad', 'leverage_flag': 0},
            {'symbol': 'VETOED', 'final_score': 0.75, 'health_score': 0.75,
             'vol_bucket': 'low', 'reason_code': 'SCORED',
             'asset_class': 'equity', 'sector': 'broad', 'leverage_flag': 0},
            {'symbol': 'BOND', 'final_score': 0.72, 'health_score': 0.72,
             'vol_bucket': 'low', 'reason_code': 'SCORED',
             'asset_class': 'bond', 'sector': 'treas_long', 'leverage_flag': 0},
        ])

    def test_vol_bucket_filter_toggle(self):
        params = {'buy_score_threshold': 0.65, 'min_health_buy': 0.6}
        llm = {}
        kept = filter_buy_candidates(self._candidates(), [], params,
                                     'risk_on_trend', llm)
        assert 'HIVOL' not in kept['symbol'].values
        kept_off = filter_buy_candidates(self._candidates(), [], params,
                                         'risk_on_trend', llm,
                                         ablation={'disable_vol_bucket_filter': True})
        assert 'HIVOL' in kept_off['symbol'].values

    def test_llm_buy_veto_toggle(self):
        params = {'buy_score_threshold': 0.65, 'min_health_buy': 0.6}
        llm = {'VETOED': {'structural_risk_veto': True}}
        kept = filter_buy_candidates(self._candidates(), [], params,
                                     'risk_on_trend', llm)
        assert 'VETOED' not in kept['symbol'].values
        kept_off = filter_buy_candidates(self._candidates(), [], params,
                                         'risk_on_trend', llm,
                                         ablation={'disable_llm_buy_veto': True})
        assert 'VETOED' in kept_off['symbol'].values

    def test_panic_buy_filter_toggle(self):
        params = {'buy_score_threshold': 0.65, 'min_health_buy': 0.6,
                  'buy_score_threshold_by_regime': {'high_vol_panic': 0.65},
                  'min_health_buy_by_regime': {'high_vol_panic': 0.6}}
        llm = {}
        kept = filter_buy_candidates(self._candidates(), [], params,
                                     'high_vol_panic', llm)
        assert set(kept['symbol']) == {'BOND'}
        kept_off = filter_buy_candidates(self._candidates(), [], params,
                                         'high_vol_panic', llm,
                                         ablation={'disable_panic_buy_filter': True})
        assert 'VETOED' in kept_off['symbol'].values  # equity passes in panic


class TestSellTriggerAblationFlags:

    def _state_and_prices(self, asset_class='equity', entry_regime=None):
        state = {'holdings': [{
            'symbol': 'XYZ', 'shares': 100, 'entry_price': 100.0,
            'peak_price': 100.0, 'asset_class': asset_class, 'sector': 'broad',
            'leverage_flag': 0, 'entry_date': '2026-01-05',
        }]}
        if entry_regime:
            state['holdings'][0]['entry_regime'] = entry_regime
        prices = pd.DataFrame({
            'symbol': ['XYZ'], 'date': ['2026-02-01'], 'close': [100.0],
        })
        return state, prices

    def test_panic_force_sell_toggle(self):
        health = [{'symbol': 'XYZ', 'health_score': 0.8}]
        params = {'trailing_stop_base': 0.10, 'sell_health_threshold': 0.35}
        state, prices = self._state_and_prices()
        sells = evaluate_holdings(state, health, prices, params,
                                  'high_vol_panic', {})
        assert [a['reason'] for a in sells] == ['REGIME_PANIC']
        state, prices = self._state_and_prices()
        none = evaluate_holdings(state, health, prices, params,
                                 'high_vol_panic', {},
                                 ablation={'disable_panic_force_sell': True})
        assert none == []

    def test_llm_sell_veto_toggle(self):
        health = [{'symbol': 'XYZ', 'health_score': 0.8}]
        params = {'trailing_stop_base': 0.10, 'sell_health_threshold': 0.35}
        llm = {'XYZ': {'structural_risk_veto': True}}
        state, prices = self._state_and_prices()
        sells = evaluate_holdings(state, health, prices, params,
                                  'risk_on_trend', llm)
        assert [a['reason'] for a in sells] == ['LLM_VETO']
        state, prices = self._state_and_prices()
        none = evaluate_holdings(state, health, prices, params,
                                 'risk_on_trend', llm,
                                 ablation={'disable_llm_sell_veto': True})
        assert none == []

    def test_reduce_regime_shift_toggle(self):
        health = [{'symbol': 'XYZ', 'health_score': 0.8}]
        params = {'trailing_stop_base': 0.10, 'sell_health_threshold': 0.35}
        state, prices = self._state_and_prices(entry_regime='risk_on_trend')
        trims = evaluate_holdings(state, health, prices, params, 'choppy', {})
        assert [a['reason'] for a in trims] == ['REGIME_SHIFT']
        state, prices = self._state_and_prices(entry_regime='risk_on_trend')
        none = evaluate_holdings(state, health, prices, params, 'choppy', {},
                                 ablation={'disable_reduce_regime_shift': True})
        assert none == []
