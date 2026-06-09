"""Tests for the continuous-line plumbing + fail-loud advance guard.

Covers:
- publish_artifacts._newest_priceable_date: mirrors the replay's pricing
  precondition (analysis + (provisional OR successor prices)). Must NOT
  false-advance onto a bare newest prefix with no analysis, and must return
  the last real day across a weekend.
- publish_artifacts._verify_extension_or_alarm: PASS on stamped + advanced,
  ALARM on missing stamp, ALARM on F_new < F_expected, PASS when nothing
  newer is priceable (weekend / no new day).
- replay_engine.run_variant provisional path: prices the newest day from a
  morning_prices.parquet (no successor prices) and reports provisional_date;
  a Monday-like newest with no analysis flat-holds with provisional_date=None.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.steps import publish_artifacts
from src.utils.three_line_replay import replay_engine


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeS3:
    """Minimal stand-in for boto3 s3 client used by S3Cache.

    keys: dict mapping S3 key -> bytes (or DataFrame/dict, auto-serialized).
    commonprefixes: list of date strings exposed by list_objects_v2 paginate.
    """

    def __init__(self, keys, dates):
        self._keys = {}
        for k, v in keys.items():
            if isinstance(v, pd.DataFrame):
                buf = io.BytesIO()
                v.to_parquet(buf, index=False)
                self._keys[k] = buf.getvalue()
            elif isinstance(v, (dict, list)):
                self._keys[k] = json.dumps(v).encode()
            elif isinstance(v, bytes):
                self._keys[k] = v
            else:
                self._keys[k] = str(v).encode()
        self._dates = list(dates)

    def get_object(self, Bucket=None, Key=None):
        if Key not in self._keys:
            raise KeyError(f"NoSuchKey: {Key}")
        return {'Body': io.BytesIO(self._keys[Key])}

    def get_paginator(self, _name):
        outer = self

        class _Pag:
            def paginate(self, **kwargs):
                yield {
                    'CommonPrefixes': [
                        {'Prefix': f'daily/{d}/'} for d in outer._dates
                    ]
                }
        return _Pag()


class FakeS3Client:
    """Stand-in for src.utils.s3_client.S3Client exposing only `.s3`."""

    def __init__(self, fake_s3):
        self.s3 = fake_s3


def _analysis_keys(date):
    """Return the three analysis artifacts for a date with minimal valid content.

    signals.parquet carries the full field set _build_expert_signals reads so
    run_variant's provisional iteration completes.
    """
    feats = pd.DataFrame([{'symbol': 'SPY', 'date': date, 'close': 100.0}])
    sigs = pd.DataFrame([{
        'date': date,
        'macro_credit_score': 0.0, 'yield_slope_10y_3m': 0.0, 'hy_spread_proxy': 0.0,
        'vol_uncertainty_score': 0.5, 'vol_regime_label': 'calm',
        'vix_percentile': 0.5, 'vvix_percentile': 0.5,
        'avg_correlation': 0.3, 'pc1_explained': 0.45,
        'entropy_score': 0.5, 'entropy_z_score': 0.0, 'entropy_shift_flag': False,
    }])
    return {
        f'daily/{date}/inference.json': {'regime': {'label': 'risk_on_trend'}},
        f'daily/{date}/features.parquet': feats,
        f'daily/{date}/signals.parquet': sigs,
    }


def _prices_for(date, symbols=('SPY',)):
    rows = [{'date': date, 'symbol': s, 'open': 100.0, 'high': 101.0,
             'low': 99.0, 'close': 100.5, 'volume': 1000} for s in symbols]
    return pd.DataFrame(rows)


def _morning_prices_for(date, symbols=('SPY',)):
    rows = [{'date': date, 'symbol': s, 'open': 100.0, 'high': 101.0,
             'low': 99.0, 'close': 100.7, 'volume': 500} for s in symbols]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# _newest_priceable_date
# --------------------------------------------------------------------------

class TestNewestPriceableDate:
    def test_real_priced_newest_via_successor_prices(self):
        """Newest day with analysis whose successor carries prices.parquet."""
        dates = ['2026-03-11', '2026-03-12', '2026-03-13']
        keys = {}
        keys.update(_analysis_keys('2026-03-12'))
        keys['daily/2026-03-13/prices.parquet'] = _prices_for('2026-03-12')
        s3c = FakeS3Client(FakeS3(keys, dates))
        assert publish_artifacts._newest_priceable_date(s3c) == '2026-03-12'

    def test_provisional_only_newest(self):
        """Newest day priceable ONLY because its own morning_prices exists
        (no successor prices file at all)."""
        dates = ['2026-03-11', '2026-03-12', '2026-03-13']
        keys = {}
        keys.update(_analysis_keys('2026-03-13'))
        keys['daily/2026-03-13/morning_prices.parquet'] = _morning_prices_for('2026-03-13')
        s3c = FakeS3Client(FakeS3(keys, dates))
        assert publish_artifacts._newest_priceable_date(s3c) == '2026-03-13'

    def test_bare_newest_prefix_no_analysis_is_skipped(self):
        """A newest prefix with NO analysis must not false-advance; falls back
        to the last real analysis+priced day."""
        dates = ['2026-03-11', '2026-03-12', '2026-03-13']
        keys = {}
        # 2026-03-12 has analysis and its successor (03-13) has prices.
        keys.update(_analysis_keys('2026-03-12'))
        keys['daily/2026-03-13/prices.parquet'] = _prices_for('2026-03-12')
        # 2026-03-13 has prices but NO analysis (bare prefix).
        s3c = FakeS3Client(FakeS3(keys, dates))
        assert publish_artifacts._newest_priceable_date(s3c) == '2026-03-12'

    def test_last_real_day_across_weekend(self):
        """Newest is a no-analysis Monday-like day; returns the last real
        (analysis + successor-prices) day before it."""
        # Fri 03-13 has analysis; Sat-equivalent successor 03-16 has prices.
        # Mon 03-16 is the newest prefix but has no analysis.
        dates = ['2026-03-12', '2026-03-13', '2026-03-16']
        keys = {}
        keys.update(_analysis_keys('2026-03-13'))
        keys['daily/2026-03-16/prices.parquet'] = _prices_for('2026-03-13')
        # 03-16 itself: no analysis, no morning_prices -> not priceable.
        s3c = FakeS3Client(FakeS3(keys, dates))
        assert publish_artifacts._newest_priceable_date(s3c) == '2026-03-13'

    def test_none_when_nothing_priceable(self):
        dates = ['2026-03-11', '2026-03-12']
        s3c = FakeS3Client(FakeS3({}, dates))
        assert publish_artifacts._newest_priceable_date(s3c) is None


# --------------------------------------------------------------------------
# _verify_extension_or_alarm
# --------------------------------------------------------------------------

def _stamped_dash(champion_frontier):
    return {
        'timeline_correction': {
            'version': 'lambda-three-line-replay-v2-optimized-canon',
            'champion_frontier': champion_frontier,
        },
        'metrics': {'canon_source': 'optimized_champion'},
    }


def _corpus_s3(newest_priceable):
    """Build a FakeS3Client whose _newest_priceable_date resolves to a chosen
    date, or None."""
    if newest_priceable is None:
        return FakeS3Client(FakeS3({}, ['2026-03-11', '2026-03-12']))
    dates = ['2026-03-11', '2026-03-12', newest_priceable]
    keys = {}
    keys.update(_analysis_keys(newest_priceable))
    keys[f'daily/{newest_priceable}/morning_prices.parquet'] = _morning_prices_for(newest_priceable)
    return FakeS3Client(FakeS3(keys, dates))


class TestVerifyExtensionOrAlarm:
    def test_pass_when_stamped_and_advanced_equal(self):
        dash = _stamped_dash('2026-03-13')
        s3c = _corpus_s3('2026-03-13')
        ok, reason = publish_artifacts._verify_extension_or_alarm(dash, s3c, 'night', '2026-03-13')
        assert ok is True, reason

    def test_alarm_when_stamp_missing(self):
        dash = {'timeline_correction': {}, 'metrics': {}}
        s3c = _corpus_s3(None)
        ok, reason = publish_artifacts._verify_extension_or_alarm(dash, s3c, 'night', '2026-03-13')
        assert ok is False
        assert 'STAMP' in reason

    def test_alarm_when_frontier_behind_expected(self):
        dash = _stamped_dash('2026-03-12')  # replay stalled one day behind
        s3c = _corpus_s3('2026-03-13')      # corpus offers a newer priceable day
        ok, reason = publish_artifacts._verify_extension_or_alarm(dash, s3c, 'night', '2026-03-13')
        assert ok is False
        assert 'FORWARD-ADVANCE' in reason

    def test_pass_when_nothing_newer_priceable(self):
        """Weekend / no-new-day: F_expected is None -> guard passes silently."""
        dash = _stamped_dash('2026-03-12')
        s3c = _corpus_s3(None)
        ok, reason = publish_artifacts._verify_extension_or_alarm(dash, s3c, 'night', '2026-03-16')
        assert ok is True, reason

    def test_alarm_when_frontier_none(self):
        dash = _stamped_dash(None)
        s3c = _corpus_s3(None)
        ok, reason = publish_artifacts._verify_extension_or_alarm(dash, s3c, 'night', '2026-03-13')
        assert ok is False
        assert 'FRONTIER' in reason


# --------------------------------------------------------------------------
# run_variant provisional path
# --------------------------------------------------------------------------

class CountingCache(replay_engine.S3Cache):
    """Real S3Cache backed by a FakeS3 (gives correct get/get_json/get_parquet)."""

    def __init__(self, fake_s3):
        super().__init__(fake_s3, bucket='test-bucket')


def _seed_portfolio_state():
    return {
        'cash': 100000.0,
        'holdings': [],
        'benchmark_shares': 0.0,
        'benchmark_start_price': 0.0,
    }


class TestRunVariantProvisional:
    def _build_keys(self, newest, with_analysis):
        """Seed portfolio at START + optional analysis/morning_prices for newest."""
        keys = {
            f'daily/{replay_engine.START_PORTFOLIO_DATE}/portfolio_state.json':
                _seed_portfolio_state(),
        }
        if with_analysis:
            keys.update(_analysis_keys(newest))
            keys[f'daily/{newest}/llm_risk.json'] = {'risks': {}}
            keys[f'daily/{newest}/morning_prices.parquet'] = _morning_prices_for(newest)
        return keys

    def _variant(self):
        return replay_engine.VariantConfig(
            name='test', decision_params={'min_order_dollars': 250},
            regime_compatibility={}, signal_overrides={}, regime_fusion_overrides={},
            decision_engine_overrides={}, ensemble_overrides={},
            transaction_cost_overrides={},
        )

    def test_provisional_newest_with_analysis(self, monkeypatch):
        """Newest day carries analysis + morning_prices and NO successor prices:
        run_variant prices it provisionally and reports provisional_date."""
        start = replay_engine.START_PORTFOLIO_DATE  # 2026-03-11
        newest = '2026-03-12'
        trading_dates = [start, newest]
        keys = self._build_keys(newest, with_analysis=True)
        cache = CountingCache(FakeS3(keys, trading_dates))

        # Stub the decision engine so the provisional iteration completes without
        # the full model stack. No trades -> flat hold; we only assert pricing.
        monkeypatch.setattr(
            replay_engine.decision_engine, 'run',
            lambda *a, **k: {'actions': [], 'regime': 'risk_on_trend', 'expert_metrics': {}},
        )

        result = replay_engine.run_variant(
            cache, self._variant(), None, trading_dates, pd.DataFrame()
        )
        assert result['provisional_date'] == newest
        assert newest in result['date_value_map']

    def test_monday_like_newest_no_analysis_flat_holds(self, monkeypatch):
        """Newest day has NO analysis (Monday-like): no provisional file probe
        match -> provisional_date is None and the day flat-holds (not priced)."""
        start = replay_engine.START_PORTFOLIO_DATE
        newest = '2026-03-16'
        trading_dates = [start, newest]
        keys = self._build_keys(newest, with_analysis=False)  # no morning_prices either
        cache = CountingCache(FakeS3(keys, trading_dates))

        monkeypatch.setattr(
            replay_engine.decision_engine, 'run',
            lambda *a, **k: {'actions': [], 'regime': 'risk_on_trend', 'expert_metrics': {}},
        )

        result = replay_engine.run_variant(
            cache, self._variant(), None, trading_dates, pd.DataFrame()
        )
        assert result['provisional_date'] is None
        assert newest not in result['date_value_map']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
