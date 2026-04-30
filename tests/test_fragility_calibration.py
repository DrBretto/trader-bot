"""Tests for the 2026-04-30 fragility historical-norm recalibration.

These tests lock both the production defaults (so a silent code change cannot
roll the audit's ship-decision back) and the recalibrated empirical baseline
(so a future drift in the input dataset is detectable). See:
- docs/plans/2026-04-30-phase2-fragility-baseline-audit.md (derivation)
- docs/POSTMORTEMS.md "stale-historical-norm" (failure mode this prevents)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.signals.fragility import (
    compute_fragility,
    AVG_CORR_MEAN,
    AVG_CORR_STD,
    PC1_MEAN,
    PC1_STD,
    RECALIBRATED_2026_04_30,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECAL_BUNDLE_PATH = REPO_ROOT / 'config' / 'decision_params.recalibrated_2026_04_30.json'


class TestFragilityCalibrationLock:
    """Lock both the production default and the empirical recalibration."""

    def test_production_defaults_unchanged(self):
        # Production constants stay at the originals until operator promotes.
        # If this test fails, someone changed defaults without the audit gate.
        assert AVG_CORR_MEAN == 0.30
        assert AVG_CORR_STD == 0.15
        assert PC1_MEAN == 0.45
        assert PC1_STD == 0.12

    def test_recalibrated_constants_present(self):
        # The empirical 900d re-derivation values must be present and
        # documented. If these change, the dataset must be re-derived and
        # the Phase 2 audit doc updated to match.
        assert RECALIBRATED_2026_04_30['AVG_CORR_MEAN'] == pytest.approx(0.4774)
        assert RECALIBRATED_2026_04_30['AVG_CORR_STD'] == pytest.approx(0.0979)
        assert RECALIBRATED_2026_04_30['PC1_MEAN'] == pytest.approx(0.6007)
        assert RECALIBRATED_2026_04_30['PC1_STD'] == pytest.approx(0.0658)
        # Provenance fields exist (so a future maintainer cannot simply
        # mutate the values without leaving a trail).
        assert 'source_dataset' in RECALIBRATED_2026_04_30
        assert 'derivation_date' in RECALIBRATED_2026_04_30

    def test_recalibrated_bundle_exists_and_matches(self):
        # The shipping bundle must carry the same values as the Python module
        # constants — divergence between the two is the failure class this
        # test exists to catch (config and code drifting apart).
        assert RECAL_BUNDLE_PATH.exists(), (
            f'Expected recalibrated bundle at {RECAL_BUNDLE_PATH}; '
            f'see docs/plans/2026-04-30-phase4-recalibration-recommendation.md'
        )
        bundle = json.loads(RECAL_BUNDLE_PATH.read_text())
        frag = bundle.get('signals', {}).get('fragility', {})
        assert frag.get('AVG_CORR_MEAN') == pytest.approx(
            RECALIBRATED_2026_04_30['AVG_CORR_MEAN']
        )
        assert frag.get('AVG_CORR_STD') == pytest.approx(
            RECALIBRATED_2026_04_30['AVG_CORR_STD']
        )
        assert frag.get('PC1_MEAN') == pytest.approx(
            RECALIBRATED_2026_04_30['PC1_MEAN']
        )
        assert frag.get('PC1_STD') == pytest.approx(
            RECALIBRATED_2026_04_30['PC1_STD']
        )

    def test_recalibrated_params_change_fragility_distinguishably(self):
        # Build a synthetic moderate-correlation panel that lands near the
        # empirical 25th percentile (avg_correlation ~ 0.40). Under current
        # constants the score is already at the gate threshold; under
        # recalibrated constants it is well below — i.e. **resolution
        # restored** at the lower tail of the empirical distribution.
        rng = np.random.default_rng(seed=42)
        n_days = 80
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'EFA', 'EEM']
        dates = pd.date_range('2024-01-01', periods=n_days)
        common = rng.normal(0.0, 0.01, n_days)
        rows = []
        w_common = 0.80
        for sym in symbols:
            idio = rng.normal(0.0, 0.01 * np.sqrt(1 - w_common ** 2), n_days)
            ret = w_common * common + idio
            px = 100.0 * np.exp(np.cumsum(ret))
            for i, d in enumerate(dates):
                rows.append({'date': d, 'symbol': sym, 'close': px[i]})
        df = pd.DataFrame(rows)

        current = compute_fragility(df)
        recal = compute_fragility(df, params=RECALIBRATED_2026_04_30)

        # Sanity-check the fixture lands in the targeted band: avg_corr
        # near the empirical 25th percentile (~0.43-0.45).
        assert 0.40 < current['avg_correlation'] < 0.50, (
            f'Synthetic panel produced avg_correlation '
            f'{current["avg_correlation"]:.3f}, expected 0.40-0.50 '
            f'(near empirical p25). Tune w_common in the fixture.'
        )

        # Current constants saturate (>0.75 — gate fires) on input that the
        # 900d empirical distribution treats as below-median (recal < 0.30).
        assert current['fragility_score'] > 0.75, (
            f'Expected current constants to saturate on near-p25 input; '
            f'got fragility_score={current["fragility_score"]:.3f}'
        )
        assert recal['fragility_score'] < 0.30, (
            f'Expected recalibrated constants to read near-p25 input as low; '
            f'got fragility_score={recal["fragility_score"]:.3f}'
        )
        gap = current['fragility_score'] - recal['fragility_score']
        assert gap > 0.40, (
            f'Expected current vs recalibrated gap > 0.40 on near-p25 input; '
            f'got gap={gap:.3f}'
        )

    def test_recalibrated_params_preserve_panic_signal(self):
        # On a high-correlation panel that genuinely is at the empirical p95
        # (avg_corr > 0.62), even the recalibrated metric must read above
        # the 0.75 gate threshold. This locks the protection property:
        # extreme cross-asset correlation is still flagged.
        rng = np.random.default_rng(seed=7)
        n_days = 80
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'EFA', 'EEM']
        dates = pd.date_range('2024-01-01', periods=n_days)
        common = rng.normal(0.0, 0.01, n_days)
        rows = []
        for sym in symbols:
            idio = rng.normal(0.0, 0.0035, n_days)
            ret = 0.85 * common + 0.15 * idio  # ~p95-level coupling
            px = 100.0 * np.exp(np.cumsum(ret))
            for i, d in enumerate(dates):
                rows.append({'date': d, 'symbol': sym, 'close': px[i]})
        df = pd.DataFrame(rows)

        recal = compute_fragility(df, params=RECALIBRATED_2026_04_30)
        assert recal['avg_correlation'] > 0.60, (
            f'Test panel did not land at p95 coupling; '
            f'got avg_correlation={recal["avg_correlation"]:.3f}'
        )
        assert recal['fragility_score'] > 0.75, (
            f'Expected recalibrated metric to fire gate at extreme coupling; '
            f'got fragility_score={recal["fragility_score"]:.3f}'
        )
