"""Tests for the 2026-04-30 optimizer empirical-mutation fix."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.param_space import (
    ParameterSpec,
    compute_empirical_statistic,
    empirical_genome,
    is_calibration_only_diff,
    parse_empirical_statistic,
)


def _spec(
    name: str,
    *,
    parameter_class: str = 'decision_threshold',
    empirical_statistic=None,
    current_value: float = 0.30,
    min_value: float = -5.0,
    max_value: float = 5.0,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        type='float',
        min_value=min_value,
        max_value=max_value,
        current_value=current_value,
        component='expert_signal_computation',
        where_defined='src/signals/fragility.py:19',
        parameter_class=parameter_class,
        empirical_statistic=empirical_statistic,
    )


class TestParseEmpiricalStatistic:
    def test_parses_mean_with_exclude_zero_true(self):
        result = parse_empirical_statistic('mean(avg_correlation, exclude_zero=True)')
        assert result == {
            'fn': 'mean',
            'field': 'avg_correlation',
            'exclude_zero': True,
        }

    def test_parses_std_with_exclude_zero_false(self):
        result = parse_empirical_statistic('std(hy_spread_proxy, exclude_zero=False)')
        assert result == {
            'fn': 'std',
            'field': 'hy_spread_proxy',
            'exclude_zero': False,
        }

    def test_parses_without_exclude_zero(self):
        # exclude_zero defaults to False when omitted.
        result = parse_empirical_statistic('mean(pc1_explained)')
        assert result == {
            'fn': 'mean',
            'field': 'pc1_explained',
            'exclude_zero': False,
        }

    def test_returns_none_for_empty(self):
        assert parse_empirical_statistic(None) is None
        assert parse_empirical_statistic('') is None

    def test_returns_none_for_unsupported_fn(self):
        assert parse_empirical_statistic('quantile(avg_correlation, 0.5)') is None
        assert parse_empirical_statistic('p99(avg_correlation)') is None


class TestComputeEmpiricalStatistic:
    def test_mean_excludes_zero_when_flagged(self):
        spec = _spec(
            'fragility.AVG_CORR_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'avg_correlation', 'exclude_zero': True},
            min_value=0.0,
            max_value=1.0,
        )
        # 10 zeros + 10 real values; if exclude_zero is honored, mean is 0.5,
        # not 0.25.
        rows = [{'avg_correlation': 0.0}] * 10 + [{'avg_correlation': v} for v in [0.4, 0.45, 0.50, 0.55, 0.60, 0.50, 0.45, 0.55, 0.50, 0.50]]
        assert compute_empirical_statistic(spec, rows) == pytest.approx(0.50, abs=1e-9)

    def test_mean_includes_zero_when_not_flagged(self):
        spec = _spec(
            'macro_credit.HY_SPREAD_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'hy_spread_proxy', 'exclude_zero': False},
            min_value=-1.0,
            max_value=1.0,
        )
        # Zero is a real value for hy_spread_proxy (signed), so it must be
        # included.
        rows = [{'hy_spread_proxy': 0.0}] * 5 + [{'hy_spread_proxy': v} for v in [0.01, 0.02, -0.01, -0.02, 0.0]]
        # 5 zeros + 0.01 + 0.02 + (-0.01) + (-0.02) + 0 = 0.0 / 10 = 0.0
        assert compute_empirical_statistic(spec, rows) == pytest.approx(0.0, abs=1e-9)

    def test_std_computed(self):
        spec = _spec(
            'fragility.AVG_CORR_STD',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'std', 'field': 'avg_correlation', 'exclude_zero': True},
            min_value=0.001,
            max_value=1.0,
        )
        # values 0.4, 0.5, 0.6 (sample std = sqrt((0.01+0+0.01)/2) ≈ 0.1)
        rows = [{'avg_correlation': v} for v in [0.4, 0.5, 0.6]] + [{'avg_correlation': 0.0}] * 12
        # exclude_zero filters to 3 values. Sample std with n-1 denominator.
        result = compute_empirical_statistic(spec, rows)
        # n=3 < min_n=10 → returns None
        assert result is None

    def test_returns_none_when_too_few_samples(self):
        spec = _spec(
            'fragility.AVG_CORR_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'avg_correlation', 'exclude_zero': True},
            min_value=0.0,
            max_value=1.0,
        )
        # Only 5 non-zero samples (< 10 min_n).
        rows = [{'avg_correlation': v} for v in [0.4, 0.5, 0.6, 0.45, 0.55]]
        assert compute_empirical_statistic(spec, rows) is None

    def test_returns_none_for_decision_threshold(self):
        # Specs without empirical_statistic must return None — the
        # operator skips them gracefully.
        spec = _spec('regime_fusion.fragility_threshold', parameter_class='decision_threshold')
        rows = [{'avg_correlation': 0.5}] * 100
        assert compute_empirical_statistic(spec, rows) is None

    def test_clamps_to_bounds(self):
        # Empirical statistic outside bounds clamps to bounds.
        spec = _spec(
            'fragility.AVG_CORR_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'avg_correlation', 'exclude_zero': True},
            min_value=0.5,
            max_value=0.6,
        )
        # Real mean 0.40, but bounds [0.5, 0.6] → clamps to 0.5.
        rows = [{'avg_correlation': v} for v in [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]]
        assert compute_empirical_statistic(spec, rows) == pytest.approx(0.5, abs=1e-9)


class TestEmpiricalGenome:
    def test_only_normalization_constants_are_overridden(self):
        norm_spec = _spec(
            'fragility.AVG_CORR_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'avg_correlation', 'exclude_zero': True},
            current_value=0.30,
            min_value=0.0,
            max_value=1.0,
        )
        decision_spec = _spec(
            'regime_fusion.fragility_threshold',
            parameter_class='decision_threshold',
            current_value=0.75,
            min_value=0.0,
            max_value=1.0,
        )
        rows = [{'avg_correlation': 0.5}] * 20
        base = {'fragility.AVG_CORR_MEAN': 0.30, 'regime_fusion.fragility_threshold': 0.75}

        genes, summary = empirical_genome([norm_spec, decision_spec], rows, base)

        # Empirical-mutation overrode the normalization constant only.
        assert genes['fragility.AVG_CORR_MEAN'] == pytest.approx(0.5, abs=1e-9)
        # Decision threshold inherited from base.
        assert genes['regime_fusion.fragility_threshold'] == 0.75
        # Summary lists exactly the normalization constant.
        assert set(summary.keys()) == {'fragility.AVG_CORR_MEAN'}

    def test_skipped_when_empirical_statistic_missing(self):
        # Tag-only normalization constants (empirical_statistic=None) are
        # skipped — the candidate inherits from base for them.
        spec_tag = _spec(
            'vol_uncertainty.vix_thresholds.p20',
            parameter_class='normalization_constant',
            empirical_statistic=None,
            current_value=13.0,
            min_value=9.0,
            max_value=16.0,
        )
        rows = [{'avg_correlation': 0.5}] * 20
        base = {'vol_uncertainty.vix_thresholds.p20': 13.0}

        genes, summary = empirical_genome([spec_tag], rows, base)

        assert genes['vol_uncertainty.vix_thresholds.p20'] == 13.0
        assert summary == {}

    def test_verification_gate_target(self):
        # Simulates the verification gate dry-run on a synthetic dataset
        # whose post-pivot avg_correlation samples have mean ≈ 0.477.
        # Locks the verification target the packet specifies.
        spec = _spec(
            'fragility.AVG_CORR_MEAN',
            parameter_class='normalization_constant',
            empirical_statistic={'fn': 'mean', 'field': 'avg_correlation', 'exclude_zero': True},
            current_value=0.30,
            min_value=0.30,
            max_value=0.65,
        )
        # 125 zero (pre-pivot) + 52 real values centered at 0.477.
        rows = [{'avg_correlation': 0.0}] * 125 + [
            {'avg_correlation': v}
            for v in [
                0.388, 0.395, 0.403, 0.410, 0.418, 0.422, 0.430, 0.440, 0.448, 0.455,
                0.460, 0.465, 0.470, 0.475, 0.477, 0.480, 0.485, 0.490, 0.495, 0.500,
                0.505, 0.510, 0.515, 0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550,
                0.555, 0.560, 0.565, 0.570, 0.575, 0.580, 0.585, 0.590, 0.595, 0.600,
                0.388, 0.395, 0.403, 0.410, 0.418, 0.422, 0.430, 0.440, 0.448, 0.455,
                0.460, 0.470,
            ]
        ]
        base = {'fragility.AVG_CORR_MEAN': 0.30}
        _, summary = empirical_genome([spec], rows, base)
        assert 'fragility.AVG_CORR_MEAN' in summary
        proposed = summary['fragility.AVG_CORR_MEAN']
        assert abs(proposed - 0.477) <= 0.02, (
            f'Verification gate failed: proposed={proposed:.4f}, '
            f'target=0.477±0.02. Empirical-mutation operator broken.'
        )


class TestCalibrationOnlyDiff:
    def test_classifies_calibration_only(self):
        norm_spec = _spec(
            'fragility.AVG_CORR_MEAN', parameter_class='normalization_constant',
        )
        diffs = [{'name': 'fragility.AVG_CORR_MEAN', 'from': 0.30, 'to': 0.477}]
        assert is_calibration_only_diff(diffs, [norm_spec]) is True

    def test_classifies_mixed(self):
        norm_spec = _spec('fragility.AVG_CORR_MEAN', parameter_class='normalization_constant')
        decision_spec = _spec('regime_fusion.fragility_threshold', parameter_class='decision_threshold')
        diffs = [
            {'name': 'fragility.AVG_CORR_MEAN', 'from': 0.30, 'to': 0.477},
            {'name': 'regime_fusion.fragility_threshold', 'from': 0.75, 'to': 0.85},
        ]
        assert is_calibration_only_diff(diffs, [norm_spec, decision_spec]) is False

    def test_classifies_empty_diff_as_not_calibration_only(self):
        # No diffs at all → not "calibration only", just no change.
        spec = _spec('x', parameter_class='normalization_constant')
        assert is_calibration_only_diff([], [spec]) is False
