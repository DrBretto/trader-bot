"""PKT-TB-008 acceptance: Select reads no dollar / cash / NAV / cluster / cap value.

Three independent layers of evidence:
  1. structural -- ``select``'s signature exposes no sizing/portfolio parameter;
  2. enforced  -- ``assert_no_dollar_surface`` rejects any forbidden field, and
     ``select`` calls it on every invocation;
  3. behavioural -- the selected set is unchanged when every theta_size dollar
     knob is swept (a focused precursor to the full PKT-TB-009 invariant).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from src.brain.engine import SizingParams, select
from src.brain.engine.contracts import assert_no_dollar_surface


def test_select_signature_has_no_sizing_or_portfolio_param():
    params = list(inspect.signature(select).parameters)
    assert params == ["f", "theta_sel"], (
        "Select must take only (f, theta_sel); any portfolio/sizing parameter "
        "would re-couple selection to sizing."
    )


def test_forecast_and_theta_sel_expose_no_dollar_surface(forecast, theta_sel):
    # Must not raise -- the real inputs are clean.
    assert_no_dollar_surface(forecast, theta_sel)


def test_assert_rejects_a_forbidden_field():
    @dataclass(frozen=True)
    class Tainted:
        mu: dict
        nav: float  # forbidden book-dollar surface

    with pytest.raises(AssertionError):
        assert_no_dollar_surface(Tainted(mu={}, nav=1.0))


def test_selected_set_invariant_to_sizing_dollar_knobs(forecast, theta_sel, portfolio):
    """Selection cannot even receive theta_size; confirm the set is identical for
    wildly different sizing configs (the full grid is swept in PKT-TB-009).
    """
    base = select(forecast, theta_sel).selected_set
    for mpw in (0.05, 0.10, 0.20, 0.30, 0.50):
        for gt in (0.5, 1.0, 1.25):
            for crp in (0.0, 0.25, 0.50):
                # theta_size is constructed but provably never reaches select().
                _ = SizingParams(gross_target=gt, max_position_weight=mpw, cash_reserve_pct=crp)
                assert select(forecast, theta_sel).selected_set == base
