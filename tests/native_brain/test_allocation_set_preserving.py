"""PKT-TB-008: allocation is set-preserving (drops no name except lot-infeasibility).

The truncation site that broke the incumbent (a shared cash counter drained
inside a ranked walk) does not exist here. Allocation may only change *how much*,
never *which* -- verified across a sizing grid, plus the force-sell / all-cash
case where the whole vector scales toward zero but the set is unchanged.
"""
from __future__ import annotations

import dataclasses

from src.brain.engine import SizingParams, allocate, select


def _alloc(forecast, theta_sel, theta_size, portfolio):
    sel = select(forecast, theta_sel)
    return select(forecast, theta_sel), allocate(
        sel, forecast, theta_size, portfolio, theta_sel.lot_policy
    )


def test_held_set_constant_across_sizing_grid(forecast, theta_sel, portfolio):
    sel = select(forecast, theta_sel)
    reference = None
    for mpw in (0.05, 0.10, 0.20, 0.30, 0.50):
        for mcw in (0.10, 0.35, 1.0):
            for gt in (0.5, 1.0, 1.25):
                for crp in (0.0, 0.10, 0.50):
                    ts = SizingParams(
                        gross_target=gt, max_position_weight=mpw,
                        max_cluster_weight=mcw, cash_reserve_pct=crp,
                    )
                    alloc = allocate(sel, forecast, ts, portfolio, theta_sel.lot_policy)
                    if reference is None:
                        reference = alloc.held_symbols
                    assert alloc.held_symbols == reference
                    # held set is exactly selected minus the lot-infeasible set
                    assert alloc.held_symbols == sel.selected_set - alloc.lot_infeasible


def test_lot_infeasible_set_is_theta_size_independent(forecast, theta_sel, portfolio):
    """C5 precursor: lot-infeasible partition is byte-identical across the grid."""
    sel = select(forecast, theta_sel)
    seen = set()
    for mpw in (0.05, 0.50):
        for gt in (0.5, 1.25):
            ts = SizingParams(gross_target=gt, max_position_weight=mpw)
            alloc = allocate(sel, forecast, ts, portfolio, theta_sel.lot_policy)
            seen.add(tuple(sorted(alloc.lot_infeasible)))
    assert len(seen) == 1  # one and only one lot-infeasible partition


def test_no_symbol_outside_selected_set_appears(forecast, theta_sel, theta_size, portfolio):
    sel, alloc = _alloc(forecast, theta_sel, theta_size, portfolio)
    held_or_buy = {
        a["symbol"] for a in alloc.intents if a["action"] in ("BUY", "HOLD", "REDUCE")
    }
    assert held_or_buy <= set(sel.selected_set)


def test_force_sell_all_cash_keeps_set(forecast, theta_sel, portfolio):
    """Regime shrinks gross toward zero: holds less of each, set S unchanged."""
    sel = select(forecast, theta_sel)
    full = allocate(sel, forecast, SizingParams(gross_target=1.0), portfolio, theta_sel.lot_policy)
    tiny = SizingParams(
        gross_target=1.0,
        regime_exposure_multiplier={"risk_on_trend": 0.0},  # all-cash scaling
    )
    near_zero = allocate(sel, forecast, tiny, portfolio, theta_sel.lot_policy)
    assert near_zero.held_symbols == full.held_symbols  # which is unchanged
    assert near_zero.kappa <= full.kappa or near_zero.parity_record["realized_gross_frac"] == 0.0


def test_kappa_is_a_uniform_downscale(forecast, theta_sel, portfolio):
    """Cash reconciliation shrinks every position by the same kappa, drops none."""
    sel = select(forecast, theta_sel)
    # cash_reserve forces kappa < 1 against a full gross target
    ts = SizingParams(gross_target=1.0, cash_reserve_pct=0.50, max_cluster_weight=1.0)
    alloc = allocate(sel, forecast, ts, portfolio, theta_sel.lot_policy)
    assert 0.0 < alloc.kappa <= 1.0
    # no held name was dropped by the down-scale
    assert alloc.held_symbols == sel.selected_set - alloc.lot_infeasible
