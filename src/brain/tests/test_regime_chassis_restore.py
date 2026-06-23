"""Regime chassis restoration tests (2026-06-23 re-freeze).

The native two-stage cutover had dropped config/regime_compatibility.json — the
regime picker's multiplier series the orthogonal members were specced to add value
ON TOP OF (PROPOSAL_ORTHOGONALITY §0). These tests pin the restoration:

  - build_forecast_bundle populates regime_score_mult from the compat table
    (sector key, asset_class fallback, 1.0 default; empty table -> all 1.0).
  - selection ranks on the regime-tilted forecast: in high_vol_panic a high-mu
    tech name is demoted below a lower-mu defensive name (the rotation that did
    not happen on 2026-06-23).
  - theta_size.regime_exposure_multiplier cuts the book-level gross in panic.
  - build_portfolio_state collapses the correlated sleeves (semis/tech/innovation
    -> tech_growth) so max_cluster_weight actually binds.
"""
from __future__ import annotations

import pandas as pd

from src.brain.forecast_adapter import build_forecast_bundle, build_portfolio_state
from src.brain.engine import SelectionParams, SizingParams, allocate, select
from src.brain.runtime import theta_from_freeze


# A minimal slice of the real regime_compatibility.json shape.
COMPAT = {
    "risk_on_trend": {"equity": 1.10, "commodity": 1.0, "sector_tech": 1.15},
    "high_vol_panic": {"equity": 0.50, "commodity": 1.10, "sector_tech": 0.40},
}

UNIVERSE = pd.DataFrame([
    {"symbol": "XLK",  "asset_class": "equity",    "sector": "sector_tech",      "eligible": 1},
    {"symbol": "SMH",  "asset_class": "equity",    "sector": "industry_semis",   "eligible": 1},
    {"symbol": "ARKK", "asset_class": "equity",    "sector": "theme_innovation", "eligible": 1},
    {"symbol": "GLD",  "asset_class": "commodity", "sector": "gold",             "eligible": 1},
    {"symbol": "SLV",  "asset_class": "commodity", "sector": "silver",           "eligible": 1},
])


def _features():
    return pd.DataFrame([
        {"date": "2026-06-23", "symbol": s, "close": 100.0, "vol_21d": 0.15}
        for s in UNIVERSE["symbol"]
    ])


def test_regime_score_mult_lookup():
    # sector key hits; asset_class fallback for sectors absent from the table; 1.0 default.
    mu = {s: 0.5 for s in UNIVERSE["symbol"]}
    f = build_forecast_bundle("2026-06-23", mu, _features(), "high_vol_panic",
                              UNIVERSE, regime_compat=COMPAT)
    assert f.regime_score_mult["XLK"] == 0.40        # sector_tech direct
    assert f.regime_score_mult["SMH"] == 0.50        # industry_semis absent -> equity fallback
    assert f.regime_score_mult["GLD"] == 1.10        # gold absent -> commodity fallback
    # empty/None compat -> all 1.0 (backward compatible: raw-mu ranking)
    f0 = build_forecast_bundle("2026-06-23", mu, _features(), "high_vol_panic",
                               UNIVERSE, regime_compat=None)
    assert all(v == 1.0 for v in f0.regime_score_mult.values())


def test_panic_rotates_tech_out():
    # XLK has the HIGHEST raw forecast; GLD lower. In panic the regime tilt must
    # demote XLK below GLD (rotation into the defensive sleeve).
    mu = {"XLK": 1.00, "SMH": 0.95, "ARKK": 0.90, "GLD": 0.60, "SLV": 0.55}
    theta_sel = SelectionParams(N=2)

    panic = build_forecast_bundle("2026-06-23", mu, _features(), "high_vol_panic",
                                  UNIVERSE, regime_compat=COMPAT)
    sel_panic = select(panic, theta_sel)
    assert "GLD" in sel_panic.selected_set          # defensive selected
    assert "XLK" not in sel_panic.selected_set      # tech rotated out despite top raw mu

    # In risk_on the same forecast keeps the tech names (no rotation).
    riskon = build_forecast_bundle("2026-06-23", mu, _features(), "risk_on_trend",
                                   UNIVERSE, regime_compat=COMPAT)
    sel_riskon = select(riskon, theta_sel)
    assert "XLK" in sel_riskon.selected_set


def test_regime_exposure_multiplier_cuts_gross_in_panic():
    _, theta_size = theta_from_freeze()
    assert theta_size.regime_exposure_multiplier["high_vol_panic"] == 0.50
    assert theta_size.regime_exposure_multiplier["risk_off_trend"] == 0.75
    assert theta_size.regime_exposure_multiplier["risk_on_trend"] == 1.0


def test_correlation_groups_collapse_tech_complex():
    state = {"cash": 50_000.0, "holdings": [
        {"symbol": "XLK", "shares": 100, "current_price": 100.0},
        {"symbol": "SMH", "shares": 100, "current_price": 100.0},
        {"symbol": "ARKK", "shares": 100, "current_price": 100.0},
        {"symbol": "GLD", "shares": 100, "current_price": 100.0},
        {"symbol": "SLV", "shares": 100, "current_price": 100.0},
    ]}
    portfolio = build_portfolio_state(state, _features(), UNIVERSE)
    # semis/tech/innovation collapse to one cluster; gold/silver to another.
    assert portfolio.cluster_of["XLK"] == "tech_growth"
    assert portfolio.cluster_of["SMH"] == "tech_growth"
    assert portfolio.cluster_of["ARKK"] == "tech_growth"
    assert portfolio.cluster_of["GLD"] == "precious_metals"
    assert portfolio.cluster_of["SLV"] == "precious_metals"


def test_cluster_cap_binds_on_collapsed_complex():
    # Five equal-weight names, three of them the tech_growth complex. With the
    # cap at 0.35 the complex must be projected down to <= 35% of NAV.
    mu = {"XLK": 1.0, "SMH": 0.99, "ARKK": 0.98, "GLD": 0.97, "SLV": 0.96}
    f = build_forecast_bundle("2026-06-23", mu, _features(), "risk_on_trend",
                              UNIVERSE, regime_compat=COMPAT)
    theta_sel = SelectionParams(N=5)
    sel = select(f, theta_sel)
    state = {"cash": 100_000.0, "holdings": []}
    portfolio = build_portfolio_state(state, _features(), UNIVERSE)
    theta_size = SizingParams(gross_target=1.0, max_position_weight=0.30,
                              max_cluster_weight=0.35, cash_reserve_pct=0.0)
    alloc = allocate(sel, f, theta_size, portfolio, theta_sel.lot_policy)
    dollars = {a["symbol"]: a.get("dollars", 0.0) for a in alloc.intents
               if a["action"] in ("BUY", "HOLD")}
    tech = sum(dollars.get(s, 0.0) for s in ("XLK", "SMH", "ARKK"))
    assert tech <= 0.36 * portfolio.nav  # bound (allow integer-lot slack)
