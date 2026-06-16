"""Shared fixtures for the native two-stage engine tests.

A small, deterministic forecast fixture plus a NAV/positions book. The
exhaustive >=30-fixture x full-grid invariant battery lives in PKT-TB-009
(``test_selection_invariant.py``); these fixtures support PKT-TB-008's
engine unit tests.
"""
from __future__ import annotations

import pytest

from src.brain.engine import (
    ForecastBundle,
    LotPolicy,
    PortfolioState,
    SelectionParams,
    SizingParams,
)

# 12-name universe spanning equity / bond / commodity, with a spread of mu_M1,
# health straddling h_min, one M4 event block, and one ineligible name.
_SYMBOLS = ["ITA", "SOXX", "XRT", "RSP", "TLT", "AGG", "MUB", "FXE", "USO", "FXI", "KRE", "SHY"]

_MU_M1 = {
    "ITA": 0.42, "SOXX": 0.38, "XRT": 0.31, "RSP": 0.27, "TLT": 0.22, "AGG": 0.18,
    "MUB": 0.15, "FXE": 0.12, "USO": 0.09, "FXI": 0.05, "KRE": 0.30, "SHY": 0.02,
}
_HEALTH = {
    "ITA": 0.82, "SOXX": 0.78, "XRT": 0.71, "RSP": 0.66, "TLT": 0.69, "AGG": 0.64,
    "MUB": 0.61, "FXE": 0.40, "USO": 0.55, "FXI": 0.72, "KRE": 0.74, "SHY": 0.90,
}  # FXE below h_min=0.60 -> gated out; USO below -> gated out
_EVENT_BLOCK = {s: False for s in _SYMBOLS}
_EVENT_BLOCK["KRE"] = True  # M4 hard exclusion despite strong mu_M1
_IDIO_VOL = {s: round(0.10 + 0.01 * i, 4) for i, s in enumerate(_SYMBOLS)}
_ELIGIBLE = {s: True for s in _SYMBOLS}
_ELIGIBLE["SHY"] = False  # ineligible (e.g. ADV/coverage gate folded in upstream)
_ASSET_CLASS = {
    "ITA": "equity", "SOXX": "equity", "XRT": "equity", "RSP": "equity",
    "TLT": "bond", "AGG": "bond", "MUB": "bond", "SHY": "bond",
    "FXE": "fx", "USO": "commodity", "FXI": "equity", "KRE": "equity",
}
_VOL_BUCKET = {s: "med" for s in _SYMBOLS}
_CLUSTER_OF = {
    "ITA": "equity", "SOXX": "equity", "XRT": "equity", "RSP": "equity", "FXI": "equity",
    "KRE": "equity", "TLT": "rates", "AGG": "rates", "MUB": "rates", "SHY": "rates",
    "FXE": "fx", "USO": "commodity",
}


@pytest.fixture
def forecast() -> ForecastBundle:
    return ForecastBundle(
        date="2026-06-16",
        mu_M1=dict(_MU_M1),
        health=dict(_HEALTH),
        regime_label="risk_on_trend",
        event_block=dict(_EVENT_BLOCK),
        idio_vol=dict(_IDIO_VOL),
        eligible=dict(_ELIGIBLE),
        asset_class=dict(_ASSET_CLASS),
        vol_bucket=dict(_VOL_BUCKET),
    )


@pytest.fixture
def theta_sel() -> SelectionParams:
    return SelectionParams(
        N=8,
        h_min=0.60,
        core_fraction=0.5,
        regime_admissibility={
            # risk_on_trend admits everything; a panic regime would admit only defensives.
            "high_vol_panic": ("bond", "commodity"),
        },
        lot_policy=LotPolicy(min_order=250.0, reference_nav=100_000.0),
    )


@pytest.fixture
def theta_size() -> SizingParams:
    return SizingParams(
        gross_target=1.0,
        max_position_weight=0.20,
        max_cluster_weight=0.35,
        cash_reserve_pct=0.10,
    )


@pytest.fixture
def portfolio() -> PortfolioState:
    marks = {s: 100.0 for s in _SYMBOLS}
    return PortfolioState(
        nav=100_000.0,
        cash=100_000.0,
        positions={},
        marks=marks,
        cluster_of=dict(_CLUSTER_OF),
        beta={s: 1.0 for s in _SYMBOLS},
        sigma={s: 0.15 for s in _SYMBOLS},
    )
