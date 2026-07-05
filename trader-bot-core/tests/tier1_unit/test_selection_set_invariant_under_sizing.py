"""Tier 1 unit-fixture — SALVAGE #1, re-pointed to the clean engine.

The one honestly-designed test in the old suite: for a fixed forecast and
selection params, the held symbol SET is invariant across the whole sizing grid
(the property the incumbent violated — the 0.20-vs-0.30 sign-flip). Ported here
to the NEW ``engine`` (``held_symbols_invariant`` + ``default_theta_size_grid``),
demoted to a Tier-1 unit test (never "proof the whole system works"), with:

  * a REAL RECORDED forecast fixture (mu captured from a live governed invoke,
    ``tests/fixtures/recorded_forecast_mu.json``);
  * a deterministic adversarial battery (tier-edge ties, near-h_min health,
    near-lot-floor);
  * the ``xfail`` strict NEGATIVE CONTROL: the incumbent truncating walk IS
    non-invariant — pinning the sign-flip so the guard would have caught it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FIXTURES  # noqa: E402

from engine import held_symbols_invariant, select, allocate  # noqa: E402
from engine.engine import default_theta_size_grid  # noqa: E402
from engine.contracts import (ForecastBundle, LotPolicy, PortfolioState,  # noqa: E402
                              SelectionParams, SizingParams)

_GRID = default_theta_size_grid()


def _bundle(mu: Dict[str, float], asset_class: Dict[str, str], *, regime="choppy",
            health=None, event=None, eligible=None) -> ForecastBundle:
    syms = list(mu)
    return ForecastBundle(
        date="2026-07-02", mu_M1=dict(mu),
        health=health or {s: 0.85 for s in syms},
        regime_label=regime,
        event_block=event or {s: False for s in syms},
        idio_vol={s: 0.10 + 0.001 * i for i, s in enumerate(syms)},
        eligible=eligible or {s: True for s in syms},
        asset_class=asset_class,
        vol_bucket={s: "med" for s in syms},
    )


def _port(syms: List[str], nav=100_000.0) -> PortfolioState:
    return PortfolioState(nav=nav, cash=nav, positions={},
                          marks={s: 100.0 for s in syms},
                          cluster_of={s: "equity" for s in syms},
                          beta={s: 1.0 for s in syms}, sigma={s: 0.15 for s in syms})


# --------------------------------------------------------------------------- #
# (A) the REAL recorded forecast — invariant must hold on genuine mu.
# --------------------------------------------------------------------------- #
def _recorded_bundle():
    rec = json.loads((FIXTURES / "recorded_forecast_mu.json").read_text())
    mu = {k: float(v) for k, v in rec["mu"].items()}
    if len(mu) < 3:
        pytest.skip("recorded mu fixture too small to exercise selection")
    import csv
    ucsv = Path(__file__).resolve().parents[2] / "config" / "universe.csv"
    acls = {}
    with open(ucsv) as f:
        for row in csv.DictReader(f):
            acls[row["symbol"].strip()] = row.get("asset_class", "equity").strip()
    asset_class = {s: acls.get(s, "equity") for s in mu}
    return _bundle(mu, asset_class), _port(list(mu))


@pytest.mark.unit
def test_selection_set_invariant_on_recorded_forecast():
    f, port = _recorded_bundle()
    theta_sel = SelectionParams(N=10, h_min=0.60)
    assert held_symbols_invariant(f, theta_sel, port, _GRID), (
        "held symbol SET changed across the sizing grid on a REAL recorded "
        "forecast — the sizing-invariance property is violated")


# --------------------------------------------------------------------------- #
# (B) deterministic adversarial battery (frozen, no RNG).
# --------------------------------------------------------------------------- #
def _battery():
    fixtures = []
    # monotone mu
    for k in range(3):
        syms = [f"real{k}_{i:02d}" for i in range(12 + k)]
        mu = {s: 0.50 - 0.03 * i for i, s in enumerate(syms)}
        fixtures.append((f"real{k}", _bundle(mu, {s: "equity" for s in syms}), _port(syms)))
    # tier-edge 3-way ties
    for k in range(2):
        syms = [f"ties{k}_{i:02d}" for i in range(14)]
        mu = {s: 0.40 - 0.05 * (i // 3) for i, s in enumerate(syms)}
        fixtures.append((f"ties{k}", _bundle(mu, {s: "equity" for s in syms}), _port(syms)))
    # near-h_min health straddle
    for k in range(2):
        syms = [f"hmin{k}_{i:02d}" for i in range(16)]
        mu = {s: 0.45 - 0.02 * i for i, s in enumerate(syms)}
        health = {s: 0.60 + (0.01 if i % 2 == 0 else -0.01) for i, s in enumerate(syms)}
        fixtures.append((f"hmin{k}", _bundle(mu, {s: "equity" for s in syms}, health=health), _port(syms)))
    # near-lot-floor (many names, tiny weights)
    for k in range(2):
        syms = [f"lot{k}_{i:02d}" for i in range(18 + k)]
        mu = {s: 0.50 - 0.01 * i for i, s in enumerate(syms)}
        p = _port(syms, nav=20_000.0 + 1_000.0 * k)
        fixtures.append((f"lot{k}", _bundle(mu, {s: "equity" for s in syms}), p))
    return fixtures


@pytest.mark.unit
@pytest.mark.parametrize("case", _battery(), ids=lambda c: c[0])
def test_selection_set_invariant_under_sizing(case):
    name, f, port = case
    theta_sel = SelectionParams(N=10, h_min=0.60)
    assert held_symbols_invariant(f, theta_sel, port, _GRID), (
        f"[{name}] held symbol SET changed across the sizing grid — sizing "
        f"invariance violated (the incumbent 0.20-vs-0.30 sign-flip class)")


# --------------------------------------------------------------------------- #
# (C) NEGATIVE CONTROL — the incumbent truncating walk IS non-invariant.
# --------------------------------------------------------------------------- #
def _incumbent_truncating_walk(f: ForecastBundle, port: PortfolioState, ts: SizingParams) -> frozenset:
    pv = port.nav
    available_cash = pv * (1.0 - ts.cash_reserve_pct)
    gross_cap = pv * ts.gross_target * ts.regime_exposure_multiplier.get(f.regime_label, 1.0)
    pos_cap = pv * ts.max_position_weight
    cluster_cap = pv * ts.max_cluster_weight
    min_order = 250.0
    ranked = sorted(f.mu_M1, key=lambda s: (-f.mu_M1[s], s))
    cluster_dollars: Dict[str, float] = {}
    deployed = 0.0
    held: List[str] = []
    for sym in ranked:
        price = port.marks.get(sym, 0.0)
        dollars = pos_cap
        gross_head = gross_cap - deployed
        if gross_head <= 0:
            continue
        dollars = min(dollars, gross_head)
        cl = port.cluster_of.get(sym, sym)
        cluster_head = cluster_cap - cluster_dollars.get(cl, 0.0)
        if cluster_head <= 0:
            continue
        dollars = min(dollars, cluster_head)
        if dollars < min_order or dollars > available_cash:
            continue
        shares = int(dollars / price) if price > 0 else 0
        if shares <= 0:
            continue
        dollars = shares * price
        if dollars < min_order or dollars > available_cash:
            continue
        held.append(sym)
        available_cash -= dollars
        deployed += dollars
        cluster_dollars[cl] = cluster_dollars.get(cl, 0.0) + dollars
    return frozenset(held)


def _incumbent_held_varies(f, port) -> bool:
    seen = set()
    for ts in _GRID:
        seen.add(_incumbent_truncating_walk(f, port, ts))
        if len(seen) > 1:
            return True
    return False


def _neg_fixture():
    syms = [f"neg_{i:02d}" for i in range(14)]
    mu = {s: 0.50 - 0.02 * i for i, s in enumerate(syms)}
    return _bundle(mu, {s: "equity" for s in syms}), _port(syms)


@pytest.mark.unit
@pytest.mark.xfail(reason="incumbent truncating walk is NON-invariant by construction "
                          "(the 0.20-vs-0.30 sign-flip); this asserts invariance and is "
                          "EXPECTED to fail — the pinned regression guard.", strict=True)
def test_incumbent_negative_control_is_invariant():
    f, port = _neg_fixture()
    assert not _incumbent_held_varies(f, port), "incumbent should be non-invariant"


@pytest.mark.unit
def test_incumbent_is_demonstrably_non_invariant():
    f, port = _neg_fixture()
    assert _incumbent_held_varies(f, port), (
        "the incumbent walk must be non-invariant for the negative control to bite")
