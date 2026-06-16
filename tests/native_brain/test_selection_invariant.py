"""PKT-TB-009 — the spine in CI: symbol-set-invariance-under-sizing.

THE test that licenses the word "New Brain" to exist on the employer-facing
surface (DESIGN_DOSSIER §3 ruling). For fixed forecast ``f`` and selection
params ``theta_sel``, the held symbol *set* must be constant across the entire
sizing grid ``theta_size`` -- the property the incumbent violated (the historical
0.20-vs-0.30 sign-flip).

Three guarantees, per Analyst sign-off §3 / condition C5:
  1. frozenset equality of held_symbols across the full theta_size grid x >=30
     fixtures (real + adversarial);
  2. the lot-infeasibility exclusion set is byte-identical across the grid per
     fixture, and Stage-1 ``min_order`` is evaluated against a theta_size-
     INDEPENDENT notional (NOT gross_target*NAV);
  3. an ``xfail`` negative control runs the incumbent truncating walk and
     asserts NON-invariance -- pinning the sign-flip so the test would have
     caught the original bug.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import pytest

from src.brain.engine import (
    ForecastBundle,
    LotPolicy,
    PortfolioState,
    SelectionParams,
    SizingParams,
    allocate,
    select,
)

# --------------------------------------------------------------------------- #
# The theta_size grid (DESIGN_DOSSIER §2.1 test spec / 02_ML_ENGINE_DESIGN §3). #
# Cartesian product = 6 x 5 x 4 x 4 x 3 = 1440 points.                          #
# --------------------------------------------------------------------------- #
_MAX_POSITION_WEIGHT = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
_MAX_CLUSTER_WEIGHT = (0.10, 0.20, 0.30, 0.50, 1.0)
_GROSS_TARGET = (0.5, 0.75, 1.0, 1.25)
_CASH_RESERVE = (0.0, 0.10, 0.25, 0.50)
_REGIME_EXPOSURE = (0.25, 0.5, 1.0)


def _theta_size_grid() -> List[SizingParams]:
    grid: List[SizingParams] = []
    for mpw, mcw, gt, crp, rem in itertools.product(
        _MAX_POSITION_WEIGHT, _MAX_CLUSTER_WEIGHT, _GROSS_TARGET, _CASH_RESERVE, _REGIME_EXPOSURE
    ):
        grid.append(SizingParams(
            gross_target=gt,
            max_position_weight=mpw,
            max_cluster_weight=mcw,
            cash_reserve_pct=crp,
            regime_exposure_multiplier={"r": rem},   # keyed to the fixtures' regime label 'r'
        ))
    return grid


_THETA_SIZE_GRID = _theta_size_grid()


# --------------------------------------------------------------------------- #
# >=30 frozen fixtures: real + adversarial (tier-edge ties, near-h_min health,  #
# near-min_order lotting, panic force-sell books, all-cash, single-cluster).    #
# Each fixture is (ForecastBundle, PortfolioState, SelectionParams). Built       #
# deterministically (no RNG) so the battery is frozen.                          #
# --------------------------------------------------------------------------- #
_CLUSTERS = ("equity", "rates", "fx", "commodity")
_CLASSES = {"equity": "equity", "rates": "bond", "fx": "fx", "commodity": "commodity"}


def _mk_fixture(
    name: str,
    n_names: int,
    *,
    mu_fn,
    health_fn,
    eligible_fn=lambda i: True,
    event_fn=lambda i: False,
    cluster_fn=lambda i: _CLUSTERS[i % len(_CLUSTERS)],
    positions_fn=lambda i, sym: 0,
    nav: float = 100_000.0,
    lot_policy: LotPolicy = LotPolicy(min_order=250.0, reference_nav=100_000.0),
    N: int = 10,
    h_min: float = 0.60,
) -> Tuple[str, ForecastBundle, PortfolioState, SelectionParams]:
    syms = [f"{name}_{i:02d}" for i in range(n_names)]
    mu = {syms[i]: float(mu_fn(i)) for i in range(n_names)}
    health = {syms[i]: float(health_fn(i)) for i in range(n_names)}
    event_block = {syms[i]: bool(event_fn(i)) for i in range(n_names)}
    eligible = {syms[i]: bool(eligible_fn(i)) for i in range(n_names)}
    idio_vol = {syms[i]: 0.10 + 0.001 * i for i in range(n_names)}
    cluster_of = {syms[i]: cluster_fn(i) for i in range(n_names)}
    asset_class = {syms[i]: _CLASSES[cluster_of[syms[i]]] for i in range(n_names)}
    vol_bucket = {syms[i]: "med" for i in range(n_names)}
    positions = {syms[i]: int(positions_fn(i, syms[i])) for i in range(n_names)}
    positions = {s: v for s, v in positions.items() if v != 0}
    f = ForecastBundle(
        date="2026-06-16",
        mu_M1=mu, health=health, regime_label="r", event_block=event_block,
        idio_vol=idio_vol, eligible=eligible, asset_class=asset_class, vol_bucket=vol_bucket,
    )
    port = PortfolioState(
        nav=nav, cash=nav, positions=positions,
        marks={s: 100.0 for s in syms},
        cluster_of=cluster_of,
        beta={s: 1.0 for s in syms}, sigma={s: 0.15 for s in syms},
    )
    theta_sel = SelectionParams(N=N, h_min=h_min, core_fraction=0.5, lot_policy=lot_policy)
    return name, f, port, theta_sel


def _build_fixtures() -> List[Tuple[str, ForecastBundle, PortfolioState, SelectionParams]]:
    fx: List = []
    # (1-6) real-ish: monotone mu, healthy, varied size / cluster spread
    for k in range(6):
        fx.append(_mk_fixture(
            f"real{k}", 12 + k,
            mu_fn=lambda i: 0.50 - 0.03 * i,
            health_fn=lambda i: 0.90 - 0.01 * i,
        ))
    # (7-11) tier-edge ties: blocks of equal mu (tiebreak by idio_vol, symbol)
    for k in range(5):
        fx.append(_mk_fixture(
            f"ties{k}", 14,
            mu_fn=lambda i: 0.40 - 0.05 * (i // 3),   # 3-way ties
            health_fn=lambda i: 0.80,
        ))
    # (12-16) near-h_min health straddle (some just above, some just below)
    for k in range(5):
        fx.append(_mk_fixture(
            f"hmin{k}", 16,
            mu_fn=lambda i: 0.45 - 0.02 * i,
            health_fn=lambda i, k=k: 0.60 + (0.005 if (i + k) % 2 == 0 else -0.005) * (1 + i % 3),
        ))
    # (17-21) near-min_order lotting: many names -> tiny w_target near the lot floor
    for k in range(5):
        fx.append(_mk_fixture(
            f"lot{k}", 18 + k,
            mu_fn=lambda i: 0.50 - 0.01 * i,
            health_fn=lambda i: 0.85,
            N=14,
            lot_policy=LotPolicy(min_order=250.0, reference_nav=20_000.0 + 1_000.0 * k),
        ))
    # (22-25) panic force-sell books: existing positions, regime exposure swept toward 0
    for k in range(4):
        fx.append(_mk_fixture(
            f"panic{k}", 12,
            mu_fn=lambda i: 0.40 - 0.02 * i,
            health_fn=lambda i: 0.75,
            positions_fn=lambda i, sym: 50 + 5 * i,
        ))
    # (26-28) all-cash, single-cluster-dominant
    for k in range(3):
        fx.append(_mk_fixture(
            f"onecluster{k}", 13,
            mu_fn=lambda i: 0.48 - 0.02 * i,
            health_fn=lambda i: 0.82,
            cluster_fn=lambda i: "equity",
        ))
    # (29-32) mixed eligibility + M4 event blocks interleaved
    for k in range(4):
        fx.append(_mk_fixture(
            f"gates{k}", 15,
            mu_fn=lambda i: 0.46 - 0.015 * i,
            health_fn=lambda i: 0.78,
            eligible_fn=lambda i: (i % 4 != 0),
            event_fn=lambda i, k=k: (i % 5 == k % 5),
        ))
    return fx


_FIXTURES = _build_fixtures()
assert len(_FIXTURES) >= 30, f"need >=30 fixtures, have {len(_FIXTURES)}"


# --------------------------------------------------------------------------- #
# The negative control: a faithful transcription of the incumbent truncating    #
# walk (src/steps/decision_engine.py:1264-1343, verified against source). The    #
# shared ``available_cash`` is drained INSIDE the ranked walk and per-cluster /  #
# gross headroom forces ``continue`` -- so a bigger cap changes WHICH names fit. #
# This is the 0.20-vs-0.30 sign-flip mechanism, reproduced to be pinned.         #
# --------------------------------------------------------------------------- #
def _incumbent_truncating_walk(f: ForecastBundle, port: PortfolioState, ts: SizingParams) -> frozenset:
    pv = port.nav
    available_cash = pv * (1.0 - ts.cash_reserve_pct)        # shared, drained below
    gross_cap = pv * ts.gross_target * ts.regime_exposure_multiplier.get(f.regime_label, 1.0)
    pos_cap_dollars = pv * ts.max_position_weight
    cluster_cap_dollars = pv * ts.max_cluster_weight
    min_order = 250.0
    # rank by the incumbent's score (here mu_M1), same descending walk
    ranked = sorted(f.mu_M1, key=lambda s: (-f.mu_M1[s], s))
    cluster_dollars: Dict[str, float] = {}
    deployed = 0.0
    held: List[str] = []
    for sym in ranked:
        price = port.marks.get(sym, 0.0)
        # incumbent per-name target sizing (weight * pv), the cash-competing amount
        dollars = pos_cap_dollars
        # clamp by gross-exposure ceiling (decision_engine.py:1303-1307)
        gross_head = gross_cap - deployed
        if gross_head <= 0:
            continue
        dollars = min(dollars, gross_head)
        # clamp by correlated-cluster cap (decision_engine.py:1309-1314)
        cl = port.cluster_of.get(sym, sym)
        cluster_head = cluster_cap_dollars - cluster_dollars.get(cl, 0.0)
        if cluster_head <= 0:
            continue
        dollars = min(dollars, cluster_head)
        # drop sub-min / unaffordable (decision_engine.py:1316-1325) -> truncation
        if dollars < min_order or dollars > available_cash:
            continue
        shares = int(dollars / price) if price > 0 else 0
        if shares <= 0:
            continue
        dollars = shares * price
        if dollars < min_order or dollars > available_cash:
            continue
        held.append(sym)
        available_cash -= dollars                  # <-- shared cash drained inside the walk
        deployed += dollars
        cluster_dollars[cl] = cluster_dollars.get(cl, 0.0) + dollars
    return frozenset(held)


# --------------------------------------------------------------------------- #
# THE invariant: native engine held_symbols constant across the whole grid.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", _FIXTURES, ids=[fx[0] for fx in _FIXTURES])
def test_selection_set_invariant_under_sizing(fixture):
    name, f, port, theta_sel = fixture
    sel = select(f, theta_sel)
    reference_held = None
    reference_lot = None
    for ts in _THETA_SIZE_GRID:
        alloc = allocate(sel, f, ts, port, theta_sel.lot_policy)
        if reference_held is None:
            reference_held = alloc.held_symbols
            reference_lot = alloc.lot_infeasible
            continue
        # (1) frozenset equality of held_symbols across every grid point
        assert alloc.held_symbols == reference_held, (
            f"[{name}] held_symbols changed under sizing: "
            f"{sorted(alloc.held_symbols)} != {sorted(reference_held)} at {ts}"
        )
        # (2/C5) lot-infeasibility set byte-identical across the grid
        assert alloc.lot_infeasible == reference_lot, (
            f"[{name}] lot-infeasible set changed under sizing (C5 violation): "
            f"{sorted(alloc.lot_infeasible)} != {sorted(reference_lot)} at {ts}"
        )
    # held == selected minus the (theta_size-independent) lot-infeasible set
    assert reference_held == sel.selected_set - reference_lot


def test_min_order_notional_is_theta_size_independent():
    """C5: Stage-1 lot eligibility uses a theta_size-INDEPENDENT notional.

    ``LotPolicy.reference_notional`` is a function of (w_target) only; it cannot
    read gross_target / NAV / kappa. Demonstrate the lot-feasibility decision is
    identical for the same name under wildly different theta_size.
    """
    lp = LotPolicy(min_order=250.0, reference_nav=10_000.0)
    # boundary weight: 0.025 * 10_000 = 250 -> feasible; just below -> infeasible
    assert lp.is_lot_feasible(0.0251) is True
    assert lp.is_lot_feasible(0.0249) is False
    # the decision is taken against reference_nav, never the live book / theta_size
    import inspect
    params = list(inspect.signature(lp.reference_notional).parameters)
    assert params == ["w_target"], "lot notional must depend only on w_target (theta_size-free)"


# --------------------------------------------------------------------------- #
# Negative control: the incumbent loop IS non-invariant. xfail = we EXPECT the   #
# 'invariance' assertion to fail (the bug is present in the incumbent). If the    #
# incumbent ever became invariant this would XPASS and flag the guard is stale.   #
# --------------------------------------------------------------------------- #
def _incumbent_held_varies(f, port) -> bool:
    seen = set()
    for ts in _THETA_SIZE_GRID:
        seen.add(_incumbent_truncating_walk(f, port, ts))
        if len(seen) > 1:
            return True
    return False


@pytest.mark.xfail(reason="incumbent ranked walk is NON-invariant by construction "
                          "(decision_engine.py:1313-1325 sign-flip); the assertion below "
                          "is EXPECTED to fail -- that is the pinned regression guard.",
                   strict=True)
def test_incumbent_negative_control_is_invariant():
    # A representative single-cluster, cash-competing book where bigger caps eat
    # cash faster and truncate a *different* set -- the historical mechanism.
    _, f, port, _ = _mk_fixture(
        "neg", 14,
        mu_fn=lambda i: 0.50 - 0.02 * i,
        health_fn=lambda i: 0.85,
        cluster_fn=lambda i: "equity",
    )
    # This asserts invariance and is EXPECTED TO FAIL (xfail strict) because the
    # incumbent held set DOES change with sizing.
    assert not _incumbent_held_varies(f, port), "incumbent should be non-invariant"


def test_incumbent_is_demonstrably_non_invariant():
    """Positive confirmation (not xfail): the incumbent walk changes the held set
    across the grid -- proving the negative control pins a real defect.
    """
    _, f, port, _ = _mk_fixture(
        "neg", 14,
        mu_fn=lambda i: 0.50 - 0.02 * i,
        health_fn=lambda i: 0.85,
        cluster_fn=lambda i: "equity",
    )
    assert _incumbent_held_varies(f, port), (
        "the incumbent truncating walk must be non-invariant for the negative "
        "control to be meaningful"
    )
