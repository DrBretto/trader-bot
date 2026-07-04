"""Stage 1 -- SELECTION: size-independent set construction.

``select(f, theta_sel) -> SelectionResult``

The held symbol *set* is computed as a pure function of the forecast inputs and
the hand-frozen selection parameters. There is no parameter through which any
sizing or book-dollar value can enter, so the result is literally independent of
``theta_size`` (DESIGN_DOSSIER §2.1, layer 2). ``mu_M1`` is the PRIMARY ordering
key -- not an <=8% tilt on an incumbent ranker. Health is a binary eligibility
gate, regime an admissibility gate, M4 a hard exclusion. No allocation, no cash
counter, no truncation site exists in this pass.
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

from .contracts import (
    ForecastBundle,
    SelectionParams,
    SelectionResult,
    assert_no_dollar_surface,
)


def _regime_admits(theta_sel: SelectionParams, regime_label: str, asset_class: str) -> bool:
    """Regime is an admissibility GATE (set decision), never a per-name multiplier."""
    admissible = theta_sel.regime_admissibility.get(regime_label)
    if not admissible:           # regime absent from map -> admits all classes
        return True
    return asset_class in admissible


def _eligible(f: ForecastBundle, theta_sel: SelectionParams, sym: str) -> bool:
    """Stage-1 eligibility: every gate here is a SET decision, reads no dollar."""
    if not f.eligible.get(sym, False):                      # universe/tradability gate
        return False
    if f.health.get(sym, 0.0) < theta_sel.h_min:            # health binary eligibility gate
        return False
    if f.event_block.get(sym, False):                       # M4 hard exclusion
        return False
    if not _regime_admits(theta_sel, f.regime_label, f.asset_class.get(sym, "")):
        return False
    return True


def _regime_adjusted_mu(f: ForecastBundle, sym: str) -> float:
    """The regime-tilted forecast score: ``mu_M1 x regime_score_mult``.

    This restores the chassis regime picker into the New Brain's PRIMARY ordering
    key -- the faithful port of the original ``decision_engine.score_candidates``
    (``final_score = base_score x regime_multiplier``), with the orthogonal M1
    forecast as the base instead of health. In ``high_vol_panic`` the tech/equity
    names are knocked (x0.4..0.5) and fall out of the top-N while defensives rise;
    in ``risk_on_trend`` tech is lifted (x1.15). ``regime_score_mult`` defaults to
    1.0 per symbol (empty map) -> identical to the pre-restore raw-mu ranking.

    Selection holds the top-N by this score, which over the health-eligible set is
    the positive-mu region; the multiplier is strictly positive so it preserves the
    sign of mu and only re-weights its magnitude (a negative-mu name stays below the
    cut either way).
    """
    return float(f.mu_M1[sym]) * float(f.regime_score_mult.get(sym, 1.0))


def _ordering_key(f: ForecastBundle, sym: str):
    """Frozen ordering: regime-adjusted mu_M1 desc, then idio_vol asc, then symbol asc."""
    return (-_regime_adjusted_mu(f, sym), float(f.idio_vol.get(sym, 0.0)), sym)


def select(f: ForecastBundle, theta_sel: SelectionParams) -> SelectionResult:
    """Compute the held symbol set. Pure function of ``(f, theta_sel)``.

    Enforces 'Select reads no dollar value' at runtime via
    ``assert_no_dollar_surface`` over its only two inputs.
    """
    # The guarantee, enforced -- not just asserted in a docstring.
    assert_no_dollar_surface(f, theta_sel)

    # (1) eligibility filter -> E
    eligible: List[str] = [s for s in f.mu_M1 if _eligible(f, theta_sel, s)]

    # (2) sort E by the frozen ordering key (mu_M1 PRIMARY)
    eligible.sort(key=lambda s: _ordering_key(f, s))

    # (3) set cut S = top-N by rank
    selected: List[str] = eligible[: max(0, int(theta_sel.N))]

    # (4) tier assignment from the frozen Core fraction (rank-based, deterministic)
    core_count = min(len(selected), max(0, round(theta_sel.N * theta_sel.core_fraction)))
    tier = {
        sym: ("core" if i < core_count else "satellite")
        for i, sym in enumerate(selected)
    }

    # (5) emit dimensionless w_target: Core full, Satellite half, normalised over S
    raw = {sym: (1.0 if tier[sym] == "core" else 0.5) for sym in selected}
    total = sum(raw.values())
    w_target = {sym: (raw[sym] / total if total > 0 else 0.0) for sym in selected}

    forecast_meta = {
        "date": f.date,
        "regime_label": f.regime_label,
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "mu_M1": {sym: float(f.mu_M1[sym]) for sym in selected},
        "regime_score_mult": {sym: float(f.regime_score_mult.get(sym, 1.0)) for sym in selected},
        "regime_adjusted_mu": {sym: _regime_adjusted_mu(f, sym) for sym in selected},
        "core_count": core_count,
    }

    return SelectionResult(
        selected_set=frozenset(selected),
        ordered=tuple(selected),
        tier=tier,
        w_target=w_target,
        forecast_meta=forecast_meta,
        theta_sel_hash=theta_sel.content_hash(),
    )
