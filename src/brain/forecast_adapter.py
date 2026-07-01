"""Adapt the chassis night-pipeline artifacts into the engine's frozen contracts.

The native two-stage engine (PKT-TB-008) consumes a ``ForecastBundle`` (f) and a
``PortfolioState``. This module builds them from what the night Lambda already
has on hand:

  - ``mu_map``        : full-universe M1 ``mu`` from the frozen brain's forward
                        inference (the forecast-ledger record; the PRIMARY
                        ordering key). NOT a tilt on the incumbent ranker.
  - ``features_df``   : per-symbol close / vols (idio_vol tiebreak, marks).
  - ``regime_label``  : the chassis fused regime (enters Stage-2 only; θ_size's
                        regime_exposure_multiplier is frozen {} so it is 1.0).
  - ``universe_df``   : config/universe.csv — eligibility + asset_class (the
                        sizing-independent universe/tradability gate).
  - ``health_map``    : per-symbol health_score in [0,1] from the chassis
                        decisions / holdings; the Stage-1 binary eligibility gate.

Frozen-contract notes (no parameter invented outside FREEZE_ORB1):
  - ``event_block`` is set all-False in the go-live adapter: θ_sel froze NO M4
    hard-exclusion threshold, so M4 enters as the attribution ladder's event-DAMP
    rung (E), not as a Stage-1 hard cut. Setting a threshold here would be an
    un-frozen θ (a LIVE_PREREG §3 no-mid-stream violation). Recorded, not invented.
  - ``health`` for a symbol with NO reported health signal is UNKNOWN, not
    healthy: it is set to the explicit, named ``UNKNOWN_HEALTH_DEFAULT`` (a
    documented benefit-of-doubt pass), applied IDENTICALLY to fresh candidates and
    held positions and counted, so the h_min gate excludes only names with an
    explicit sub-threshold health while the incumbency effect stays auditable
    (PKT-3 / ISSUE-11 — no more silent ``.get(sym, 1.0)``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from .engine import ForecastBundle, PortfolioState

# Built-in correlation grouping: the granular universe.csv `sector` taxonomy
# splits one correlated risk factor into separate buckets (industry_semis vs
# sector_tech vs theme_innovation are three sectors; gold vs silver are two), so
# the Stage-2 `max_cluster_weight` cap never binds on the real complex. These
# groups collapse the ~0.9-correlated sleeves so the cap binds. Overridable via
# config/correlation_groups.json. Sectors not listed map to themselves.
_DEFAULT_CORRELATION_GROUPS: Dict[str, tuple] = {
    "tech_growth": ("industry_semis", "sector_tech", "theme_innovation"),
    "precious_metals": ("gold", "silver"),
}

# ---------------------------------------------------------------- health=None rule
# EXPLICIT rule (PKT-3 / ISSUE-11), replacing the silent ``health_map.get(sym, 1.0)``.
# A symbol with NO reported health_score (None / absent from health_map) is in a
# distinct UNKNOWN state — it is NOT asserted healthy. It is assigned this named
# default, a deliberate benefit-of-doubt PASS, for one documented reason: on a live
# book, CUTTING a name because its health is *absent* (not measured low) would churn
# the portfolio on missing data — strictly worse than holding through the gap. The
# fix here is not to change that pass, but to make it (a) explicit + named instead of
# a magic literal, (b) COUNTED and logged so the effect is observable, and (c) applied
# IDENTICALLY to fresh candidates and held positions so the incumbency asymmetry the
# audit flagged (incumbents tend to omit health, candidates tend to carry real scores)
# is auditable rather than hidden. A symbol with an explicit sub-threshold health is
# still cut by the h_min gate — only genuinely UNKNOWN health gets the pass.
UNKNOWN_HEALTH_DEFAULT = 1.0


def _config_root() -> Path:
    """Repo root (or ${LAMBDA_TASK_ROOT} in the image) holding config/."""
    task_root = os.environ.get("LAMBDA_TASK_ROOT")
    if task_root and (Path(task_root) / "config").exists():
        return Path(task_root)
    return Path(__file__).resolve().parents[2]


def _load_correlation_groups() -> Dict[str, str]:
    """Return a flat ``sector -> group`` map. Reads config/correlation_groups.json
    ({group: [sectors]}) when present; otherwise the built-in default. Fail-soft:
    a missing/garbled file falls back to the default, never raises."""
    groups = _DEFAULT_CORRELATION_GROUPS
    try:
        p = _config_root() / "config" / "correlation_groups.json"
        if p.exists():
            loaded = json.loads(p.read_text())
            if isinstance(loaded, dict) and loaded:
                groups = {g: tuple(members) for g, members in loaded.items()}
    except Exception:  # noqa: BLE001 — never let grouping config crash the night
        groups = _DEFAULT_CORRELATION_GROUPS
    sector_to_group: Dict[str, str] = {}
    for group, members in groups.items():
        for sector in members:
            sector_to_group[str(sector)] = str(group)
    return sector_to_group


def _regime_mult_for(
    regime_compat: Optional[Mapping[str, Mapping[str, float]]],
    regime_label: str,
    sector: str,
    asset_class: str,
) -> float:
    """The chassis regime x sector compatibility multiplier — the faithful port of
    the original ``decision_engine.score_candidates.get_multiplier`` lookup:
    try the fine sector key, fall back to asset_class, then 1.0."""
    compat = (regime_compat or {}).get(regime_label, {})
    if not compat:
        return 1.0
    return float(compat.get(sector, compat.get(asset_class, 1.0)))


def _vol_bucket(v: Optional[float]) -> str:
    if v is None:
        return "med"
    if v < 0.12:
        return "low"
    if v > 0.30:
        return "high"
    return "med"


def _features_by_symbol(features_df) -> Dict[str, dict]:
    """Latest row per symbol as a plain dict (handles multi-date frames)."""
    if features_df is None or len(features_df) == 0:
        return {}
    df = features_df
    if "date" in df.columns:
        df = df.sort_values("date").groupby("symbol", as_index=False).tail(1)
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        out[str(row["symbol"])] = {k: row.get(k) for k in df.columns}
    return out


def build_forecast_bundle(
    date: str,
    mu_map: Mapping[str, float],
    features_df,
    regime_label: str,
    universe_df,
    health_map: Optional[Mapping[str, float]] = None,
    regime_compat: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> ForecastBundle:
    """Build ``f`` for one decision date. ``mu_map`` is the frozen brain's
    full-universe M1 forecast (the PRIMARY ordering key).

    ``regime_compat`` is the chassis ``regime_compatibility`` table (per
    regime, per sector/asset_class multiplier). When provided, each symbol gets a
    ``regime_score_mult`` so Stage-1 ranks on the regime-tilted forecast (the
    restored chassis socket). When ``None``/empty the mult is 1.0 for all symbols
    -> identical to the pre-restore raw-mu ranking.
    """
    health_map = dict(health_map or {})
    n_unknown_health = 0  # symbols with NO reported health -> UNKNOWN_HEALTH_DEFAULT
    elig = {str(s): bool(int(e)) for s, e in
            zip(universe_df["symbol"], universe_df["eligible"])}
    acls = {str(s): str(a) for s, a in
            zip(universe_df["symbol"], universe_df["asset_class"])}
    secs = {str(s): str(sec) for s, sec in
            zip(universe_df["symbol"], universe_df["sector"])} \
        if "sector" in getattr(universe_df, "columns", []) else {}
    feats = _features_by_symbol(features_df)

    mu_M1: Dict[str, float] = {}
    health: Dict[str, float] = {}
    event_block: Dict[str, bool] = {}
    idio_vol: Dict[str, float] = {}
    eligible: Dict[str, bool] = {}
    asset_class: Dict[str, str] = {}
    vol_bucket: Dict[str, str] = {}
    regime_score_mult: Dict[str, float] = {}

    for sym, mu in mu_map.items():
        sym = str(sym)
        try:
            muf = float(mu)
        except (TypeError, ValueError):
            continue
        if muf != muf:                          # drop NaN mu (never select on NaN)
            continue
        mu_M1[sym] = muf
        # EXPLICIT health=None rule (see UNKNOWN_HEALTH_DEFAULT above): a reported
        # score is honored (a real sub-threshold health can cut the name); an
        # absent/None health is the UNKNOWN state -> the counted benefit-of-doubt
        # default, never a silent .get(...,1.0).
        reported = health_map.get(sym, None)
        if reported is None:
            health[sym] = UNKNOWN_HEALTH_DEFAULT
            n_unknown_health += 1
        else:
            health[sym] = float(reported)
        event_block[sym] = False  # no frozen M4 hard-exclusion threshold (see module docstring)
        f = feats.get(sym, {})
        v21 = f.get("vol_21d")
        idio_vol[sym] = float(v21) if v21 is not None and v21 == v21 else 0.0
        eligible[sym] = elig.get(sym, False)
        acl = acls.get(sym, "equity")
        asset_class[sym] = acl
        vol_bucket[sym] = _vol_bucket(idio_vol[sym])
        regime_score_mult[sym] = _regime_mult_for(
            regime_compat, str(regime_label or "neutral"), secs.get(sym, ""), acl)

    # Observability for the health=None rule: how many symbols fell to the
    # UNKNOWN_HEALTH_DEFAULT this build (the incumbency-effect signal made auditable).
    if n_unknown_health:
        print(f"  [HEALTH] {n_unknown_health}/{len(mu_M1)} symbols had UNKNOWN health "
              f"-> UNKNOWN_HEALTH_DEFAULT={UNKNOWN_HEALTH_DEFAULT} (benefit-of-doubt "
              "pass; explicit rule, PKT-3/ISSUE-11); "
              f"{len(mu_M1) - n_unknown_health} carried a reported health_score.")

    return ForecastBundle(
        date=date,
        mu_M1=mu_M1,
        health=health,
        regime_label=str(regime_label or "neutral"),
        event_block=event_block,
        idio_vol=idio_vol,
        eligible=eligible,
        asset_class=asset_class,
        vol_bucket=vol_bucket,
        regime_score_mult=regime_score_mult,
    )


def build_portfolio_state(
    portfolio_state: dict,
    features_df,
    universe_df,
) -> PortfolioState:
    """Build the Stage-2 ``PortfolioState`` from the chassis portfolio + marks."""
    def _finite(x):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return xf if xf == xf else None        # drop NaN/inf-as-NaN

    feats = _features_by_symbol(features_df)
    marks: Dict[str, float] = {}
    for sym, f in feats.items():
        c = _finite(f.get("close"))
        if c is not None and c > 0:
            marks[sym] = c

    positions: Dict[str, int] = {}
    for h in portfolio_state.get("holdings", []) or []:
        sym = str(h.get("symbol", ""))
        if not sym:
            continue
        sh = _finite(h.get("shares")) or 0.0
        positions[sym] = int(sh)
        if sym not in marks:
            # NaN current_price must NOT leak into the book (NaN is truthy) — it
            # would make NAV NaN and crash the frozen engine's int() lot-solve.
            cp = (_finite(h.get("current_price")) or _finite(h.get("close_price"))
                  or _finite(h.get("entry_price")))
            if cp is not None and cp > 0:
                marks[sym] = cp

    # Map the granular universe.csv `sector` to its correlation group so the
    # Stage-2 max_cluster_weight cap binds on the real complex (e.g. SMH/SOXX/XLK/
    # ARKK -> one `tech_growth` cluster instead of three). Ungrouped sectors map
    # to themselves (their own cluster), preserving prior behaviour for them.
    sector_to_group = _load_correlation_groups()
    cluster_of = {
        str(s): sector_to_group.get(str(sec), str(sec))
        for s, sec in zip(universe_df["symbol"], universe_df["sector"])
    }

    cash = _finite(portfolio_state.get("cash")) or 0.0
    nav = cash + sum(positions.get(s, 0) * marks.get(s, 0.0) for s in positions)
    if not (nav == nav) or nav <= 0:
        # Fall back to the published portfolio_value / a frozen reference so the
        # allocator never divides by a zero or NaN book.
        nav = (_finite(portfolio_state.get("portfolio_value")) or 0.0) or 100_000.0

    return PortfolioState(
        nav=nav,
        cash=cash,
        positions=positions,
        marks=marks,
        cluster_of=cluster_of,
    )
