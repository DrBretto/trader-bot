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
  - ``health`` defaults to 1.0 (pass) for symbols lacking a health signal, so the
    h_min gate excludes only names with an explicit sub-threshold health.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

from .engine import ForecastBundle, PortfolioState


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
) -> ForecastBundle:
    """Build ``f`` for one decision date. ``mu_map`` is the frozen brain's
    full-universe M1 forecast (the PRIMARY ordering key)."""
    health_map = dict(health_map or {})
    elig = {str(s): bool(int(e)) for s, e in
            zip(universe_df["symbol"], universe_df["eligible"])}
    acls = {str(s): str(a) for s, a in
            zip(universe_df["symbol"], universe_df["asset_class"])}
    feats = _features_by_symbol(features_df)

    mu_M1: Dict[str, float] = {}
    health: Dict[str, float] = {}
    event_block: Dict[str, bool] = {}
    idio_vol: Dict[str, float] = {}
    eligible: Dict[str, bool] = {}
    asset_class: Dict[str, str] = {}
    vol_bucket: Dict[str, str] = {}

    for sym, mu in mu_map.items():
        sym = str(sym)
        mu_M1[sym] = float(mu)
        health[sym] = float(health_map.get(sym, 1.0))
        event_block[sym] = False  # no frozen M4 hard-exclusion threshold (see module docstring)
        f = feats.get(sym, {})
        v21 = f.get("vol_21d")
        idio_vol[sym] = float(v21) if v21 is not None and v21 == v21 else 0.0
        eligible[sym] = elig.get(sym, False)
        asset_class[sym] = acls.get(sym, "equity")
        vol_bucket[sym] = _vol_bucket(idio_vol[sym])

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
    )


def build_portfolio_state(
    portfolio_state: dict,
    features_df,
    universe_df,
) -> PortfolioState:
    """Build the Stage-2 ``PortfolioState`` from the chassis portfolio + marks."""
    feats = _features_by_symbol(features_df)
    marks: Dict[str, float] = {}
    for sym, f in feats.items():
        c = f.get("close")
        if c is not None and c == c:
            marks[sym] = float(c)

    positions: Dict[str, int] = {}
    for h in portfolio_state.get("holdings", []) or []:
        sym = str(h.get("symbol", ""))
        if not sym:
            continue
        positions[sym] = int(float(h.get("shares", 0) or 0))
        cp = h.get("current_price") or h.get("close_price") or h.get("entry_price")
        if sym not in marks and cp:
            marks[sym] = float(cp)

    cluster_of = {str(s): str(sec) for s, sec in
                  zip(universe_df["symbol"], universe_df["sector"])}

    cash = float(portfolio_state.get("cash", 0.0) or 0.0)
    nav = cash + sum(positions.get(s, 0) * marks.get(s, 0.0) for s in positions)
    if nav <= 0:
        # Fall back to the published portfolio_value / a frozen reference so the
        # allocator never divides by a zero book.
        nav = float(portfolio_state.get("portfolio_value", 0.0) or 0.0) or 100_000.0

    return PortfolioState(
        nav=nav,
        cash=cash,
        positions=positions,
        marks=marks,
        cluster_of=cluster_of,
    )
