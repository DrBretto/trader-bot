"""publish/challenger.py — the ONE challenger mechanism (exactly one, by invariant).

Extracted VERBATIM (logic unchanged) from ``replay/driver.py::_shadow_intents`` so
the M1-conviction tilt has a single canonical home. ``replay/driver.py`` imports
``shadow_tilt_intents`` from here; there is no second copy.

CHALLENGER (per P4 / the clean-rebuild architecture): the shadow line is the
M1-conviction tilt of the SAME ``run_engine`` selection — same mu (0.0-diff), same
selected grid, **tilt only** (Stage-2 weights re-shaped by mu conviction, membership
untouched). It is NOT a different engine.

ONE MECHANISM ONLY: ``src/brain/tilt_live.py`` is an UNCONFIRMED second challenger
mechanism and is deliberately NOT carried into ``trader-bot-core/`` (it stays a DROP
candidate outside the clean core). The clean core holds exactly this one tilt.
"""
from __future__ import annotations

from typing import List

import numpy as np


def challenger_target_weights(engine_out, f, canon_gross: float,
                              tilt_gain: float = 0.5) -> dict:
    """The ONE challenger as TARGET WEIGHTS: the M1-conviction tilt of the SAME
    selection, at the SAME gross exposure the canon achieved.

    Same held names (membership untouched), same total invested fraction
    (``canon_gross``) — the ONLY difference is a Stage-2 re-shape of the per-name
    weights by ``exp(gain * z(mu))``. Returned as ``{sym: target_weight}`` where the
    weights sum to ``canon_gross`` (the rest is cash), so the caller rebalances the
    challenger book to these targets value-conservingly. This replaces the earlier
    BUY-only intent generator, which never sold trimmed names and drove the shadow
    book's cash arbitrarily negative over a multi-day replay (the challenger bug)."""
    sel = engine_out.selection
    alloc = engine_out.allocation
    held = [s for s in sel.ordered if s in alloc.held_symbols]
    if not held:
        return {}
    mus = np.array([float(f.mu_M1.get(s, 0.0)) for s in held])
    z = (mus - mus.mean()) / (mus.std() + 1e-12) if len(mus) > 1 else np.zeros_like(mus)
    tilt = np.exp(tilt_gain * z)
    base = np.array([float(sel.w_target.get(s, 0.0)) for s in held])
    w = base * tilt
    w = w / w.sum() if w.sum() > 0 else (base / base.sum() if base.sum() > 0 else base)
    return {s: float(wi * canon_gross) for s, wi in zip(held, w)}


def shadow_tilt_intents(engine_out, f, tilt_gain: float = 0.5) -> List[dict]:
    """The ONE challenger mechanism: an M1-conviction tilt of the SAME selection.

    Same mu (0.0-diff), same selected grid (membership = engine_out.selection),
    **tilt only**: re-weight the held names by exp(gain * z(mu)) and re-derive
    integer-lot BUY/SELL/HOLD/REDUCE actions against the same book marks. Nothing
    about selection/eligibility is touched — this is a Stage-2 re-shape, not a
    different engine."""
    sel = engine_out.selection
    alloc = engine_out.allocation
    held = [s for s in sel.ordered if s in alloc.held_symbols]
    if not held:
        return []
    mus = np.array([float(f.mu_M1.get(s, 0.0)) for s in held])
    z = (mus - mus.mean()) / (mus.std() + 1e-12) if len(mus) > 1 else np.zeros_like(mus)
    tilt = np.exp(tilt_gain * z)
    base = np.array([float(sel.w_target.get(s, 0.0)) for s in held])
    w = base * tilt
    w = w / w.sum() if w.sum() > 0 else base
    # size against the same realized book the engine sized against
    price = {it["symbol"]: float(it["price"])
             for it in alloc.intents if it.get("price")}
    # gross budget = sum of engine's target dollars over held (same book posture)
    gross = sum(abs(float(it.get("dollars", 0.0))) for it in alloc.intents
                if str(it.get("action")).upper() in ("BUY", "HOLD"))
    intents: List[dict] = []
    for wi, s in zip(w, held):
        p = price.get(s, 0.0)
        if p <= 0:
            continue
        target_shares = int((gross * wi) / p)
        intents.append({"action": "BUY", "symbol": s, "shares": target_shares,
                        "price": p, "dollars": round(target_shares * p, 2),
                        "reason": f"SHADOW_M1_TILT_{tilt_gain}"})
    return intents
