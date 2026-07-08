"""No-churn exposure hysteresis — the clean-engine port of the production
``src/steps/decision_engine.py:575-702`` ``compute_exposure_trims`` (PKT-TB-004).

PKT-TRADER-BOT-SEED-CANON-BY-REPLAY §Pinned-3 pins THIS hysteresis: "trims TO
target + ``hysteresis_gap``; a re-fire requires the gap to re-open", ported into
the clean engine "so the reconstructed book does not churn ~50%/day — it holds
like the real system."

The production function emits pro-rata REDUCE *intents* against a live book. The
clean marking machinery (``replay/driver.py``) is WEIGHTS-based (§Pinned-1: one
valuation machinery marks every line by target weights), so this port exposes the
IDENTICAL mechanism as the *effective book gross* the canon weights are scaled to,
rather than as trim intents. The behaviour is byte-faithful to the original:

  * SELL-SIDE ONLY. Like the original, it only ever trims an OVER-exposed book
    DOWN; it never chases an under-exposed book UP toward a risen target. So when
    the fresh engine gross target RISES (e.g. a risk-on regime), the book HOLDS
    its current gross (no buy-up churn); when the target FALLS, the book only
    trims once the breach clears ``trigger_gap``, and then only down to
    ``target + hysteresis_gap`` (never below target).
  * NO-CHURN. ``over = current > target + trigger_gap``; a consecutive-breach
    counter is carried on the book state (``exposure_trim_consecutive_over``,
    the same key + ``persistence_days`` gate as the original); a trim re-arms the
    counter to 0, so a re-fire requires the gap to re-open by at least
    ``trigger_gap - hysteresis_gap`` (the original's exact no-churn property).
  * ``skip_regimes`` / ``target_floor`` behave as in the original.

This is the ONLY hold in the reconstruction: membership still rotates with the
two-stage selection (that is the model's own regime, not churn to be damped), and
this hysteresis bounds the gross-exposure swing that drove the broken attempt's
~50%/day (and up to 134%/day) turnover.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if f == f else default          # reject NaN
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# Default no-churn config — the production defaults from compute_exposure_trims
# (trigger_gap 0.10, hysteresis_gap 0.05, persistence_days 1). Enabled here BY
# DESIGN: the reconstruction ports the hysteresis to HOLD the book (§Pinned-3),
# whereas production ships it gated-off (absent config = production unchanged).
DEFAULT_HYSTERESIS_CFG: Dict[str, Any] = {
    "enabled": True,
    "trigger_gap": 0.10,
    "hysteresis_gap": 0.05,
    "persistence_days": 1,
    "target_floor": 0.0,
    "skip_regimes": [],
}


def held_gross(
    current_gross_frac: float,
    target_gross_frac: float,
    state: Dict[str, Any],
    regime_label: str,
    cfg: Optional[Mapping[str, Any]] = None,
) -> float:
    """Return the EFFECTIVE gross fraction the book should carry this day — the
    no-churn hysteresis hold of ``current_gross_frac`` against the fresh
    ``target_gross_frac``.

    Byte-faithful to ``compute_exposure_trims`` (decision_engine.py:610-702):

      * disabled / skip-regime  -> pass the fresh target through (no hold).
      * ``over`` breach + persistence -> trim TO ``target + hysteresis_gap`` and
        re-arm the counter (a re-fire needs the gap to re-open).
      * otherwise               -> HOLD ``current_gross_frac`` (never chase up).

    ``state`` is mutated in place (the consecutive-over counter), exactly as the
    original mutates ``portfolio_state``.
    """
    cfg = cfg or DEFAULT_HYSTERESIS_CFG
    if not cfg.get("enabled"):
        return target_gross_frac
    if regime_label in set(cfg.get("skip_regimes", []) or []):
        return target_gross_frac

    trigger_gap = _safe_float(cfg.get("trigger_gap"), 0.10)
    hysteresis_gap = min(_safe_float(cfg.get("hysteresis_gap"), 0.05), trigger_gap)
    persistence_days = max(int(cfg.get("persistence_days", 1) or 1), 1)
    target_floor = _safe_float(cfg.get("target_floor"), 0.0)
    target = _clip(_safe_float(target_gross_frac, 1.0), target_floor, 1.0)
    current = _safe_float(current_gross_frac, 0.0)

    counter_key = "exposure_trim_consecutive_over"
    over = current > target + trigger_gap
    counter = int(state.get(counter_key, 0) or 0)
    counter = counter + 1 if over else 0
    state[counter_key] = counter

    if not over or counter < persistence_days:
        # HOLD: the book keeps its current gross (the sell-side hysteresis never
        # forces a buy-up toward a risen target). On the FIRST marked day the book
        # has no prior gross to hold (current == 0), so fall through to the fresh
        # target — there is nothing to churn yet.
        return current if current > 0 else target

    # Over-exposed past the trigger for persistence_days -> trim TO target +
    # hysteresis_gap (never below target). Re-arm so a re-fire needs a fresh breach.
    state[counter_key] = 0
    return target + hysteresis_gap
