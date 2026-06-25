"""Canonical hybrid-replay anchor for the dashboard equity curve.

The 2026-05-06 known-bugs-fixed algorithm comparison v2 run produced a
source-faithful counterfactual line ("hybrid_current_fixed") for
2026-03-12 -> 2026-05-05. Operator promoted that line as the canonical
display: the dashboard chart and metrics block should reflect this
counterfactual, not the live broker line.

To make this stable across future Lambda runs:
1. _load_daily_states overrides per-day `value`, `cash`, `holdings_count`,
   and zeros `external_cashflow` for dates in HYBRID_SEGMENT.
2. On SEAM_CASHFLOW_DATE (the first post-segment date), an injected
   `external_cashflow` of SEAM_CASHFLOW_VALUE bridges broker raw to
   the hybrid value at the seam. Net cumulative_external_cashflow on
   the seam day equals SEAM_CASHFLOW_VALUE (the prior bridge entries
   on 2026-03-12 / 2026-04-22 are zeroed out by override #1).
3. Going forward, broker daily moves carry the canonical line via the
   existing continuity-cashflow math. No further per-day patching.

Run dir evidence: book-factory/use_lane_outputs/runs/
20260506_trader-bot-mar11-known-bugs-fixed-algorithm-comparison-v2/
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

SEAM_CASHFLOW_DATE = "2026-05-06"
SEAM_CASHFLOW_VALUE = -10131.0727

# ----------------------------------------------------------- PKT-TB-012 freeze
# The in-sample optimized champion is RETIRED as the live algorithm as of
# 2026-06-11 (D-AUTO-20260616 call #1). Its displayed line through 2026-06-11 is
# frozen as a byte-immutable static table and is NEVER recomputed/restarted/
# zeroed and NEVER spliced for computation of the forward line (D-AUTO call #2;
# DESIGN_DOSSIER §2 STAGE 4 / §4 Attack 1; LIVE_PREREG §1). The New Brain
# (native two-stage engine) is the primary displayed line forward of 2026-06-12,
# re-anchored C0-continuous to the 06-11 terminal (display continuity only — NOT
# a computational splice; the forward line is the realized book's own returns
# chained onto the frozen terminal).
NEW_BRAIN_BOUNDARY_DATE = "2026-06-11"          # champion owns <= this; New Brain owns >
CHAMPION_FREEZE_FILENAME = "champion_freeze_20260611.json"


def _champion_freeze_path() -> Path:
    # config/champion_freeze_20260611.json, resolved for repo or Lambda image.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    cand = repo_root / "config" / CHAMPION_FREEZE_FILENAME
    return cand


def load_champion_freeze() -> Optional[Dict[str, Any]]:
    """Load the frozen champion static table. Returns None if absent (the
    extender then falls back to its prior behavior rather than crashing)."""
    p = _champion_freeze_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def champion_freeze_map() -> Tuple[Dict[str, float], Optional[str], Optional[float]]:
    """(date->value map, terminal_date, terminal_value) of the frozen champion.

    The terminal value (~$114.9k at 2026-06-11) is the C0 anchor the New Brain
    forward line continues from."""
    fz = load_champion_freeze()
    if not fz:
        return {}, None, None
    curve: List[Dict[str, Any]] = fz.get("curve", [])
    m = {row["date"]: float(row["value"]) for row in curve}
    return m, fz.get("terminal_date"), (
        float(fz["terminal_value"]) if fz.get("terminal_value") is not None else None
    )


# NOTE (clean core, 2026-06-25): the band-aid forward New-Brain freeze loader
# (PKT-TRADER-BOT-REGIME-PICKER-NIGHTLY-RECALC-FIX-V1) is REMOVED. It wrapped the
# very recompute the clean-core dossier deletes; the displayed forward line is now
# the STORED equity ledger (src/canon/equity_ledger.py), read by
# src/canon/equity_line.py — not a frozen table read by a recompute. The champion
# freeze (<= 2026-06-11) above is preserved as gated infra (byte-immutable anchor).


# Per-day override values for 2026-03-12 -> 2026-05-05.
# value = hybrid_current_fixed.timeline[d].ending_value
# cash  = hybrid_current_fixed.timeline[d].ending_cash
# holdings_count = len(hybrid_current_fixed.timeline[d].holdings_at_close)
HYBRID_SEGMENT: Dict[str, Dict[str, Any]] = {
    "2026-03-12": {"value": 102422.19, "cash": 56214.39, "holdings_count": 8},
    "2026-03-13": {"value": 102540.86, "cash": 96769.42, "holdings_count": 2},
    "2026-03-17": {"value": 102508.68, "cash": 96769.42, "holdings_count": 2},
    "2026-03-18": {"value": 102316.02, "cash": 96769.42, "holdings_count": 2},
    "2026-03-19": {"value": 102023.92, "cash": 97759.82, "holdings_count": 1},
    "2026-03-20": {"value": 102008.41, "cash": 98746.71, "holdings_count": 3},
    "2026-03-24": {"value": 101974.48, "cash": 97124.92, "holdings_count": 1},
    "2026-03-25": {"value": 102207.32, "cash": 100286.08, "holdings_count": 2},
    "2026-03-26": {"value": 102243.76, "cash": 102243.76, "holdings_count": 0},
    "2026-03-27": {"value": 102323.12, "cash": 98701.44, "holdings_count": 4},
    "2026-04-01": {"value": 102407.58, "cash": 102407.58, "holdings_count": 0},
    "2026-04-02": {"value": 102397.22, "cash": 100226.8, "holdings_count": 1},
    "2026-04-07": {"value": 102672.0, "cash": 47761.19, "holdings_count": 8},
    "2026-04-08": {"value": 103864.86, "cash": 47761.19, "holdings_count": 8},
    "2026-04-09": {"value": 103972.44, "cash": 47761.19, "holdings_count": 8},
    "2026-04-10": {"value": 103708.14, "cash": 47761.19, "holdings_count": 8},
    "2026-04-14": {"value": 105560.31, "cash": 47761.19, "holdings_count": 8},
    "2026-04-15": {"value": 106216.87, "cash": 47761.19, "holdings_count": 8},
    "2026-04-16": {"value": 106388.53, "cash": 47761.19, "holdings_count": 8},
    "2026-04-17": {"value": 107333.74, "cash": 47761.19, "holdings_count": 8},
    "2026-04-21": {"value": 106926.66, "cash": 47761.19, "holdings_count": 8},
    "2026-04-22": {"value": 107477.89, "cash": 47761.19, "holdings_count": 8},
    "2026-04-23": {"value": 106565.91, "cash": 47761.19, "holdings_count": 8},
    "2026-04-24": {"value": 106873.45, "cash": 47761.19, "holdings_count": 8},
    "2026-04-28": {"value": 106308.71, "cash": 47761.19, "holdings_count": 8},
    "2026-04-29": {"value": 106091.84, "cash": 47761.19, "holdings_count": 8},
    "2026-04-30": {"value": 106802.69, "cash": 47761.19, "holdings_count": 8},
    "2026-05-01": {"value": 107004.95, "cash": 47761.19, "holdings_count": 8},
    "2026-05-05": {"value": 107235.83, "cash": 47761.19, "holdings_count": 8},
}
