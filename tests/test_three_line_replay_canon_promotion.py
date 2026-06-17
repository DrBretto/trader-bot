"""Regression tests for the PKT-TB-012 New-Brain canon (freeze + re-anchor).

This file previously pinned the optimized-champion canon promotion. PKT-TB-012
RETIRES that canon: per D-AUTO-20260616 and the DESIGN_DOSSIER §3 Attack-5 ruling
the in-sample optimized champion is no longer the live algorithm and no longer
the displayed canon line. The champion line through 2026-06-11 is frozen
byte-immutable (config/champion_freeze_20260611.json, never recomputed) and the
New Brain (native two-stage engine) is the primary line forward, re-anchored
C0-continuous to the frozen terminal.

These tests pin the new contract the extender must keep stable:

1. `equity_curve[i].value` == the frozen champion table on every row <= the
   boundary (byte-immutable; never recomputed).
2. `equity_curve[i].value` forward of the boundary is the New Brain line,
   re-anchored to the frozen terminal (`new_brain_value` populated).
3. `metrics.canon_source` == 'new_brain'; the line is no longer branded champion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.three_line_replay import extender
from src.utils.canonical_replay_anchor import (
    champion_freeze_map,
    NEW_BRAIN_BOUNDARY_DATE,
)


def _seed_dash() -> Dict[str, Any]:
    """A dashboard whose <= boundary rows use real frozen-table dates plus two
    forward realized-book continuity rows."""
    fmap, _tdate, tval = champion_freeze_map()
    # Pick a few real frozen dates (sorted) ending at the boundary terminal.
    frozen_dates = sorted(d for d in fmap if d <= NEW_BRAIN_BOUNDARY_DATE)
    sample = frozen_dates[-3:]  # ..., boundary
    rows: List[Dict[str, Any]] = []
    for d in sample:
        # Earlier frozen rows are seeded with an arbitrary pre-freeze value to
        # prove the extender OVERWRITES them with the frozen table (byte-
        # immutability under recompute). The boundary row carries the realized
        # book value (~terminal) — the base the forward re-anchor chains from.
        seed_v = tval if d == NEW_BRAIN_BOUNDARY_DATE else 1.0
        rows.append({"date": d, "value": seed_v, "benchmark": 95000.0,
                     "cumulative_external_cashflow": 0.0})
    # forward realized-book continuity rows (New Brain era)
    rows.append({"date": "2026-06-12", "value": tval * 1.01,
                 "benchmark": 95000.0, "cumulative_external_cashflow": 0.0})
    rows.append({"date": "2026-06-13", "value": tval * 1.01 * 0.99,
                 "benchmark": 95000.0, "cumulative_external_cashflow": 0.0})
    return {
        "snapshot": {"id": "stub", "date": "2026-06-13", "phase": "morning", "timestamp": "stub"},
        "metrics": {"total_value": 1.0, "timestamp": "stub"},
        "equity_curve": rows,
        "drawdowns": [], "monthly_returns": [], "trades": [],
        "trade_summary": {}, "holdings": [], "round_trips": [],
    }


class TestNewBrainCanon:
    def _run(self):
        # engine actually drove the forward dates -> New Brain brand attaches.
        return extender.extend_dashboard(
            s3_client=None, dash=_seed_dash(),
            engine_driven_dates={"2026-06-12", "2026-06-13"})

    def test_frozen_champion_is_byte_immutable(self):
        """value <= boundary equals the frozen static table, never recomputed."""
        fmap, _td, _tv = champion_freeze_map()
        out = self._run()
        for row in out["equity_curve"]:
            d = row["date"]
            if d <= NEW_BRAIN_BOUNDARY_DATE and d in fmap:
                assert row["value"] == fmap[d]
                assert row["champion_frozen_value"] == fmap[d]
                assert row["new_brain_value"] is None

    def test_new_brain_line_reanchored_forward(self):
        """Forward of the boundary the New Brain line is re-anchored C0-continuous
        to the frozen terminal."""
        fmap, _td, tval = champion_freeze_map()
        out = self._run()
        rows = {r["date"]: r for r in out["equity_curve"]}
        # 06-12 realized return = (tval*1.01)/tval - 1 = 0.01 -> anchored to tval.
        assert abs(rows["2026-06-12"]["new_brain_value"] - tval * 1.01) < 0.01
        assert rows["2026-06-12"]["value"] == rows["2026-06-12"]["new_brain_value"]
        assert rows["2026-06-12"]["champion_frozen_value"] is None

    def test_canon_source_is_new_brain(self):
        out = self._run()
        assert out["metrics"]["canon_source"] == "new_brain"
        assert out["timeline_correction"]["canon_source"] == "new_brain"

    def test_brand_and_forward_confirmed(self):
        out = self._run()
        tc = out["timeline_correction"]
        assert tc["brand"] == "New Brain"           # forward line present -> authorized
        assert tc["forward_confirmed"] is False
        assert tc["champion_frozen"]["byte_immutable"] is True

    def test_optimized_value_field_still_emitted(self):
        """`optimized_value` stays populated (mirrors `value`) for the desktop
        chart's `optimizedValue ?? correctedValue` fallback."""
        out = self._run()
        for row in out["equity_curve"]:
            assert row.get("optimized_value") == row["value"]
