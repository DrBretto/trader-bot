"""Regression tests for the forecast substrate-currency freshness gate
(`_freshness_gate_verdict`), PKT-TRADER-BOT-TRUE-REPLAY-CANON-AND-SHADOW.

The fix under test: the gate compares the store's max bar against the max real bar
actually available in S3's settled `prices.parquet` files (`max_available_bar`), not
against folder labels. A `daily/<today>/` folder with no settled bar (or a file that
only carries bars through D-1) leaves `max_available_bar == store max` => NO false
abort before today's close; a genuine freeze — a real newer settled bar the store
failed to reach — still fails loud.
"""
from src.brain.runtime import _freshness_gate_verdict


def test_phantom_today_folder_is_not_stale():
    """`daily/2026-07-02/` folder exists (in `gap`) but has no settled bar, so the
    max real bar available equals the store max: a not-yet-closed day, NOT a freeze.
    This is the false-abort fixed."""
    fresh = {"gap": ["2026-07-02"], "gap_settled": [], "bars_added": 0,
             "max_available_bar": "2026-07-01"}
    v = _freshness_gate_verdict("2026-07-01", "2026-07-01", fresh)
    assert v["stale"] is False, v


def test_d_minus_1_file_is_not_stale():
    """A `daily/2026-07-02/prices.parquet` that only carries bars through 07-01
    (D-1 convention) leaves max_available_bar == store max => not stale even though a
    newer settled *folder* is present and 0 net bars were added."""
    fresh = {"gap": ["2026-07-02"], "gap_settled": ["2026-07-02"], "bars_added": 0,
             "max_available_bar": "2026-07-01"}
    v = _freshness_gate_verdict("2026-07-01", "2026-07-01", fresh)
    assert v["stale"] is False, v


def test_genuine_freeze_still_fails_loud():
    """A real settled bar (07-01) is available in S3 but the store only reached
    06-27 — the splice froze (ISSUE-01). Must trip."""
    fresh = {"gap": ["2026-06-30", "2026-07-01"],
             "gap_settled": ["2026-06-30", "2026-07-01"], "bars_added": 0,
             "max_available_bar": "2026-07-01"}
    v = _freshness_gate_verdict("2026-07-01", "2026-06-27", fresh)
    assert v["stale"] is True
    assert any("freeze" in r or "STALE" in r for r in v["reasons"]), v


def test_fresh_store_passes():
    """Settled bars available and the store reached the newest one."""
    fresh = {"gap": ["2026-07-01", "2026-07-02"],
             "gap_settled": ["2026-07-01"], "bars_added": 14,
             "max_available_bar": "2026-07-01"}
    v = _freshness_gate_verdict("2026-07-01", "2026-07-01", fresh)
    assert v["stale"] is False, v


def test_frozen_at_seed_trips_lag_invariant():
    """Store stuck at the 06-10 seed while the run marches on: invariant 1
    (max settled bar > K trading days behind the run date) fires."""
    fresh = {"gap": [], "gap_settled": [], "bars_added": 0,
             "max_available_bar": "2026-06-10"}
    v = _freshness_gate_verdict("2026-07-01", "2026-06-10", fresh)
    assert v["stale"] is True
    assert any("STALE" in r for r in v["reasons"]), v


def test_forced_stale_flag_trips():
    """diagnose_forecast_freshness(force_stale=True) must still fire the gate."""
    fresh = {"forced_stale": True, "gap": ["__forced__"],
             "gap_settled": ["__forced__"], "bars_added": 0}
    v = _freshness_gate_verdict("2026-07-01", "2026-07-01", fresh)
    assert v["stale"] is True


def test_empty_store_is_stale():
    fresh = {"gap": [], "gap_settled": [], "bars_added": 0, "max_available_bar": None}
    v = _freshness_gate_verdict("2026-07-01", None, fresh)
    assert v["stale"] is True


def test_backward_compat_missing_max_available_bar():
    """An older extend dict without max_available_bar falls back to the settled-gap
    heuristic (settled folder present, 0 bars) so a legacy freeze still trips."""
    fresh = {"gap": ["2026-07-01"], "gap_settled": ["2026-07-01"], "bars_added": 0}
    v = _freshness_gate_verdict("2026-07-01", "2026-06-27", fresh)
    assert v["stale"] is True
