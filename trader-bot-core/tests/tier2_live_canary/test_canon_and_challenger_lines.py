"""Tier 2 live-canary — promoted TILT canon is replay-derived (NOT the internal
sim book) and the dotted-yellow two-stage comparison is populated.

``canon_line_is_not_sim_book`` pins the #1 recurring failure on this project
(CLAUDE.md banner): confusing the displayed canon LINE with the ~$97k internal
sim book. It reads the live promoted-ledger terminal + the live
``portfolio_state.sim_book_value`` and asserts they are genuinely different AND
that the line is ledger-derived (holdings-marked), never the sim book.

``comparison_line_present_nonempty`` asserts the two-stage comparison artifact
exists, is nonempty, and its last point is the expected settled trading day.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FeedUnreachable  # noqa: E402


@pytest.mark.live_canary
def test_canon_line_is_not_sim_book(reality):
    term = reality.canon_terminal()                     # live promoted TILT leaf
    line_value = float(term["value"])

    # the line must be ledger-derived (holdings×settled), never the sim book.
    seg = term.get("segment")
    model = str(term.get("model_id", ""))
    assert seg == "new_brain", f"canon terminal segment is {seg!r}, not the ledger line"
    assert model.startswith("tilt-canon@replay@"), (
        f"canon terminal model_id {model!r} is not a promoted TILT replay leaf")

    # the ~$97k sim book, read from live prod portfolio_state — must be DIFFERENT.
    ps = reality.portfolio_state(term["date"])
    sim_book = ps.get("sim_book_value")
    assert sim_book is not None, "portfolio_state carried no sim_book_value to distinguish"
    assert abs(line_value - float(sim_book)) > 100.0, (
        f"canon line {line_value:.2f} is within $100 of the sim book "
        f"{float(sim_book):.2f} — the line is being confused with the ~$97k book "
        f"(the #1 recurring failure, CLAUDE.md banner)")


@pytest.mark.live_canary
def test_comparison_line_present_nonempty(reality):
    ts = reality.shadow_timeseries()
    series = ts.get("shadow_A")
    assert isinstance(series, list) and len(series) >= 1, (
        f"two-stage comparison (shadow_A) missing or empty: {type(series).__name__}")

    last = series[-1]
    assert isinstance(last, (list, tuple)) and len(last) >= 2, (
        f"malformed two-stage comparison point {last!r}")
    last_date, last_val = str(last[0]), float(last[1])
    assert last_val > 0, f"two-stage comparison last value non-positive: {last_val}"

    # its last point should be the expected settled trading day = the canon frontier.
    expected = reality.canon_terminal()["date"]
    assert last_date == expected, (
        f"two-stage comparison last point {last_date} != expected settled day {expected} "
        f"(comparison is stale / not advancing)")
