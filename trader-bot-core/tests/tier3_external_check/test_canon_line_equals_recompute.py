"""Tier 3 external-cross-check — canon line == Σ(shares × settled close).

The canon LINE is never a stored dollar level trusted on read; it is
``Book.value = cash + Σ(shares × settled close)`` (replay.driver / P4-P6). This
test RE-EXECUTES that marking machinery independently over the real settled
window (fresh in-memory ledger, live substrate) and asserts the recompute
reproduces the STORED clean canon terminal within 0.5% — proving the stored line
traces to real holdings marked at real settled closes, not a planted or stale
value. The settled closes it marks off are tied to Yahoo reality by the second
assertion (and universe-wide by ``price_equals_external_feed``).

Respects the CLAUDE.md banner: this NEVER compares the line's dollar LEVEL to the
~$97k sim book — it recomputes the line from its own holdings×settled machinery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import (FeedUnreachable, reconstruct_canon_terminal,  # noqa: E402
                      yahoo_close_on)

TOL = 0.005  # 0.5%


@pytest.mark.external_check
def test_canon_line_equals_recompute(reality):
    stored = reality.canon_terminal()                       # live clean-ledger leaf
    recomp = reconstruct_canon_terminal()                   # independent re-mark

    assert recomp["date"] == stored["date"], (
        f"recompute terminal date {recomp['date']} != stored canon terminal "
        f"{stored['date']} — the clean ledger frontier is not the recomputed frontier")

    rel = abs(recomp["recomputed_value"] - stored["value"]) / stored["value"]
    assert rel < TOL, (
        f"canon line {stored['value']:.2f} diverges {rel:.4%} from the "
        f"holdings×settled recompute {recomp['recomputed_value']:.2f} on "
        f"{stored['date']} — the stored line does not trace to Σ shares·settled")

    # Tie the marks to Yahoo reality: SPY's settled close used in the marking
    # window must equal Yahoo's settled close for the terminal date (<0.5%).
    try:
        y = yahoo_close_on("SPY", stored["date"])
    except FeedUnreachable as e:
        pytest.fail(f"Yahoo unreachable — cannot tie the recompute marks to reality: {e}")
    assert y is not None, f"Yahoo carried no SPY bar for {stored['date']}"
    df = reality.read_prices(reality.latest_settled_date())
    import pandas as pd
    spy = df[(df["symbol"] == "SPY")].assign(_d=pd.to_datetime(df[df["symbol"] == "SPY"]["date"]).dt.strftime("%Y-%m-%d"))
    store_close = float(spy[spy["_d"] == stored["date"]]["close"].iloc[0])
    rel_spy = abs(store_close - y) / y
    assert rel_spy < TOL, (
        f"settled SPY close feeding the recompute (store={store_close:.4f}) diverges "
        f"{rel_spy:.4%} from Yahoo ({y:.4f}) — marks are not real settled prices")
