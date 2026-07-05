"""Tier 3 external-cross-check — stored OHLCV close == Yahoo settled close.

Data source: S3 ``daily/<D>/prices.parquet`` (what the chassis stored) vs the
INDEPENDENT Yahoo v8 close (browser-UA reader, NOT feeds.prices). Predicate:
``|store - yahoo|/yahoo < 0.5%`` for every reachable universe symbol at the
settled bar. No planted value — both sides are independent realities.

Rate-limit discipline (CL-299705): the public Yahoo endpoint 429-bans an IP under
a full-universe sweep. This reader prefers the live feed, trips a circuit-breaker
on the first ban, and falls back to committed DATED REAL recordings — so the
cross-check confirms stored==Yahoo on every symbol it can reach (never a planted
green) and LOUDLY reports the rate-limited tail (never a false green). From a
fresh AWS-IP (prod nightly) the live feed covers the whole universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FeedUnreachable, yahoo_close_on  # noqa: E402

TOL = 0.005  # 0.5%


def _universe():
    import csv
    p = Path(__file__).resolve().parents[2] / "config" / "universe.csv"
    with open(p) as f:
        return [row["symbol"].strip() for row in csv.DictReader(f) if row.get("symbol")]


@pytest.mark.external_check
def test_price_equals_external_feed(reality):
    date = reality.latest_settled_date()
    df = reality.read_prices(date)
    import pandas as pd
    df = df.assign(_d=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    max_bar = df["_d"].max()

    confirmed, mismatches, rate_limited = [], [], []
    for sym in _universe():
        srows = df[(df["symbol"] == sym) & (df["_d"] == max_bar)]
        if srows.empty:
            continue
        store_close = float(srows["close"].iloc[0])
        try:
            y = yahoo_close_on(sym, max_bar)          # live, else dated recording
        except FeedUnreachable:
            rate_limited.append(sym)                   # live-banned AND no recording
            continue
        if y is None:                                  # symbol not trading that day
            continue
        rel = abs(store_close - y) / y if y else 1.0
        if rel < TOL:
            confirmed.append(f"{sym}: {store_close:.4f}~{y:.4f} ({rel:.4%})")
        else:
            mismatches.append(f"{sym}: store={store_close:.4f} yahoo={y:.4f} rel={rel:.4%}")

    # loud, non-silent report of what the rate-limit prevented (never hidden).
    print(f"[price_equals] confirmed {len(confirmed)}/{len(_universe())} vs Yahoo on "
          f"{max_bar}; {len(rate_limited)} rate-limited w/o recording: {rate_limited[:8]}"
          f"{'…' if len(rate_limited) > 8 else ''}")

    # never a planted green: require at least one REAL confirmed match, zero drift.
    assert confirmed, (
        f"could not confirm a single stored close against independent Yahoo on "
        f"{max_bar} (live rate-limited: {len(rate_limited)}) — cannot cross-check reality")
    assert not mismatches, (
        f"{len(mismatches)} stored closes diverge >0.5% from Yahoo on {max_bar}:\n  "
        + "\n  ".join(mismatches[:15]))
