"""feeds/produce.py — settled-bar PRODUCTION for the clean night path (CL-708001).

Why this exists
---------------
The clean night pipeline (``app.night.run_night`` → ``decide.cutover.run_cutover``
→ ``production_forecaster``) is a pure CONSUMER of settled bars:
``store.ohlcv_store._extend_ohlcv_from_s3`` downloads ``daily/<D>/prices.parquet``
from S3 and splices it into the OHLCV store, then the freshness gate checks
substrate currency. Nothing in the clean path ever PRODUCED those files — that was
the old chassis step (``src/steps/publish_artifacts.py`` writing ``prices.parquet``,
fed by ``src/steps/ingest_prices.py``). The P9 cutover (2026-07-05) pointed prod at
the clean thin router and left the settled-bar production step un-rebuilt — the
same class of gap the morning path had. After the night of 2026-07-03 (the last
``prices.parquet``, bars through 07-02) no new settled bar was written, so the store
froze at 07-02 and every scheduled night correctly ABORTED stale (the freshness gate
saw 07-02 fall four trading days behind the run date).

This module restores that production step ON the deployed night path. It fetches
settled OHLCV for the universe via ``feeds.prices`` (Yahoo v8 raw PRIMARY +
yfinance / Alpha Vantage / stooq fallbacks — the working P1 feeds) and writes
``daily/<D>/prices.parquet`` to S3 so the store can advance.

Abort-never-degrade, never a silent no-op writer
------------------------------------------------
It NEVER fabricates a bar. If no source produced a fresh settled bar for the run
date (a genuine feed outage), it writes NOTHING and returns ``produced=False`` with
the per-source ingest report — the freshness gate then aborts and the outage
surfaces honestly. The freshness/abort gate is untouched; this fixes the substrate
the gate reads.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# The anchor symbol the freshness gate keys on; its settled bar is the currency
# proof. If NO source can serve SPY, the substrate is genuinely down — surface it,
# never write a placeholder.
ANCHOR_SYMBOL = "SPY"

# Columns the OHLCV store's extend (forecast.inference.extend_ohlcv) consumes.
_PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def produce_settled_prices(
    settled: str,
    universe_df,
    s3_client,
    *,
    bucket: str,
    alphavantage_key: Optional[str] = None,
    lookback_days: int = 400,
    tolerance_td: int = 3,
) -> Dict[str, Any]:
    """Fetch settled OHLCV for the universe and write ``daily/<D>/prices.parquet``.

    ``settled`` is the run's latest settled trading day (the store must reach within
    ``tolerance_td`` of it). ``universe_df`` carries a ``symbol`` column.
    ``s3_client`` is a ``src.utils.s3_client.S3Client`` (has ``file_exists`` /
    ``read_parquet`` / ``write_parquet``).

    Returns a diagnostic dict; ``produced`` is True iff a real settled bar was
    written. Idempotent — a settled bar already present for ``settled`` is a no-op.
    """
    import pandas as pd
    from feeds import prices as FEEDS
    from store.ohlcv_store import _weekday_trading_days_between

    symbols = [str(s) for s in universe_df["symbol"].tolist()]
    out: Dict[str, Any] = {
        "settled": settled, "produced": False, "reason": "",
        "written_key": None, "feed_max_bar": None,
        "source_counts": {}, "failed_symbols": [], "fallback_events": [],
    }

    # ---- idempotency: a settled bar for `settled` already in S3? ----
    existing_key = f"daily/{settled}/prices.parquet"
    if s3_client.file_exists(existing_key):
        try:
            ex = s3_client.read_parquet(existing_key)
            ex_spy = ex[ex["symbol"] == ANCHOR_SYMBOL]
            ex_max = pd.to_datetime(ex_spy["date"]).max() if len(ex_spy) else None
            if ex_max is not None and pd.notna(ex_max) and ex_max.strftime("%Y-%m-%d") >= settled:
                out["produced"] = False
                out["already_present"] = True
                out["feed_max_bar"] = ex_max.strftime("%Y-%m-%d")
                out["reason"] = (f"daily/{settled}/prices.parquet already carries a settled "
                                 f"{ANCHOR_SYMBOL} bar {ex_max.strftime('%Y-%m-%d')} — no-op")
                return out
        except Exception as e:  # noqa: BLE001 — unreadable existing file: re-produce
            out["existing_unreadable"] = f"{type(e).__name__}: {e}"

    # ---- fetch settled OHLCV (Yahoo v8 PRIMARY + yfinance/AV/stooq fallbacks) ----
    df, report = FEEDS.run_with_report(
        symbols, alphavantage_key=alphavantage_key, lookback_days=lookback_days)
    out["source_counts"] = dict(report.source_counts)
    out["failed_symbols"] = list(report.failed_symbols)
    out["fallback_events"] = list(report.fallback_events)
    out["served_by_anchor"] = report.served_by.get(ANCHOR_SYMBOL)

    if df is None or len(df) == 0:
        out["reason"] = ("ALL feeds returned zero rows for the entire universe "
                         "(feed outage) — NOT writing a bar; freshness gate will abort")
        return out

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    spy = df[df["symbol"] == ANCHOR_SYMBOL].sort_values("date")
    if len(spy) == 0:
        out["reason"] = (f"{ANCHOR_SYMBOL} (anchor) not served by ANY source — feed down; "
                         "NOT writing a bar; freshness gate will abort")
        return out

    feed_max_ts = spy["date"].max()
    feed_max = feed_max_ts.strftime("%Y-%m-%d")
    out["feed_max_bar"] = feed_max

    # ---- reality gate: the freshest fetched settled bar must be within tolerance
    #      of the run date, else the feeds are genuinely stale — surface, don't
    #      write. (The producer never masks a real outage with an old bar.) ----
    lag = _weekday_trading_days_between(feed_max, settled)
    if lag > tolerance_td:
        out["reason"] = (f"freshest fetched {ANCHOR_SYMBOL} bar {feed_max} is {lag} trading "
                         f"days behind run date {settled} (tolerance {tolerance_td}) — feeds "
                         "stale; NOT writing a bar; freshness gate will abort")
        return out

    # ---- write daily/<feed_max>/prices.parquet (folder label == max bar date, the
    #      store convention). The full-lookback frame carries every bar from the
    #      store frontier forward, so one write advances the store across a
    #      multi-day gap in a single extend. ----
    write_key = f"daily/{feed_max}/prices.parquet"
    frame = (df[_PRICE_COLS].sort_values(["symbol", "date"]).reset_index(drop=True))
    ok = s3_client.write_parquet(frame, write_key)
    if not ok:
        out["reason"] = f"S3 write FAILED for {write_key}"
        return out

    out["produced"] = True
    out["written_key"] = write_key
    out["rows"] = int(len(frame))
    out["symbols"] = int(frame["symbol"].nunique())
    out["reason"] = (f"wrote {write_key}: {frame['symbol'].nunique()} symbols, "
                     f"max {ANCHOR_SYMBOL} bar {feed_max}, {len(frame)} rows "
                     f"(sources: {out['source_counts']})")
    return out
