"""OHLCV store — substrate currency, S3 splice, and the freeze-on-contact
detector (PKT-TRADER-BOT-FORECAST-SPINE-RELOCATE, P3).

Relocated VERBATIM from ``src/brain/runtime.py`` (the local-HEAD version that
carries the dtype-splice / freshness false-abort fix). The only edits are the
relocation itself — no ``sys.path.append(runs/pkt_tb_00X/...)`` and first-class
imports. Logic (the freeze-detector keyed on real bar dates, the two-invariant
staleness verdict, the ISSUE-01 dtype-mismatch class) is byte-preserved.

The dtype coerce-or-RAISE **at the extend() boundary** lives in
``forecast.inference.extend_ohlcv`` (relocated verbatim from the same brain) and
is driven here through ``FI.extend_ohlcv`` by ``_extend_ohlcv_from_s3`` — the
splice orchestration + freeze detector this module owns.

``_freshness_gate_verdict`` computes the substrate-currency verdict; the enforcing
gate that RAISES + fires SNS is ``decide.freshness_gate`` (which imports the
verdict + watermark from here).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# The single S3 bucket the chassis publishes daily/<D>/prices.parquet into.
S3_BUCKET = "investment-system-data"

# Trading days the store may lag the run date before the staleness gate ABORTS.
_STALE_TOLERANCE_TD = 3


def _ohlcv_watermark(SL):
    """Return ``(max_date_str_or_None, row_count)`` for the in-store SPY panel —
    the substrate-currency watermark the freshness gate keys on."""
    import pandas as pd
    spy = SL.CACHE_OHLCV / "SPY.parquet"
    if not spy.exists():
        return None, 0
    df = pd.read_parquet(spy, columns=["date"])
    d = pd.to_datetime(df["date"])
    return (d.max().strftime("%Y-%m-%d") if len(d) else None), int(len(df))


def _extend_ohlcv_from_s3(SL, FI) -> dict:
    """Bring the brain's OHLCV store current: fetch the chassis's published
    daily/<D>/prices.parquet for every date newer than the seed store and splice
    them in (extend_ohlcv). Uses the Lambda execution role's default creds.

    Returns a diagnostic dict (store watermark before/after, gap, bars added,
    per-date errors). Failures are LOGGED with the exact exception — NEVER
    silently swallowed (ISSUE-01/F-D1). The aggregate substrate-currency gate in
    ``production_forecaster`` is what ABORTS a stale night; per-date faults are
    collected here so that gate can name them."""
    import boto3
    import traceback
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    last_ohlcv, n_before = _ohlcv_watermark(SL)
    result = {"last_ohlcv_before": last_ohlcv, "rows_before": n_before,
              "dates_listed": 0, "gap": [], "downloaded": 0, "extended": 0,
              "bars_added": 0, "errors": []}

    try:
        pg = s3.get_paginator("list_objects_v2")
        dates = set()
        for page in pg.paginate(Bucket=S3_BUCKET, Prefix="daily/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                d = cp["Prefix"].split("/")[-2]
                if len(d) == 10 and d[4] == "-":
                    dates.add(d)
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        msg = f"S3 list daily/ FAILED: {type(e).__name__}: {e}"
        print(f"  [FRESHNESS] {msg}")
        result["errors"].append(msg)
        dates = set()
    result["dates_listed"] = len(dates)

    gap = sorted(d for d in dates if last_ohlcv is None or d > last_ohlcv)
    result["gap"] = gap

    # A `daily/<D>/` FOLDER newer than the store is NOT proof of a newer SETTLED
    # trading day: the morning/midday runs create `daily/<today>/` carrying only
    # morning artifacts (morning_prices.parquet, portfolio_state.json) hours before
    # today's session settles a real OHLCV bar (prices.parquet). Counting those
    # phantom folders as "newer daily dates available but 0 bars spliced" is exactly
    # the false-abort the frozen-substrate gate must NOT fire on (a not-yet-closed
    # day is not a freeze). The Sunday folders (06-15/22/29) are the same shape.
    # The staleness signal is defined ONLY over dates that carry a settled
    # prices.parquet bar; a genuine freeze (a real bar exists but won't splice)
    # still trips the gate. HEAD each candidate so warm-cache invokes don't miss it.
    gap_settled: List[str] = []
    for d in gap:
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=f"daily/{d}/prices.parquet")
            gap_settled.append(d)
        except Exception:  # noqa: BLE001 — no settled bar for d (phantom/Sunday folder)
            continue
    result["gap_settled"] = gap_settled
    print(f"  [FRESHNESS] store SPY max={last_ohlcv} rows={n_before}; listed "
          f"{len(dates)} daily dates; gap={len(gap)} "
          f"[{gap[0] if gap else '-'}..{gap[-1] if gap else '-'}]; "
          f"settled-gap={len(gap_settled)} "
          f"[{gap_settled[0] if gap_settled else '-'}..{gap_settled[-1] if gap_settled else '-'}]")

    import pandas as pd
    max_available_bar: Optional[str] = last_ohlcv
    for d in gap_settled:
        dst = SL.CACHE_DAILY / d / "prices.parquet"
        if not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(S3_BUCKET, f"daily/{d}/prices.parquet", str(dst))
                result["downloaded"] += 1
            except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
                msg = f"download daily/{d}/prices.parquet FAILED: {type(e).__name__}: {e}"
                print(f"  [FRESHNESS] {msg}")
                result["errors"].append(msg)
                continue
        # The freeze detector keys on the actual bar dates INSIDE the settled files,
        # not the folder labels: a `daily/D/prices.parquet` may carry bars only
        # through D-1, so a folder's mere existence never counts as a newer bar. The
        # max real bar found here is the "expected settled day" the gate compares the
        # store against — the store MUST reach it or a genuine freeze is live.
        try:
            bd = pd.to_datetime(pd.read_parquet(dst, columns=["date"])["date"])
            if len(bd):
                fdmax = bd.max().strftime("%Y-%m-%d")
                if max_available_bar is None or fdmax > max_available_bar:
                    max_available_bar = fdmax
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            result["errors"].append(f"read-bar daily/{d}/prices.parquet: {type(e).__name__}: {e}")
        try:
            FI.extend_ohlcv(d)
            result["extended"] += 1
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            print(f"  [FRESHNESS] extend_ohlcv({d}) FAILED: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}")
            result["errors"].append(f"extend_ohlcv({d}): {type(e).__name__}: {e}")
            continue

    last_after, n_after = _ohlcv_watermark(SL)
    result["last_ohlcv_after"] = last_after
    result["rows_after"] = n_after
    result["bars_added"] = n_after - n_before
    result["max_available_bar"] = max_available_bar
    print(f"  [FRESHNESS] after extend: SPY max={last_after} rows={n_after} "
          f"(+{result['bars_added']} bars; {result['extended']}/{len(gap_settled)} settled dates "
          f"extended; {result['downloaded']} downloaded; max_available_bar={max_available_bar}; "
          f"{len(result['errors'])} errors)")
    return result


def _weekday_trading_days_between(d0: str, d1: str) -> int:
    """Weekday count strictly between two YYYY-MM-DD dates (a cheap NYSE proxy;
    holidays make this CONSERVATIVE — it never fires falsely loud). 0 if d1<=d0."""
    import datetime as _dt
    try:
        a = _dt.date.fromisoformat(str(d0)[:10])
        b = _dt.date.fromisoformat(str(d1)[:10])
    except Exception:  # noqa: BLE001
        return 0
    if b <= a:
        return 0
    n, cur = 0, a
    while cur < b:
        cur += _dt.timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def _freshness_gate_verdict(run_date: str, ohlcv_max_date: Optional[str],
                            fresh: dict) -> dict:
    """Substrate-currency verdict. Two independent invariants, either trips ABORT:
      1. max settled OHLCV bar within K trading days of the run date, AND
      2. bar-count advanced when the S3 gap was non-empty (mu is a pure function of
         the price/vol OHLCV panel -> bars-advanced is the causal proxy for
         mu-freshness; a no-op splice IS the frozen-`mu` signature)."""
    reasons: List[str] = []

    # (0) diagnostic force: diagnose_forecast_freshness(force_stale=True) skips the
    #     extend to prove the gate still fires over a deliberately frozen store.
    if fresh.get("forced_stale"):
        reasons.append("forced_stale diagnostic: extend skipped to exercise the gate")

    if not ohlcv_max_date:
        reasons.append("OHLCV store empty — no SPY watermark to trust")
    else:
        # (1) the store's max bar must be within K trading days of the run date.
        #     A store frozen at the seed (06-10) while the run marches on trips this
        #     within ~4 trading days; a not-yet-closed run date lags by <= 1 and does
        #     NOT fire (tolerance K).
        lag = _weekday_trading_days_between(ohlcv_max_date, run_date)
        if lag > _STALE_TOLERANCE_TD:
            reasons.append(
                f"OHLCV substrate STALE: max settled bar {ohlcv_max_date} is {lag} "
                f"trading days behind run date {run_date} (tolerance {_STALE_TOLERANCE_TD})")

        # (2) THE freeze-on-contact detector, keyed on the expected settled day:
        #     the max real bar actually present in S3's settled prices.parquet files
        #     (max_available_bar). If a bar newer than the store exists but the store
        #     did not reach it, the splice froze (the genuine ISSUE-01 signature) —
        #     fail loud immediately, before invariant (1)'s lag even accrues. This
        #     compares BAR DATES, never folder labels, so a phantom `daily/<today>/`
        #     folder (no settled bar) and a `daily/D/` file that only carries bars
        #     through D-1 both leave max_available_bar == the store max => NO false
        #     abort on a not-yet-closed day.
        max_available = fresh.get("max_available_bar")
        if max_available and max_available > ohlcv_max_date:
            reasons.append(
                f"OHLCV freeze: settled bar {max_available} is available in S3 but the "
                f"store only reached {ohlcv_max_date} — the splice froze (frozen-mu signature)")
        elif max_available is None:
            # backward-compat: an older extend dict without the bar-date probe.
            # Fall back to the settled-gap heuristic (a settled folder present but 0
            # bars spliced) so a legacy freeze still trips.
            gap_settled = fresh.get("gap_settled")
            if gap_settled is None:
                gap_settled = fresh.get("gap") or []
            if gap_settled and fresh.get("bars_added", 0) <= 0:
                reasons.append(
                    f"OHLCV extend NO-OP: {len(gap_settled)} newer SETTLED daily date(s) "
                    f"[{gap_settled[0]}..{gap_settled[-1]}] but 0 bars spliced — frozen-substrate signature")

    return {"stale": bool(reasons), "reasons": reasons,
            "run_date": run_date, "ohlcv_max_date": ohlcv_max_date,
            "bars_added": fresh.get("bars_added", 0),
            "max_available_bar": fresh.get("max_available_bar"),
            "gap_len": len(fresh.get("gap") or []),
            "gap_settled_len": len(fresh.get("gap_settled") or [])}
