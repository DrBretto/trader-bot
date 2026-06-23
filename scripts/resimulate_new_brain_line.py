#!/usr/bin/env python3
"""Re-simulate the New Brain canon line under a chosen engine variant.

The forward `new_brain_value` line on the dashboard is the accumulated daily book
values the LIVE engine produced. The cutover shipped regime-blind (regime
chassis dropped), so that line is contaminated by the implementation bug. This
re-runs the same forward book day-by-day with NO look-ahead — same frozen-brain
forecasts (production_forecaster, parity-checked to 1e-7 against the recorded
nightlies), same prices, same fill convention — but through the corrected engine,
to recover what the line WOULD have been had the algorithm executed as designed.

Two variants (`--variant`):
  blind  -> regime_compat passed as None: reproduces the deployed buggy engine.
            Used to VALIDATE the harness (it should track the published line).
  fixed  -> regime_compat = config/regime_compatibility.json: the corrected,
            regime-aware engine (the restored chassis).

No look-ahead: for decision date D the forecaster reads OHLCV <= D-1; D's fill is
the morning OPEN of D (priced from the successor day's prices.parquet, or the
provisional morning_prices.parquet for the newest date); marks are D's CLOSE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses

from src.brain.forecast_adapter import build_forecast_bundle, build_portfolio_state
from src.brain.engine import run_engine
from src.brain.engine.contracts import SizingParams
from src.brain.runtime import theta_from_freeze

# The recorded, point-in-time forecast ledger (one row per night, timestamped
# BEFORE outcomes, model_sha-stamped). This is the no-look-ahead mu the live
# engine used each night — using it (rather than re-running the forecaster, which
# refuses past dates now that the OHLCV store extends past them) keeps the re-sim
# strictly faithful to what was knowable at decision time.
_LEDGER = (Path(__file__).resolve().parents[1] / "runs" / "pkt_tb_007_orthogonal_brain"
           / "shadow" / "state" / "ledgers" / "forecast_ledger.jsonl")


def _recorded_mu():
    out = {}
    for ln in _LEDGER.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rec = json.loads(ln)
        if rec.get("date") and rec.get("mu"):
            out[rec["date"]] = rec["mu"]   # last record per date wins
    return out

BUCKET = "investment-system-data"
BUY_GAP = 0.05
MIN_ORDER = 250.0

# The New-Brain canon era. Engine-driven decision dates (wrote
# brain_selected_universe.json); 06-22 (Mon) had no engine decision -> mark-only.
DECISION_DATES = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-23"]
ALL_DATES = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-22", "2026-06-23"]
SEED_DATE = "2026-06-16"
# D's OHLC bar lives in its SUCCESSOR's prices.parquet (written that night,
# covers through the prior close); the newest date is priced provisionally.
SUCCESSOR = {
    "2026-06-17": "2026-06-18", "2026-06-18": "2026-06-19",
    "2026-06-19": "2026-06-20", "2026-06-20": "2026-06-22",
    "2026-06-22": "2026-06-23",
}

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _get_json(key):
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def _get_parquet(key):
    import io
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


# The comprehensive prices.parquet written on 06-22 night carries full OHLC
# history through the 06-22 close (06-17..06-22 real bars). The newest date
# (06-23) has no settled close yet -> its provisional morning bar.
_HIST_PRICES_KEY = "daily/2026-06-23/prices.parquet"
_NEWEST = "2026-06-23"
_hist_cache = {}


def _ohlc(date):
    """{'open':..,'close':..} per symbol for `date`."""
    if date == _NEWEST:
        df = _get_parquet(f"daily/{date}/morning_prices.parquet")
    else:
        if "hist" not in _hist_cache:
            _hist_cache["hist"] = _get_parquet(_HIST_PRICES_KEY)
        df = _hist_cache["hist"]
    df = df.copy()
    df["ds"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    sub = df[df["ds"] == date]
    return {r["symbol"]: {"open": float(r["open"]), "close": float(r["close"])}
            for _, r in sub.iterrows()}


def _seed_book():
    st = _get_json(f"daily/{SEED_DATE}/portfolio_state.json")
    return {"cash": float(st["cash"]),
            "holdings": [{"symbol": h["symbol"], "shares": int(h["shares"]),
                          "current_price": float(h.get("current_price",
                                                        h.get("entry_price", 0)) or 0)}
                         for h in st.get("holdings", [])]}


def _book_value(book, marks):
    return book["cash"] + sum(h["shares"] * marks.get(h["symbol"], 0.0) for h in book["holdings"])


def _fill(book, intents, ohlc, universe_df):
    """Apply the New Brain intents at the morning OPEN (faithful to
    morning_executor: 5% buy-gap guard, whole-share floor, min-order, cash cap;
    REDUCE/SELL use the intent's own share delta — no halving)."""
    secs = dict(zip(universe_df["symbol"].astype(str), universe_df["sector"].astype(str)))
    pos = {h["symbol"]: h for h in book["holdings"]}
    for it in intents:
        sym = it["symbol"]; act = it["action"]
        q = ohlc.get(sym)
        if q is None:
            continue
        op = q["open"]
        if act == "BUY":
            ip = float(it.get("price", 0) or 0)
            if ip <= 0 or op <= 0:
                continue
            if abs(op - ip) / ip > BUY_GAP:          # price-gap guard
                continue
            dollars = float(it.get("dollars", 0) or it.get("shares", 0) * ip)
            sh = int(dollars / op)
            if sh <= 0 or sh * op < MIN_ORDER:
                continue
            if sh * op > book["cash"]:
                sh = int(book["cash"] / op)
                if sh <= 0:
                    continue
            book["cash"] -= sh * op
            if sym in pos:
                pos[sym]["shares"] += sh
            else:
                pos[sym] = {"symbol": sym, "shares": sh, "current_price": op}
                book["holdings"].append(pos[sym])
        elif act in ("REDUCE", "SELL"):
            h = pos.get(sym)
            if h is None:
                continue
            sh = min(int(it.get("shares", h["shares"])), h["shares"])
            if sh <= 0:
                continue
            book["cash"] += sh * op
            h["shares"] -= sh
    book["holdings"] = [h for h in book["holdings"] if h["shares"] > 0]
    return book


def _mark(book, ohlc):
    for h in book["holdings"]:
        q = ohlc.get(h["symbol"])
        if q is not None:
            h["current_price"] = q["close"]


# Last good DISPLAYED canon value before the broken implementation took the line
# (06-16, the day before the first brain_selected_universe). The re-sim re-anchors
# the corrected book's RETURNS to this so totals are in actual displayed dollars
# and the line BEFORE 06-17 is left untouched (cumulative_external_cashflow == 0,
# so the displayed canon line tracks raw book returns one-for-one).
DISPLAY_ANCHOR = 117985.68


def resimulate(variant, field):
    # variant: baseline = regime-blind engine as deployed; fixed = restored chassis.
    # field:   derived  = the committee's concentrated daily pool (the bug);
    #          full     = the ORIGINAL working field (the full config universe the
    #                     system selected from before the committee narrowed it).
    regime_compat = None
    if variant == "fixed":
        regime_compat = json.loads(
            (Path(__file__).resolve().parents[1] / "config" / "regime_compatibility.json").read_text())
    universe_df = pd.read_csv(
        Path(__file__).resolve().parents[1] / "config" / "universe.csv")
    theta_sel, theta_size = theta_from_freeze()
    raw_sector = dict(zip(universe_df["symbol"].astype(str), universe_df["sector"].astype(str)))
    if variant == "baseline":
        theta_size = dataclasses.replace(theta_size, regime_exposure_multiplier={})

    book = _seed_book()
    # raw value of the seed book at the 06-16 close -> the denominator that maps
    # the corrected raw book onto the displayed scale.
    _mark(book, _ohlc(SEED_DATE))
    raw_seed = _book_value(book, {h["symbol"]: h["current_price"] for h in book["holdings"]})

    mu_by_date = _recorded_mu()
    raw_line, disp_line = {}, {}
    for date in ALL_DATES:
        ohlc = _ohlc(date)
        if date in DECISION_DATES:
            mu = mu_by_date[date]
            dec = _get_json(f"daily/{date}/decisions.json")
            regime = dec.get("regime") or "risk_on_trend"
            feats = _get_parquet(f"daily/{date}/features.parquet")
            uni = universe_df.copy()
            if field == "derived":
                pool = set(_get_json(f"daily/{date}/brain_selected_universe.json")["selected_universe"])
                uni["eligible"] = uni["symbol"].isin(pool).astype(int)
            # field == "full": keep config/universe.csv eligibility (the original
            # working field the engine selects its top-N from).
            f = build_forecast_bundle(date, mu, feats, regime, uni,
                                      health_map=None, regime_compat=regime_compat)
            portfolio = build_portfolio_state(deepcopy(book), feats, uni)
            if variant == "baseline":
                portfolio = dataclasses.replace(portfolio, cluster_of=raw_sector)
            out = run_engine(f, theta_sel, theta_size, portfolio)
            book = _fill(book, out.trade_intents["actions"], ohlc, uni)
        else:
            regime = "(carry)"
        _mark(book, ohlc)
        raw = _book_value(book, {h["symbol"]: h["current_price"] for h in book["holdings"]})
        disp = DISPLAY_ANCHOR * raw / raw_seed
        raw_line[date], disp_line[date] = round(raw, 2), round(disp, 2)
        held = sorted((h["symbol"], h["shares"]) for h in book["holdings"])
        print(f"{date} [{regime:14}] TOTAL=${disp:,.0f}  (raw ${raw:,.0f})  cash=${book['cash']:,.0f}")
        print(f"               held={held}")
    return {"raw": raw_line, "displayed": disp_line, "raw_seed": round(raw_seed, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "fixed"], required=True)
    ap.add_argument("--field", choices=["derived", "full"], default="derived")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"# variant={args.variant} field={args.field}  (anchor 06-16 displayed ${DISPLAY_ANCHOR:,.0f})")
    res = resimulate(args.variant, args.field)
    print("\nDISPLAYED TOTALS:", json.dumps(res["displayed"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("wrote", args.out)
