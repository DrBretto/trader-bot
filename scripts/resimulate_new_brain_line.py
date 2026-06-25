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
# 06-24 is a CARRY (mark-only) day: the morning trigger was DISABLED on 06-24, so
# the 06-24 intents were never executed — the book HELD its 06-23 positions. The
# honest 06-24 line is therefore the held 06-23 book marked at the 06-24 SETTLED
# close (which landed in daily/2026-06-25/prices.parquet), NOT a counterfactual
# rotation and NOT a hand-set flat value. 06-25 has no settled close yet (no
# successor prices, no provisional morning bar) -> not priceable; the dashboard
# extender flat-holds it from 06-24 (the system's own priceability rule).
DECISION_DATES = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-23"]
ALL_DATES = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-22", "2026-06-23", "2026-06-24"]
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
# Per-date SETTLED price source for days whose real close arrived after the
# comprehensive _HIST_PRICES_KEY file was written. 06-24's settled bar lives in
# the 06-25 night prices file (06-23 stays provisional so its displayed value is
# the accepted gate $117,862 — R0 keep-the-line-exactly).
_SETTLED_PRICE_KEY = {"2026-06-24": "daily/2026-06-25/prices.parquet"}
_hist_cache = {}


def _ohlc(date):
    """{'open':..,'close':..} per symbol for `date`."""
    if date in _SETTLED_PRICE_KEY:
        key = _SETTLED_PRICE_KEY[date]
        if key not in _hist_cache:
            _hist_cache[key] = _get_parquet(key)
        df = _hist_cache[key]
    elif date == _NEWEST:
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


def resimulate(variant, field="full"):
    # variant: baseline = regime-blind engine as deployed; fixed = restored chassis.
    # field:   ALWAYS full. The committee-narrowed "derived" pool is GARBAGE and has
    #          been removed (PKT-TRADER-BOT-REGIME-PICKER-NIGHTLY-RECALC-FIX-V1,
    #          operator fact #3): all selection is from the FULL config/universe.csv,
    #          never a narrowed pool. The parameter is retained only so the legacy
    #          call signature does not break; any value other than "full" is rejected.
    if field != "full":
        raise ValueError(
            "resimulate: the narrowed/committee universe ('derived') is removed — "
            "selection is always from the full config/universe.csv (field='full')")
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
            # FULL universe always: keep config/universe.csv eligibility (the original
            # working field the engine selects its top-N from). The narrowed committee
            # pool is removed — never intersect eligibility with a selected pool.
            uni = universe_df.copy()
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
    # Terminal corrected book (last date's marks). This is the corrected position
    # the correctly-implemented engine would be holding at the end of the resim
    # window — used to reconcile the live sim book to the corrected history so the
    # corrected line is durable (the broken concentrated book is replaced, not
    # carried forward into a phantom drawdown).
    terminal_book = {
        "cash": round(float(book["cash"]), 6),
        "holdings": [
            {"symbol": h["symbol"], "shares": int(h["shares"]),
             "current_price": round(float(h["current_price"]), 6)}
            for h in sorted(book["holdings"], key=lambda x: x["symbol"])
        ],
        "marked_at": ALL_DATES[-1],
    }
    return {"raw": raw_line, "displayed": disp_line, "raw_seed": round(raw_seed, 2),
            "terminal_book": terminal_book}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "fixed"], required=True)
    # Full universe only. 'derived' (the narrowed committee pool) is removed.
    ap.add_argument("--field", choices=["full"], default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"# variant={args.variant} field={args.field}  (anchor 06-16 displayed ${DISPLAY_ANCHOR:,.0f})")
    res = resimulate(args.variant, args.field)
    print("\nDISPLAYED TOTALS:", json.dumps(res["displayed"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("wrote", args.out)
