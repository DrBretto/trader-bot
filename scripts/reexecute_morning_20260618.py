#!/usr/bin/env python3
"""One-shot recovery for 2026-06-18: the reset bug made the morning run execute
today's intents against an empty $100k book. This replays the SAME intents and
the SAME captured morning prices against the REAL carried-forward book (06-17),
so the day reflects what actually should have happened — including the REDUCE
ARKK that got skipped on the empty book.

Uses the real production morning_executor (simulated mode); only fetch_morning_quotes
is monkeypatched to return the already-captured daily/2026-06-18/morning_prices.parquet
instead of re-fetching live (after-close) quotes.

Env: AWS_PROFILE=personal. Run from repo root with the project venv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.s3_client import S3Client
from src.steps import paper_trader, morning_executor, ingest_prices, publish_artifacts
from src.steps.publish_artifacts import build_dashboard_data, _build_snapshot_meta, _can_publish_dashboard, _verify_extension_or_alarm
from src.utils.three_line_replay.extender import extend_dashboard
from src.utils.dashboard_metrics import attach_new_brain_surface, sanitize_nan_for_json

BUCKET = "investment-system-data"
RUN_DATE = "2026-06-18"
DASH_KEYS = ["dashboard/data/dashboard.json", "dashboard/dashboard.json"]


def main(commit: bool) -> int:
    s3 = S3Client(BUCKET)

    # Captured morning quotes -> the price map the executor would have used.
    mp = s3.read_parquet(f"daily/{RUN_DATE}/morning_prices.parquet")
    quotes = mp[["symbol", "close"]].rename(columns={"close": "price"}).copy()

    def _fake_fetch(symbols, *a, **k):
        return quotes[quotes["symbol"].isin(set(symbols))].reset_index(drop=True)

    morning_executor.ingest_prices.fetch_morning_quotes = _fake_fetch

    # Minimal config (simulated mode -> broker=None).
    dp = s3.read_json("config/decision_params.active.json") or {}
    config = {
        "decision_params": dp.get("decision_params", {}),
        "transaction_cost_overrides": dp.get("transaction_costs"),
    }

    starting = paper_trader.load_portfolio_state(s3)
    print(f"START book: cash={starting.get('cash'):.2f} "
          f"holdings={[h['symbol'] for h in starting.get('holdings', [])]} "
          f"benchmark={starting.get('benchmark_value')}")

    result = morning_executor.run(BUCKET, config)
    pf = result["portfolio_state"]
    trades = result["trades"]

    print("\n--- replay validation log ---")
    for line in result["validation_log"]:
        print("  ", line)

    print("\n--- RESULT book (real, post today's trades) ---")
    print(f"  cash           = {pf.get('cash'):.2f}")
    print(f"  portfolio_value= {pf.get('portfolio_value'):.2f}")
    print(f"  benchmark_value= {pf.get('benchmark_value'):.2f}")
    for h in pf.get("holdings", []):
        print(f"    {h['symbol']:5s} {h['shares']:>8.3f} sh @ {h.get('current_price',0):.2f}"
              f"  mv={h.get('market_value',0):.2f}")
    print(f"  trades today   = {[(t['action'], t['symbol'], t['shares']) for t in trades]}")

    # sanity gate
    assert len(pf.get("holdings", [])) >= 5, "too few holdings — aborting"
    assert 80000 < pf["portfolio_value"] < 120000, f"sim value out of range: {pf['portfolio_value']}"

    if not commit:
        print("\n(dry run — pass 'commit' to write)")
        return 0

    # 1. Persist the real book (role-marked published shape).
    pf["date"] = RUN_DATE
    pf["trades_today"] = trades
    s3.write_json(paper_trader.to_published_state(pf), f"daily/{RUN_DATE}/portfolio_state.json")
    print("\nwrote daily/%s/portfolio_state.json" % RUN_DATE)

    # 2. Overwrite trades.jsonl with the real trades (replace phantom fills).
    import boto3
    raw = "\n".join(json.dumps(t, default=str) for t in trades) + ("\n" if trades else "")
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key=f"daily/{RUN_DATE}/trades.jsonl",
        Body=raw.encode(), ContentType="application/x-ndjson",
    )
    print("rewrote daily/%s/trades.jsonl (%d trades)" % (RUN_DATE, len(trades)))

    # 3. Rebuild dashboard from the corrected book.
    latest = s3.read_json("daily/latest.json") or {}
    nd = latest.get("intents_date", latest.get("date", RUN_DATE))
    ni = s3.read_json(f"daily/{nd}/inference.json") or {}
    ndec = s3.read_json(f"daily/{nd}/decisions.json") or {}
    nw = s3.read_json(f"daily/{nd}/weather_blurb.json") or {}
    es = None
    try:
        sig = s3.read_parquet(f"daily/{nd}/signals.parquet")
        if len(sig) > 0:
            r = sig.iloc[0]
            es = {
                "macro_credit": {"macro_credit_score": float(r.get("macro_credit_score", 0)),
                                  "yield_slope_10y_3m": float(r.get("yield_slope_10y_3m", 0)),
                                  "hy_spread_proxy": float(r.get("hy_spread_proxy", 0))},
                "vol_uncertainty": {"vol_uncertainty_score": float(r.get("vol_uncertainty_score", 0.5)),
                                     "vol_regime_label": str(r.get("vol_regime_label", "calm")),
                                     "vix_percentile": float(r.get("vix_percentile", 0.5)),
                                     "vvix_percentile": float(r.get("vvix_percentile", 0.5))},
                "fragility": {"fragility_score": float(r.get("fragility_score", 0.5)),
                               "avg_correlation": float(r.get("avg_correlation", 0)),
                               "pc1_explained": float(r.get("pc1_explained", 0))},
                "entropy_shift": {"entropy_score": float(r.get("entropy_score", 0.5)),
                                   "entropy_z_score": float(r.get("entropy_z_score", 0)),
                                   "entropy_shift_flag": bool(r.get("entropy_shift_flag", False))},
            }
    except Exception as e:
        print("WARN expert_signals:", e)

    if not _can_publish_dashboard(es):
        sys.exit("ABORT: expert_signals null — would skip dashboard publish")

    meta = _build_snapshot_meta(RUN_DATE, "morning", pf)
    dash = build_dashboard_data(pf, ni, ndec, nw, s3, expert_signals=es, snapshot_meta=meta)
    dash = extend_dashboard(s3.s3, dash)
    try:
        shadow = s3.read_json("dashboard/shadow_timeseries.json")
    except Exception:
        shadow = None
    dash = attach_new_brain_surface(dash, shadow)
    dash = sanitize_nan_for_json(dash)
    ok, reason = _verify_extension_or_alarm(dash, s3, "morning", RUN_DATE)
    if not ok:
        sys.exit(f"ABORT: advance guard failed — {reason}")

    last = dash["equity_curve"][-1]
    print("\n--- rebuilt dashboard tail ---")
    for p in dash["equity_curve"][-3:]:
        print("  ", p["date"], "MAIN=", round(p["value"], 1), "SPY=", round(p["benchmark"], 1))

    for k in DASH_KEYS:
        cur = s3.read_json(k)
        if cur is not None:
            s3.write_json(cur, f"dashboard/backups/{k.replace('/', '_')}.reexec-{RUN_DATE}.json")
    for k in DASH_KEYS:
        s3.write_json(dash, k)
        print("wrote", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(commit=("commit" in sys.argv)))
