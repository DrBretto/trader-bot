"""PKT-TB-007 Analyst — part B follow-ups: universe coverage across the boundary,
signals schema in the backfill era, per-file repair epochs."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "runs/pkt_tb_006_clean_sheet_brain/prototype/cache/s3/daily"
LISTING = json.loads((Path(__file__).resolve().parent / "s3_daily_listing.json").read_text())
W = sorted(d for d in LISTING if "2025-08-04" <= d <= "2026-06-10")

print("== features/signals/prices coverage probes ==")
for d in ["2025-08-05", "2025-11-14", "2026-01-28", "2026-01-31", "2026-03-12", "2026-06-09"]:
    f = CACHE / d / "features.parquet"
    s = CACHE / d / "signals.parquet"
    p = CACHE / d / "prices.parquet"
    if not f.exists():
        print(f"  {d}: missing features")
        continue
    fd = pd.read_parquet(f)
    sd = pd.read_parquet(s)
    pdf = pd.read_parquet(p)
    n_sym = fd["symbol"].nunique() if "symbol" in fd.columns else -1
    latest = str(pd.to_datetime(fd["date"]).max().date())
    n_sym_latest = fd[pd.to_datetime(fd["date"]) == pd.to_datetime(fd["date"]).max()]["symbol"].nunique()
    print(f"  {d}: features syms={n_sym} latest_date={latest} syms@latest={n_sym_latest} "
          f"rows={len(fd)} cols={len(fd.columns)}; signals rows={len(sd)} cols={list(sd.columns)[:6]}...; "
          f"prices syms={pdf['symbol'].nunique()} range={pd.to_datetime(pdf['date']).min().date()}..{pd.to_datetime(pdf['date']).max().date()}")

print("\n== signals.parquet required-column check (replay _build_expert_signals) ==")
need = ["avg_correlation", "pc1_explained", "macro_credit_score", "yield_slope_10y_3m",
        "hy_spread_proxy", "vol_uncertainty_score", "vol_regime_label", "vix_percentile",
        "vvix_percentile"]
for d in ["2025-08-05", "2025-12-15", "2026-01-28", "2026-03-12"]:
    sd = pd.read_parquet(CACHE / d / "signals.parquet")
    missing = [c for c in need if c not in sd.columns]
    ent = [c for c in sd.columns if "entropy" in c]
    print(f"  {d}: missing={missing} entropy_cols={ent} n_rows={len(sd)}")

print("\n== asset_health count by date (inference.json) ==")
counts = []
for d in W:
    p = CACHE / d / "inference.json"
    if not p.exists():
        continue
    inf = json.loads(p.read_text())
    counts.append((d, len(inf.get("asset_health", []))))
cc = Counter(n for _, n in counts)
print("  distribution:", dict(cc))
trans, prev = [], None
for d, n in counts:
    if n != prev:
        trans.append((d, n))
        prev = n
print("  transitions:", trans)

print("\n== per-file repair epochs (which files written on which dates) ==")
by_file = Counter()
for d in W:
    for fname, meta in LISTING[d].items():
        lm = datetime.fromisoformat(meta["lm"]).date()
        ddate = datetime.strptime(d, "%Y-%m-%d").date()
        lag = (lm - ddate).days
        bucket = str(lm) if lag > 3 else "same_week"
        by_file[(fname, bucket)] += 1
for (fname, bucket), n in sorted(by_file.items()):
    if bucket != "same_week":
        print(f"  {fname} written {bucket}: {n} dirs")
print("  (same-week writes):", {f: n for (f, b), n in by_file.items() if b == "same_week"})
