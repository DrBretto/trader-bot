"""PKT-TB-007 Analyst — artifact depth + caveat map (part B analysis).

1. inference.json structure across the TB-004 backfill boundary (one-hot heuristic
   2025 artifacts vs real model probs from 2026-01-31).
2. Repaired-timeline epochs: S3 LastModified vs dir date.
3. Paired-day counts for the replay plan (decision date needs inference+features+
   signals; priced from the NEXT dir's prices.parquet).
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "runs/pkt_tb_006_clean_sheet_brain/prototype/cache/s3/daily"
LISTING = json.loads((Path(__file__).resolve().parent / "s3_daily_listing.json").read_text())

W = sorted(d for d in LISTING if "2025-08-04" <= d <= "2026-06-10")

# ---------- 1. inference.json structure across the boundary ----------
print("== inference.json structure probe ==")
probe_dates = ["2025-08-05", "2025-10-15", "2025-12-15", "2026-01-15", "2026-01-28",
               "2026-01-30", "2026-01-31", "2026-02-03", "2026-02-05", "2026-03-12",
               "2026-06-09"]
for d in probe_dates:
    p = CACHE / d / "inference.json"
    if not p.exists():
        print(f"  {d}: NO inference.json")
        continue
    inf = json.loads(p.read_text())
    reg = inf.get("regime", {})
    gru = reg.get("gru_prediction") or {}
    tra = reg.get("transformer_prediction") or {}
    gp = gru.get("probs") or {}
    tp = tra.get("probs") or {}
    probs = reg.get("probs") or {}
    onehot = (sorted(probs.values()) and max(probs.values()) == 1.0
              and sum(probs.values()) == 1.0) if probs else None
    n_health = len(inf.get("asset_health", []))
    print(f"  {d}: label={reg.get('label')} conf={reg.get('confidence')} "
          f"probs_onehot={onehot} per_model_probs={'YES' if gp and tp else 'NO'} "
          f"keys={sorted(reg.keys())[:8]} n_asset_health={n_health}")

# Scan the whole window: classify each date's inference as one-hot vs soft, per-model present
print("\n== full-window inference classification ==")
classes = {}
for d in W:
    p = CACHE / d / "inference.json"
    if not p.exists():
        classes[d] = "absent"
        continue
    try:
        inf = json.loads(p.read_text())
    except Exception:
        classes[d] = "unreadable"
        continue
    reg = inf.get("regime", {})
    probs = reg.get("probs") or {}
    gp = (reg.get("gru_prediction") or {}).get("probs")
    onehot = bool(probs) and max(probs.values()) == 1.0
    if gp:
        classes[d] = "full_model"
    elif onehot:
        classes[d] = "onehot_backfill"
    else:
        classes[d] = "soft_no_permodel"
trans = []
prev = None
for d in W:
    c = classes[d]
    if c != prev:
        trans.append((d, c))
        prev = c
print("  transitions:", trans)
print("  counts:", Counter(classes.values()))

# ---------- 2. repaired-timeline epochs ----------
print("\n== S3 LastModified epochs (write-lag per dir, max over files) ==")
lag_rows = []
for d in W:
    lms = []
    for fname, meta in LISTING[d].items():
        lm = datetime.fromisoformat(meta["lm"]).date()
        lms.append((fname, lm))
    if not lms:
        continue
    ddate = datetime.strptime(d, "%Y-%m-%d").date()
    max_lag = max((lm - ddate).days for _, lm in lms)
    min_lag = min((lm - ddate).days for _, lm in lms)
    lag_rows.append({"date": d, "min_lag_days": min_lag, "max_lag_days": max_lag,
                     "lm_dates": sorted({str(lm) for _, lm in lms})})
df = pd.DataFrame(lag_rows)
print("  dirs with max write-lag > 3 days:", int((df.max_lag_days > 3).sum()), "of", len(df))
# epoch clustering: group by the set of LastModified dates
epoch_counter = Counter(tuple(r["lm_dates"]) for r in lag_rows)
print("  top write-date clusters:")
for lmset, n in epoch_counter.most_common(12):
    ds = [r["date"] for r in lag_rows if tuple(r["lm_dates"]) == lmset]
    print(f"    written {list(lmset)}: {n} dirs ({ds[0]}..{ds[-1]})")
df.to_csv(Path(__file__).resolve().parent / "write_lag_epochs.csv", index=False)

# ---------- 3. paired-day counts (replay plan simulation) ----------
print("\n== replay plan: viable decision dates ==")
def has(d, f):
    return (CACHE / d / f).exists()

dates_with_prices = [d for d in W if has(d, "prices.parquet")]
print("  dirs with prices.parquet:", len(dates_with_prices),
      f"({dates_with_prices[0]}..{dates_with_prices[-1]})")

# run_variant plan: decision dates are trading_dates[i+1] for i in 0..n-3,
# priced from trading_dates[i+2]'s prices.parquet. A decision date executes
# only if it ALSO has inference+features+signals (else 'continue' = flat-hold).
td = dates_with_prices
plan = [(td[i + 1], td[i + 2]) for i in range(len(td) - 2)]
viable = [d for d, nxt in plan
          if has(d, "inference.json") and has(d, "features.parquet") and has(d, "signals.parquet")]
skipped = [d for d, _ in plan if d not in viable]
print("  plan decision dates:", len(plan), "; viable (full inputs):", len(viable),
      "; flat-hold skips:", len(skipped))
print("  skipped dates:", skipped)
hold = [d for d in viable if d >= "2026-03-11"]
pre = [d for d in viable if d < "2026-03-11"]
print(f"  viable pre-holdout (<2026-03-11): {len(pre)}  ({pre[0]}..{pre[-1]})")
print(f"  viable holdout (>=2026-03-11): {len(hold)}  ({hold[0]}..{hold[-1]})")
dow = Counter(pd.Timestamp(d).day_name() for d in viable)
print("  day-of-week mix:", dict(dow))

# llm_risk availability among viable dates
llm_have = [d for d in viable if has(d, "llm_risk.json")]
print(f"  viable dates WITH llm_risk.json: {len(llm_have)} "
      f"(first {llm_have[0] if llm_have else None})")

# ---------- 4. seed portfolio at 2025-08-04 ----------
print("\n== portfolio_state.json @ 2025-08-04 ==")
ps = json.loads((CACHE / "2025-08-04" / "portfolio_state.json").read_text())
print("  keys:", sorted(ps.keys()))
print("  cash:", ps.get("cash"), " portfolio_value:", ps.get("portfolio_value"),
      " n_holdings:", len(ps.get("holdings", [])))
for h in ps.get("holdings", [])[:12]:
    print("   ", {k: h.get(k) for k in ("symbol", "shares", "entry_price", "entry_date")})
