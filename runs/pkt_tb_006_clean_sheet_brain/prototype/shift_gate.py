"""PKT-TB-006 SYN-1 prototype — GDELT distribution-shift gate (TOURNAMENT §4.6.2,
BUILD_SPEC §2.2 / §15 step 5).

Runs immediately after the top-up backfill, BEFORE any GDELT-fed training:

  top-up window   2026-02-05 -> 2026-06-10
  reference       2025-09-01 -> 2026-01-31

Checks (all numbers recorded; PASS/BREACH per check):
  1. record-count median ratio  median(n_records top-up)/median(n_records ref) in [0.5, 2.0]
  2. |mean tone shift| <= 2 x reference daily-tone sd
  3. no G1 bucket mean share shifted > 5 x its reference sd

Result committed to prototype/gdelt_shift_gate.json.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

PROTO = Path(__file__).resolve().parent
FEATURES = PROTO / "store" / "gdelt_features.parquet"
OUT = PROTO / "gdelt_shift_gate.json"

TOPUP = ("2026-02-05", "2026-06-10")
REFERENCE = ("2025-09-01", "2026-01-31")
RATIO_BOUNDS = (0.5, 2.0)
TONE_SD_MULT = 2.0
SHARE_SD_MULT = 5.0


def run_gate(features_path: Path = FEATURES, out_path: Path = OUT) -> dict:
    df = pd.read_parquet(features_path)
    df["gdelt_date"] = pd.to_datetime(df["gdelt_date"])

    def window(lo: str, hi: str) -> pd.DataFrame:
        return df[(df["gdelt_date"] >= lo) & (df["gdelt_date"] <= hi)]

    ref = window(*REFERENCE)
    top = window(*TOPUP)

    result: dict = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "reference_window": {"start": REFERENCE[0], "end": REFERENCE[1], "n_days": int(len(ref))},
        "topup_window": {"start": TOPUP[0], "end": TOPUP[1], "n_days": int(len(top))},
        "checks": {},
    }

    # 1. record-count median ratio
    med_ref = float(ref["n_records"].median())
    med_top = float(top["n_records"].median())
    ratio = med_top / med_ref
    result["checks"]["record_count_median_ratio"] = {
        "median_reference": med_ref,
        "median_topup": med_top,
        "ratio": ratio,
        "bounds": list(RATIO_BOUNDS),
        "verdict": "PASS" if RATIO_BOUNDS[0] <= ratio <= RATIO_BOUNDS[1] else "BREACH",
    }

    # 2. mean tone shift vs reference daily-tone sd
    ref_mean = float(ref["g3_tone_mean"].mean())
    top_mean = float(top["g3_tone_mean"].mean())
    ref_sd = float(ref["g3_tone_mean"].std())
    shift = abs(top_mean - ref_mean)
    result["checks"]["tone_shift"] = {
        "reference_mean_tone": ref_mean,
        "topup_mean_tone": top_mean,
        "reference_daily_tone_sd": ref_sd,
        "abs_shift": shift,
        "limit": TONE_SD_MULT * ref_sd,
        "shift_in_sd_units": shift / ref_sd if ref_sd > 0 else None,
        "verdict": "PASS" if shift <= TONE_SD_MULT * ref_sd else "BREACH",
    }

    # 3. per-bucket G1 mean-share shift vs reference share sd
    share_cols = sorted(c for c in df.columns
                        if c.startswith("g1_share_") and c != "g1_share_entropy")
    buckets = {}
    worst = (None, 0.0)
    any_breach = False
    for c in share_cols:
        b = c.removeprefix("g1_share_")
        r_mean, r_sd = float(ref[c].mean()), float(ref[c].std())
        t_mean = float(top[c].mean())
        shift_sd = abs(t_mean - r_mean) / r_sd if r_sd > 0 else float("inf")
        verdict = "PASS" if shift_sd <= SHARE_SD_MULT else "BREACH"
        any_breach |= verdict == "BREACH"
        if shift_sd > worst[1]:
            worst = (b, shift_sd)
        buckets[b] = {
            "reference_mean_share": r_mean,
            "reference_share_sd": r_sd,
            "topup_mean_share": t_mean,
            "shift_in_sd_units": shift_sd,
            "limit_sd_units": SHARE_SD_MULT,
            "verdict": verdict,
        }
    result["checks"]["g1_bucket_share_shift"] = {
        "per_bucket": buckets,
        "worst_bucket": worst[0],
        "worst_shift_in_sd_units": worst[1],
        "verdict": "BREACH" if any_breach else "PASS",
    }

    verdicts = [v["verdict"] for v in result["checks"].values()]
    result["overall"] = "PASS" if all(v == "PASS" for v in verdicts) else "BREACH"
    out_path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run_gate()
    c = r["checks"]
    print(f"GDELT distribution-shift gate -> {r['overall']}")
    rc = c["record_count_median_ratio"]
    print(f"  1. record-count median ratio  {rc['ratio']:.3f}  "
          f"(ref med {rc['median_reference']:.0f}, top-up med {rc['median_topup']:.0f}, "
          f"bounds {rc['bounds']})  {rc['verdict']}")
    ts = c["tone_shift"]
    print(f"  2. tone shift |{ts['abs_shift']:.3f}| vs limit {ts['limit']:.3f} "
          f"({ts['shift_in_sd_units']:.2f} sd)  {ts['verdict']}")
    bs = c["g1_bucket_share_shift"]
    n_breach = sum(1 for b in bs["per_bucket"].values() if b["verdict"] == "BREACH")
    print(f"  3. G1 bucket share shifts: {n_breach}/{len(bs['per_bucket'])} breach; "
          f"worst {bs['worst_bucket']} at {bs['worst_shift_in_sd_units']:.2f} sd "
          f"(limit {SHARE_SD_MULT})  {bs['verdict']}")
    if n_breach:
        for b, info in bs["per_bucket"].items():
            if info["verdict"] == "BREACH":
                print(f"       BREACH {b}: {info['shift_in_sd_units']:.2f} sd "
                      f"(ref {info['reference_mean_share']:.4f} -> top-up {info['topup_mean_share']:.4f})")
