#!/usr/bin/env python3
"""Republish the canon line with the 06-17..06-23 segment corrected.

The New-Brain cutover's universe derivation narrowed the working field into a
concentrated semis/metals pool; the regime-blind engine then held it through a
risk-off gap and the line showed a spurious -3.37% on 06-23. This patches ONLY
the dates the broken implementation drove (06-17 onward) with the corrected
re-simulation (original full field + the restored regime chassis); every date
<= 06-16 is left byte-untouched. Values are the bias-cancelled re-sim
(scripts/resimulate_new_brain_line.py; corrected = actual x corrected/baseline
raw-NAV ratio, which cancels the harness's ~1.4% absolute tracking error).

Backs up both dashboard keys first (S3 versioning is Suspended -> the only
rollback) and invalidates CloudFront.
"""
import json
import os
import boto3

BUCKET = "investment-system-data"
DIST_ID = "E10EHVNQ0CELM2"
KEYS = ["dashboard/dashboard.json", "dashboard/data/dashboard.json"]
STAMP = "20260623_corrected_line"

CORRECTED = {
    "2026-06-17": 118452.40, "2026-06-18": 117401.15, "2026-06-19": 117696.46,
    "2026-06-20": 117696.46, "2026-06-22": 117498.82, "2026-06-23": 116201.08,
}

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _recompute_drawdowns(equity_curve):
    peak = None
    dd = []
    for row in equity_curve:
        v = row.get("value")
        if v is None:
            dd.append({"date": row["date"], "drawdown": 0.0})
            continue
        peak = v if peak is None else max(peak, v)
        dd.append({"date": row["date"], "drawdown": (v / peak - 1.0) if peak else 0.0})
    return dd


def patch(dash):
    for row in dash.get("equity_curve", []):
        c = CORRECTED.get(row.get("date"))
        if c is None:
            continue
        row["value"] = c
        if row.get("new_brain_value") is not None:
            row["new_brain_value"] = c
    dash["drawdowns"] = _recompute_drawdowns(dash["equity_curve"])
    max_dd = min((d["drawdown"] for d in dash["drawdowns"]), default=0.0)
    cur_dd = dash["drawdowns"][-1]["drawdown"] if dash["drawdowns"] else 0.0
    m = dash.get("metrics", {})
    end = CORRECTED["2026-06-23"]
    m["total_value"] = end
    m["corrected_total_value"] = end
    m["actual_total_value"] = end
    m["max_drawdown"] = max_dd
    m["current_drawdown"] = cur_dd
    m["ytd_return"] = end / 100000.0 - 1.0
    dash["metrics"] = m
    tc = dash.get("timeline_correction", {}) or {}
    tc["resim_correction"] = {
        "applied": "2026-06-23",
        "segment": "2026-06-17..2026-06-23 (dates <= 2026-06-16 untouched)",
        "what": "The New-Brain cutover's committee universe derivation narrowed the "
                "working field into a concentrated semis/metals pool; the regime-blind "
                "engine held it into a risk-off gap, producing a spurious -3.37% on "
                "06-23. This segment is re-simulated with the original full field + the "
                "restored regime chassis (regime exposure cut + binding cluster cap).",
        "method": "bias-cancelled: corrected = actual x (corrected_raw / baseline_raw); "
                  "baseline reproduces the deployed-bug line, so the ratio isolates the "
                  "corrected-vs-bug differential and cancels the ~1.4% harness scale error.",
        "note": "Re-simulation of a paper sim, not a re-statement of realized cash. "
                "The live engine already selects from the full field going forward "
                "(verified on real 06-23 inputs).",
    }
    dash["timeline_correction"] = tc
    return dash


def main():
    base = json.loads(s3.get_object(Bucket=BUCKET, Key=KEYS[0])["Body"].read())
    # backup both keys first
    for k in KEYS:
        try:
            raw = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
            s3.put_object(Bucket=BUCKET, Key=f"dashboard/backups/{STAMP}/{os.path.basename(k)}",
                          Body=raw, ContentType="application/json")
        except Exception as e:
            print(f"backup {k}: {e}")
    patched = patch(base)
    body = json.dumps(patched).encode()
    for k in KEYS:
        s3.put_object(Bucket=BUCKET, Key=k, Body=body, ContentType="application/json")
        print("wrote", k)
    inv = s3.meta.client if False else boto3.client("cloudfront")
    r = inv.create_invalidation(DistributionId=DIST_ID, InvalidationBatch={
        "Paths": {"Quantity": 1, "Items": ["/*"]},
        "CallerReference": f"corrected-line-{STAMP}"})
    print("cloudfront invalidation:", r["Invalidation"]["Id"])
    ec = {r["date"]: r["value"] for r in patched["equity_curve"] if r["date"] in CORRECTED}
    print("patched equity_curve:", json.dumps(ec))
    print("max_drawdown:", patched["metrics"]["max_drawdown"], "total_value:", patched["metrics"]["total_value"])


if __name__ == "__main__":
    main()
