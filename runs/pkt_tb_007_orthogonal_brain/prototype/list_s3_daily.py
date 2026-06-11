"""PKT-TB-007 Analyst — S3 daily/ listing for artifact-depth audit (part B)."""
import boto3, json
from collections import defaultdict

s3 = boto3.Session(profile_name="personal", region_name="us-east-1").client("s3")
pag = s3.get_paginator("list_objects_v2")
listing = defaultdict(dict)
for page in pag.paginate(Bucket="investment-system-data", Prefix="daily/"):
    for obj in page.get("Contents", []) or []:
        parts = obj["Key"].split("/")
        if len(parts) != 3 or not parts[2]:
            continue
        d, fname = parts[1], parts[2]
        listing[d][fname] = {"size": obj["Size"], "lm": obj["LastModified"].isoformat()}
out = {d: dict(listing[d]) for d in sorted(listing)}
with open("runs/pkt_tb_007_orthogonal_brain/prototype/s3_daily_listing.json", "w") as f:
    json.dump(out, f, indent=0, default=str)

dates = sorted(listing)
print("total date dirs:", len(dates), "first:", dates[0], "last:", dates[-1])
w = [d for d in dates if "2025-08-04" <= d <= "2026-06-10"]
print("in window 2025-08-04..2026-06-10:", len(w))
need = ["prices.parquet", "context.parquet", "features.parquet", "signals.parquet",
        "inference.json", "llm_risk.json", "portfolio_state.json"]
complete = [d for d in w if all(f in listing[d] for f in need)]
print("complete (all 7 files):", len(complete))
missing = {d: [f for f in need if f not in listing[d]]
           for d in w if any(f not in listing[d] for f in need)}
print("dates with missing files:", len(missing))
for d, m in sorted(missing.items()):
    print(" ", d, m)
