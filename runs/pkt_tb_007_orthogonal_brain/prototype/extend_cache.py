"""PKT-TB-007 Analyst — extend the TB-006 disk cache back to 2025-08-04 (part B).

Downloads daily/<D>/{prices,context,features,signals}.parquet + inference.json +
llm_risk.json + portfolio_state.json for 2025-08-04..2026-06-10 into the TB-006
prototype cache (runs/pkt_tb_006_clean_sheet_brain/prototype/cache/s3/daily/),
skip-existing. Reports wall-clock + bytes downloaded.
"""
import json
import time
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "runs/pkt_tb_006_clean_sheet_brain/prototype/cache/s3"
LISTING = Path(__file__).resolve().parent / "s3_daily_listing.json"
BUCKET = "investment-system-data"
FILES = ["prices.parquet", "context.parquet", "features.parquet", "signals.parquet",
         "inference.json", "llm_risk.json", "portfolio_state.json"]
START, END = "2025-08-04", "2026-06-10"

listing = json.loads(LISTING.read_text())
s3 = boto3.Session(profile_name="personal", region_name="us-east-1").client("s3")

t0 = time.time()
n_dl = n_skip = n_absent = 0
bytes_dl = 0
for d in sorted(listing):
    if not (START <= d <= END):
        continue
    for fname in FILES:
        if fname not in listing[d]:
            n_absent += 1
            continue
        p = CACHE / "daily" / d / fname
        if p.exists():
            n_skip += 1
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        data = s3.get_object(Bucket=BUCKET, Key=f"daily/{d}/{fname}")["Body"].read()
        p.write_bytes(data)
        n_dl += 1
        bytes_dl += len(data)

dt = time.time() - t0
print(f"downloaded={n_dl} files / {bytes_dl/1e6:.1f} MB; skipped(existing)={n_skip}; "
      f"absent_in_s3={n_absent}; wall_clock={dt:.1f}s")
