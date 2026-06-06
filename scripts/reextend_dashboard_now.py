#!/usr/bin/env python3
"""One-shot: re-run the three-line replay extender against the live
dashboard.json and republish, WITHOUT waiting for a Lambda rebuild.

Use this to push a corrected displayed line immediately after changing the
extender/replay code in the working tree. The nightly Lambda will produce the
same output once its image is rebuilt (the extender is now wired into the
committed publish path — see src/steps/publish_artifacts.py).

Safety:
- Backs up both live dashboard keys to dashboard/backups/ before overwriting
  (S3 versioning is Suspended; this is the only rollback).
- Aborts unless the re-extension stamped timeline_correction + optimized canon
  (extend_dashboard is internally defensive and returns the dash unchanged on
  failure; without this guard a silent no-op could republish stale data).

Env: AWS_PROFILE=personal AWS_REGION=us-east-1. Run from repo root.
"""
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.three_line_replay.extender import extend_dashboard  # noqa: E402

BUCKET = "investment-system-data"
REGION = os.environ.get("AWS_REGION", "us-east-1")
DIST_ID = "E10EHVNQ0CELM2"
SERVE_KEY = "dashboard/dashboard.json"
KEYS = ["dashboard/dashboard.json", "dashboard/data/dashboard.json"]


def main(stamp: str) -> int:
    s3 = boto3.client("s3", region_name=REGION)

    base = json.loads(s3.get_object(Bucket=BUCKET, Key=SERVE_KEY)["Body"].read())

    # Back up both keys first (no S3 versioning -> this is the only rollback).
    for k in KEYS:
        name = k.replace("/", "_")
        s3.copy_object(
            Bucket=BUCKET, Key=f"dashboard/backups/{name}.{stamp}.json",
            CopySource={"Bucket": BUCKET, "Key": k},
        )
        print("backed up", k)

    out = extend_dashboard(s3, base)

    tc = out.get("timeline_correction", {})
    if tc.get("version") != "lambda-three-line-replay-v2-optimized-canon":
        sys.exit("ABORT: extender did not stamp timeline_correction — extension failed")
    if out.get("metrics", {}).get("canon_source") != "optimized_champion":
        sys.exit("ABORT: canon_source not optimized_champion — not publishing")

    body = json.dumps(out, indent=2, default=str, allow_nan=False).encode()
    for k in KEYS:
        s3.put_object(Bucket=BUCKET, Key=k, Body=body, ContentType="application/json")
        print("wrote", k, len(body), "bytes")

    cf = boto3.client("cloudfront", region_name=REGION)
    inv = cf.create_invalidation(
        DistributionId=DIST_ID,
        InvalidationBatch={
            "Paths": {"Quantity": 2, "Items": ["/dashboard.json", "/data/dashboard.json"]},
            "CallerReference": f"reextend-{stamp}",
        },
    )
    print("invalidation", inv["Invalidation"]["Id"], inv["Invalidation"]["Status"])
    print("final displayed value:", out["equity_curve"][-1]["value"])
    return 0


if __name__ == "__main__":
    # Caller supplies a stamp so reruns/resume are deterministic.
    stamp = sys.argv[1] if len(sys.argv) > 1 else "manual"
    raise SystemExit(main(stamp))
