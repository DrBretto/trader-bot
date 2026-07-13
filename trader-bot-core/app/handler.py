"""Thin router for the clean spine (trader-bot-core) — P8.

A ROUTER, nothing else: it parses the invocation and dispatches to the
corresponding clean-core path. It performs NO data-science compute itself (no
pandas / numpy / torch, no feature math, no inference, no allocation) — every
heavy step lives behind a clean-core module the branch calls. Keeping the handler
this thin is the Lambda-thin-handler discipline and is verifiable by inspection
(``grep`` finds no compute here; the imports are lazy and per-branch so the cold
module carries no heavy deps).

Invocation is routed on ``event['source']`` (backward-compatible with the prod
event contract):

    morning-execution          → app.morning  (PRESERVED path; zero morning change)
    midday-check               → app.midday   (PRESERVED path)
    daily-health / healthcheck → monitors.run_daily_health_check (three-line email)
    ops-probe / *-diag /       → app.ops_probes.dispatch (governed, non-destructive
      canary / watchdog-diag                              reality-test branches)
    shadow-publish             → app.shadow_publish (publish canon + comparison mirrors)
    <anything else>            → app.night    (feeds→store→freshness_gate→features→
                                               forecast→engine→decide→lines→publish)

The night path emits ``daily/<D>/trade_intents.json`` in the EXACT schema the
morning executor already consumes (built by ``engine.build_trade_intents`` — the
morning path is untouched). Cutting production over to this handler is P9; this
module is the forward-running entrypoint P9 points the Lambda at.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

# Ops-probe / diag sources routed to app.ops_probes (governed, non-destructive).
_OPS_PROBE_SOURCES = frozenset({
    "ops-probe", "forecast-diag", "freshness-diag", "regime-diag",
    "canary", "config-canary", "advance-challenger",
    "watchdog-diag", "publish-revert-diag",
})
# Health-report sources routed to the three-line watchdog + daily email.
_HEALTH_SOURCES = frozenset({"daily-health", "healthcheck"})


def _bucket_region(event: dict) -> Tuple[str, str]:
    bucket = event.get("bucket", os.environ.get("S3_BUCKET", "investment-system-data"))
    region = (event.get("region") or os.environ.get("AWS_REGION")
              or os.environ.get("AWS_REGION_NAME") or "us-east-1")
    return bucket, region


def _ok(phase: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": 200,
            "body": json.dumps({"status": "success", "phase": phase, **payload},
                               default=str)}


def lambda_handler(event: dict, context) -> Dict[str, Any]:
    """Route one invocation to its clean-core path. Pure dispatch — the heavy
    lifting is behind the module each branch lazily imports."""
    event = event or {}
    source = event.get("source", "eventbridge-scheduled")
    bucket, region = _bucket_region(event)

    if source == "morning-execution":
        from app.morning import run_morning
        return run_morning(event, bucket, region)

    if source == "midday-check":
        from app.midday import run_midday
        return run_midday(event, bucket, region)

    if source in _HEALTH_SOURCES:
        from chassis.utils.s3_client import S3Client
        from monitors.watchdog import run_daily_health_check
        status = run_daily_health_check(S3Client(bucket, region),
                                        today=event.get("today"))
        return _ok("daily-health", {"ok": status.get("ok"),
                                    "subject": status.get("subject"),
                                    "lines": status.get("lines")})

    if source in _OPS_PROBE_SOURCES:
        from app.ops_probes import dispatch as ops_dispatch
        result = ops_dispatch(source, event, bucket, region)
        return _ok(result.get("phase", "ops-probe"), {"result": result})

    if source == "shadow-publish":
        # Publish the stored canon + comparison mirrors. NOT a second run_night.
        from app.shadow_publish import run_shadow_publish
        return _ok("shadow-publish", run_shadow_publish(event, bucket, region))

    # default: the night forward pipeline
    from app.night import run_night
    return run_night(event, bucket, region)


# For local testing / manual invoke.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trader-bot-core thin router")
    parser.add_argument("--source", default="daily-health",
                        help="invocation source (event['source'])")
    parser.add_argument("--bucket", default="investment-system-data")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--today", default=None)
    args = parser.parse_args()
    ev = {"source": args.source, "bucket": args.bucket, "region": args.region}
    if args.today:
        ev["today"] = args.today
    print(json.dumps(lambda_handler(ev, None), indent=2))
