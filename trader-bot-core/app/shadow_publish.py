"""Publish the canonical TILT mirror and dotted two-stage comparison.

The clean router (``app/handler.py``) dispatches ``source=shadow-publish`` here
(NOT to ``run_night``). This is a thin publish step — no data-science compute; the
promoted replay ledger has already stored TILT in ``value`` and the reconstructed
two-stage book in ``comparison``. This publisher mirrors those stored facts into the
legacy shadow payload shape consumed by the frontend.

It reads the active ``lines.ledger`` cache and writes
``dashboard/shadow_timeseries.json``:

  * ``live_line`` = canonical TILT ``value`` on the real settled-session grid.
  * ``shadow_A``  = reconstructed two-stage ``comparison`` for the dotted yellow line.

Every other key in the existing payload (the ``shadow_I/R/F/E/U/B`` attribution
ladder, ``ic_series``, ``organ_ledger``, ``stats``, prereg pointers) is preserved
unchanged — CU-02 owns only the challenger dotted line + clearing the contaminated
canon-mirror point. A correctness guard REFUSES to publish if either terminal is the
known-contaminated value, so the fix can never silently re-publish contamination.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Tuple

SHADOW_KEY = "dashboard/shadow_timeseries.json"
DEFAULT_START_DATE = "2026-06-11"           # split terminal / rebase anchor

# The pre-P6 contaminated canon terminal (121147.52), matched at cent precision.
# Same sentinel the watchdog carries (monitors.watchdog.CONTAMINATED_TERMINAL_VALUES).
CONTAMINATED_TERMINAL_CENTS = frozenset({12114752})


def _cents(v: float) -> int:
    return int(round(float(v) * 100))


def _series_from_cache(cache_body: bytes, start_date: str
                       ) -> Tuple[List[List[Any]], List[List[Any]], str]:
    """Fold the active ledger into canon TILT and two-stage comparison mirrors."""
    rows = [json.loads(ln) for ln in cache_body.decode().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["date"])
    live_line = [[r["date"], round(float(r["value"]), 2)]
                 for r in rows if r["date"] >= start_date]
    shadow_A = [[r["date"], round(float(r["comparison"]), 2)]
                for r in rows if r.get("comparison") is not None]
    if not live_line:
        raise RuntimeError("active ledger has no canon leaves at/after start_date "
                           f"{start_date} — cannot publish live_line")
    if not shadow_A:
        raise RuntimeError("active ledger carries no two-stage comparison values — "
                           "cannot publish the dotted line")
    return live_line, shadow_A, live_line[-1][0]


def build_payload(existing: Dict[str, Any], cache_body: bytes) -> Dict[str, Any]:
    """Merge the corrected challenger + canon-mirror series into the existing
    ``shadow_timeseries.json`` payload, preserving every other key. Pure (no I/O) so
    it is unit-testable; the correctness guard lives here."""
    payload = dict(existing) if existing else {}
    start_date = payload.get("start_date") or DEFAULT_START_DATE
    live_line, shadow_A, latest_settled = _series_from_cache(cache_body, start_date)

    if _cents(live_line[-1][1]) in CONTAMINATED_TERMINAL_CENTS:
        raise RuntimeError(
            f"REFUSING to publish: live_line terminal {live_line[-1][1]} is the "
            f"known-contaminated value — the active ledger did not clear it")
    if _cents(shadow_A[-1][1]) in CONTAMINATED_TERMINAL_CENTS:
        raise RuntimeError(
            f"REFUSING to publish: shadow_A terminal {shadow_A[-1][1]} is the "
            f"known-contaminated value")

    payload["live_line"] = live_line
    payload["shadow_A"] = shadow_A
    payload["provisional_date"] = latest_settled
    payload["start_date"] = start_date
    payload.setdefault("schema", "shadow_timeseries.v2")
    payload["as_of"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    payload["challenger_source"] = (
        "two-stage comparison (active promoted ledger comparison)"
    )
    return payload


def run_shadow_publish(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Publish canonical TILT + dotted two-stage mirrors to
    ``dashboard/shadow_timeseries.json``. Returns a status dict (routed through the
    handler's ``_ok``)."""
    import boto3
    from lines.ledger import EquityLedger

    s3 = boto3.client("s3", region_name=region)
    ledger = EquityLedger(s3, bucket)
    cache_body = ledger.read_cache()
    if not cache_body:
        raise RuntimeError("active canon cache is empty/unreadable — refuse to publish")

    try:
        existing = json.loads(
            s3.get_object(Bucket=bucket, Key=SHADOW_KEY)["Body"].read())
    except Exception:  # noqa: BLE001 — first publish / missing file is fine
        existing = {}

    payload = build_payload(existing, cache_body)
    s3.put_object(
        Bucket=bucket, Key=SHADOW_KEY,
        Body=json.dumps(payload, sort_keys=True, default=float).encode(),
        ContentType="application/json")

    ll_term = payload["live_line"][-1]
    sa_term = payload["shadow_A"][-1]
    print(f"  shadow-publish OK: live_line terminal {ll_term} (TILT canon); "
          f"shadow_A terminal {sa_term} (two-stage comparison); "
          f"{len(payload['shadow_A'])} comparison points",
          flush=True)
    return {
        "phase": "shadow-publish",
        "published_key": SHADOW_KEY,
        "challenger_source": payload["challenger_source"],
        "live_line_terminal": ll_term,
        "shadow_A_terminal": sa_term,
        "n_live_line": len(payload["live_line"]),
        "n_shadow_A": len(payload["shadow_A"]),
        "provisional_date": payload["provisional_date"],
        "contaminated_cleared": True,
    }


if __name__ == "__main__":     # local dry-run: read + build, print, DO NOT write
    import argparse
    import boto3
    ap = argparse.ArgumentParser(description="shadow-publish challenger (dry-run)")
    ap.add_argument("--bucket", default="investment-system-data")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--apply", action="store_true", help="actually write to S3")
    args = ap.parse_args()
    _s3 = boto3.client("s3", region_name=args.region)
    from lines.ledger import EquityLedger
    _cache = EquityLedger(_s3, args.bucket).read_cache()
    try:
        _exist = json.loads(_s3.get_object(Bucket=args.bucket, Key=SHADOW_KEY)["Body"].read())
    except Exception:
        _exist = {}
    _pl = build_payload(_exist, _cache)
    print(json.dumps({
        "live_line_terminal": _pl["live_line"][-1],
        "shadow_A_terminal": _pl["shadow_A"][-1],
        "n_live_line": len(_pl["live_line"]),
        "n_shadow_A": len(_pl["shadow_A"]),
        "preserved_keys": sorted(k for k in _pl if k not in
                                 ("live_line", "shadow_A", "as_of", "provisional_date",
                                  "challenger_source")),
    }, indent=2, default=str))
    if args.apply:
        run_shadow_publish({}, args.bucket, args.region)
