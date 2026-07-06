"""CU-02 — shadow-publish: publish the P6 INDEPENDENT challenger (dotted) line.

The clean router (``app/handler.py``) dispatches ``source=shadow-publish`` here
(NOT to ``run_night``). This is a thin publish step — no data-science compute; the
independent incumbent+tilt challenger has ALREADY been computed and stored in the
corrected clean ledger's ``comparison`` column (written by
``replay/seed_canon.seed_canon_by_replay`` with ``independent_challenger=True`` —
the ported incumbent selection + M1 tilt, run INDEPENDENTLY, NOT the coupled
``publish.challenger`` M1-tilt-of-two-stage). This publisher surfaces that corrected
series onto the dashboard.

It reads the supersede-folded ``canon/equity_ledger_clean_v2/`` cache and writes
``dashboard/shadow_timeseries.json``:

  * ``shadow_A``  = the INDEPENDENT challenger series (``comparison`` column) — the
                    dotted line the frontend renders (``lines/seed.py`` reads
                    ``shadow_A`` as the comparison line; the tier-2 canary reads the
                    dotted line from ``shadow_A``).
  * ``live_line`` = the corrected canon ``value`` series on the REAL trading-day grid
                    — this OVERWRITES the known-contaminated ``121147.52`` terminal
                    (which sat on the old contaminated grid with phantom
                    weekend/holiday points).

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
    """Fold the clean_v2 cache (ndjson displayed leaves) into the two published
    series: ``live_line`` = canon ``value`` on the real settled grid (dates >=
    ``start_date``), ``shadow_A`` = independent challenger ``comparison`` (every leaf
    that carries one). Returns (live_line, shadow_A, latest_settled)."""
    rows = [json.loads(ln) for ln in cache_body.decode().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["date"])
    live_line = [[r["date"], round(float(r["value"]), 2)]
                 for r in rows if r["date"] >= start_date]
    shadow_A = [[r["date"], round(float(r["comparison"]), 2)]
                for r in rows if r.get("comparison") is not None]
    if not live_line:
        raise RuntimeError("clean_v2 has no canon leaves at/after start_date "
                           f"{start_date} — cannot publish live_line")
    if not shadow_A:
        raise RuntimeError("clean_v2 carries no challenger comparison values — "
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
            f"known-contaminated value — the corrected clean_v2 ledger did not clear it")
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
    payload["challenger_source"] = ("independent incumbent+tilt "
                                    "(canon/equity_ledger_clean_v2 comparison, P6)")
    return payload


def run_shadow_publish(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Publish the independent challenger dotted line + corrected canon mirror to
    ``dashboard/shadow_timeseries.json``. Returns a status dict (routed through the
    handler's ``_ok``)."""
    import boto3
    from lines.ledger import EquityLedger

    s3 = boto3.client("s3", region_name=region)
    ledger = EquityLedger(s3, bucket)          # defaults to clean_v2 prefix
    cache_body = ledger.read_cache()
    if not cache_body:
        raise RuntimeError("clean_v2 cache is empty/unreadable — cannot publish "
                           "the challenger (refuse to write an empty line)")

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
    print(f"  shadow-publish OK: live_line terminal {ll_term} (canon, "
          f"contaminated 121147.52 cleared); shadow_A terminal {sa_term} "
          f"(independent challenger); {len(payload['shadow_A'])} challenger points",
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
