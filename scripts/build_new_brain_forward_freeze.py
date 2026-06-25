#!/usr/bin/env python3
"""Build the byte-static New-Brain forward-line freeze table.

PKT-TRADER-BOT-REGIME-PICKER-NIGHTLY-RECALC-FIX-V1.

The displayed forward line (> 2026-06-11) was being re-derived on every publish by
chaining the per-day ``sim_book_value`` (the dead ~$95k internal book — NEVER the
line) onto the frozen 06-11 champion terminal. The live night/midday runs overwrite
each day's ``sim_book_value``, so every overnight publish re-chained — and re-broke —
the whole forward line (the recurring cliff).

This freezes the CORRECT forward line as a byte-static table, read (never recomputed)
by ``extend_dashboard`` exactly like the gated champion freeze (<= 2026-06-11). The
nightly recompute then cannot revert it.

Segments of the table:
  - 2026-06-12 .. 2026-06-16: the accepted PRE-ENGINE incumbent line, snapshotted
    verbatim from the live (operator-accepted) dashboard.json. The New-Brain era
    starts 2026-06-17 (first brain_selected_universe.json), so these dates predate
    the regime/universe bug and are the accepted truth.
  - 2026-06-17 .. 2026-06-24: the regime-ON + FULL-universe replay
    (resimulate_new_brain_line.py, variant=fixed) — the validated correct line
    (gate: 06-23 = $117,861.91 ~= the accepted $117,862). 06-24 is a mark-only
    CARRY: the morning trigger was disabled, so no trades executed and the book
    HELD its 06-23 positions, marked at the 06-24 SETTLED close — the honest real
    value (~$115.8k; 06-24 was a real high_vol_panic down day), NOT a hand-set or
    flat-carried number.
  - 2026-06-25 (beyond FRONTIER): NOT in the table. 06-25 has no settled close yet
    (no successor prices, no provisional morning bar) -> not priceable; the
    dashboard extender flat-holds it from the 06-24 value. Nothing is hand-set.

Run (laptop, .venv with pyarrow + torch):
    AWS_REGION=us-east-1 .venv/bin/python scripts/build_new_brain_forward_freeze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3

from resimulate_new_brain_line import resimulate  # noqa: E402  (same dir, sys.path)

REPO = Path(__file__).resolve().parents[1]
BUCKET = "investment-system-data"
OUT = REPO / "config" / "new_brain_forward_freeze_20260625.json"

BOUNDARY_AFTER = "2026-06-11"     # champion owns <= this; this table owns >
FRONTIER_DATE = "2026-06-24"      # newest SETTLED/priceable date this freeze covers
PRE_ENGINE_DATES = ["2026-06-12", "2026-06-13", "2026-06-15", "2026-06-16"]


def _live_dashboard():
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=BUCKET, Key="dashboard/dashboard.json")["Body"].read()
    return json.loads(body)


def main() -> None:
    # 1. Accepted pre-engine incumbent segment, snapshotted from the accepted line.
    dash = _live_dashboard()
    live_by_date = {r["date"]: r.get("value") for r in dash.get("equity_curve", [])}
    pre_engine = {}
    for d in PRE_ENGINE_DATES:
        v = live_by_date.get(d)
        if v is None:
            raise SystemExit(f"pre-engine date {d} absent from live dashboard equity_curve")
        pre_engine[d] = round(float(v), 2)

    # 2. Regime-ON + FULL-universe replay (the corrected New-Brain era 06-17..06-24;
    #    06-24 is a mark-only held-book carry at its settled close).
    res = resimulate("fixed", "full")
    displayed = {d: round(float(v), 2) for d, v in res["displayed"].items()}

    gate = displayed.get("2026-06-23")
    if gate is None or abs(gate - 117862.0) > 5.0:
        raise SystemExit(f"GATE FAILED: 06-23 = {gate} (expected ~117862)")
    if "2026-06-24" not in displayed:
        raise SystemExit("06-24 missing from replay (expected mark-only carry)")

    merged = {}
    merged.update(pre_engine)
    merged.update(displayed)
    curve = [{"date": d, "value": merged[d]} for d in sorted(merged)]

    table = {
        "schema": "new_brain_forward_freeze.v1",
        "meta": {
            "packet": "PKT-TRADER-BOT-REGIME-PICKER-NIGHTLY-RECALC-FIX-V1",
            "built_on": "2026-06-25",
            "purpose": (
                "Byte-static forward New-Brain line (> 2026-06-11), READ by "
                "extend_dashboard and NEVER recomputed — the durable lock that "
                "stops the nightly recalc from re-deriving (and re-breaking) the "
                "line from the dead sim_book_value. Mirrors the gated champion "
                "freeze (<= 2026-06-11) extended forward."),
            "basis": (
                "CHAMPION/replay only. 06-12..06-16 = accepted pre-engine incumbent "
                "line snapshotted from the accepted dashboard.json; 06-17..06-24 = "
                "regime-ON + FULL-universe replay (resimulate_new_brain_line.py "
                "variant=fixed; the one chassis-restored regime picker, full "
                "config/universe.csv, no narrowed pool). 06-24 is a mark-only held-book "
                "carry at its settled close (morning trigger disabled -> no execution -> "
                "book held 06-23 positions); it is a REAL high_vol_panic down day "
                "(~115.8k), not a bug and not a hand-set value. 06-25 is not yet "
                "priceable (no settled close) and flat-holds 06-24 via the extender."),
            "gate": "2026-06-23 == 117861.91 (~= accepted 117862)",
            "finding": ("06-24's ~1.6% drop is REAL market movement (06-23 risk_on book "
                        "caught in a high_vol_panic session), NOT the nightly-recalc bug. "
                        "The bug was the REVERSION (line re-derived from sim_book every "
                        "publish); that is what this lock kills."),
            "never": "sim_book_value is NEVER the line and NEVER the return base.",
        },
        "boundary_after": BOUNDARY_AFTER,
        "frontier_date": FRONTIER_DATE,
        "curve": curve,
    }
    OUT.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {OUT}")
    for row in curve:
        print(f"  {row['date']}  {row['value']:,.2f}")


if __name__ == "__main__":
    main()
