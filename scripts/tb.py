#!/usr/bin/env python3
"""tb — the trader-bot operator CLI (seed).

PKT-TB-IR-01 (Stage S1) ships ONE verb: ``tb correct``. It writes a single
append-only, attributed CorrectionEvent to the ``corrections/`` store. It does
NOT — and must not — edit any rendered ``dashboard.json``; the nightly pipeline
reads the overlay and reproduces the correction on every regenerate. The fuller
``tb`` surface (add/compare/promote/retire/rebuild) arrives with S2..S7.

Usage:
  tb correct --set 2026-06-23=116201.08 [--set DATE=VALUE ...] \\
             --reason "why this correction" \\
             --reason-code RESIM_SEGMENT \\
             --method "how the corrected values were produced" \\
             [--from 2026-06-23=120000.0 ...] \\
             [--model-id canon] [--provenance constructed] \\
             [--sim-semantics paper_resim] [--supersedes corr:...] \\
             [--dry-run]

Env: AWS_PROFILE=personal AWS_REGION=us-east-1. The correction is written with
the operator's interactive credentials — which CAN write corrections/ but, after
IR-01's bucket-policy deny, CANNOT write the rendered dashboard keys. That is the
point: the honest path (a correction event) is the only path.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402

from src.utils.corrections import (  # noqa: E402
    BUCKET, CorrectionStore, build_event, resolve_overlay,
)


def _resolve_identity() -> str:
    """Resolve a real, non-anonymous identity for issued_by (never anonymous)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        email = subprocess.check_output(
            ["git", "-C", repo, "config", "user.email"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if email:
            return email
    except Exception:  # noqa: BLE001
        pass
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if user:
        return f"{user}@{socket.gethostname()}"
    raise SystemExit("tb correct: cannot resolve a non-anonymous identity for issued_by "
                     "(set git config user.email)")


def _parse_kv(items, label):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--{label} expects DATE=VALUE, got {it!r}")
        d, v = it.split("=", 1)
        d = d.strip()
        try:
            out[d] = float(v)
        except ValueError:
            raise SystemExit(f"--{label} value for {d} is not a number: {v!r}")
    return out


def cmd_correct(args) -> int:
    sets = _parse_kv(args.set, "set")
    from_values = _parse_kv(args.from_value, "from")
    issued_by = args.issued_by or _resolve_identity()

    event = build_event(
        issued_by=issued_by,
        reason_code=args.reason_code,
        why=args.reason,
        method=args.method,
        dates_to_values=sets,
        from_values=from_values,
        model_id=args.model_id,
        supersedes=args.supersedes,
        sim_semantics=args.sim_semantics,
        provenance=args.provenance,
    )

    print("CorrectionEvent:")
    print(json.dumps(event, indent=2, sort_keys=True))

    if args.dry_run:
        print("\n[dry-run] not written.")
        return 0

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    store = CorrectionStore(s3, bucket=BUCKET)
    key = store.append(event)
    print(f"\nwrote (write-once): s3://{BUCKET}/{key}")

    # Show the resulting active overlay for this model so the operator sees the
    # supersede-chain head (and any conflict) immediately.
    overlay = store.active_overlay(args.model_id)
    affected = {d: overlay[d] for d in sets if d in overlay}
    print("active overlay (head per corrected date):")
    print(json.dumps(affected, indent=2, sort_keys=True))
    conflicts = [d for d, o in affected.items() if o.get("conflict")]
    if conflicts:
        print(f"WARNING: conflict on dates {conflicts} — prior head retained, marker surfaced. "
              f"Use --supersedes <head corr_id> to override intentionally.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tb", description="trader-bot operator CLI (seed)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correct", help="write one append-only attributed correction event")
    c.add_argument("--set", action="append", required=True, metavar="DATE=VALUE",
                   help="corrected value for a date (repeatable)")
    c.add_argument("--from", dest="from_value", action="append", metavar="DATE=VALUE",
                   help="pre-correction value for a date (attribution only; repeatable)")
    c.add_argument("--reason", required=True, help="why this correction (human text)")
    c.add_argument("--reason-code", default="MANUAL_RESIM",
                   help="reason code enum (RESIM_SEGMENT|SPLIT_BASIS|ANCHOR_PROMOTE|SEAM|FREEZE|MANUAL_RESIM)")
    c.add_argument("--method", default="", help="how the corrected values were produced")
    c.add_argument("--model-id", default="canon", help="canon line alias (S1) or model id (S2+)")
    c.add_argument("--provenance", default="constructed", choices=("measured", "constructed"))
    c.add_argument("--sim-semantics", default="paper_resim",
                   choices=("paper_sim", "paper_resim", "paper_anchor", "paper_seam"))
    c.add_argument("--supersedes", default=None, help="corr_id of the prior chain head to supersede")
    c.add_argument("--issued-by", default=None, help="override identity (default: git user.email)")
    c.add_argument("--dry-run", action="store_true", help="print the event; do not write")
    c.set_defaults(func=cmd_correct)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
