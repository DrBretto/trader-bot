"""FP-08-2 — seed the equity ledger from the LIVE rendered chart (one-time backfill).

The clean-core genesis. The displayed line the operator certifies as accepted truth
is **photographed verbatim** into write-once ledger leaves — never re-derived. This
job:

  * reads the LIVE S3 ``dashboard/dashboard.json`` (``equity_curve[].value`` +
    ``[].benchmark`` + the per-segment markers) and ``dashboard/shadow_timeseries.json``
    ``shadow_A`` (the raw dotted comparison the frontend rebases-to-100);
  * appends one write-once leaf per trading day, in date order, into
    ``src/canon/equity_ledger.py`` — value/benchmark/comparison/segment per the
    dossier §2 schema;
  * writes ``canon/equity_ledger/meta.json`` provenance (incl. the honest
    FREEZE_ORB1 circular-origin disclosure);
  * asserts EXACT three-line parity vs the rendered chart and the operator's
    terminal pin, or aborts.

**G-SEED-SOURCE / K-1:** this NEVER calls ``extend_dashboard`` /
``champion_freeze_map`` / ``_apply_canonical_overrides`` / any return re-chaining.
It reads only the already-rendered live values. The repo ``frontend/dist|public``
snapshots are FORBIDDEN as source — live S3 only.

Run (laptop, AWS creds):
    AWS_REGION=us-east-1 python -m src.canon.equity_seed --dry-run
    AWS_REGION=us-east-1 python -m src.canon.equity_seed --apply --terminal-pin 116126.12
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lines.ledger import (
    BUCKET, LEDGER_PREFIX, EquityLedger, build_leaf, fold_cache,
)
from lines.line import (
    build_line_view, assert_seed_parity_against_rendered,
)

DASHBOARD_KEY = "dashboard/dashboard.json"
SHADOW_KEY = "dashboard/shadow_timeseries.json"
META_KEY = LEDGER_PREFIX + "meta.json"

_REPO = Path(__file__).resolve().parents[2]

# Per-segment provenance for the seed leaves (photographed, not recomputed).
_SEG_MODEL = {
    "frozen_champion": ("champion_freeze_20260611", "champion_freeze"),
    "incumbent": ("incumbent", "incumbent_algorithm"),
    "new_brain": ("FREEZE_ORB1@4ae7b79d", "native_two_stage"),
}


def _segment_of(row: Dict[str, Any]) -> str:
    if row.get("champion_frozen_value") is not None:
        return "frozen_champion"
    if row.get("new_brain_value") is not None:
        return "new_brain"
    if row.get("incumbent_value") is not None:
        return "incumbent"
    raise ValueError(f"row {row.get('date')} has no segment marker — cannot seed honestly")


def _sha256_file(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_meta(rendered: Dict[str, Any], terminal_value: float, n_leaves: int) -> Dict[str, Any]:
    """Provenance, including the honest FREEZE_ORB1 circular-origin disclosure."""
    return {
        "schema": "equity_ledger_meta.v1",
        "seeded_from": {
            "source": "live S3 dashboard/dashboard.json + shadow_timeseries.json (G-SEED-SOURCE)",
            "snapshot_id": rendered.get("snapshot", {}).get("id"),
            "snapshot_date": rendered.get("snapshot", {}).get("date"),
            "method": "photographed verbatim; NO extend_dashboard / champion_freeze_map / recompute",
        },
        "displayed_terminal": terminal_value,
        "terminal_anchor_internal": 114772.39,
        "n_leaves": n_leaves,
        "freeze_provenance": {
            "FREEZE_ORB1": {
                "path": "brain/FREEZE_ORB1.json",
                "sha256": _sha256_file(_REPO / "brain" / "FREEZE_ORB1.json"),
                "role": "the LIVE engine freeze (regime-restored native_two_stage)",
            },
            "champion_freeze_20260611": {
                "path": "config/champion_freeze_20260611.json",
                "sha256": _sha256_file(_REPO / "config" / "champion_freeze_20260611.json"),
                "role": "the displayed historical anchor through 2026-06-11 (terminal 114772.39)",
            },
            "disclosure": (
                "The champion freeze table was ORIGINALLY minted circularly from a prior "
                "recompute; the operator accepted it as the record and it is carried "
                "verbatim, byte-immutable. It is NOT independent evidence of itself. "
                "terminal_anchor_internal (114772.39) is internal to the historical "
                "segment; displayed_terminal is the only displayed number."
            ),
        },
    }


def _rendered_to_leaves(
    rendered: Dict[str, Any],
    shadow_A: Dict[str, float],
    issued_by: str,
) -> List[Dict[str, Any]]:
    """Photograph each rendered equity_curve row into a (not-yet-stored) leaf payload."""
    ec = sorted(rendered["equity_curve"], key=lambda r: r["date"])
    leaves: List[Dict[str, Any]] = []
    for r in ec:
        seg = _segment_of(r)
        model_id, source = _SEG_MODEL[seg]
        leaves.append({
            "date": r["date"],
            "value": float(r["value"]),
            "benchmark": float(r["benchmark"]),
            "comparison": (float(shadow_A[r["date"]]) if r["date"] in shadow_A else None),
            "segment": seg,
            "model_id": model_id,
            "source": source,
            "issued_by": issued_by,
        })
    return leaves


def seed(s3_client, *, issued_by: str, terminal_pin: float,
         apply: bool, bucket: str = BUCKET) -> Dict[str, Any]:
    """Seed the ledger from live S3. Returns a report dict. Idempotent when the
    ledger is already seeded to the same terminal; refuses to clobber a divergent
    existing ledger."""
    rendered = json.loads(
        s3_client.get_object(Bucket=bucket, Key=DASHBOARD_KEY)["Body"].read())
    try:
        shadow = json.loads(
            s3_client.get_object(Bucket=bucket, Key=SHADOW_KEY)["Body"].read())
        shadow_A = {d: float(v) for d, v in (shadow.get("shadow_A") or [])}
    except Exception:  # noqa: BLE001 — comparison line optional
        shadow_A = {}

    leaf_payloads = _rendered_to_leaves(rendered, shadow_A, issued_by)
    terminal_date = leaf_payloads[-1]["date"]
    terminal_value = leaf_payloads[-1]["value"]
    if terminal_value != terminal_pin:
        raise ValueError(
            f"rendered terminal {terminal_value} != operator pin {terminal_pin} "
            f"(G-TERMINAL-PIN) — refusing to seed a drifted line")

    # Build the would-be chain (in memory) now so we can verify any existing leaves
    # are a byte-identical prefix of our target chain (deterministic content_sha).
    chain: List[Dict[str, Any]] = []
    prev = None
    for p in leaf_payloads:
        leaf = build_leaf(
            date=p["date"], value=p["value"], benchmark=p["benchmark"],
            comparison=p["comparison"], segment=p["segment"], model_id=p["model_id"],
            source=p["source"], issued_by=p["issued_by"],
            prev_date=(prev["date"] if prev else None),
            prev_content_hash=(prev["content_sha"] if prev else None),
        )
        chain.append(leaf)
        prev = leaf
    chain_sha_by_date = {leaf["date"]: leaf["content_sha"] for leaf in chain}

    ledger = EquityLedger(s3_client, bucket)
    existing = ledger.read_manifest()
    existing_front = existing.get("frontier")
    already = False
    if existing_front is not None:
        fdate, fsha = existing_front.get("date"), existing_front.get("content_sha")
        if fdate == terminal_date and fsha == chain_sha_by_date.get(terminal_date):
            already = True
        elif fdate in chain_sha_by_date and fsha == chain_sha_by_date[fdate]:
            # A byte-identical PREFIX (e.g. an interrupted seed) — safe to complete:
            # the written leaves match our chain exactly; we re-put the full set
            # (idempotent) and rebuild manifest+cache from the leaves.
            pass
        else:
            raise RuntimeError(
                f"ledger already has frontier {existing_front} that does NOT match "
                f"this seed's chain; refusing to clobber. Inspect before re-seeding.")

    report: Dict[str, Any] = {
        "n_rows": len(leaf_payloads),
        "terminal_date": terminal_date,
        "terminal_value": terminal_value,
        "comparison_points": sum(1 for p in leaf_payloads if p["comparison"] is not None),
        "segments": {},
        "applied": False,
        "already_seeded": already,
    }
    for p in leaf_payloads:
        report["segments"][p["segment"]] = report["segments"].get(p["segment"], 0) + 1

    # Verify parity over the in-memory chain BEFORE writing anything.
    cache_rows = [json.loads(line) for line in fold_cache(chain).decode().splitlines() if line]
    line_view = build_line_view(cache_rows)
    assert_seed_parity_against_rendered(
        line_view, rendered["equity_curve"], terminal_pin=terminal_pin)
    report["parity"] = "EXACT (value + benchmark per date, terminal pinned)"
    report["line_metrics"] = line_view["line_metrics"]

    if not apply:
        return report

    if not already:
        # Bulk seed: write every leaf write-once (the in-memory `chain` already has
        # the exact prev-hash links + content_shas), then rebuild the manifest+cache
        # ONCE from the leaves. This avoids the O(n^2) per-append re-fold; the
        # restore-from-facts rebuild proves the leaves alone reconstruct the chain.
        for leaf in chain:
            ledger._put_leaf_write_once(leaf)
    # provenance + a re-fold to guarantee the cache matches the leaves
    meta = _build_meta(rendered, terminal_value, len(leaf_payloads))
    s3_client.put_object(
        Bucket=bucket, Key=META_KEY,
        Body=json.dumps(meta, indent=2, sort_keys=True, allow_nan=False).encode(),
        ContentType="application/json")
    ledger.rebuild(write=True)
    report["applied"] = True
    report["meta_key"] = META_KEY
    return report


def main() -> None:
    import boto3
    ap = argparse.ArgumentParser(description="Seed the equity ledger from the live chart")
    ap.add_argument("--apply", action="store_true", help="write leaves (default: dry-run)")
    ap.add_argument("--terminal-pin", type=float, required=True,
                    help="the certified current-line terminal value (G-TERMINAL-PIN)")
    ap.add_argument("--issued-by", default="drbretto82@gmail.com")
    ap.add_argument("--bucket", default=BUCKET)
    args = ap.parse_args()
    s3 = boto3.client("s3")
    report = seed(s3, issued_by=args.issued_by, terminal_pin=args.terminal_pin,
                  apply=args.apply, bucket=args.bucket)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
