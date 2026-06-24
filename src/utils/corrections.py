"""Append-only, attributed CORRECTION OVERLAY (PKT-TB-IR-01 — Stage S1).

A correction is a first-class **append-only attributed event** written *write-once*
to the ``corrections/`` store — it is NEVER an edit to a rendered ``dashboard.json``.
The nightly extender reads the *active overlay* as an INPUT and applies it after
re-anchoring, so a correction **reproduces on every regenerate** instead of being
erased the next night. That is the exact failure mode (``republish_corrected_line``)
this kills: the patch used to hit the OUTPUT; now it is an INPUT upstream of the line.

Resolution rule (DESIGN_DOSSIER §4): a date's value is the **head of its
supersede-chain** — the latest event that validly supersedes the prior head. A
``supersedes: null`` event landing on a date that already has an active chain is a
**conflict**: we keep the prior head value and surface a visible ``conflict`` marker
(never a silent pick, never a dropped point). Likewise an event that claims to
supersede a corr_id that is not the current head is an orphan → conflict, prior head
retained.

Stage S1 scope: this is the small correction-overlay read on the EXISTING pipeline.
It is NOT the S2 append-only fact-log, the S3 pure fold, or the S6 registry. The
displayed canon line is addressed here by the alias ``model_id = "canon"``; per-model
ids arrive with the fold in S2+.

Honesty constraints baked in:
  * ``sim_semantics`` enum has NO realized-cash member — corrections are paper re-sims.
  * ``issued_by`` is never anonymous (the writer must resolve an identity or fail).
  * leaves are content-addressed + written with ``IfNoneMatch='*'`` → write-once;
    re-writing identical content is an idempotent no-op, different content cannot
    clobber an existing leaf.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BUCKET = "investment-system-data"
CORRECTIONS_PREFIX = "corrections/"
LOG_KEY = "corrections/_log.jsonl"

# The displayed canon line's alias for Stage S1 (per-model ids arrive in S2+).
CANON_MODEL_ID = "canon"

# sim_semantics: paper-only; there is deliberately NO realized-cash member.
SIM_SEMANTICS = ("paper_sim", "paper_resim", "paper_anchor", "paper_seam")
PROVENANCE = ("measured", "constructed")

# Reason codes (extensible; constrained so a typo is caught).
REASON_CODES = (
    "RESIM_SEGMENT", "SPLIT_BASIS", "ANCHOR_PROMOTE", "SEAM", "FREEZE", "MANUAL_RESIM",
)

# The canon value fields the frontend reads off each equity_curve row. A correction
# sets every one that is PRESENT and non-None on the corrected date (mirrors the
# retired republish script's CANON_FIELDS so the plotted line actually moves), and
# leaves None fields (e.g. champion_frozen_value/incumbent_value on forward dates)
# untouched so it never resurrects a field that wasn't there.
CANON_VALUE_FIELDS = (
    "value", "optimized_value", "new_brain_value", "champion_frozen_value",
    "incumbent_value", "corrected_value", "hybrid_value", "actual_value",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(obj: Dict[str, Any]) -> bytes:
    """Deterministic serialization for content-addressing (sorted keys, no NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha(event_without_sha: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(event_without_sha)).hexdigest()


def build_event(
    *,
    issued_by: str,
    reason_code: str,
    why: str,
    method: str,
    dates_to_values: Dict[str, float],
    from_values: Optional[Dict[str, float]] = None,
    model_id: str = CANON_MODEL_ID,
    supersedes: Optional[str] = None,
    sim_semantics: str = "paper_resim",
    provenance: str = "constructed",
    issued_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a fully-formed, content-addressed CorrectionEvent (not yet stored).

    ``dates_to_values`` maps YYYY-MM-DD -> corrected value (the ``to``). ``from_values``
    optionally records the pre-correction value per date (attribution only).
    """
    if not issued_by or not issued_by.strip() or issued_by.strip().lower() in {"unknown", "anonymous"}:
        raise ValueError("issued_by must be a real, non-anonymous identity")
    if reason_code not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {REASON_CODES}, got {reason_code!r}")
    if sim_semantics not in SIM_SEMANTICS:
        raise ValueError(f"sim_semantics must be one of {SIM_SEMANTICS} (no realized-cash member), got {sim_semantics!r}")
    if provenance not in PROVENANCE:
        raise ValueError(f"provenance must be one of {PROVENANCE}, got {provenance!r}")
    if not why or not why.strip():
        raise ValueError("why is required (a correction must say why)")
    if not dates_to_values:
        raise ValueError("at least one date->value is required")

    dates = sorted(dates_to_values.keys())
    from_values = from_values or {}
    sets = {
        d: {"from": from_values.get(d), "to": float(dates_to_values[d])}
        for d in dates
    }
    issued_at = issued_at or _utcnow_iso()
    body = {
        "kind": "correction_event",
        "schema": "correction_event.v1",
        "issued_at": issued_at,
        "issued_by": issued_by.strip(),
        "reason_code": reason_code,
        "why": why.strip(),
        "method": method.strip() if method else "",
        "scope": {"model_id": model_id, "dates": dates},
        "sets": sets,
        "supersedes": supersedes,
        "sim_semantics": sim_semantics,
        "provenance": provenance,
    }
    sha = content_sha(body)
    body["content_sha"] = sha
    body["corr_id"] = f"corr:{issued_at}:{sha[:12]}"
    return body


def _leaf_key(event: Dict[str, Any]) -> str:
    issued_date = str(event.get("issued_at", ""))[:10] or "undated"
    return f"{CORRECTIONS_PREFIX}{issued_date}/{event['content_sha'][:16]}.json"


class CorrectionStore:
    """Write-once, append-only correction store on S3 (raw boto3 client).

    Accepts a raw boto3 S3 client (the same object ``extend_dashboard`` already
    receives as ``s3_client``), so the overlay read attaches with no new wiring.
    """

    def __init__(self, s3_client, bucket: str = BUCKET):
        self.s3 = s3_client
        self.bucket = bucket

    # ---- writes (append-only, write-once) --------------------------------
    def append(self, event: Dict[str, Any]) -> str:
        """Write the event leaf write-once (IfNoneMatch) and append the log line.

        Returns the leaf key. Re-appending byte-identical content is a harmless
        no-op; different content can never overwrite an existing leaf.
        """
        if "content_sha" not in event or "corr_id" not in event:
            raise ValueError("event must be built via build_event()")
        key = _leaf_key(event)
        body = json.dumps(event, indent=2, sort_keys=True, allow_nan=False).encode()
        try:
            self.s3.put_object(
                Bucket=self.bucket, Key=key, Body=body,
                ContentType="application/json", IfNoneMatch="*",
            )
        except Exception as e:  # noqa: BLE001
            # PreconditionFailed => the content-addressed leaf already exists
            # (identical correction already recorded). That is write-once doing
            # its job, not an error.
            if "PreconditionFailed" in str(type(e)) or "PreconditionFailed" in str(e):
                logger.info("correction leaf already present (idempotent): %s", key)
            else:
                raise
        # Append to the rolling index (a convenience; resolution can rebuild from
        # the prefix if the log is missing).
        self._append_log({
            "corr_id": event["corr_id"], "issued_at": event["issued_at"],
            "issued_by": event["issued_by"], "model_id": event["scope"]["model_id"],
            "dates": event["scope"]["dates"], "supersedes": event.get("supersedes"),
            "content_sha": event["content_sha"], "key": key,
        })
        return key

    def _append_log(self, row: Dict[str, Any]) -> None:
        try:
            existing = b""
            try:
                existing = self.s3.get_object(Bucket=self.bucket, Key=LOG_KEY)["Body"].read()
            except Exception:  # noqa: BLE001 — missing log is fine
                existing = b""
            line = (json.dumps(row, sort_keys=True) + "\n").encode()
            self.s3.put_object(
                Bucket=self.bucket, Key=LOG_KEY, Body=existing + line,
                ContentType="application/x-ndjson",
            )
        except Exception as e:  # noqa: BLE001 — the log is non-load-bearing
            logger.warning("correction log append failed (non-fatal): %s", e)

    # ---- reads -----------------------------------------------------------
    def list_events(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load all correction events by listing the corrections/ prefix.

        Robust to a missing/partial ``_log.jsonl`` (the leaves are the truth).
        """
        events: List[Dict[str, Any]] = []
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": CORRECTIONS_PREFIX}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                k = obj["Key"]
                if not k.endswith(".json") or k == LOG_KEY:
                    continue
                try:
                    raw = self.s3.get_object(Bucket=self.bucket, Key=k)["Body"].read()
                    ev = json.loads(raw)
                except Exception as e:  # noqa: BLE001
                    logger.warning("skip unreadable correction leaf %s: %s", k, e)
                    continue
                if model_id is None or ev.get("scope", {}).get("model_id") == model_id:
                    events.append(ev)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        events.sort(key=lambda e: (str(e.get("issued_at", "")), str(e.get("corr_id", ""))))
        return events

    def active_overlay(self, model_id: str = CANON_MODEL_ID) -> Dict[str, Dict[str, Any]]:
        """Resolve the supersede-chain head per date.

        Returns ``{date: {value, corr_id, conflict, prior_value}}``. ``conflict`` is
        True when a competing ``supersedes:null`` (or orphan) event landed on a date
        that already had an active chain — the prior head value is retained and the
        flag surfaced (never a silent pick).
        """
        events = self.list_events(model_id)
        return resolve_overlay(events)


def resolve_overlay(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pure resolver (no I/O) — supersede-chain head per date + conflict marker.

    ``events`` must be sorted by issued_at (oldest first).
    """
    # head_by_date: date -> {"event": ev, "conflict": bool}
    head: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        corr_id = ev.get("corr_id")
        supersedes = ev.get("supersedes")
        sets = ev.get("sets", {}) or {}
        for d in ev.get("scope", {}).get("dates", []) or []:
            if d not in sets:
                continue
            cur = head.get(d)
            if cur is None:
                # First event for this date.
                if supersedes is None:
                    head[d] = {"event": ev, "conflict": False}
                else:
                    # Supersedes something we've never seen for this date -> orphan.
                    head[d] = {"event": ev, "conflict": True}
            else:
                cur_id = cur["event"].get("corr_id")
                if supersedes == cur_id:
                    # Valid supersession: new head, clears prior conflict.
                    head[d] = {"event": ev, "conflict": False}
                else:
                    # supersedes:null onto an active chain, or supersedes a non-head
                    # -> conflict. Keep the prior head value; surface the flag.
                    cur["conflict"] = True
    overlay: Dict[str, Dict[str, Any]] = {}
    for d, h in head.items():
        ev = h["event"]
        s = ev["sets"][d]
        overlay[d] = {
            "value": float(s["to"]),
            "prior_value": s.get("from"),
            "corr_id": ev.get("corr_id"),
            "conflict": bool(h["conflict"]),
        }
    return overlay


# ---- the extender hook ---------------------------------------------------
def _recompute_drawdowns(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    peak = None
    dd: List[Dict[str, Any]] = []
    for row in equity_curve:
        v = row.get("value")
        if v is None:
            dd.append({"date": row["date"], "drawdown": 0.0})
            continue
        peak = v if peak is None else max(peak, v)
        dd.append({"date": row["date"], "drawdown": (v / peak - 1.0) if peak else 0.0})
    return dd


def apply_correction_overlay(dash: Dict[str, Any], overlay: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply a resolved overlay to a dashboard dict IN PLACE (pure given overlay).

    Sets the canon value fields on each corrected date, recomputes drawdowns +
    hero metrics from the corrected primary line, and stamps
    ``timeline_correction.overlay``. Idempotent: applying the same overlay twice
    yields the same dashboard. A no-op when the overlay is empty.
    """
    if not overlay:
        return dash
    equity_curve = dash.get("equity_curve", []) or []
    applied, conflicts = [], []
    for row in equity_curve:
        d = row.get("date")
        ov = overlay.get(d)
        if ov is None:
            continue
        cv = ov["value"]
        for fld in CANON_VALUE_FIELDS:
            if row.get(fld) is not None:
                row[fld] = cv
        row["correction"] = {"corr_id": ov["corr_id"], "conflict": ov["conflict"]}
        applied.append(ov["corr_id"])
        if ov["conflict"]:
            conflicts.append({"date": d, "corr_id": ov["corr_id"]})

    # Recompute drawdowns + hero metrics from the corrected primary line.
    dash["drawdowns"] = _recompute_drawdowns(equity_curve)
    valued = [r for r in equity_curve if r.get("value") is not None]
    if valued:
        today = valued[-1]["date"]
        total_value = valued[-1]["value"]

        def find_v(target):
            for r in valued:
                if r["date"] >= target:
                    return r["value"]
            return None
        ytd_base = find_v(f"{today[:4]}-01-01")
        mtd_base = find_v(f"{today[:7]}-01")
        m = dash.setdefault("metrics", {})
        m["total_value"] = total_value
        m["corrected_total_value"] = total_value
        m["actual_total_value"] = total_value
        if ytd_base and ytd_base > 0:
            m["ytd_return"] = total_value / ytd_base - 1.0
        if mtd_base and mtd_base > 0:
            m["mtd_return"] = total_value / mtd_base - 1.0
        dd_vals = [p["drawdown"] for p in dash["drawdowns"]]
        m["max_drawdown"] = min(dd_vals, default=0.0)
        m["current_drawdown"] = dash["drawdowns"][-1]["drawdown"] if dash["drawdowns"] else 0.0

    tc = dash.setdefault("timeline_correction", {})
    tc["overlay"] = {
        "applied_corr_ids": sorted(set(applied)),
        "n_dates": len(applied),
        "conflicts": conflicts,
        "source": "corrections/ append-only overlay (PKT-TB-IR-01)",
        "note": ("Corrections are append-only attributed inputs read every "
                 "regenerate; they cannot be silently reverted. Paper re-sim, "
                 "never realized cash."),
    }
    return dash


def apply_active_overlay(s3_client, dash: Dict[str, Any], model_id: str = CANON_MODEL_ID) -> Dict[str, Any]:
    """Load the active overlay from S3 and apply it to ``dash`` in place.

    Defensive: any failure logs and returns ``dash`` unchanged (mirrors the rest
    of ``extend_dashboard``), so a corrections-store hiccup degrades to the raw
    re-anchored line rather than crashing the publish.
    """
    try:
        store = CorrectionStore(s3_client)
        overlay = store.active_overlay(model_id)
        return apply_correction_overlay(dash, overlay)
    except Exception as exc:  # noqa: BLE001
        logger.warning("correction overlay skipped (non-fatal): %s", exc)
        return dash
