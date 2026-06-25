"""Canonical, append-only, content-addressed EQUITY LEDGER (FP-08-1).

The single persisted artifact the clean-core rebuild stands on. The displayed
equity line stops being a recomputed view and becomes a **stored fact**: one
write-once leaf per trading day, appended at the frontier, never recomputed. The
old failure class ("history moves every other day") is structurally impossible
here because nothing recomputes the line — a settled leaf is immutable and the
append path is incapable of rewriting it.

Layout (S3 ``investment-system-data``, versioning OFF — durability is
content-addressing + write-once + immutable leaves, NOT versioning):

  canon/equity_ledger/points/<date>/<sha16>.json  -- write-once per-day leaves
  canon/equity_ledger/_manifest.json              -- ordered (date -> content_sha) chain
  canon/equity_ledger/equity_history.jsonl        -- chart-consumed CACHE (a pure fold)

This module reuses the write-once / content-addressing primitives already tested
in ``src/utils/corrections.py`` (``content_sha`` / ``_canonical_bytes`` /
``_utcnow_iso`` + the ``IfNoneMatch="*"`` put pattern) rather than reinventing them.

Two binding guards from the architecture dossier (§2, §5) are enforced + tested,
not asserted in a comment:

  * **G-APPEND-ONLY-FRONTIER** — an append can only extend the frontier. A
    write of different content to an already-settled date is rejected; a write of
    a date at or before the frontier is rejected at the frontier gate. The append
    path is structurally incapable of rewriting a settled leaf or manifest entry.
  * **G-CACHE-PROJECTION** — ``equity_history.jsonl`` is a PURE deterministic fold
    over the leaves+manifest. It is never written independently; regenerating it
    yields byte-identical output, so it can never become an independent float.

Scope note (FP-08-1): this builds the primitive in isolation with unit tests. It
does NOT seed from production, wire into any live render path, or delete existing
code (FP-08-2/-3/-4). Supersede-chain resolution for first-class corrections
lands in FP-08-5; restore here walks the prev-hash frontier chain (no corrections
exist yet).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

# Reuse the already-tested write-once / content-addressing primitives.
from src.utils.corrections import _canonical_bytes, _utcnow_iso, content_sha

logger = logging.getLogger(__name__)

BUCKET = "investment-system-data"
LEDGER_PREFIX = "canon/equity_ledger/"
POINTS_PREFIX = LEDGER_PREFIX + "points/"
MANIFEST_KEY = LEDGER_PREFIX + "_manifest.json"
CACHE_KEY = LEDGER_PREFIX + "equity_history.jsonl"

LEAF_SCHEMA = "equity_point.v1"
MANIFEST_SCHEMA = "equity_manifest.v1"

# The three displayed lines (dossier §2) + the segment a leaf belongs to.
SEGMENTS = ("frozen_champion", "new_brain", "incumbent")

# Fields that define a leaf's CONTENT identity → content_sha. ``written_at`` (the
# wall-clock of the write event) and ``content_sha`` itself are excluded so the
# same equity fact reduces to the same content-addressed key, which is exactly
# what makes an identical re-append an idempotent no-op under ``IfNoneMatch="*"``.
_HASHED_FIELDS = (
    "schema", "date", "value", "benchmark", "comparison", "segment",
    "model_id", "source", "prev_date", "prev_content_hash", "issued_by",
    "supersedes",
)

# The chart-consumed columns the cache fold projects out of each leaf.
_CACHE_FIELDS = ("date", "value", "benchmark", "comparison", "segment", "model_id")


class LedgerError(Exception):
    """Base for ledger-append violations."""


class FrontierViolation(LedgerError):
    """An append whose date is at or before the frontier (out-of-order / non-frontier)."""


class AppendOnlyViolation(LedgerError):
    """An append of DIFFERENT content to an already-settled date (G-APPEND-ONLY-FRONTIER)."""


class LedgerIntegrityError(LedgerError):
    """The on-S3 leaves do not form a single valid hash chain (restore-from-facts failed)."""


# --------------------------------------------------------------------------- #
# leaf construction + content-addressing
# --------------------------------------------------------------------------- #
def _content_sha_of(leaf: Dict[str, Any]) -> str:
    """content_sha over the identity fields only (excludes written_at / content_sha)."""
    payload = {k: leaf[k] for k in _HASHED_FIELDS}
    return content_sha(payload)


def build_leaf(
    *,
    date: str,
    value: float,
    benchmark: float,
    comparison: Optional[float],
    segment: str,
    model_id: str,
    source: str,
    issued_by: str,
    prev_date: Optional[str] = None,
    prev_content_hash: Optional[str] = None,
    supersedes: Optional[str] = None,
    written_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a fully-formed, content-addressed ``equity_point.v1`` leaf (not stored).

    ``value`` and ``benchmark`` are required and must be finite. ``comparison``
    (the dotted tilt_adapter shadow line) is OPTIONAL — it only exists for the
    recent shadow window, so leaves before the comparison series begins carry
    ``comparison=None`` (a JSON null, not NaN). Any provided numeric column must be
    finite (``allow_nan=False`` in the canonical bytes rejects NaN/inf). The chain
    links ``prev_date``/``prev_content_hash`` point at the prior frontier leaf,
    making the ledger a tamper-evident chain of record.
    """
    if not issued_by or not issued_by.strip() or issued_by.strip().lower() in {"unknown", "anonymous"}:
        raise ValueError("issued_by must be a real, non-anonymous identity")
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}, got {segment!r}")
    if not source or not source.strip():
        raise ValueError("source is required")
    if not date or len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")

    leaf = {
        "schema": LEAF_SCHEMA,
        "date": date,
        "value": float(value),
        "benchmark": float(benchmark),
        "comparison": float(comparison) if comparison is not None else None,
        "segment": segment,
        "model_id": model_id,
        "source": source,
        "prev_date": prev_date,
        "prev_content_hash": prev_content_hash,
        "issued_by": issued_by.strip(),
        "supersedes": supersedes,
    }
    # Force the canonical-bytes path now so NaN/inf is rejected eagerly with a
    # clear error rather than at put time.
    _canonical_bytes({k: leaf[k] for k in _HASHED_FIELDS})
    leaf["content_sha"] = _content_sha_of(leaf)
    leaf["written_at"] = written_at or _utcnow_iso()
    return leaf


def _leaf_key(leaf: Dict[str, Any]) -> str:
    return f"{POINTS_PREFIX}{leaf['date']}/{leaf['content_sha'][:16]}.json"


# --------------------------------------------------------------------------- #
# the cache fold — a PURE deterministic projection (G-CACHE-PROJECTION)
# --------------------------------------------------------------------------- #
def fold_cache(ordered_leaves: List[Dict[str, Any]]) -> bytes:
    """Deterministically fold an ordered leaf chain into ``equity_history.jsonl``.

    Pure function of the leaves: no wall-clock, no I/O. One JSON object per line,
    canonically serialized (sorted keys, no NaN), in chain order. Regenerating
    from the same leaves yields byte-identical output — the cache can never
    diverge from the leaves it projects (G-CACHE-PROJECTION).
    """
    lines = []
    for leaf in ordered_leaves:
        row = {k: leaf[k] for k in _CACHE_FIELDS}
        lines.append(_canonical_bytes(row).decode())
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


# --------------------------------------------------------------------------- #
# the ledger store
# --------------------------------------------------------------------------- #
class EquityLedger:
    """Append-only, content-addressed equity ledger on S3 (raw boto3 client).

    Accepts the same raw boto3 S3 client the rest of the publish path already
    holds, so the ledger attaches with no new wiring.
    """

    def __init__(self, s3_client, bucket: str = BUCKET):
        self.s3 = s3_client
        self.bucket = bucket

    # ---- low-level S3 helpers ------------------------------------------- #
    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception:  # noqa: BLE001 — missing key is a normal "not yet" state
            return None
        return json.loads(raw)

    def _put_leaf_write_once(self, leaf: Dict[str, Any]) -> bool:
        """Write a leaf with ``IfNoneMatch="*"``. Returns True if newly written,
        False if the content-addressed key already existed (idempotent no-op)."""
        key = _leaf_key(leaf)
        body = json.dumps(leaf, indent=2, sort_keys=True, allow_nan=False).encode()
        try:
            self.s3.put_object(
                Bucket=self.bucket, Key=key, Body=body,
                ContentType="application/json", IfNoneMatch="*",
            )
            return True
        except Exception as e:  # noqa: BLE001
            # PreconditionFailed => the content-addressed leaf already exists with
            # identical content. Write-once doing its job, not an error.
            if "PreconditionFailed" in str(type(e)) or "PreconditionFailed" in str(e):
                logger.info("equity leaf already present (idempotent no-op): %s", key)
                return False
            raise

    def _put_manifest(self, manifest: Dict[str, Any]) -> None:
        body = _canonical_bytes(manifest)
        self.s3.put_object(
            Bucket=self.bucket, Key=MANIFEST_KEY, Body=body,
            ContentType="application/json",
        )

    def _refold_cache(self) -> bytes:
        """Re-derive the cache from the leaves (the ONLY writer of CACHE_KEY)."""
        leaves = self._ordered_chain_from_leaves()
        body = fold_cache(leaves)
        self.s3.put_object(
            Bucket=self.bucket, Key=CACHE_KEY, Body=body,
            ContentType="application/x-ndjson",
        )
        return body

    # ---- reads ---------------------------------------------------------- #
    def read_manifest(self) -> Dict[str, Any]:
        m = self._get_json(MANIFEST_KEY)
        if m is None:
            return {"schema": MANIFEST_SCHEMA, "entries": [], "frontier": None}
        return m

    def read_cache(self) -> bytes:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=CACHE_KEY)["Body"].read()
        except Exception:  # noqa: BLE001
            return b""

    def frontier(self) -> Optional[Dict[str, Any]]:
        return self.read_manifest().get("frontier")

    def list_leaves(self) -> List[Dict[str, Any]]:
        """Load every leaf by listing the points/ prefix (the leaves are the truth)."""
        leaves: List[Dict[str, Any]] = []
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": POINTS_PREFIX}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                k = obj["Key"]
                if not k.endswith(".json"):
                    continue
                try:
                    raw = self.s3.get_object(Bucket=self.bucket, Key=k)["Body"].read()
                    leaves.append(json.loads(raw))
                except Exception as e:  # noqa: BLE001
                    logger.warning("skip unreadable equity leaf %s: %s", k, e)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return leaves

    def get_leaf(self, content_sha_value: str) -> Optional[Dict[str, Any]]:
        for leaf in self.list_leaves():
            if leaf.get("content_sha") == content_sha_value:
                return leaf
        return None

    # ---- restore-from-facts -------------------------------------------- #
    def _ordered_chain_from_leaves(self) -> List[Dict[str, Any]]:
        """Reconstruct the canonical chain from the leaves, resolving supersedes.

        The leaves ARE the truth. Each date's displayed value is the HEAD of its
        supersede chain — the active leaf no other leaf supersedes (FP-08-5). The
        original value is retained as a superseded leaf (still on S3), so a
        correction sticks and the old value is never lost. The chain is the active
        leaf per date, in date order; integrity (content_sha recompute, one active
        leaf per date, supersede targets present, strictly increasing dates) is
        verified or it raises rather than guessing.
        """
        leaves = self.list_leaves()
        if not leaves:
            return []
        by_sha: Dict[str, Dict[str, Any]] = {}
        for leaf in leaves:
            # integrity: recompute each leaf's content_sha from its identity fields
            if _content_sha_of(leaf) != leaf["content_sha"]:
                raise LedgerIntegrityError(
                    f"leaf content_sha mismatch (tampered): {leaf['content_sha']}")
            by_sha[leaf["content_sha"]] = leaf

        superseded = {leaf["supersedes"] for leaf in leaves if leaf.get("supersedes")}
        for s in superseded:
            if s not in by_sha:
                raise LedgerIntegrityError(f"leaf supersedes unknown content {s}")

        # The active leaf for each date is the one nothing supersedes.
        by_date: Dict[str, Dict[str, Any]] = {}
        for leaf in leaves:
            if leaf["content_sha"] in superseded:
                continue  # an old value retained behind a correction
            d = leaf["date"]
            if d in by_date:
                raise LedgerIntegrityError(
                    f"two active (non-superseded) leaves for {d} — ambiguous head")
            by_date[d] = leaf

        chain = [by_date[d] for d in sorted(by_date)]
        for a, b in zip(chain, chain[1:]):
            if not (a["date"] < b["date"]):
                raise LedgerIntegrityError(
                    f"non-increasing dates in chain: {a['date']} !< {b['date']}")
        return chain

    def _manifest_from_chain(self, chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        entries = [
            {"date": leaf["date"], "content_sha": leaf["content_sha"], "key": _leaf_key(leaf)}
            for leaf in chain
        ]
        frontier = (
            {"date": chain[-1]["date"], "content_sha": chain[-1]["content_sha"]}
            if chain else None
        )
        return {"schema": MANIFEST_SCHEMA, "entries": entries, "frontier": frontier}

    def rebuild(self, *, write: bool = True) -> Dict[str, Any]:
        """Restore the manifest + cache by listing the points/ prefix.

        The leaves are the source of truth; the manifest and cache are pure
        projections over them. Returns the rebuilt manifest. When ``write`` is
        True the rebuilt manifest + cache are persisted (restore-from-facts).
        """
        chain = self._ordered_chain_from_leaves()
        manifest = self._manifest_from_chain(chain)
        if write:
            self._put_manifest(manifest)
            self.s3.put_object(
                Bucket=self.bucket, Key=CACHE_KEY, Body=fold_cache(chain),
                ContentType="application/x-ndjson",
            )
        return manifest

    # ---- the append (the only mutation) -------------------------------- #
    def append(
        self,
        *,
        date: str,
        value: float,
        benchmark: float,
        comparison: float,
        segment: str,
        model_id: str,
        source: str,
        issued_by: str,
        supersedes: Optional[str] = None,
        written_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append one frontier leaf, advance the manifest, re-fold the cache.

        The frontier gate (G-APPEND-ONLY-FRONTIER):
          * date > frontier  -> a new frontier leaf is written + the chain advances.
          * date == frontier -> idempotent no-op iff the content is identical;
                                a different-content write to the settled date is rejected.
          * date <  frontier -> rejected (out-of-order / non-frontier append).
        Genesis (empty ledger) accepts any first date.
        """
        manifest = self.read_manifest()
        entries: List[Dict[str, Any]] = manifest.get("entries", []) or []
        front = manifest.get("frontier")

        if front is not None and date < front["date"]:
            raise FrontierViolation(
                f"append date {date} <= frontier {front['date']}: ledger is append-only at the frontier"
            )

        if front is not None and date == front["date"]:
            # Re-append at the settled frontier date. Reconstruct the leaf with the
            # SAME chain links the settled leaf used (the entry before the frontier)
            # so identical content reproduces the same content_sha.
            prev = entries[-2] if len(entries) >= 2 else None
            candidate = build_leaf(
                date=date, value=value, benchmark=benchmark, comparison=comparison,
                segment=segment, model_id=model_id, source=source, issued_by=issued_by,
                prev_date=(prev["date"] if prev else None),
                prev_content_hash=(prev["content_sha"] if prev else None),
                supersedes=supersedes, written_at=written_at,
            )
            if candidate["content_sha"] == front["content_sha"]:
                # Identical content → idempotent no-op (write-once already holds it).
                self._put_leaf_write_once(candidate)
                logger.info("idempotent re-append of settled date %s", date)
                return candidate
            raise AppendOnlyViolation(
                f"date {date} is already settled with content {front['content_sha'][:16]}; "
                f"a different value cannot rewrite history (G-APPEND-ONLY-FRONTIER)"
            )

        # Normal frontier append (date > frontier, or genesis).
        leaf = build_leaf(
            date=date, value=value, benchmark=benchmark, comparison=comparison,
            segment=segment, model_id=model_id, source=source, issued_by=issued_by,
            prev_date=(front["date"] if front else None),
            prev_content_hash=(front["content_sha"] if front else None),
            supersedes=supersedes, written_at=written_at,
        )
        self._put_leaf_write_once(leaf)
        entries.append(
            {"date": leaf["date"], "content_sha": leaf["content_sha"], "key": _leaf_key(leaf)}
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "entries": entries,
            "frontier": {"date": leaf["date"], "content_sha": leaf["content_sha"]},
        }
        self._put_manifest(manifest)
        self._refold_cache()
        return leaf

    # ---- first-class correction (FP-08-5) ------------------------------ #
    def correct(
        self,
        *,
        date: str,
        value: float,
        benchmark: float,
        comparison: Optional[float],
        issued_by: str,
        why: str,
        reason_code: str,
        model_id: Optional[str] = None,
        source: str = "correction",
        written_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Correct a settled date's displayed value with a NEW attributed leaf.

        A correction is first-class and durable (FP-08-5, history-protection §5): it
        is a NEW immutable supersede-leaf at the corrected date carrying the new
        value + ``issued_by`` (anonymous rejected) + ``why`` + ``reason_code`` +
        ``supersedes:<prior head>``. The prior value is RETAINED as a superseded
        leaf (still on S3), never deleted. Because the render reads the ledger (an
        immutable input), the correction sticks across a night regenerate — it can
        never be silently reverted. The chain links match the corrected date's
        position so the rest of the line is undisturbed.
        """
        if not why or not why.strip():
            raise ValueError("a correction must carry a non-empty why")
        manifest = self.read_manifest()
        entry = next((e for e in (manifest.get("entries") or []) if e["date"] == date), None)
        if entry is None:
            raise LedgerError(f"cannot correct {date}: not present in the ledger")
        active = by_sha = None
        for leaf in self.list_leaves():
            if leaf["content_sha"] == entry["content_sha"]:
                active = leaf
                break
        if active is None:
            raise LedgerIntegrityError(f"active leaf for {date} not found on S3")

        leaf = build_leaf(
            date=date, value=value, benchmark=benchmark, comparison=comparison,
            segment=active["segment"], model_id=model_id or active["model_id"],
            source=source, issued_by=issued_by,
            prev_date=active.get("prev_date"),
            prev_content_hash=active.get("prev_content_hash"),
            supersedes=active["content_sha"], written_at=written_at,
        )
        # Attribution carried as NON-hashed metadata (does not change content_sha,
        # so it never perturbs existing leaves' identities).
        leaf["why"] = why.strip()
        leaf["reason_code"] = reason_code
        self._put_leaf_write_once(leaf)
        # Rebuild manifest + cache from the leaves (supersede-aware) so the
        # correction becomes head-of-chain for its date.
        self.rebuild(write=True)
        return leaf
