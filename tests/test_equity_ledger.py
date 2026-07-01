"""FP-08-1 — canonical equity ledger primitive (unit suite).

Covers every Acceptance-test item from PKT-TRADER-BOT-FP08-1, especially the two
binding guards:

  G-APPEND-ONLY-FRONTIER  the append path is structurally incapable of rewriting a
                          settled leaf or manifest entry — a different-content write
                          to a settled date is rejected, and a date at/before the
                          frontier is rejected at the frontier gate.
  G-CACHE-PROJECTION      equity_history.jsonl is a PURE deterministic fold over the
                          leaves; regenerating it is byte-identical and it is never
                          written independently of the leaves.

Runs against an in-memory FakeS3 with IfNoneMatch write-once (the same stub shape
test_correction_overlay uses) — no real S3, no production data.
"""
import json

import pytest

from src.canon.equity_ledger import (
    CACHE_KEY,
    MANIFEST_KEY,
    POINTS_PREFIX,
    AppendOnlyViolation,
    EquityLedger,
    FrontierViolation,
    LedgerIntegrityError,
    build_leaf,
    fold_cache,
)


# --------------------------------------------------------------------------- #
# in-memory boto3-S3 stub with IfNoneMatch write-once + put accounting
# --------------------------------------------------------------------------- #
class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class FakeS3:
    def __init__(self):
        self.store = {}
        self.put_calls = []          # every put_object Key (for no-op accounting)
        self.precondition_rejects = 0

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch == "*" and Key in self.store:
            self.precondition_rejects += 1
            raise Exception("PreconditionFailed: key exists (IfNoneMatch=*)")
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.put_calls.append(Key)
        return {}

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise Exception("NoSuchKey")
        return {"Body": _Body(self.store[Key])}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _append(ledger, date, value, *, benchmark=None, comparison=None,
            segment="new_brain", model_id="FREEZE_ORB1@abc123",
            source="native_two_stage", issued_by="op@example.com", **kw):
    return ledger.append(
        date=date, value=value,
        benchmark=value * 0.9 if benchmark is None else benchmark,
        comparison=value * 0.8 if comparison is None else comparison,
        segment=segment, model_id=model_id, source=source, issued_by=issued_by, **kw,
    )


def _seed_three(ledger):
    _append(ledger, "2026-06-21", 100000.0)
    _append(ledger, "2026-06-22", 110000.0)
    return _append(ledger, "2026-06-23", 117873.57)


# --------------------------------------------------------------------------- #
# Acceptance #1 — an append writes a frontier leaf + advances the manifest
# --------------------------------------------------------------------------- #
def test_append_writes_frontier_leaf_and_advances_manifest():
    s3 = FakeS3()
    led = EquityLedger(s3)
    leaf = _append(led, "2026-06-21", 100000.0)

    expected_key = f"{POINTS_PREFIX}2026-06-21/{leaf['content_sha'][:16]}.json"
    assert expected_key in s3.store, "leaf written at points/<date>/<sha16>.json"

    manifest = led.read_manifest()
    assert manifest["frontier"] == {"date": "2026-06-21", "content_sha": leaf["content_sha"]}
    assert manifest["entries"] == [
        {"date": "2026-06-21", "content_sha": leaf["content_sha"], "key": expected_key}
    ]

    # second day advances the frontier and chains to the prior leaf
    leaf2 = _append(led, "2026-06-22", 110000.0)
    assert leaf2["prev_date"] == "2026-06-21"
    assert leaf2["prev_content_hash"] == leaf["content_sha"]
    m2 = led.read_manifest()
    assert m2["frontier"]["date"] == "2026-06-22"
    assert [e["date"] for e in m2["entries"]] == ["2026-06-21", "2026-06-22"]


# --------------------------------------------------------------------------- #
# Acceptance #2 — idempotent same-content no-op; different content rejected
# (G-APPEND-ONLY-FRONTIER)
# --------------------------------------------------------------------------- #
def test_same_date_identical_content_is_idempotent_noop():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _append(led, "2026-06-21", 100000.0)
    leaf = _append(led, "2026-06-22", 110000.0)

    puts_before = len(s3.put_calls)
    manifest_before = led.read_manifest()

    # re-append the SAME date with identical content (even a different written_at)
    again = _append(led, "2026-06-22", 110000.0, written_at="2099-01-01T00:00:00+00:00")
    assert again["content_sha"] == leaf["content_sha"]
    # the content-addressed leaf already exists → IfNoneMatch rejected the re-put
    assert s3.precondition_rejects == 1
    # no NEW manifest/cache writes for a no-op (only the rejected leaf put was attempted)
    assert led.read_manifest() == manifest_before
    assert MANIFEST_KEY not in s3.put_calls[puts_before:]
    assert CACHE_KEY not in s3.put_calls[puts_before:]


def test_different_content_to_settled_date_is_rejected():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _append(led, "2026-06-21", 100000.0)
    _append(led, "2026-06-22", 110000.0)

    manifest_before = led.read_manifest()
    with pytest.raises(AppendOnlyViolation):
        _append(led, "2026-06-22", 999999.99)  # rewrite of a settled date
    # history untouched: no leaf added, manifest unchanged
    assert led.read_manifest() == manifest_before


# --------------------------------------------------------------------------- #
# Acceptance #3 — out-of-order / non-frontier append rejected at the frontier gate
# --------------------------------------------------------------------------- #
def test_out_of_order_append_rejected_at_frontier_gate():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _append(led, "2026-06-21", 100000.0)
    _append(led, "2026-06-23", 117873.57)  # frontier = 2026-06-23

    manifest_before = led.read_manifest()
    with pytest.raises(FrontierViolation):
        _append(led, "2026-06-22", 111111.11)  # date < frontier
    with pytest.raises(FrontierViolation):
        _append(led, "2026-06-21", 100000.0)   # date < frontier even if identical to an old leaf
    assert led.read_manifest() == manifest_before


# --------------------------------------------------------------------------- #
# Acceptance #4 — rebuild from the points/ prefix reproduces the exact chain
# --------------------------------------------------------------------------- #
def test_rebuild_from_leaves_reproduces_exact_chain():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)

    manifest_live = led.read_manifest()
    cache_live = led.read_cache()

    # blow away the projections; the leaves remain the truth
    del s3.store[MANIFEST_KEY]
    del s3.store[CACHE_KEY]

    rebuilt = led.rebuild(write=True)
    assert rebuilt == manifest_live, "restore-from-facts reproduces the exact manifest chain"
    assert led.read_cache() == cache_live, "restore-from-facts reproduces the exact cache"


def test_rebuild_detects_a_tampered_leaf():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)
    # corrupt one leaf's value without fixing its content_sha → chain integrity fails
    key = next(k for k in s3.store if k.startswith(POINTS_PREFIX))
    leaf = json.loads(s3.store[key])
    leaf["value"] = leaf["value"] + 1.0
    s3.store[key] = json.dumps(leaf).encode()
    with pytest.raises(LedgerIntegrityError):
        led.rebuild(write=False)


# --------------------------------------------------------------------------- #
# Acceptance #5 — the cache is a PURE deterministic fold (G-CACHE-PROJECTION)
# --------------------------------------------------------------------------- #
def test_cache_is_a_pure_deterministic_fold():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)

    chain = led._ordered_chain_from_leaves()
    folded = fold_cache(chain)

    # the persisted cache equals the fold over the leaves...
    assert led.read_cache() == folded
    # ...and re-folding is byte-identical (deterministic, no wall-clock)
    assert fold_cache(chain) == folded
    # the fold is a projection of the chart columns, in chain (date) order
    rows = [json.loads(line) for line in folded.decode().splitlines()]
    assert [r["date"] for r in rows] == ["2026-06-21", "2026-06-22", "2026-06-23"]
    assert set(rows[0].keys()) == {"date", "value", "benchmark", "comparison", "segment", "model_id"}
    assert rows[-1]["value"] == 117873.57


def test_cache_never_diverges_from_leaves_across_appends():
    s3 = FakeS3()
    led = EquityLedger(s3)
    for d, v in [("2026-06-21", 100000.0), ("2026-06-22", 110000.0), ("2026-06-23", 117873.57)]:
        _append(led, d, v)
        # invariant after every append: cache == fold over the current leaf chain
        assert led.read_cache() == fold_cache(led._ordered_chain_from_leaves())


# --------------------------------------------------------------------------- #
# Acceptance #6 — each leaf carries the full dossier §2 schema
# --------------------------------------------------------------------------- #
def test_leaf_carries_full_schema():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _append(led, "2026-06-21", 100000.0)
    leaf = _append(led, "2026-06-22", 110000.0, benchmark=105000.0, comparison=102000.0)

    for field in ("value", "benchmark", "comparison", "segment", "model_id", "source",
                  "prev_date", "prev_content_hash", "content_sha", "issued_by",
                  "written_at", "supersedes", "schema", "date"):
        assert field in leaf, f"leaf missing required field {field!r}"
    assert leaf["schema"] == "equity_point.v1"
    assert leaf["value"] == 110000.0
    assert leaf["benchmark"] == 105000.0
    assert leaf["comparison"] == 102000.0
    assert leaf["segment"] == "new_brain"
    assert leaf["prev_date"] == "2026-06-21"
    assert leaf["prev_content_hash"] is not None
    assert leaf["supersedes"] is None
    assert len(leaf["content_sha"]) == 64


# --------------------------------------------------------------------------- #
# build_leaf validation + content-addressing
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# FP-08-5 — first-class corrections (attributed supersede-leaves that STICK)
# --------------------------------------------------------------------------- #
def test_correction_supersedes_and_sticks_across_rebuild():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)  # 06-21,06-22,06-23

    before_value = led.read_manifest()  # frontier 06-23
    # correct 06-22 to a new value
    corr = led.correct(
        date="2026-06-22", value=111111.11, benchmark=99000.0, comparison=None,
        issued_by="op@example.com", why="settled-price fix", reason_code="RESIM_SEGMENT",
    )
    rows = [json.loads(line) for line in led.read_cache().decode().splitlines()]
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-06-22"]["value"] == 111111.11, "correction is head-of-chain"
    # old value retained as a superseded leaf (still on S3)
    leaves = led.list_leaves()
    vals_0622 = sorted(l["value"] for l in leaves if l["date"] == "2026-06-22")
    assert vals_0622 == [110000.0, 111111.11], "old value retained, not deleted"
    # the correction sticks across a full restore-from-facts rebuild
    led.rebuild(write=True)
    rows2 = [json.loads(line) for line in led.read_cache().decode().splitlines()]
    assert {r["date"]: r["value"] for r in rows2}["2026-06-22"] == 111111.11
    # frontier/terminal unchanged (06-23 still last)
    assert led.frontier()["date"] == "2026-06-23"
    # attribution present on the correction leaf
    assert corr["why"] == "settled-price fix" and corr["reason_code"] == "RESIM_SEGMENT"


def test_correction_requires_identity_and_reason():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)
    with pytest.raises(ValueError):
        led.correct(date="2026-06-22", value=1.0, benchmark=1.0, comparison=None,
                    issued_by="anonymous", why="x", reason_code="SEAM")
    with pytest.raises(ValueError):
        led.correct(date="2026-06-22", value=1.0, benchmark=1.0, comparison=None,
                    issued_by="op@example.com", why="", reason_code="SEAM")


def test_build_leaf_rejects_anonymous_issuer_and_bad_segment():
    base = dict(date="2026-06-21", value=1.0, benchmark=1.0, comparison=1.0,
                model_id="m", source="native_two_stage")
    for bad in ["", "  ", "unknown", "anonymous"]:
        with pytest.raises(ValueError):
            build_leaf(segment="new_brain", issued_by=bad, **base)
    with pytest.raises(ValueError):
        build_leaf(segment="not_a_segment", issued_by="op@example.com", **base)


def test_build_leaf_rejects_non_finite_values():
    with pytest.raises(ValueError):
        build_leaf(date="2026-06-21", value=float("nan"), benchmark=1.0, comparison=1.0,
                   segment="new_brain", model_id="m", source="native_two_stage",
                   issued_by="op@example.com")


def test_content_addressing_excludes_written_at():
    # identical content but different written_at → SAME content_sha (so identical
    # facts reduce to the same key, which is what makes re-append idempotent)
    a = build_leaf(date="2026-06-21", value=1.0, benchmark=2.0, comparison=3.0,
                   segment="new_brain", model_id="m", source="s", issued_by="op@example.com",
                   written_at="2026-01-01T00:00:00+00:00")
    b = build_leaf(date="2026-06-21", value=1.0, benchmark=2.0, comparison=3.0,
                   segment="new_brain", model_id="m", source="s", issued_by="op@example.com",
                   written_at="2099-12-31T23:59:59+00:00")
    assert a["content_sha"] == b["content_sha"]
    # but a different value → different content_sha
    c = build_leaf(date="2026-06-21", value=1.5, benchmark=2.0, comparison=3.0,
                   segment="new_brain", model_id="m", source="s", issued_by="op@example.com")
    assert c["content_sha"] != a["content_sha"]


# --------------------------------------------------------------------------- #
# Regression — the Lambda's old boto3 lacks S3 IfNoneMatch; the append MUST
# still advance the line (root cause of "line frozen / didn't run more than one
# day in a row", 2026-06-26: ParamValidationError on IfNoneMatch was swallowed
# as "line holds", so the frontier never moved).
# --------------------------------------------------------------------------- #
class FakeS3OldSdk(FakeS3):
    """A boto3 too old to know the S3 ``IfNoneMatch`` conditional-write param —
    it raises a ParamValidationError before writing, exactly like the Lambda."""

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch is not None:
            raise Exception(
                "ParamValidationError: Parameter validation failed: "
                'Unknown parameter in input: "IfNoneMatch"'
            )
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.put_calls.append(Key)
        return {}

    def head_object(self, Bucket, Key):
        if Key not in self.store:
            raise Exception("NoSuchKey (404)")
        return {"ContentLength": len(self.store[Key])}


def test_append_advances_line_on_old_sdk_without_ifnonematch():
    s3 = FakeS3OldSdk()
    led = EquityLedger(s3)

    leaf = _append(led, "2026-06-21", 100000.0)
    expected_key = f"{POINTS_PREFIX}2026-06-21/{leaf['content_sha'][:16]}.json"
    assert expected_key in s3.store, "leaf written even though IfNoneMatch is unsupported"

    # the frontier ADVANCES across consecutive days (the whole point)
    _append(led, "2026-06-22", 110000.0)
    leaf3 = _append(led, "2026-06-23", 117873.57)
    m = led.read_manifest()
    assert m["frontier"]["date"] == "2026-06-23"
    assert [e["date"] for e in m["entries"]] == ["2026-06-21", "2026-06-22", "2026-06-23"]
    assert leaf3["prev_date"] == "2026-06-22"


def test_old_sdk_same_content_reappend_is_noop_via_head():
    s3 = FakeS3OldSdk()
    led = EquityLedger(s3)
    _append(led, "2026-06-21", 100000.0)
    leaf = _append(led, "2026-06-22", 110000.0)

    puts_before = len(s3.put_calls)
    manifest_before = led.read_manifest()

    # re-append identical content → HEAD finds the leaf → idempotent no-op
    again = _append(led, "2026-06-22", 110000.0, written_at="2099-01-01T00:00:00+00:00")
    assert again["content_sha"] == leaf["content_sha"]
    assert led.read_manifest() == manifest_before
    assert MANIFEST_KEY not in s3.put_calls[puts_before:]
    assert CACHE_KEY not in s3.put_calls[puts_before:]
