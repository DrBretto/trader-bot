"""PKT-TB-IR-01 — correction overlay invariants (INV-1 / INV-3 / INV-10).

INV-1  append-only / write-once content-addressed corrections store.
INV-3  a correction SURVIVES every regenerate (the cliff never reverts); the
       pre-correction value is unreachable once the overlay is applied.
INV-10 corrections are attributed (non-anonymous issued_by, content_sha) and
       resolve by supersede-chain head (never last-writer-wins; conflicts are
       surfaced, never silently picked).
"""
import json

import pytest

from src.utils.corrections import (
    CANON_VALUE_FIELDS,
    CorrectionStore,
    apply_correction_overlay,
    build_event,
    resolve_overlay,
)


# ---- a minimal in-memory boto3-S3 stub with IfNoneMatch write-once ----------
class _Body:
    def __init__(self, b): self._b = b
    def read(self): return self._b


class FakeS3Boto:
    def __init__(self):
        self.store = {}

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch == "*" and Key in self.store:
            raise Exception("PreconditionFailed: key exists (IfNoneMatch=*)")
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {}

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise Exception("NoSuchKey")
        return {"Body": _Body(self.store[Key])}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


# ---- build_event: attribution + validation (INV-10) -------------------------
def test_build_event_is_attributed_and_content_addressed():
    ev = build_event(
        issued_by="op@example.com", reason_code="RESIM_SEGMENT",
        why="risk-off gap re-sim", method="bias-cancelled re-sim",
        dates_to_values={"2026-06-23": 116201.08},
    )
    assert ev["issued_by"] == "op@example.com"
    assert len(ev["content_sha"]) == 64
    assert ev["corr_id"].startswith("corr:")
    assert ev["sim_semantics"] == "paper_resim"
    assert ev["provenance"] == "constructed"
    assert ev["sets"]["2026-06-23"]["to"] == 116201.08


def test_build_event_rejects_anonymous_identity():
    for bad in ["", "  ", "unknown", "anonymous"]:
        with pytest.raises(ValueError):
            build_event(issued_by=bad, reason_code="RESIM_SEGMENT", why="x",
                        method="", dates_to_values={"2026-06-23": 1.0})


def test_sim_semantics_has_no_realized_cash_member():
    from src.utils.corrections import SIM_SEMANTICS
    assert "realized" not in " ".join(SIM_SEMANTICS).lower()
    with pytest.raises(ValueError):
        build_event(issued_by="op@example.com", reason_code="RESIM_SEGMENT", why="x",
                    method="", dates_to_values={"2026-06-23": 1.0},
                    sim_semantics="realized_cash")


# ---- supersede-chain resolution (INV-10) ------------------------------------
def test_resolve_head_wins_over_prior():
    e1 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="v1",
                     method="", dates_to_values={"2026-06-23": 100.0},
                     issued_at="2026-06-23T10:00:00+00:00")
    e2 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="v2 supersedes v1",
                     method="", dates_to_values={"2026-06-23": 116.0},
                     supersedes=e1["corr_id"], issued_at="2026-06-23T11:00:00+00:00")
    ov = resolve_overlay([e1, e2])
    assert ov["2026-06-23"]["value"] == 116.0
    assert ov["2026-06-23"]["conflict"] is False
    assert ov["2026-06-23"]["corr_id"] == e2["corr_id"]


def test_resolve_conflict_when_null_lands_on_active_chain():
    e1 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="v1",
                     method="", dates_to_values={"2026-06-23": 100.0},
                     issued_at="2026-06-23T10:00:00+00:00")
    e2 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="rival, supersedes nothing",
                     method="", dates_to_values={"2026-06-23": 999.0},
                     supersedes=None, issued_at="2026-06-23T11:00:00+00:00")
    ov = resolve_overlay([e1, e2])
    # Prior head retained (NOT last-writer-wins), conflict surfaced.
    assert ov["2026-06-23"]["value"] == 100.0
    assert ov["2026-06-23"]["conflict"] is True


def test_resolve_orphan_supersede_is_conflict():
    e1 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="orphan",
                     method="", dates_to_values={"2026-06-23": 50.0},
                     supersedes="corr:does-not-exist", issued_at="2026-06-23T10:00:00+00:00")
    ov = resolve_overlay([e1])
    assert ov["2026-06-23"]["conflict"] is True


# ---- apply overlay + the cliff-survives-regenerate proof (INV-3) ------------
def _dash_with_cliff():
    # The pre-correction line ends on the spurious 06-23 cliff.
    ec = [
        {"date": "2026-06-16", "value": 119000.0, "optimized_value": 119000.0,
         "new_brain_value": 119000.0, "benchmark": 100000.0},
        {"date": "2026-06-22", "value": 118000.0, "optimized_value": 118000.0,
         "new_brain_value": 118000.0, "benchmark": 100500.0},
        {"date": "2026-06-23", "value": 113900.0, "optimized_value": 113900.0,
         "new_brain_value": 113900.0, "benchmark": 100600.0},  # the cliff
    ]
    return {"equity_curve": ec, "metrics": {"total_value": 113900.0}, "drawdowns": [], "timeline_correction": {}}


def test_overlay_applies_and_is_idempotent():
    overlay = {"2026-06-23": {"value": 116201.08, "prior_value": 113900.0,
                              "corr_id": "corr:x", "conflict": False}}
    d1 = apply_correction_overlay(_dash_with_cliff(), overlay)
    last = d1["equity_curve"][-1]
    for fld in ("value", "optimized_value", "new_brain_value"):
        assert last[fld] == 116201.08
    assert d1["metrics"]["total_value"] == 116201.08
    assert last["correction"]["corr_id"] == "corr:x"
    # Idempotent: applying again yields the same dashboard.
    d2 = apply_correction_overlay(d1, overlay)
    assert d2["equity_curve"][-1]["value"] == 116201.08
    assert d2["metrics"]["total_value"] == 116201.08


def test_cliff_survives_regenerate_via_store(monkeypatch):
    """End-to-end through the store: write a correction once, then simulate TWO
    nightly regenerates — each rebuilds the raw cliff line, then applies the
    active overlay read from the store. The corrected value persists; the cliff
    value (113900) is unreachable after the overlay both times."""
    s3 = FakeS3Boto()
    store = CorrectionStore(s3, bucket="investment-system-data")
    ev = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT",
                     why="06-23 risk-off gap re-sim", method="bias-cancelled",
                     dates_to_values={"2026-06-23": 116201.08},
                     from_values={"2026-06-23": 113900.0})
    store.append(ev)

    def regenerate():
        # The nightly path always rebuilds the RAW cliff line from inputs...
        dash = _dash_with_cliff()
        # ...then applies the active overlay as an INPUT (the IR-01 hook).
        overlay = store.active_overlay("canon")
        return apply_correction_overlay(dash, overlay)

    for _ in range(2):  # two consecutive nightly regenerates
        out = regenerate()
        vals = [r["value"] for r in out["equity_curve"]]
        assert 116201.08 in vals
        assert 113900.0 not in vals          # the cliff never comes back
        assert out["metrics"]["total_value"] == 116201.08


def test_store_is_write_once_and_append_only(monkeypatch):
    s3 = FakeS3Boto()
    store = CorrectionStore(s3, bucket="investment-system-data")
    ev = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="x", method="",
                     dates_to_values={"2026-06-23": 116201.08})
    key = store.append(ev)
    # Re-appending the identical content-addressed event is an idempotent no-op
    # (write-once does not raise out of append).
    key2 = store.append(ev)
    assert key == key2
    # The leaf bytes are unchanged (no overwrite).
    leaves = [k for k in s3.store if k.endswith(".json") and k != "corrections/_log.jsonl"]
    assert len(leaves) == 1
    stored = json.loads(s3.store[leaves[0]])
    assert stored["content_sha"] == ev["content_sha"]
    # A different correction is a different key (cannot clobber).
    ev2 = build_event(issued_by="op@e", reason_code="RESIM_SEGMENT", why="y", method="",
                      dates_to_values={"2026-06-23": 117000.0})
    store.append(ev2)
    leaves2 = [k for k in s3.store if k.endswith(".json") and k != "corrections/_log.jsonl"]
    assert len(leaves2) == 2


def test_canon_value_fields_only_touch_present_fields():
    overlay = {"2026-06-23": {"value": 116201.08, "prior_value": None,
                              "corr_id": "corr:x", "conflict": False}}
    dash = {"equity_curve": [
        {"date": "2026-06-23", "value": 113900.0, "champion_frozen_value": None}],
        "metrics": {}, "drawdowns": [], "timeline_correction": {}}
    apply_correction_overlay(dash, overlay)
    row = dash["equity_curve"][0]
    assert row["value"] == 116201.08
    assert row["champion_frozen_value"] is None  # None field not resurrected
    assert "champion_frozen_value" in CANON_VALUE_FIELDS
