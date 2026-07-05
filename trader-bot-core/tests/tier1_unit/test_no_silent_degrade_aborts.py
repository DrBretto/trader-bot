"""Tier 1 unit-fixture (FAULT-INJECTED on the real code path) —
a degraded / empty publish ABORTS + ALARMS and writes NO artifact.

The two-week failure had NO test confirming the pipeline refuses to ship a
degraded feed. This drives the REAL publish guards (``publish.dashboard``) with a
poisoned/empty line and asserts they (1) RAISE + log an ALARM on a point-reduced
overwrite, and (2) HOLD last-known-good with NO ``put_object`` on an empty
publish. Nothing is mocked green — the assertions are on the real guard's real
abort behaviour under injected degradation.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from publish import dashboard as D  # noqa: E402
from publish.dashboard import DestructivePublishError  # noqa: E402


class _FakeS3:
    """A fake S3 whose CURRENT published dashboard is populated (a good line to
    protect) and that RECORDS every write — so a degraded publish that writes
    nothing is observable."""

    def __init__(self, current_points: int):
        body = json.dumps({
            "equity_curve": [{"date": f"2026-06-{10 + i:02d}", "value": 100.0 + i}
                             for i in range(current_points)]
        }).encode()
        self._body = body
        self.puts = []

    def get_object(self, Bucket=None, Key=None):
        return {"Body": io.BytesIO(self._body)}

    def put_object(self, **kw):
        self.puts.append(kw)


@pytest.mark.unit
@pytest.mark.fault_injection
def test_never_shrink_guard_raises_and_alarms(caplog):
    """A point-reduced overwrite RAISES DestructivePublishError + logs an ALARM,
    leaving the good line intact."""
    s3 = _FakeS3(current_points=15)
    degraded = {"equity_curve": [{"date": "2026-06-10", "value": 100.0}]}  # 15 -> 1
    with caplog.at_level(logging.ERROR):
        with pytest.raises(DestructivePublishError):
            D.assert_publish_not_shrunk(degraded, s3)
    assert any("ALARM" in r.getMessage() for r in caplog.records), (
        "never-shrink abort did not emit an ALARM (no alert on the degrade)")
    assert s3.puts == [], "a degraded publish wrote an artifact — the good line was clobbered"


@pytest.mark.unit
@pytest.mark.fault_injection
def test_empty_publish_holds_and_writes_no_artifact():
    """An empty publish HOLDS (published=False) and writes NOTHING."""
    s3 = _FakeS3(current_points=15)
    empty = {"equity_curve": []}
    result = D.publish_line(empty, s3, phase="night", run_date="2026-07-02")
    assert result["published"] is False and result["held"] is True, (
        f"empty publish was not held: {result}")
    assert s3.puts == [], "empty publish wrote an artifact — a silent degrade shipped"


@pytest.mark.unit
@pytest.mark.fault_injection
def test_point_reduced_publish_holds_and_writes_no_artifact():
    """A populated-but-point-reduced publish HOLDS and writes NOTHING."""
    s3 = _FakeS3(current_points=15)
    reduced = {"equity_curve": [{"date": "2026-06-10", "value": 100.0}]}  # 15 -> 1
    result = D.publish_line(reduced, s3, phase="night", run_date="2026-07-02")
    assert result["published"] is False and result["held"] is True, (
        f"point-reduced publish was not held: {result}")
    assert s3.puts == [], "point-reduced publish wrote an artifact"
