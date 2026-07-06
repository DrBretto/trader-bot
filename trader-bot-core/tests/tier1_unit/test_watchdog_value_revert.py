"""Tier 1 unit-fixture (FAULT-INJECTED on the real code path) —
the watchdog flags a published line whose DATE stayed current while its VALUE
reverted to a contaminated terminal.

The three-line watchdog caught staleness/non-advancement but was BLIND to the
2026-07-06 incident: the preserved morning publish path read the retired
``canon/equity_ledger/`` and re-published ``dashboard.json`` with the pre-P6
contaminated terminal (07-02 = 121147.52) on top of the night's corrected line
(114271.38) — same date, reverted value. This drives the REAL
``monitors.watchdog.check_value_revert`` and asserts it (1) fires on the known
contaminated sentinel, (2) fires on any published terminal diverging from the
corrected ledger for the same date, and (3) does NOT fire when the published
terminal matches the corrected ledger. Nothing is mocked green — the assertions
are on the real check's real behaviour under injected contamination.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitors import watchdog as W  # noqa: E402

# The corrected clean_v2 terminal (07-02) the night path publishes.
_LEDGER_TERM = {"date": "2026-07-02", "value": 114271.38326324461,
                "benchmark": 107973.34057066578, "segment": "new_brain"}
# The contaminated pre-P6 terminal the retired ledger carried for the same date.
_CONTAMINATED_VALUE = 121147.52434461439


class _FakeS3:
    """Serves a chosen published dashboard terminal + the corrected ledger cache."""

    def __init__(self, published_terminal):
        self._published = published_terminal

    def read_json(self, key):
        if key == W.PUBLISHED_DASHBOARD_KEY:
            return {"equity_curve": [self._published]} if self._published else {}
        return {}

    def read_jsonl(self, key):
        if key == W.LEDGER_CACHE_KEY:
            return [_LEDGER_TERM]
        return []


def test_contaminated_sentinel_terminal_is_flagged():
    s3 = _FakeS3({"date": "2026-07-02", "value": _CONTAMINATED_VALUE})
    out = W.check_value_revert(s3)
    assert out["reverted"] is True
    assert "CONTAMINATED" in out["reason"]


def test_nonsentinel_divergence_from_ledger_is_flagged():
    # A wrong value that is NOT the known sentinel must still trip via the
    # corrected-ledger cross-check for the same date.
    s3 = _FakeS3({"date": "2026-07-02", "value": 999999.99})
    out = W.check_value_revert(s3)
    assert out["reverted"] is True
    assert "DIVERGES" in out["reason"]


def test_matching_corrected_terminal_is_not_flagged():
    s3 = _FakeS3({"date": _LEDGER_TERM["date"], "value": _LEDGER_TERM["value"]})
    out = W.check_value_revert(s3)
    assert out["reverted"] is False


def test_provisional_later_date_is_not_a_revert():
    # An intraday provisional dot on a LATER date is legitimately ahead of the
    # settled ledger and must not be mistaken for a revert.
    s3 = _FakeS3({"date": "2026-07-03", "value": 114500.00})
    out = W.check_value_revert(s3)
    assert out["reverted"] is False
