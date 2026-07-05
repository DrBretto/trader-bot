"""Tier 3 external-cross-check — GDELT ingest counts ≈ the live GDELT source.

The single worst class of theater in the old suite: the only GDELT tests
asserted ``gdelt_doc_count == 0`` — green precisely when GDELT was dead. This
test does the opposite: it re-fetches the raw GDELT v2 GKG files for a fixed
recorded date INDEPENDENTLY (raw record count, not the ingest's parser) and
asserts the ingest's stored doc count for that date is within ±5% of the live
source. A frozen or wrong-URL ingest (the real failure mode) makes this diverge.

Data source: the ingest's stored daily aggregate (``store/gdelt_cache/daily/
<YMD>.json``, the new-spine cache the forward path builds) vs a live re-fetch
from ``data.gdeltproject.org/gdeltv2``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FeedUnreachable, gdelt_v2_gkg_record_count  # noqa: E402

TOL = 0.05  # ±5%
_CACHE = Path(__file__).resolve().parents[2] / "store" / "gdelt_cache" / "daily"


def _fixed_cross_check_date() -> str:
    """A stable, complete cached GDELT day: the second-most-recent cached date
    (the most recent may be a partial in-progress UTC day)."""
    days = sorted(p.stem for p in _CACHE.glob("*.json") if p.stem.isdigit())
    if len(days) < 2:
        pytest.skip("no complete cached GDELT day to cross-check")
    ymd = days[-2]
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


@pytest.mark.external_check
def test_gdelt_ingest_matches_source():
    date = _fixed_cross_check_date()
    ymd = date.replace("-", "")
    stored = json.loads((_CACHE / f"{ymd}.json").read_text())
    stored_n = int(stored.get("n_records") or 0)
    assert stored_n > 0, (
        f"ingest recorded n_records=0 for {date} — a dead/zero GDELT feed "
        f"(the exact failure the old suite certified green)")

    hours = tuple(stored.get("files", {}).get("gkg_ok") or
                  ("000000", "060000", "120000", "180000"))
    try:
        live = gdelt_v2_gkg_record_count(date, hours=hours)
    except FeedUnreachable as e:
        pytest.fail(f"GDELT v2 source unreachable — cannot cross-check ingest: {e}")

    rel = abs(live["records"] - stored_n) / stored_n
    assert rel < TOL, (
        f"GDELT ingest {stored_n} records for {date} diverges {rel:.2%} from the "
        f"live source re-count {live['records']} ({live['hours_fetched']} hours) "
        f"— ingest is frozen/wrong-URL")
