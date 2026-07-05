"""The reality-assertion rule, enforced as a test (grep-proof).

LOAD-BEARING: no test in the live-canary or external-check tiers may assert a
value it (or a fixture) planted. This scans those tiers and fails if any file
mocks the reality source — the exact theater being removed (the old suite mocked
every input with ``_FakeRawS3`` / ``patch(...)`` and asserted the failure value
as correct). Every live/external test must instead read the live reader
(``reality``) or call an independent real feed.

Fault-injection tests are exempt from the "must read live reality" requirement —
they DELIBERATELY inject a broken reality to prove a canary goes RED — but they
still may not use unittest.mock/_FakeRawS3 (they use the typed FrozenStoreReader
shim, which injects a dated real-shaped fault, not a planted green).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_LIVE = _HERE / "tier2_live_canary"
_EXT = _HERE / "tier3_external_check"

# the theater signatures the old 570-suite was built from.
_THEATER = re.compile(r"_FakeRawS3|unittest\.mock|from mock\b|import mock\b|\bpatch\(|MagicMock|mock\.patch")
# at least one independent-reality touchpoint every live/external test must have.
_REALITY = re.compile(r"\breality\b|yahoo_v8_daily|yahoo_close_on|gdelt_v2_gkg_record_count|"
                      r"reconstruct_canon_terminal|freshness_verdict|shadow_timeseries|"
                      r"canon_terminal|read_prices|store_watermark|get_json|"
                      r"gdelt_features\.parquet|gdelt_cache")


def _tier_files():
    return sorted(list(_LIVE.glob("test_*.py")) + list(_EXT.glob("test_*.py")))


def test_reality_tiers_contain_no_mocked_inputs():
    offenders = []
    for f in _tier_files():
        txt = f.read_text()
        if _THEATER.search(txt):
            offenders.append(f.name)
    assert not offenders, (
        f"live/external tier files mock the reality source (planted-value theater): {offenders}")


def test_every_live_and_external_test_reads_a_real_source():
    missing = []
    for f in _tier_files():
        if not _REALITY.search(f.read_text()):
            missing.append(f.name)
    assert not missing, (
        f"live/external tier files with no independent-reality read: {missing}")


def test_old_570_suite_is_retired_as_acceptance_surface():
    """The suite's own pytest.ini scopes collection to this directory only — the
    repo-root tests/ 570-suite is not the acceptance gate."""
    ini = (_HERE / "pytest.ini").read_text()
    assert "testpaths = ." in ini, "suite is not scoped to trader-bot-core/tests/"
    # and no live/external test imports the retired theater package.
    for f in _tier_files():
        assert "src.brain.tests" not in f.read_text(), f"{f.name} imports the retired theater suite"
