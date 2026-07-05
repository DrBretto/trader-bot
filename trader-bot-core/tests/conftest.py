"""Reality suite (P7) — collection + fixtures.

Puts the core package dir AND the parent repo on sys.path (some core modules
import ``src.utils.*``, exactly as the deployed brain resolves them), registers
the three tier markers, and exposes a session-scoped live S3 reader.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CORE = Path(__file__).resolve().parents[1]
_REPO = _CORE.parent
for _p in (str(_CORE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _reality import LiveS3Reader  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Tier-1 deterministic unit-fixture (real recorded data). Every commit.")
    config.addinivalue_line("markers", "live_canary: Tier-2 assert against TODAY's real S3/CloudWatch. Nightly + post-pipeline.")
    config.addinivalue_line("markers", "external_check: Tier-3 call the real feed (Yahoo/GDELT) independently. Nightly.")
    config.addinivalue_line("markers", "fault_injection: proves a canary goes RED on a deliberately broken reality.")


@pytest.fixture(scope="session")
def reality() -> LiveS3Reader:
    """The live S3 reader — TODAY's real production output. If AWS is
    unreachable the live/external tests ERROR (never silently pass)."""
    return LiveS3Reader()
