"""Midday check — PRESERVED path.

The thin router routes ``midday-check`` here. Like the morning path, P8 does not
rebuild the midday checker; it delegates to the production midday phase so the
behavior is provably untouched. P9 owns any relocation into the clean core.
"""
from __future__ import annotations

from typing import Any, Dict


def run_midday(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    from src.handler import _run_midday_check
    return _run_midday_check(event, bucket, region)
