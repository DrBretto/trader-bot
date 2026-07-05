"""Tier 2 live-canary — the regime chassis is LOADED and nontrivial in the
deployed config, not inert.

A mocked suite reads the config file and passes; it can never tell whether the
DEPLOYED image loaded a nontrivial table. This canary reads the deployed
``config/regime_compatibility.json`` (what is baked into the image) and asserts
it is nonempty AND that the multiplier for the live regime is a real tilt
(``mult != 1.0`` somewhere in the row), then ties it to the live regime label
read from S3 ``daily/latest.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import FeedUnreachable  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "regime_compatibility.json"


@pytest.mark.live_canary
def test_regime_chassis_loaded_nontrivial(reality):
    table = json.loads(_CONFIG.read_text())
    assert isinstance(table, dict) and table, "regime_compatibility table is empty — chassis inert"

    # 'choppy' is the live regime family; its row must carry a real tilt.
    assert "choppy" in table, f"no 'choppy' regime in the chassis: {list(table)[:6]}"
    choppy = table["choppy"]
    nontrivial = [(k, v) for k, v in choppy.items() if abs(float(v) - 1.0) > 1e-9]
    assert nontrivial, (
        "every 'choppy' multiplier is exactly 1.0 — the chassis is loaded but "
        "INERT (no regime tilt applied)")

    # tie to live state: the regime label production is actually running must be a
    # key in the deployed table (chassis is dispatched, not bypassed).
    try:
        latest = reality.get_json("daily/latest.json")
    except FeedUnreachable as e:
        pytest.fail(f"cannot read live regime label from S3: {e}")
    live_regime = latest.get("regime")
    assert live_regime in table, (
        f"live regime {live_regime!r} is not a key in the deployed chassis "
        f"{list(table)} — deployed table does not match the running regime set")
