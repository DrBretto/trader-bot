"""Morning execution — PRESERVED path (zero morning-path change).

The thin router routes ``morning-execution`` here. P8 does NOT rebuild the morning
executor: it consumes ``daily/<D>/trade_intents.json`` in the exact schema the
night path (``engine.build_trade_intents``) emits, and that contract is unchanged.
This delegates byte-for-byte to the production morning phase so the morning path
is provably untouched; P9 owns any relocation of the morning executor into the
clean core.
"""
from __future__ import annotations

from typing import Any, Dict


def run_morning(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    from src.handler import _run_morning_phase
    return _run_morning_phase(event, bucket, region)
