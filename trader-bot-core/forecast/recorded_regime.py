"""Recorded-regime resolver for the P6 seed-by-replay reconstruction
(PKT-TRADER-BOT-SEED-CANON-BY-REPLAY).

LOCKED OPERATOR SCOPE #3: the reconstruction of the post-split window uses the
**RECORDED regime labels read from the daily files** — the real regime the
algorithm consumed — NOT the deterministic ``forecast.regime.regime(D)`` picker
(that picker is FORWARD-only). This module reads that recorded regime, per settled
trading day, straight from the production ``daily/<D>/`` artifacts on S3.

WHY A RESOLVER (not a raw ``decisions.json`` read): the production date-stamping
across the post-split window is internally inconsistent — a UTC bug displaced
several Friday-night decisions onto the following **Saturday** folder, and the
intermittent freeze skipped some nights entirely. So for a handful of real settled
trading days (the Mondays 2026-06-15 / -22 / -29) the day's own
``daily/<D>/decisions.json`` is ABSENT. The regime the algorithm actually consumed
on those days is still recorded — under the folder whose intents the morning
executor ran, which each day's own ``daily/<D>/morning_execution.json`` names
explicitly in ``intents_date``. Following that execution record is reading the
recorded regime, NOT guessing: it is the algorithm's own log of which decision it
executed that morning.

RESOLUTION ORDER for a settled trading day D (fail-loud; never invents a label):
  1. ``daily/<D>/decisions.json`` present -> its fused regime
     (``expert_metrics.final_regime_label`` -> ``regime``). This is the dominant
     folder==execution-day case.
  2. else ``daily/<D>/morning_execution.json['intents_date'] = F`` ->
     ``daily/<F>/decisions.json`` fused regime (the UTC-displaced case).
  3. else RAISE ``RecordedRegimeUnavailable`` — surfaced to the operator, never a
     silent ``'neutral'`` fallback (the packet's explicit STOP condition).

The fused label (``decisions.expert_metrics.final_regime_label``) is what the
decision engine actually applied (``src/steps/decision_engine.py``:
``regime_label = fusion['final_regime_label']``), which is what drives
``regime_score_mult`` / admissibility / exposure inside ``run_engine``. The
pre-fusion ``inference.json['regime']['label']`` (the raw trained-ensemble output)
is deliberately NOT used — it is not what the algorithm consumed.

``VERIFIED_WINDOW`` bakes the resolved map for the reconstruction window as an
auditable constant. ``recorded_regime(D)`` reads S3 live; the seed orchestrator
cross-checks its live resolution against ``VERIFIED_WINDOW`` so a drift in the
production artifacts cannot silently change the seeded line.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

S3_BUCKET = "investment-system-data"

# The recorded fused regime the algorithm consumed, per real settled trading day of
# the post-split reconstruction window (2026-06-12 .. 2026-07-02, 14 NY trading
# days; 06-19 Juneteenth has no settled bar). Resolved from the production daily/
# artifacts on 2026-07-05 via the resolution order above and captured here for
# reproducibility + audit. The three Mondays resolve through morning_execution
# (UTC-displaced Fri-night decision); 2026-06-24 resolves from its own night
# decision (morning_execution artifact absent, night decision present).
VERIFIED_WINDOW: Dict[str, str] = {
    "2026-06-12": "risk_on_trend",
    "2026-06-15": "risk_on_trend",   # via morning_execution -> daily/2026-06-13
    "2026-06-16": "risk_on_trend",
    "2026-06-17": "choppy",
    "2026-06-18": "high_vol_panic",
    "2026-06-22": "choppy",          # via morning_execution -> daily/2026-06-20
    "2026-06-23": "risk_on_trend",
    "2026-06-24": "high_vol_panic",  # own night decision (dec.date 2026-06-23)
    "2026-06-25": "choppy",
    "2026-06-26": "risk_off_trend",
    "2026-06-29": "high_vol_panic",  # via morning_execution -> daily/2026-06-27
    "2026-06-30": "choppy",
    "2026-07-01": "choppy",
    "2026-07-02": "choppy",
}


class RecordedRegimeUnavailable(RuntimeError):
    """The recorded regime for a settled trading day cannot be read from the daily
    files and cannot be resolved through the execution record — surfaced to the
    operator (the packet STOP condition), never stubbed to 'neutral'."""


def _s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _get_json(s3, key: str) -> Optional[dict]:
    try:
        return json.loads(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 — missing key is a normal "not present" state
        return None


def _fused_regime(dec: Optional[dict]) -> Optional[str]:
    if not dec:
        return None
    lbl = (dec.get("expert_metrics", {}) or {}).get("final_regime_label") or dec.get("regime")
    return str(lbl) if lbl else None


def recorded_regime(date: str, s3=None) -> str:
    """Resolve the recorded fused regime the algorithm consumed on settled trading
    day ``date``, reading the production ``daily/<date>/`` artifacts. Fail-loud
    (raises ``RecordedRegimeUnavailable``) — never returns a guessed label."""
    s3 = s3 or _s3()

    # (1) the day's own night decision (folder == execution day)
    reg = _fused_regime(_get_json(s3, f"daily/{date}/decisions.json"))
    if reg:
        return reg

    # (2) the execution record: which decision the morning executor actually ran
    me = _get_json(s3, f"daily/{date}/morning_execution.json")
    idate = me.get("intents_date") if me else None
    if idate:
        reg = _fused_regime(_get_json(s3, f"daily/{idate}/decisions.json"))
        if reg:
            return reg

    raise RecordedRegimeUnavailable(
        f"recorded regime for settled trading day {date} is unavailable: no "
        f"daily/{date}/decisions.json and no resolvable morning_execution "
        f"intents_date -> decision. STOP; do NOT guess or stub to neutral.")


def recorded_regime_map(dates, s3=None, verify: bool = True) -> Dict[str, str]:
    """Resolve the recorded regime for every date in ``dates``. When ``verify`` is
    set, any date present in ``VERIFIED_WINDOW`` must match its live resolution —
    a mismatch RAISES (a drift in the production artifacts cannot silently move the
    seeded line)."""
    s3 = s3 or _s3()
    out: Dict[str, str] = {}
    for d in dates:
        r = recorded_regime(d, s3=s3)
        if verify and d in VERIFIED_WINDOW and r != VERIFIED_WINDOW[d]:
            raise RecordedRegimeUnavailable(
                f"recorded regime for {d} resolved live to {r!r} but VERIFIED_WINDOW "
                f"pins {VERIFIED_WINDOW[d]!r} — production daily/ artifacts drifted; "
                f"STOP and re-verify before seeding.")
        out[d] = r
    return out
