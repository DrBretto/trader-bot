"""Ops probes — the governed, read-mostly, NON-DESTRUCTIVE diag branches.

These are the ``*-diag`` invokes used to reality-test the clean spine from the
AWS-IP Lambda context (the P1/P3 pattern) without touching a single S3 object or
the live pipeline. ``app.handler`` dispatches ``ops-probe`` / ``*-diag`` /
``canary`` invocations here; each branch delegates to a clean-core module and
returns machine-checkable acceptance rows.

Branches:
  * ``forecast-diag``   → forecast-freshness diagnostic (extend → panel → mu),
                          watermark + gate verdict, writes no S3 object.
  * ``freshness-diag``  → forces the substrate-currency gate over a deliberately
                          frozen store (``force_stale``) to prove it fires.
  * ``regime-diag``     → regime-chassis probe (chassis loaded, mu re-ranked).
  * ``canary``          → the post-pipeline reality-canary tier (live), alert off.
  * ``watchdog-diag``   → the P8 three-line health reality-test: a LIVE health
                          check (✓ all three lines, real advancing dates) AND a
                          deliberately skipped/stale night that trips the
                          missed-run/stale alarm (SNS fires, ✗ in the email).

NOTHING here writes an S3 object or mutates production state. ``watchdog-diag``
sends a CLEARLY-MARKED reality-test SNS on the stale branch — that firing IS the
alarm proof (it never fires on the live-✓ branch).
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ diag helpers
def _pending(event: dict) -> Optional[List[str]]:
    pending = (event or {}).get("pending")
    if isinstance(pending, str):
        return [pending]
    return pending


def forecast_diag(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Forecast-freshness diagnostic: extend OHLCV → panel → inference, report the
    watermark before/after + the staleness-gate verdict. Writes NO S3 object."""
    from decide.cutover import diagnose_forecast_freshness
    result = diagnose_forecast_freshness(pending=_pending(event),
                                         force_stale=bool(event.get("force_stale")))
    return {"phase": "forecast-diag", "nondestructive": True, "result": result}


def freshness_diag(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Prove the substrate-currency gate FIRES over a deliberately frozen store:
    ``force_stale=True`` skips the extend and the gate must flag stale."""
    from decide.cutover import diagnose_forecast_freshness
    result = diagnose_forecast_freshness(pending=_pending(event), force_stale=True)
    gate = (result or {}).get("gate", {})
    return {"phase": "freshness-diag", "nondestructive": True,
            "gate_fires_on_frozen_store": bool(gate.get("stale")),
            "result": result}


def regime_diag(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Regime-chassis probe: chassis loaded + observably re-ranks a fresh mu."""
    from decide.cutover import diagnose_regime_chassis
    result = diagnose_regime_chassis(pending=_pending(event))
    return {"phase": "regime-diag", "nondestructive": True, "result": result}


def canary(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Run the post-pipeline reality-canary tier (live) with alerting OFF (a diag
    canary must not page). Returns the tier result."""
    from monitors.canary_gate import run_post_pipeline_canaries
    tier = (event or {}).get("tier", "live")
    result = run_post_pipeline_canaries(tier=tier, alert=False)
    return {"phase": "canary", "nondestructive": True, "result": result}


# ---------------------------------------------------- the P8 watchdog reality-test
def _far_future_settled(days_ahead: int = 7) -> str:
    """A weekday ``days_ahead`` weekdays after today — the ``today`` we hand the
    health check to SIMULATE a run of nights with no advancing chassis (a skipped
    night). Uses only the real, fixed S3 chassis/line dates; nothing is written."""
    d = _dt.date.today()
    stepped = 0
    while stepped < days_ahead:
        d += _dt.timedelta(days=1)
        if d.weekday() < 5:
            stepped += 1
    return d.isoformat()


def _fire_reality_test_sns(subject: str, body: str, region: str) -> Dict[str, Any]:
    """Publish the alarm to the REAL SNS topic and capture the MessageId. Clearly
    marked ``[REALITY-TEST]`` so the operator knows it is the P8 alarm proof, not
    a live incident. Fail-soft — a publish error is reported, never raised."""
    try:
        import boto3
        from src.utils.sns_alerts import get_sns_topic_arn
        sns = boto3.client("sns", region_name=region)
        resp = sns.publish(
            TopicArn=get_sns_topic_arn(region),
            Subject=("[REALITY-TEST] " + subject)[:100],
            Message=("*** P8 WATCHDOG REALITY-TEST — deliberately-injected stale "
                     "night; NOT a live incident. This proves the missed-run/stale "
                     "alarm fires. ***\n\n" + body),
        )
        return {"fired": True, "message_id": resp.get("MessageId"), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"fired": False, "message_id": None, "error": f"{type(e).__name__}: {e}"}


def watchdog_diag(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """The P8 three-line watchdog reality-test (NON-DESTRUCTIVE; writes no S3).

    (A) LIVE health check against real S3: the daily email shows ✓ for all three
        lines (canon, SPY, challenger) with real advancing dates. No SNS is fired
        on the ✓ branch.
    (B) Deliberately-skipped-night injection: run the SAME check with ``today``
        pushed ``days_ahead`` weekdays forward so the fixed real chassis/line dates
        now lag — the check flags ✗ and the missed-run/stale alarm FIRES (a real,
        clearly-marked reality-test SNS with a captured MessageId).
    """
    from src.utils.s3_client import S3Client
    s3 = S3Client(bucket, region)
    today = (event or {}).get("today") or _dt.date.today().isoformat()
    days_ahead = int((event or {}).get("days_ahead", 7))

    # (A) LIVE ✓ branch — capture the email, DO NOT fire SNS.
    live_cap: Dict[str, Any] = {}
    live = run_health = None  # placeholders for clarity
    from monitors.watchdog import run_daily_health_check
    live = run_daily_health_check(
        s3, today=today,
        alert=lambda subj, b: live_cap.update(subject=subj, body=b))
    live_lines = live.get("lines", {})
    live_ok = bool(live.get("ok"))
    all_three_current = live_ok and all(
        (not v.get("stale")) and v.get("populated")
        for v in live_lines.values())

    # (B) SKIPPED-NIGHT injection — chassis/lines now lag; ✗ + fire the alarm.
    injected_today = _far_future_settled(days_ahead)
    stale_cap: Dict[str, Any] = {}
    stale = run_daily_health_check(
        s3, today=injected_today,
        alert=lambda subj, b: stale_cap.update(subject=subj, body=b))
    stale_detected = not stale.get("ok")
    sns = (_fire_reality_test_sns(stale_cap.get("subject", ""),
                                  stale_cap.get("body", ""), region)
           if stale_detected else {"fired": False, "message_id": None,
                                    "error": "no stale detected — alarm not fired"})

    acceptance = {
        "live_health_all_three_lines_current": all_three_current,
        "live_dates": {name: v.get("at") for name, v in live_lines.items()},
        "skipped_night_trips_alarm": stale_detected,
        "sns_fired": sns.get("fired"),
        "sns_message_id": sns.get("message_id"),
    }
    result = {
        "phase": "watchdog-diag",
        "nondestructive": True,
        "live": {"ok": live_ok, "subject": live_cap.get("subject"),
                 "body": live_cap.get("body"), "lines": live_lines},
        "stale_injection": {"injected_today": injected_today,
                            "days_ahead": days_ahead,
                            "detected_stale": stale_detected,
                            "subject": stale_cap.get("subject"),
                            "body": stale_cap.get("body"),
                            "sns": sns},
        "acceptance": acceptance,
        "all_pass": bool(all_three_current and stale_detected and sns.get("fired")),
    }
    return result


# ----------------------------------------------------------------- dispatch table
_PROBES = {
    "forecast-diag": forecast_diag,
    "freshness-diag": freshness_diag,
    "regime-diag": regime_diag,
    "canary": canary,
    "watchdog-diag": watchdog_diag,
}


def dispatch(source: str, event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Route an ops-probe ``source`` to its branch. Unknown sources return an
    honest error row (never raise into the router)."""
    fn = _PROBES.get(source)
    if fn is None:
        return {"phase": "ops-probe", "error": f"unknown ops probe: {source!r}",
                "known": sorted(_PROBES)}
    return fn(event or {}, bucket, region)
