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


def advance_challenger(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Refresh both model lines from replay and publish the promoted TILT canon.

    The EventBridge source name is retained for compatibility, but this is no longer
    a comparison-only patch. It is the sole post-split line writer: TILT is canonical
    ``value`` and the reconstructed two-stage book is ``comparison``. Dry-run remains
    the default; commit requires cent-level continuity and a complete settled grid.
    """
    import boto3
    from lines.replay_refresh import refresh_promoted_ledger

    s3 = boto3.client("s3", region_name=region)
    out = refresh_promoted_ledger(
        s3,
        bucket=bucket,
        commit=bool(event.get("commit")),
    )
    out["phase"] = "advance-challenger"
    if not event.get("commit") or not out.get("committed"):
        return out

    from publish.dashboard import build_publish_surface, publish_line
    dashboard = build_publish_surface(s3)
    out["dashboard_publish"] = publish_line(
        dashboard,
        s3,
        phase="replay-refresh",
        run_date=out["latest_settled"],
    )
    from app.shadow_publish import run_shadow_publish
    out["shadow_publish"] = run_shadow_publish(
        {"run_date": out["latest_settled"]}, bucket, region
    )
    return out


def config_canary(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """CL-708150 empty-but-critical config canary probe. With no args it runs the
    LIVE verdict (alerting OFF — a diag must not page) and confirms the load-bearing
    tables are non-empty. With ``inject_empty`` (a name or list from
    {regime_compatibility, regime_admissibility}) it PROVES the canary fails loud on
    a deliberately-emptied table WITHOUT touching the real config — the un-fakeable
    acceptance test. ``fires_on_injected_empty`` is True iff the injected run went
    RED."""
    from monitors.config_canary import run_config_canary
    inj = event.get("inject_empty")
    if isinstance(inj, str):
        inj = [inj]
    live = run_config_canary(alert=False)
    out = {"phase": "config-canary", "nondestructive": True, "live": live}
    if inj:
        injected = run_config_canary(alert=False, force_empty=inj)
        out["injected_empty"] = inj
        out["injected_result"] = injected
        out["fires_on_injected_empty"] = not injected.get("ok")
    return out


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
        from chassis.utils.sns_alerts import get_sns_topic_arn
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
    from chassis.utils.s3_client import S3Client
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


# ------------------------------------------- CU-04 publish-revert fault-injection
def _fire_publish_revert_test_sns(reason: str, region: str) -> Dict[str, Any]:
    """Publish the CU-04 guard-block alarm to the REAL SNS topic, CLEARLY MARKED as
    a reality-test. Its firing IS the 'gate alarms on block' proof. Fail-soft."""
    try:
        import boto3
        from chassis.utils.sns_alerts import get_sns_topic_arn
        sns = boto3.client("sns", region_name=region)
        resp = sns.publish(
            TopicArn=get_sns_topic_arn(region),
            Subject="[REALITY-TEST] CU-04 publish-time value-revert guard BLOCKED a publish"[:100],
            Message=("*** CU-04 PUBLISH-REVERT REALITY-TEST — a deliberately-injected "
                     "contaminated/reverted value was refused by the publish gate; NOT a "
                     "live incident. This proves the guard BLOCKS + ALARMS at publish "
                     "time (would have stopped the original 121147.52 revert). ***\n\n"
                     + reason),
        )
        return {"fired": True, "message_id": resp.get("MessageId"), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"fired": False, "message_id": None, "error": f"{type(e).__name__}: {e}"}


def publish_revert_diag(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """CU-04 reality-test (NON-DESTRUCTIVE; writes NO S3): inject a contaminated /
    reverted value at the PUBLISH gate and prove it BLOCKS + the alarm fires; and
    prove the watchdog now flags a CHALLENGER + SPY value-revert (not just canon).

    Uses the real guard/gate functions against the active canonical ledger on
    S3; only the injected values are synthetic (the fault). Nothing is written, so
    the live dashboard line is untouched — the block is proven by the guard's
    (ok=False) verdict, not by inspecting a mutated object.
    """
    from chassis.utils.s3_client import S3Client
    from chassis.steps.publish_artifacts import guard_publish_not_reverted, _verify_ledger_or_hold
    from monitors.watchdog import _ledger_terminal, evaluate_value_revert
    s3 = S3Client(bucket, region)

    corr, cerr = _ledger_terminal(s3)
    if corr is None:
        return {"phase": "publish-revert-diag", "nondestructive": True,
                "error": f"canonical ledger unreadable: {cerr}"}
    d = corr.get("date")
    base_term = {"date": d, "value": corr.get("value"), "benchmark": corr.get("benchmark")}
    CONTAM = 121147.52

    def dd(term):  # a one-point dashboard_data carrying `term`
        return {"equity_curve": [term]}

    # --- (1) publish-time guard: correct line passes; injected faults BLOCK ---
    ok_base, r_base = guard_publish_not_reverted(dd(dict(base_term)), s3, d)
    ok_c, r_c = guard_publish_not_reverted(dd({**base_term, "value": CONTAM}), s3, d)
    ok_r, r_r = guard_publish_not_reverted(dd({**base_term, "value": base_term["value"] + 5000.0}), s3, d)
    ok_s, r_s = guard_publish_not_reverted(dd({**base_term, "benchmark": base_term["benchmark"] + 5000.0}), s3, d)
    # full gate blocks the write on the injected contaminated terminal too
    gate_c_ok, gate_c_reason = _verify_ledger_or_hold(dd({**base_term, "value": CONTAM}), s3, "morning", d)

    guard_pass_on_correct = bool(ok_base)
    guard_blocks_contaminated = not ok_c
    guard_blocks_reverted_canon = not ok_r
    guard_blocks_reverted_spy = not ok_s
    gate_holds_write = not gate_c_ok

    # fire the marked reality-test alarm iff the guard blocked the contaminated value
    sns = (_fire_publish_revert_test_sns(r_c, region) if guard_blocks_contaminated
           else {"fired": False, "message_id": None, "error": "guard did not block contaminated"})

    # --- (2) watchdog detection dry-run: challenger + SPY revert flagged ---
    clean = evaluate_value_revert(
        dd(dict(base_term)),
        {"shadow_A": [[d, corr.get("comparison")]]}, corr)
    spy_inj = evaluate_value_revert(
        dd({**base_term, "benchmark": base_term["benchmark"] + 6000.0}),
        {"shadow_A": [[d, corr.get("comparison")]]}, corr)
    chal_inj = evaluate_value_revert(
        dd(dict(base_term)),
        {"shadow_A": [[d, CONTAM]]}, corr)
    wd_flags_spy = bool(spy_inj["lines"]["SPY"]["reverted"]) and not clean["reverted"]
    wd_flags_challenger = bool(chal_inj["lines"]["challenger"]["reverted"]) and not clean["reverted"]

    acceptance = {
        "guard_no_false_positive_on_correct_line": guard_pass_on_correct,
        "publish_guard_BLOCKS_contaminated_121147_52": guard_blocks_contaminated,
        "publish_guard_BLOCKS_reverted_canon": guard_blocks_reverted_canon,
        "publish_guard_BLOCKS_reverted_SPY": guard_blocks_reverted_spy,
        "full_gate_HOLDS_the_write": gate_holds_write,
        "alarm_fired_on_block": bool(sns.get("fired")),
        "watchdog_flags_SPY_revert": wd_flags_spy,
        "watchdog_flags_challenger_revert": wd_flags_challenger,
    }
    return {
        "phase": "publish-revert-diag",
        "nondestructive": True,
        "corrected_terminal": {"date": d, "value": corr.get("value"),
                               "benchmark": corr.get("benchmark"),
                               "comparison": corr.get("comparison")},
        "publish_guard": {"correct_line": [ok_base, r_base],
                          "injected_contaminated": [ok_c, r_c],
                          "injected_reverted_canon": [ok_r, r_r],
                          "injected_reverted_spy": [ok_s, r_s],
                          "full_gate_on_contaminated": [gate_c_ok, gate_c_reason]},
        "alarm": sns,
        "watchdog_dryrun": {"clean_reverted": clean["reverted"],
                            "spy_injected": spy_inj["lines"]["SPY"],
                            "challenger_injected": chal_inj["lines"]["challenger"]},
        "acceptance": acceptance,
        "all_pass": all(acceptance.values()),
    }


# ----------------------------------------------------------------- dispatch table
_PROBES = {
    "forecast-diag": forecast_diag,
    "freshness-diag": freshness_diag,
    "regime-diag": regime_diag,
    "canary": canary,
    "config-canary": config_canary,
    "advance-challenger": advance_challenger,
    "watchdog-diag": watchdog_diag,
    "publish-revert-diag": publish_revert_diag,
}


def dispatch(source: str, event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Route an ops-probe ``source`` to its branch. Unknown sources return an
    honest error row (never raise into the router)."""
    fn = _PROBES.get(source)
    if fn is None:
        return {"phase": "ops-probe", "error": f"unknown ops probe: {source!r}",
                "known": sorted(_PROBES)}
    return fn(event or {}, bucket, region)
