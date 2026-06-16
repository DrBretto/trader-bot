"""Brain liveness alarms (PKT-TB-012; DESIGN_DOSSIER reports/03 §4).

The laptop shadow's single worst failure was invisible: launchd asleep, nobody
paged, the line silently stops advancing. The cloud cutover replaces that with
two alarms:

  1. **stale-publish alarm** (the single most important new alarm): the brain's
     last publish is more than one trading day behind the chassis. ``check_stale_
     publish`` reads ``dashboard/shadow_timeseries.json.as_of`` and
     ``daily/latest.json.intents_date`` and raises SNS CRITICAL when the brain
     line falls behind — exactly the laptop-asleep failure, made loud.
  2. **missed-run alarm**: no successful night invocation in N hours. This is a
     CloudWatch metric-filter / "no datapoints" alarm on the night rule (infra,
     wired by the operator); ``emit_run_heartbeat`` puts the custom metric the
     alarm watches so a missed run trips it.

``stale_publish_handler`` is the Lambda entry the operator points a small
EventBridge schedule at. All functions fail soft: an alarm path that itself
errors logs and returns rather than crashing.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict, Optional

STALE_TRADING_DAYS = 1          # brain may lag the chassis by at most this much
BRAIN_NAMESPACE = "TraderBot/Brain"


def _trading_days_between(d0: str, d1: str) -> int:
    """Weekday count strictly between two YYYY-MM-DD dates (a cheap NYSE proxy;
    holidays make this conservative, never falsely loud)."""
    try:
        a = _dt.date.fromisoformat(d0)
        b = _dt.date.fromisoformat(d1)
    except Exception:  # noqa: BLE001
        return 0
    if b <= a:
        return 0
    n = 0
    cur = a
    while cur < b:
        cur += _dt.timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def check_stale_publish(
    s3,
    alert: Optional[Callable[[str, str], Any]] = None,
    max_trading_days: int = STALE_TRADING_DAYS,
) -> Dict[str, Any]:
    """Compare the brain's last publish to the chassis's latest intents date.

    Returns a status dict; raises SNS CRITICAL via ``alert`` when the brain line
    is stale by more than ``max_trading_days`` trading days. ``s3`` is an
    S3Client; ``alert`` defaults to src.utils.sns_alerts.send_alert.
    """
    status: Dict[str, Any] = {"stale": False, "reason": "", "brain_as_of": None,
                              "chassis_date": None, "lag_trading_days": 0}
    try:
        shadow = s3.read_json("dashboard/shadow_timeseries.json") or {}
        latest = s3.read_json("daily/latest.json") or {}
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"could not read state: {type(e).__name__}: {e}"
        return status

    as_of = (shadow.get("as_of") or "")[:10]            # YYYY-MM-DD prefix of ISO
    chassis_date = latest.get("intents_date") or latest.get("date")
    status["brain_as_of"] = as_of
    status["chassis_date"] = chassis_date
    if not as_of or not chassis_date:
        status["reason"] = "missing brain as_of or chassis intents_date"
        return status

    lag = _trading_days_between(as_of, chassis_date)
    status["lag_trading_days"] = lag
    if lag > max_trading_days:
        status["stale"] = True
        status["reason"] = (f"brain last publish {as_of} is {lag} trading days "
                            f"behind chassis {chassis_date} (> {max_trading_days})")
        _fire(alert, status["reason"], as_of, chassis_date, lag)
    return status


def _fire(alert, reason: str, as_of: str, chassis: str, lag: int) -> None:
    body = (
        f"STALE-PUBLISH: {reason}\n\n"
        f"brain shadow_timeseries.as_of = {as_of}\n"
        f"chassis daily/latest.intents_date = {chassis}\n"
        f"lag = {lag} trading day(s)\n\n"
        "This is the laptop-asleep failure mode made loud: the New Brain line has "
        "stopped advancing while the chassis kept running. Check the night Lambda "
        "/ CloudWatch."
    )
    try:
        if alert is not None:
            alert("[TraderBot] CRITICAL: New Brain publish is STALE (>1 trading day behind)", body)
            return
        from src.utils.sns_alerts import send_alert
        send_alert(
            subject="[TraderBot] CRITICAL: New Brain publish is STALE (>1 trading day behind)",
            body=body,
        )
    except Exception as e:  # noqa: BLE001 — alerting must never crash the check
        print(f"  stale-publish alert failed (non-fatal): {e}")


def emit_run_heartbeat(region: str = "us-east-1", ok: bool = True) -> None:
    """Put the custom CloudWatch metric the missed-run alarm watches. Call at the
    end of a successful night brain run; a "no datapoints in N hours" alarm on
    ``TraderBot/Brain BrainNightOK`` then trips when a run is missed. Fail-soft."""
    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=region)
        cw.put_metric_data(
            Namespace=BRAIN_NAMESPACE,
            MetricData=[{
                "MetricName": "BrainNightOK",
                "Value": 1.0 if ok else 0.0,
                "Unit": "Count",
            }],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  brain heartbeat metric failed (non-fatal): {e}")


def stale_publish_handler(event: dict, context) -> dict:
    """Lambda entry for the scheduled stale-publish check (operator points a
    small EventBridge schedule here). Returns the status dict."""
    import os
    from src.utils.s3_client import S3Client
    bucket = (event or {}).get("bucket") or os.environ.get("S3_BUCKET", "investment-system-data")
    region = (event or {}).get("region") or os.environ.get("AWS_REGION", "us-east-1")
    s3 = S3Client(bucket, region)
    return check_stale_publish(s3)
