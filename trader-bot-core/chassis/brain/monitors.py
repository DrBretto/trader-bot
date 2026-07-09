"""Brain liveness alarms (PKT-TB-012; DESIGN_DOSSIER reports/03 §4).

The laptop shadow's single worst failure was invisible: launchd asleep, nobody
paged, the line silently stops advancing. The cloud cutover replaces that with
two alarms:

  1. **stale-publish alarm** (the single most important new alarm): the
     challenger's last plotted date is more than one trading day behind the
     chassis. ``check_stale_publish`` reads the actual
     ``dashboard/shadow_timeseries.json`` series and ``daily/latest.json`` and
     raises SNS CRITICAL when the displayed line falls behind — exactly the
     laptop-asleep / fresh-file-stale-line failure, made loud.
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
CHALLENGER_SERIES_KEYS = (
    "shadow_A",
    "shadow_F",
    "shadow_U",
    "shadow_E",
    "shadow_B",
    "shadow_R",
    "shadow_I",
)


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
    """Compare the plotted challenger line to the chassis's latest intents date.

    Returns a status dict; raises SNS CRITICAL via ``alert`` when the brain line
    is stale by more than ``max_trading_days`` trading days. ``s3`` is an
    S3Client; ``alert`` defaults to src.utils.sns_alerts.send_alert.
    """
    status: Dict[str, Any] = {"stale": False, "reason": "", "brain_as_of": None,
                              "challenger_last_date": None,
                              "chassis_date": None, "lag_trading_days": 0}
    try:
        shadow = s3.read_json("dashboard/shadow_timeseries.json") or {}
        latest = s3.read_json("daily/latest.json") or {}
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"could not read state: {type(e).__name__}: {e}"
        return status

    as_of = (shadow.get("as_of") or "")[:10]            # YYYY-MM-DD prefix of ISO
    challenger_last = _last_challenger_date(shadow)
    chassis_date = latest.get("intents_date") or latest.get("date")
    status["brain_as_of"] = as_of
    status["challenger_last_date"] = challenger_last
    status["chassis_date"] = chassis_date
    if not challenger_last or not chassis_date:
        status["reason"] = "missing challenger series date or chassis intents_date"
        return status

    lag = _trading_days_between(challenger_last, chassis_date)
    status["lag_trading_days"] = lag
    if lag > max_trading_days:
        status["stale"] = True
        status["reason"] = (f"challenger last plotted date {challenger_last} is {lag} trading days "
                            f"behind chassis {chassis_date} (> {max_trading_days})")
        _fire(alert, status["reason"], as_of, challenger_last, chassis_date, lag)
    return status


def _last_challenger_date(shadow: Dict[str, Any]) -> Optional[str]:
    for key in CHALLENGER_SERIES_KEYS:
        points = shadow.get(key) or []
        if points:
            last = points[-1]
            if isinstance(last, (list, tuple)) and last:
                return str(last[0])
            if isinstance(last, dict) and last.get("date"):
                return str(last["date"])
    return None


def _fire(alert, reason: str, as_of: str, challenger_last: str, chassis: str, lag: int) -> None:
    body = (
        f"STALE-PUBLISH: {reason}\n\n"
        f"brain shadow_timeseries.as_of = {as_of}\n"
        f"challenger last plotted date = {challenger_last}\n"
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
        from chassis.utils.sns_alerts import send_alert
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
    from chassis.utils.s3_client import S3Client
    bucket = (event or {}).get("bucket") or os.environ.get("S3_BUCKET", "investment-system-data")
    region = (event or {}).get("region") or os.environ.get("AWS_REGION", "us-east-1")
    s3 = S3Client(bucket, region)
    return check_stale_publish(s3)


# --------------------------------------------------------------------------- #
# Daily all-three-lines health report (the anti-betrayal watchdog).
#
# An independent check that reads the actual published S3 surfaces and verifies
# each of the THREE displayed lines (canon + SPY benchmark live in the equity
# ledger; the challenger lives in shadow_timeseries.json) actually advanced to
# the latest processed trading day. It ALWAYS emails a status — a positive "✓ all
# current" each day, or a loud "✗ stale" — so the operator never again has to
# eyeball the chart days later to discover the pipeline silently died.
# --------------------------------------------------------------------------- #
def _latest_weekday_on_or_before(today: _dt.date) -> str:
    d = today
    while d.weekday() >= 5:  # Sat/Sun -> step back to Friday
        d -= _dt.timedelta(days=1)
    return d.isoformat()


def check_canon_fresh(s3, chassis_date: str, max_trading_days: int = STALE_TRADING_DAYS) -> Dict[str, Any]:
    """Is the canon equity line (and its SPY benchmark) advanced to the chassis date?

    Reads the ledger cache (``equity_history.jsonl``) terminal date and compares it
    to the chassis's latest processed date.
    """
    status: Dict[str, Any] = {"stale": True, "last_date": None, "lag_trading_days": None}
    try:
        rows = s3.read_jsonl("canon/equity_ledger/equity_history.jsonl") or []
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"could not read equity ledger: {type(e).__name__}: {e}"
        return status
    if not rows:
        status["reason"] = "equity ledger empty / unseeded"
        return status
    last = rows[-1].get("date")
    status["last_date"] = last
    if not chassis_date:
        status["reason"] = "no chassis date to compare"
        return status
    lag = _trading_days_between(last, chassis_date)
    status["lag_trading_days"] = lag
    status["stale"] = lag > max_trading_days
    return status


def check_substrate_fresh(s3, lookback: int = 6, min_identical: int = 3) -> Dict[str, Any]:
    """SUBSTRATE-CURRENCY check (ISSUE-01/02/13) — the one the publish-recency
    checks above STRUCTURALLY cannot make.

    ``check_stale_publish`` / ``check_canon_fresh`` key on *publish recency* (a leaf
    written today, a line whose last plotted date is current). Both stayed GREEN for
    two weeks over a frozen OHLCV substrate: the chassis published a fresh leaf every
    night while the brain's ``mu`` was byte-frozen, so the SAME 10-name
    ``selected_universe`` shipped daily. Recency-of-record is not currency-of-content.

    This check keys on CONTENT: it reads the last ``lookback`` trading-day leaves'
    ``brain_selected_universe.json`` and flags the frozen-``mu`` signature — the
    selected universe byte-identical across the most recent ``min_identical`` DISTINCT
    trading days. That is the S3-observable fingerprint of a frozen substrate; the
    night's fail-loud gate is the primary defense, this is the independent watchdog."""
    status: Dict[str, Any] = {"stale": False, "reason": "", "identical_run": 0,
                              "days_checked": 0, "selected_universe": None,
                              "dates": []}
    try:
        dates = s3.list_daily_dates(max_days=lookback)
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"could not list daily dates: {type(e).__name__}: {e}"
        return status
    if not dates:
        status["reason"] = "no daily leaves found"
        return status

    # newest-first; read each leaf's selected_universe fingerprint
    fps = []  # (date, tuple(selected_universe)) newest-first
    for d in reversed(dates):
        try:
            bsu = s3.read_json(f"daily/{d}/brain_selected_universe.json") or {}
        except Exception:  # noqa: BLE001
            bsu = {}
        sel = bsu.get("selected_universe")
        if sel:
            fps.append((d, tuple(sel)))
    status["days_checked"] = len(fps)
    if len(fps) < min_identical:
        status["reason"] = (f"only {len(fps)} leaf universe(s) available "
                            f"(< {min_identical}); cannot judge substrate freshness")
        return status

    # length of the leading run of byte-identical selected_universe (newest-first)
    head = fps[0][1]
    run = 1
    for _, sel in fps[1:]:
        if sel == head:
            run += 1
        else:
            break
    status["identical_run"] = run
    status["selected_universe"] = list(head)
    status["dates"] = [d for d, _ in fps[:run]]
    if run >= min_identical:
        status["stale"] = True
        status["reason"] = (
            f"FROZEN-SUBSTRATE signature: brain selected_universe byte-identical across "
            f"{run} consecutive trading days {status['dates']} — mu is not advancing "
            f"(a frozen OHLCV substrate behind a fresh-looking publish)")
    return status


def run_daily_health_check(s3, alert: Optional[Callable[[str, str], Any]] = None,
                           today: Optional[str] = None) -> Dict[str, Any]:
    """Check all three lines + chassis liveness and ALWAYS email a status.

    Returns a status dict and sends one email: ``✓`` when every line is current,
    ``CRITICAL ✗`` naming whichever is stale. Fail-soft: never raises.
    """
    out: Dict[str, Any] = {"ok": False, "lines": {}}
    try:
        latest = s3.read_json("daily/latest.json") or {}
    except Exception as e:  # noqa: BLE001
        latest = {}
        out["latest_read_error"] = f"{type(e).__name__}: {e}"
    chassis_date = latest.get("intents_date") or latest.get("date")
    today_str = today or _dt.date.today().isoformat()
    expected = _latest_weekday_on_or_before(_dt.date.fromisoformat(today_str))

    # 1) chassis liveness: did the night pipeline run at all recently?
    chassis_lag = _trading_days_between(chassis_date, expected) if chassis_date else None
    chassis_stale = (chassis_date is None) or (chassis_lag is not None and chassis_lag > STALE_TRADING_DAYS)

    # 2) the three lines (canon+benchmark via the ledger; challenger via shadow)
    canon = check_canon_fresh(s3, chassis_date)
    shadow = check_stale_publish(s3, alert=lambda *a, **k: None)  # don't double-alert here

    lines = {
        "canon (New Brain)": {"stale": canon.get("stale"), "at": canon.get("last_date"),
                              "lag": canon.get("lag_trading_days")},
        "SPY benchmark": {"stale": canon.get("stale"), "at": canon.get("last_date"),
                          "lag": canon.get("lag_trading_days")},
        "challenger (dotted)": {"stale": shadow.get("stale"), "at": shadow.get("challenger_last_date"),
                                "lag": shadow.get("lag_trading_days")},
    }
    out["lines"] = lines
    out["chassis"] = {"date": chassis_date, "expected": expected, "stale": chassis_stale}

    # 3) SUBSTRATE currency (ISSUE-01/02/13) — the check publish-recency cannot make:
    #    is `mu` actually advancing, or is a frozen substrate shipping the same
    #    selected_universe behind fresh-looking leaves?
    substrate = check_substrate_fresh(s3)
    out["substrate"] = substrate

    any_stale = (chassis_stale or any(v.get("stale") for v in lines.values())
                 or substrate.get("stale"))
    out["ok"] = not any_stale

    def _mark(v):
        return "✗ STALE" if v.get("stale") else "✓"
    body_lines = [
        f"Expected latest trading day: {expected}",
        f"Pipeline last processed (chassis): {chassis_date}  {'✗ STALE' if chassis_stale else '✓'}",
        "",
        "Three displayed lines:",
    ]
    for name, v in lines.items():
        body_lines.append(f"  {_mark(v):<8} {name:<22} at {v.get('at')}  (lag {v.get('lag')} td)")
    body_lines += [
        "",
        "Substrate currency (is mu actually advancing?):",
        f"  {'✗ FROZEN' if substrate.get('stale') else '✓':<8} "
        f"brain selected_universe identical-run = {substrate.get('identical_run')} "
        f"day(s) over {substrate.get('days_checked')} checked"
        + (f"  [{substrate.get('reason')}]" if substrate.get('stale') else ""),
    ]
    body = "\n".join(body_lines)

    if any_stale:
        subject = ("[TraderBot] CRITICAL: forecast substrate FROZEN"
                   if substrate.get("stale") and not (chassis_stale or
                       any(v.get("stale") for v in lines.values()))
                   else "[TraderBot] CRITICAL: a displayed line did NOT advance")
        body = ("ONE OR MORE FRESHNESS CHECKS ARE STALE — the dashboard/forecast is "
                "not current.\n\n"
                + body +
                "\n\nCheck CloudWatch /aws/lambda/investment-system-daily-pipeline "
                "and the shadow-publish run.")
    else:
        subject = f"[TraderBot] daily health ✓ all three lines current ({expected})"
        body = "All three displayed lines advanced to the latest trading day.\n\n" + body

    try:
        if alert is not None:
            alert(subject, body)
        else:
            from chassis.utils.sns_alerts import send_alert
            send_alert(subject=subject, body=body)
    except Exception as e:  # noqa: BLE001 — emailing must never crash the check
        print(f"  daily health email failed (non-fatal): {e}")
    out["subject"] = subject
    return out


def daily_health_handler(event: dict, context) -> dict:
    """Lambda entry for the scheduled daily three-line health report."""
    import os
    from chassis.utils.s3_client import S3Client
    bucket = (event or {}).get("bucket") or os.environ.get("S3_BUCKET", "investment-system-data")
    region = (event or {}).get("region") or os.environ.get("AWS_REGION", "us-east-1")
    s3 = S3Client(bucket, region)
    return run_daily_health_check(s3)
