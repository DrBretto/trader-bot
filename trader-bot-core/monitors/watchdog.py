"""Three-line watchdog + daily health email (clean spine — P8).

The two-week silent freeze happened because NOTHING watched whether the three
displayed lines actually kept advancing. This module is the always-on version of
the P7 live-canaries: an independent check that reads the *actual* published S3
surfaces and verifies each of the THREE displayed lines advanced to the latest
settled trading day, is POPULATED (its terminal value is present, not null), and
is not stale — then ALWAYS emails a ✓/✗ status so a freeze/degrade is impossible
to miss.

The three lines and where they live:

  * **canon (New Brain)** — the ledger terminal ``value`` (content-addressed
    ``canon/equity_ledger/equity_history.jsonl``).
  * **SPY benchmark** — the ledger terminal ``benchmark`` (rides the SAME
    settled-day grid as canon, checked for its own populated-ness independently).
  * **challenger (dotted)** — the last plotted point of the shadow series
    (``dashboard/shadow_timeseries.json``).

Relocated as KEEP-code from ``src/brain/monitors.py`` (the anti-betrayal
watchdog) with two clean-spine sharpenings the P8 packet asks for: (a) each line
carries its OWN populated check (terminal value / benchmark / challenger point is
present, not just a fresh date), and (b) canon and SPY are reported as genuinely
independent lines. The substrate-currency check is imported VERBATIM from
``monitors.substrate`` (no duplication). All functions fail soft — an alarm path
that itself errors logs and returns rather than crashing the caller.

The two alarms:

  1. **stale alarm** — any displayed line more than ``STALE_TRADING_DAYS`` trading
     days behind the expected settled day, or unpopulated: SNS CRITICAL + ✗ email.
  2. **missed-run alarm** — no successful night invocation. ``emit_run_heartbeat``
     puts the custom CloudWatch metric a "no datapoints in N hours" alarm watches;
     in-band, a chassis whose ``intents_date`` stops advancing trips the stale
     alarm the SAME day (the skipped-night signal that was invisible before).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict, Optional, Tuple

from .substrate import check_substrate_fresh  # VERBATIM freeze-signature check

STALE_TRADING_DAYS = 1          # a line may lag the expected settled day by at most this
BRAIN_NAMESPACE = "TraderBot/Brain"
# MORNING/MIDDAY PATH REVERT HOTFIX (2026-07-06): the LIVE canon ledger is the
# corrected, replay-seeded clean_v2 (canon/equity_ledger_clean_v2/) — the same
# ledger the clean-core night path reads/appends/publishes. The watchdog must
# check the ledger production actually runs on, not the retired contaminated one.
LEDGER_CACHE_KEY = "canon/equity_ledger_clean_v2/equity_history.jsonl"
SHADOW_KEY = "dashboard/shadow_timeseries.json"
LATEST_KEY = "daily/latest.json"
# The PUBLISHED dashboard the frontend serves — the value-revert check reads its
# terminal and confirms it still matches the corrected ledger terminal (below).
PUBLISHED_DASHBOARD_KEY = "dashboard/dashboard.json"
# A published terminal must equal the corrected-ledger terminal for the SAME date
# within this many dollars. A publish path that reverts the line to a different
# source (the exact failure this hotfix closes) diverges by thousands, not cents.
VALUE_REVERT_EPS = 1.0
# Known-contaminated terminal values (the pre-P6 line the retired
# canon/equity_ledger/ carried). A published terminal landing on one of these is
# a hard contamination signal even if — for any reason — the corrected-ledger
# cross-check is unavailable. Compared at cent precision (the line is dollars).
CONTAMINATED_TERMINAL_VALUES = frozenset({12114752})  # 121147.52 * 100 (07-02 pre-P6)
CHALLENGER_SERIES_KEYS = (
    "shadow_A", "shadow_F", "shadow_U", "shadow_E",
    "shadow_B", "shadow_R", "shadow_I", "live_line",
)


# --------------------------------------------------------------------------- #
# Calendar helpers (cheap NYSE proxy — holidays make it CONSERVATIVE, never
# falsely loud: a holiday can only make the "expected" day one trading day too
# far ahead, and STALE_TRADING_DAYS absorbs exactly that).
# --------------------------------------------------------------------------- #
def _trading_days_between(d0: Optional[str], d1: Optional[str]) -> int:
    """Weekday count strictly between two YYYY-MM-DD dates."""
    try:
        a = _dt.date.fromisoformat(str(d0))
        b = _dt.date.fromisoformat(str(d1))
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


def _latest_weekday_on_or_before(today: _dt.date) -> str:
    d = today
    while d.weekday() >= 5:  # Sat/Sun -> step back to Friday
        d -= _dt.timedelta(days=1)
    return d.isoformat()


# --------------------------------------------------------------------------- #
# Per-line freshness+populated checks. Each returns
# {stale, last_date, value, populated, lag_trading_days, reason}.
# --------------------------------------------------------------------------- #
def _ledger_terminal(s3) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Terminal leaf of the content-addressed equity ledger, or (None, reason)."""
    try:
        rows = s3.read_jsonl(LEDGER_CACHE_KEY) or []
    except Exception as e:  # noqa: BLE001
        return None, f"could not read equity ledger: {type(e).__name__}: {e}"
    if not rows:
        return None, "equity ledger empty / unseeded"
    return rows[-1], None


def _line_status(last_date: Optional[str], value: Any, expected: str,
                 max_trading_days: int) -> Dict[str, Any]:
    """Shared roll-up: a line is stale if it is unpopulated OR its date lags the
    expected settled day by more than ``max_trading_days``."""
    populated = value is not None
    lag = _trading_days_between(last_date, expected) if last_date else None
    date_stale = (last_date is None) or (lag is not None and lag > max_trading_days)
    stale = date_stale or not populated
    reason = ""
    if not populated:
        reason = "terminal value is null/absent (line not populated)"
    elif date_stale:
        reason = (f"last date {last_date} lags expected {expected} by "
                  f"{lag} trading day(s) (> {max_trading_days})")
    return {"stale": stale, "last_date": last_date, "value": value,
            "populated": populated, "lag_trading_days": lag, "reason": reason}


def check_canon_line(s3, expected: str, terminal: Optional[Dict[str, Any]] = None,
                     max_trading_days: int = STALE_TRADING_DAYS) -> Dict[str, Any]:
    """canon (New Brain): ledger terminal ``value`` advanced + populated."""
    if terminal is None:
        terminal, reason = _ledger_terminal(s3)
        if terminal is None:
            return {"stale": True, "last_date": None, "value": None,
                    "populated": False, "lag_trading_days": None, "reason": reason}
    return _line_status(terminal.get("date"), terminal.get("value"),
                        expected, max_trading_days)


def check_spy_line(s3, expected: str, terminal: Optional[Dict[str, Any]] = None,
                   max_trading_days: int = STALE_TRADING_DAYS) -> Dict[str, Any]:
    """SPY benchmark: ledger terminal ``benchmark`` advanced + populated (its own
    grid-mate on the canon settled-day grid, checked independently)."""
    if terminal is None:
        terminal, reason = _ledger_terminal(s3)
        if terminal is None:
            return {"stale": True, "last_date": None, "value": None,
                    "populated": False, "lag_trading_days": None, "reason": reason}
    return _line_status(terminal.get("date"), terminal.get("benchmark"),
                        expected, max_trading_days)


def _last_challenger_point(shadow: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """(date, value) of the last plotted challenger point across the known series
    keys. Tolerant of both ``[date, value]`` pairs and ``{date, value}`` dicts."""
    for key in CHALLENGER_SERIES_KEYS:
        points = shadow.get(key) or []
        if points:
            last = points[-1]
            if isinstance(last, (list, tuple)) and last:
                return (str(last[0]), last[1] if len(last) > 1 else None)
            if isinstance(last, dict) and last.get("date"):
                return (str(last["date"]), last.get("value"))
    return (None, None)


def check_challenger_line(s3, expected: str,
                          max_trading_days: int = STALE_TRADING_DAYS) -> Dict[str, Any]:
    """challenger (dotted): last plotted shadow point advanced + populated."""
    try:
        shadow = s3.read_json(SHADOW_KEY) or {}
    except Exception as e:  # noqa: BLE001
        return {"stale": True, "last_date": None, "value": None, "populated": False,
                "lag_trading_days": None,
                "reason": f"could not read shadow series: {type(e).__name__}: {e}"}
    last_date, value = _last_challenger_point(shadow)
    st = _line_status(last_date, value, expected, max_trading_days)
    st["as_of"] = (shadow.get("as_of") or "")[:19]
    return st


def _cents(v: Any) -> Optional[int]:
    """Round a dollar value to integer cents, or None if not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return int(round(f * 100))


def check_value_revert(s3, terminal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Value-revert / contamination check (MORNING/MIDDAY PATH REVERT HOTFIX).

    The three-line watchdog caught staleness and non-advancement, but NOT a
    line whose DATE stayed current while its VALUE reverted to a known-contaminated
    terminal — exactly what a publish path reading the retired ledger did every
    weekday morning (07-02 stayed put while the value flipped 114271.38 -> 121147.52).

    This reads the PUBLISHED dashboard terminal and refuses to call the system
    healthy when that terminal (a) does not match the corrected ledger's terminal
    for the SAME date within ``VALUE_REVERT_EPS``, or (b) lands on a known
    contaminated value. Returns {reverted, published, ledger, reason}; reverted=True
    trips the CRITICAL email. Fail-soft: never raises.
    """
    out: Dict[str, Any] = {"reverted": False, "published": None, "ledger": None,
                           "reason": ""}
    try:
        dash = s3.read_json(PUBLISHED_DASHBOARD_KEY) or {}
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"could not read published dashboard: {type(e).__name__}: {e}"
        return out
    ec = dash.get("equity_curve") or []
    if not ec:
        out["reason"] = "published dashboard has no equity_curve (nothing to check)"
        return out
    pub = ec[-1] or {}
    pub_date, pub_val = pub.get("date"), pub.get("value")
    out["published"] = {"date": pub_date, "value": pub_val}

    pub_cents = _cents(pub_val)
    # (b) hard contaminated-value sentinel — fires even if the ledger read fails.
    if pub_cents is not None and pub_cents in CONTAMINATED_TERMINAL_VALUES:
        out["reverted"] = True
        out["reason"] = (f"published terminal {pub_date}={pub_val} is a KNOWN "
                         f"CONTAMINATED value (pre-P6 line) — publish path reverted "
                         f"the line to the retired ledger")
        return out

    # (a) cross-check against the corrected ledger terminal for the SAME date.
    if terminal is None:
        terminal, term_reason = _ledger_terminal(s3)
        if terminal is None:
            out["reason"] = (f"corrected ledger unreadable for value cross-check "
                             f"({term_reason}); sentinel check passed")
            return out
    out["ledger"] = {"date": terminal.get("date"), "value": terminal.get("value")}
    led_cents = _cents(terminal.get("value"))
    # Only cross-check when the published terminal is on the ledger's terminal date;
    # an intraday provisional dot on a LATER date is legitimately ahead of the
    # settled ledger and is not a revert.
    if pub_date == terminal.get("date"):
        if pub_cents is None or led_cents is None:
            out["reverted"] = True
            out["reason"] = (f"published terminal value {pub_val} or ledger value "
                             f"{terminal.get('value')} is not a finite number")
        elif abs(pub_cents - led_cents) > int(round(VALUE_REVERT_EPS * 100)):
            out["reverted"] = True
            out["reason"] = (f"published terminal {pub_date}={pub_val} DIVERGES from "
                             f"corrected ledger {terminal.get('date')}="
                             f"{terminal.get('value')} by "
                             f"${abs(pub_cents - led_cents) / 100:.2f} "
                             f"(> ${VALUE_REVERT_EPS:.2f}) — line reverted to a "
                             f"different-source value")
        else:
            out["reason"] = (f"published terminal matches corrected ledger "
                             f"({pub_date}={pub_val})")
    else:
        out["reason"] = (f"published terminal date {pub_date} != ledger terminal "
                         f"date {terminal.get('date')} — provisional/ahead, not "
                         f"cross-checked for value revert")
    return out


# --------------------------------------------------------------------------- #
# The daily health report — ALWAYS emails a ✓/✗ status.
# --------------------------------------------------------------------------- #
def run_daily_health_check(s3, alert: Optional[Callable[[str, str], Any]] = None,
                           today: Optional[str] = None) -> Dict[str, Any]:
    """Check all three lines + chassis liveness + substrate currency and ALWAYS
    email a status. Returns a status dict with real advancing dates per line and
    sends ONE email: ``✓`` when every line is current+populated, ``CRITICAL ✗``
    naming whichever is stale. Fail-soft: never raises.

    ``alert(subject, body)`` overrides the SNS sender (used by the reality-test
    to capture without emailing, and to fire a marked reality-test SNS).
    """
    out: Dict[str, Any] = {"ok": False, "lines": {}}
    try:
        latest = s3.read_json(LATEST_KEY) or {}
    except Exception as e:  # noqa: BLE001
        latest = {}
        out["latest_read_error"] = f"{type(e).__name__}: {e}"
    chassis_date = latest.get("intents_date") or latest.get("date")
    today_str = today or _dt.date.today().isoformat()
    expected = _latest_weekday_on_or_before(_dt.date.fromisoformat(today_str))

    # 1) chassis liveness — did a night invocation land recently? A chassis whose
    #    intents_date stops advancing IS the skipped-night / missed-run signal.
    chassis_lag = _trading_days_between(chassis_date, expected) if chassis_date else None
    chassis_stale = (chassis_date is None) or (chassis_lag is not None
                                               and chassis_lag > STALE_TRADING_DAYS)

    # 2) the three displayed lines (canon+SPY off one ledger terminal read).
    terminal, _term_reason = _ledger_terminal(s3)
    canon = check_canon_line(s3, expected, terminal=terminal)
    spy = check_spy_line(s3, expected, terminal=terminal)
    challenger = check_challenger_line(s3, expected)
    lines = {
        "canon (New Brain)": canon,
        "SPY benchmark": spy,
        "challenger (dotted)": challenger,
    }
    out["lines"] = {name: {"stale": v.get("stale"), "at": v.get("last_date"),
                           "populated": v.get("populated"),
                           "lag": v.get("lag_trading_days"),
                           "reason": v.get("reason")}
                    for name, v in lines.items()}
    out["chassis"] = {"date": chassis_date, "expected": expected,
                      "lag": chassis_lag, "stale": chassis_stale}

    # 3) SUBSTRATE currency — the check publish-recency structurally cannot make.
    substrate = check_substrate_fresh(s3)
    out["substrate"] = substrate

    # 4) VALUE-REVERT — the published line's DATE can stay current while its VALUE
    #    reverts to a contaminated terminal (the freshness checks above are blind to
    #    it). Cross-check the published terminal against the corrected ledger.
    value_revert = check_value_revert(s3, terminal=terminal)
    out["value_revert"] = value_revert

    any_stale = (chassis_stale or any(v.get("stale") for v in lines.values())
                 or substrate.get("stale") or value_revert.get("reverted"))
    out["ok"] = not any_stale

    # ----- compose the email body: ✓/✗ per line with real advancing dates -----
    def _mark(v: Dict[str, Any]) -> str:
        return "✗ STALE" if v.get("stale") else "✓"

    body_lines = [
        f"Expected latest settled trading day: {expected}",
        f"Pipeline last processed (chassis): {chassis_date}  "
        f"{'✗ STALE (no advancing night)' if chassis_stale else '✓'}",
        "",
        "Three displayed lines:",
    ]
    for name, v in lines.items():
        pop = "" if v.get("populated") else " [UNPOPULATED]"
        body_lines.append(
            f"  {_mark(v):<8} {name:<22} at {v.get('last_date')}  "
            f"(lag {v.get('lag_trading_days')} td){pop}"
            + (f"  — {v.get('reason')}" if v.get("stale") and v.get("reason") else "")
        )
    body_lines += [
        "",
        "Substrate currency (is mu actually advancing?):",
        f"  {'✗ FROZEN' if substrate.get('stale') else '✓':<8} "
        f"brain selected_universe identical-run = {substrate.get('identical_run')} "
        f"day(s) over {substrate.get('days_checked')} checked"
        + (f"  [{substrate.get('reason')}]" if substrate.get("stale") else ""),
    ]
    _vr_pub = value_revert.get("published") or {}
    body_lines += [
        "",
        "Published-line value integrity (did the line revert to a contaminated value?):",
        f"  {'✗ REVERTED' if value_revert.get('reverted') else '✓':<10} "
        f"published terminal {_vr_pub.get('date')}={_vr_pub.get('value')}"
        + (f"  — {value_revert.get('reason')}" if value_revert.get("reason") else ""),
    ]
    body = "\n".join(body_lines)

    if any_stale:
        only_reverted = (value_revert.get("reverted") and not chassis_stale
                         and not substrate.get("stale")
                         and not any(v.get("stale") for v in lines.values()))
        only_frozen = (substrate.get("stale") and not chassis_stale
                       and not value_revert.get("reverted")
                       and not any(v.get("stale") for v in lines.values()))
        if only_reverted:
            subject = "[TraderBot] CRITICAL: published line REVERTED to a contaminated value"
        elif only_frozen:
            subject = "[TraderBot] CRITICAL: forecast substrate FROZEN"
        else:
            subject = "[TraderBot] CRITICAL: a displayed line did NOT advance"
        body = ("ONE OR MORE FRESHNESS CHECKS ARE STALE — the dashboard/forecast is "
                "not current.\n\n" + body +
                "\n\nCheck CloudWatch /aws/lambda/investment-system-daily-pipeline "
                "and the shadow-publish run.")
    else:
        subject = f"[TraderBot] daily health ✓ all three lines current ({expected})"
        body = ("All three displayed lines advanced to the latest settled trading "
                "day and are populated.\n\n" + body)

    try:
        if alert is not None:
            alert(subject, body)
        else:
            from src.utils.sns_alerts import send_alert
            send_alert(subject=subject, body=body)
    except Exception as e:  # noqa: BLE001 — emailing must never crash the check
        print(f"  daily health email failed (non-fatal): {e}")
    out["subject"] = subject
    out["body"] = body
    return out


# --------------------------------------------------------------------------- #
# Missed-run heartbeat metric (the CloudWatch "no datapoints" alarm watches this).
# --------------------------------------------------------------------------- #
def emit_run_heartbeat(region: str = "us-east-1", ok: bool = True) -> None:
    """Put the custom CloudWatch metric the missed-run alarm watches. Call at the
    END of a successful night invocation; a "no datapoints in N hours" alarm on
    ``TraderBot/Brain BrainNightOK`` then trips when a run is missed. Fail-soft."""
    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=region)
        cw.put_metric_data(
            Namespace=BRAIN_NAMESPACE,
            MetricData=[{"MetricName": "BrainNightOK",
                         "Value": 1.0 if ok else 0.0, "Unit": "Count"}],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  brain heartbeat metric failed (non-fatal): {e}")


# --------------------------------------------------------------------------- #
# Lambda entries (an EventBridge schedule points here; see infrastructure/).
# --------------------------------------------------------------------------- #
def daily_health_handler(event: dict, context) -> dict:
    """Lambda entry for the scheduled daily three-line health report."""
    import os
    from src.utils.s3_client import S3Client
    bucket = (event or {}).get("bucket") or os.environ.get("S3_BUCKET",
                                                            "investment-system-data")
    region = (event or {}).get("region") or os.environ.get("AWS_REGION", "us-east-1")
    s3 = S3Client(bucket, region)
    return run_daily_health_check(s3, today=(event or {}).get("today"))
