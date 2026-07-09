"""APPEND-ONE-POINT-PER-DAY — the advance mechanism (clean core).

Each trading night the displayed line gains exactly ONE settled frontier leaf, and
no prior leaf is ever touched. This is what makes the line advance forever AND never
revert.

How a day's value is formed (operator's standing rule — see trader-bot/CLAUDE.md
banner, 2026-06-25):

  * **The line is the DISPLAYED, ANCHORED return series.** Today's displayed value =
    YESTERDAY'S STORED leaf value (the displayed anchor) x (1 + today's return).
  * **The return engine is real recorded holdings marked at real SETTLED prices**
    (the resim ``raw`` book, ~$95-97k). It is used ONLY to produce the daily RETURN
    (a ratio). Its dollar LEVEL is NEVER the line, NEVER a yardstick.
  * The benchmark line advances the same way off the SPY return engine.

G-LINE-VALUE-FROM-CANON: the line VALUE is the stored anchor scaled by the canon
return — the displayed value is never the raw sim-book dollars.
G-APPEND-IMPORT-WALL: this module imports ONLY the ledger + stdlib/boto3. It never
imports ``execute_trade`` / ``morning_executor`` / ``midday_checker`` — it is a pure
record of a day that already traded.

The return engine (prior-vs-today settled book marks) is computed by the night
publish path, which already holds the marked book; this module receives the returns
and writes the leaf. It reads only the ledger.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from chassis.canon.equity_ledger import BUCKET, CACHE_KEY, EquityLedger

logger = logging.getLogger(__name__)

_DEFAULT_ISSUER = "investment-system-night@trader-bot"


def _cache_tail(s3_client, bucket: str) -> Optional[Dict[str, Any]]:
    """The last ledger cache row (the stored frontier leaf's displayed columns)."""
    try:
        raw = s3_client.get_object(Bucket=bucket, Key=CACHE_KEY)["Body"].read()
    except Exception:  # noqa: BLE001
        return None
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return rows[-1] if rows else None


def _list_daily_dates(s3_client, bucket: str) -> List[str]:
    dates = set()
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": "daily/", "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3_client.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []) or []:
            p = cp.get("Prefix", "")
            d = p[len("daily/"):].rstrip("/")
            if len(d) == 10 and d[4] == "-" and d[7] == "-":
                dates.add(d)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(dates)


def _book_marks(state: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """The return-engine raw book mark + benchmark mark from a portfolio state.

    Reads the raw book value (``sim_book_value`` on published states, or
    ``portfolio_value`` on in-memory/historical states) and the SPY benchmark mark.
    These are the RETURN ENGINE only — never returned as a displayed line value.
    """
    raw = state.get("sim_book_value", state.get("portfolio_value"))
    bm = state.get("benchmark_value")
    try:
        raw = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        raw = None
    try:
        bm = float(bm) if bm is not None else None
    except (TypeError, ValueError):
        bm = None
    return raw, bm


def _prior_settled_marks(s3_client, run_date: str, bucket: str) -> Tuple[Optional[float], Optional[float]]:
    """The most recent prior trading day's settled book + benchmark marks (return base)."""
    dates = [d for d in _list_daily_dates(s3_client, bucket) if d < run_date]
    for d in reversed(dates):
        try:
            state = json.loads(
                s3_client.get_object(Bucket=bucket, Key=f"daily/{d}/portfolio_state.json")["Body"].read())
        except Exception:  # noqa: BLE001
            continue
        raw, bm = _book_marks(state)
        if raw is not None and raw > 0:
            return raw, bm
    return None, None


def _engine_model_id(s3_client, bucket: str) -> str:
    """The live engine identity for the leaf model_id (best-effort)."""
    try:
        cfg = json.loads(
            s3_client.get_object(Bucket=bucket, Key="config/brain.active.json")["Body"].read())
        sha = str(cfg.get("engine_sha", ""))[:8]
        return f"FREEZE_ORB1@{sha}" if sha else "FREEZE_ORB1"
    except Exception:  # noqa: BLE001
        return "FREEZE_ORB1"


def append_settled_point_for_publish(
    s3_client,
    run_date: str,
    portfolio_state: Dict[str, Any],
    *,
    issued_by: str = _DEFAULT_ISSUER,
    bucket: str = BUCKET,
) -> Dict[str, Any]:
    """Append today's settled frontier leaf (the advance), or no-op at/<= frontier.

    ``portfolio_state`` is the IN-MEMORY book the night path just marked at settled
    closes — its raw value is the return engine's today mark (never the line value).
    Returns a small report dict; raises nothing the caller must handle beyond
    logging (the publish path treats any failure as "line holds").
    """
    ledger = EquityLedger(s3_client, bucket)
    tail = _cache_tail(s3_client, bucket)
    if tail is None:
        return {"action": "skip", "reason": "ledger unseeded (no cache tail)"}

    front_date = tail["date"]
    if run_date <= front_date:
        return {"action": "noop", "reason": f"run_date {run_date} <= frontier {front_date}",
                "date": front_date, "value": tail["value"]}

    today_raw, today_bm = _book_marks(portfolio_state)
    prior_raw, prior_bm = _prior_settled_marks(s3_client, run_date, bucket)
    if today_raw is None or today_raw <= 0 or prior_raw is None or prior_raw <= 0:
        return {"action": "skip",
                "reason": f"return engine unavailable (today_raw={today_raw}, prior_raw={prior_raw})"}

    value_return = today_raw / prior_raw - 1.0
    # benchmark return: degrade to flat-hold if a mark is missing
    bm_return = 0.0
    if today_bm is not None and prior_bm is not None and prior_bm > 0:
        bm_return = today_bm / prior_bm - 1.0

    anchor_value = float(tail["value"])
    anchor_bm = float(tail["benchmark"])
    new_value = anchor_value * (1.0 + value_return)
    new_bm = anchor_bm * (1.0 + bm_return)

    leaf = ledger.append(
        date=run_date,
        value=new_value,
        benchmark=new_bm,
        comparison=None,  # the dotted comparison settles out-of-band (sparse shadow)
        segment="new_brain",
        model_id=_engine_model_id(s3_client, bucket),
        source="native_two_stage",
        issued_by=issued_by,
    )
    return {
        "action": "append",
        "date": run_date,
        "value": leaf["value"],
        "benchmark": leaf["benchmark"],
        "value_return": value_return,
        "benchmark_return": bm_return,
        "anchor_from": {"date": front_date, "value": anchor_value},
        "return_engine": {"today_raw": today_raw, "prior_raw": prior_raw},
    }
