"""Equity LINE read-side — the chart's three displayed lines, folded from the ledger.

FP-08-3. The displayed equity line stops being a recompute (``extend_dashboard`` +
``dashboard_metrics`` re-chain) and becomes a pure projection of the append-only
``src/canon/equity_ledger.py`` ledger. This module:

  * reads the ledger cache (``equity_history.jsonl``) — the deterministic fold over
    the write-once leaves — and renders the dashboard ``equity_curve`` rows + the
    line-derived metrics (drawdowns, monthly returns, ytd/mtd, sharpe, max-dd);
  * exposes ``assert_ledger_parity`` — the per-publish parity-or-hold gate: the
    rendered line must equal the ledger value/benchmark EXACTLY (no arithmetic
    between the stored fact and the displayed number), and the terminal must match
    the operator-certified pin, or the publish HOLDS last-known-good.

The metric math reproduces the retired ``extender.extend_dashboard`` exactly so the
repoint changes NO displayed number (the parity requirement). Nothing here reads
``sim_book_value`` or recomputes a curve — the values are the stored leaves.

Pure + import-light: stdlib + ``equity_ledger`` only. No torch/pandas, no
``execute_trade``/morning/midday imports (it is read on the lightweight publish path).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from lines.ledger import BUCKET, CACHE_KEY, EquityLedger

# Segment -> the per-segment marker field the frontend PerformanceChart colors by.
_SEGMENT_MARKER = {
    "frozen_champion": "champion_frozen_value",
    "new_brain": "new_brain_value",
    "incumbent": "incumbent_value",
}
_ALL_MARKERS = ("champion_frozen_value", "new_brain_value", "incumbent_value")


def read_cache_rows(s3_client, bucket: str = BUCKET) -> List[Dict[str, Any]]:
    """Read the ledger cache (``equity_history.jsonl``) into ordered row dicts.

    Each row: ``{date, value, benchmark, comparison, segment, model_id}`` — the
    deterministic fold the ledger persists. Empty list if the cache is absent.
    """
    try:
        raw = s3_client.get_object(Bucket=bucket, Key=CACHE_KEY)["Body"].read()
    except Exception:  # noqa: BLE001 — absent cache is a normal "not seeded yet" state
        return []
    rows = []
    for line in raw.decode().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _equity_curve_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render the dashboard ``equity_curve`` rows from ledger rows.

    Reproduces the field shape the retired extender emitted so every chart
    (EquityCurve, PerformanceChart) renders byte-identically: ``value`` +
    ``benchmark`` are the two main lines; ``optimized_value`` is the legacy alias
    (== value); the per-segment markers (champion_frozen/new_brain/incumbent) carry
    the value on their own segment and null elsewhere so the segmented chart colors
    correctly. ``cumulative_external_cashflow`` is 0.0 (the ledger is cashflow-clean).
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        seg = r.get("segment")
        marker_field = _SEGMENT_MARKER.get(seg)
        row: Dict[str, Any] = {
            "date": r["date"],
            "value": r["value"],
            "benchmark": r["benchmark"],
            "cumulative_external_cashflow": 0.0,
            "optimized_value": r["value"],
        }
        for m in _ALL_MARKERS:
            row[m] = r["value"] if m == marker_field else None
        out.append(row)
    return out


def _drawdowns(values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    peak = 0.0
    dd: List[Dict[str, Any]] = []
    for r in values:
        v = r["value"]
        if v > peak:
            peak = v
        dd.append({"date": r["date"], "drawdown": (v - peak) / peak if peak > 0 else 0.0})
    return dd


def _monthly_returns(values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    prev = None
    for r in values:
        v = r["value"]
        if prev is not None and prev > 0:
            ds = datetime.strptime(r["date"], "%Y-%m-%d")
            buckets[(ds.year, ds.month)].append((v - prev) / prev)
        prev = v
    out = []
    for (year, month), rets in sorted(buckets.items()):
        compound = 1.0
        for x in rets:
            compound *= (1 + x)
        out.append({"year": year, "month": month,
                    "return_pct": compound - 1, "observations": len(rets)})
    return out


def _line_metrics(values: List[Dict[str, Any]], drawdowns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {}
    today = values[-1]["date"]
    total_value = values[-1]["value"]

    def find_v(target: str) -> Optional[float]:
        for r in values:
            if r["date"] >= target:
                return r["value"]
        return None

    ytd_base = find_v(f"{today[:4]}-01-01")
    mtd_base = find_v(f"{today[:7]}-01")
    ytd_return = (total_value / ytd_base - 1) if ytd_base and ytd_base > 0 else 0.0
    mtd_return = (total_value / mtd_base - 1) if mtd_base and mtd_base > 0 else 0.0

    daily_returns: List[float] = []
    prev = None
    for r in values:
        if prev is not None and prev > 0:
            daily_returns.append((r["value"] - prev) / prev)
        prev = r["value"]
    sharpe = None
    if len(daily_returns) >= 60:
        sd = stdev(daily_returns)
        sharpe = (mean(daily_returns) / sd * math.sqrt(252)) if sd > 0 else None

    max_dd = min((d["drawdown"] for d in drawdowns), default=0.0)
    current_dd = drawdowns[-1]["drawdown"] if drawdowns else 0.0
    return {
        "total_value": total_value,
        "ytd_return": ytd_return,
        "mtd_return": mtd_return,
        "sharpe_ratio": sharpe,
        "sharpe_observations": len(daily_returns),
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
    }


def build_line_view(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold ledger rows into the dashboard line surface.

    Returns ``{equity_curve, drawdowns, monthly_returns, line_metrics, terminal}``
    — a pure projection of the stored leaves. No recompute, no sim_book.
    """
    equity_curve = _equity_curve_from_rows(rows)
    values = [{"date": r["date"], "value": r["value"]} for r in rows]
    drawdowns = _drawdowns(values)
    monthly = _monthly_returns(values)
    metrics = _line_metrics(values, drawdowns)
    terminal = rows[-1] if rows else None
    return {
        "equity_curve": equity_curve,
        "drawdowns": drawdowns,
        "monthly_returns": monthly,
        "line_metrics": metrics,
        "terminal": ({"date": terminal["date"], "value": terminal["value"]} if terminal else None),
    }


def load_line_view(s3_client, bucket: str = BUCKET) -> Dict[str, Any]:
    """Read the ledger cache and fold it into the dashboard line surface."""
    return build_line_view(read_cache_rows(s3_client, bucket))


class LedgerParityError(Exception):
    """The rendered line does not match the stored ledger (publish must HOLD)."""


def assert_ledger_parity(
    line_view: Dict[str, Any],
    *,
    terminal_pin: Optional[float] = None,
    expected_terminal_date: Optional[str] = None,
) -> None:
    """The per-publish parity-or-hold gate (G-CACHE-PROJECTION, G-TERMINAL-PIN).

    The line view is a pure fold of the leaves, so value/benchmark parity is
    structural — what can still go wrong is an EMPTY or short ledger, or a terminal
    that does not match the operator-certified pin. Raises ``LedgerParityError`` on
    any of those; the publish path catches it and holds last-known-good + alarms.
    """
    ec = line_view.get("equity_curve") or []
    if not ec:
        raise LedgerParityError("ledger empty — no line to display (not seeded?)")
    term = line_view.get("terminal")
    if term is None:
        raise LedgerParityError("ledger has no terminal frontier")
    if expected_terminal_date is not None and term["date"] != expected_terminal_date:
        raise LedgerParityError(
            f"terminal date {term['date']} != expected {expected_terminal_date}"
        )
    if terminal_pin is not None and term["value"] != terminal_pin:
        raise LedgerParityError(
            f"terminal value {term['value']} != pinned {terminal_pin} "
            "(the displayed line drifted from the certified pin)"
        )


def assert_seed_parity_against_rendered(
    line_view: Dict[str, Any],
    rendered_equity_curve: List[Dict[str, Any]],
    *,
    terminal_pin: float,
) -> None:
    """Seed-time three-line parity: the ledger MUST reproduce the rendered chart
    value-for-value (EXACT — no arithmetic between source and stored value).

    Asserts per-date ``value`` and ``benchmark`` equality against the live rendered
    ``dashboard.json`` ``equity_curve`` and the terminal hard-pin. The comparison
    column is photographed verbatim from ``shadow_A`` into the leaves, so its parity
    is structural and checked at seed build time.
    """
    led = {r["date"]: r for r in line_view["equity_curve"]}
    rendered = {r["date"]: r for r in rendered_equity_curve}
    if set(led) != set(rendered):
        missing = sorted(set(rendered) - set(led))
        extra = sorted(set(led) - set(rendered))
        raise LedgerParityError(
            f"date sets differ — missing {missing[:5]}... extra {extra[:5]}..."
        )
    for d, rr in rendered.items():
        lr = led[d]
        if lr["value"] != rr.get("value"):
            raise LedgerParityError(
                f"{d}: ledger value {lr['value']} != rendered {rr.get('value')}"
            )
        if lr["benchmark"] != rr.get("benchmark"):
            raise LedgerParityError(
                f"{d}: ledger benchmark {lr['benchmark']} != rendered {rr.get('benchmark')}"
            )
    term = line_view["terminal"]
    if term["value"] != terminal_pin:
        raise LedgerParityError(
            f"terminal {term['value']} != pin {terminal_pin}"
        )
