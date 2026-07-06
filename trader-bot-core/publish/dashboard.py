"""publish/dashboard.py — the NON-DESTRUCTIVE publish gates (framework invariant 3).

Extracted (logic-preserving) from the 1181-line ``src/steps/publish_artifacts.py``:
the parity-or-hold gate (``_verify_ledger_or_hold``) + the line build path
(``build_dashboard_data``'s ledger-fold half) + the never-shrink guard
(``_assert_cache_not_shrunk``'s publish-surface analogue).

INVARIANT 3 (the exact failure that wiped the challenger line earlier this session):
**a degraded or empty publish must be STRUCTURALLY unable to overwrite a populated
line.** Two independent guards enforce it, both fail-loud, neither ever silently
wipes the good line:

  * **parity-or-hold** (``verify_ledger_or_hold``) — an empty / cache-drifted /
    date-regressed line HOLDS last-known-good (returns ``ok=False``, never raises,
    no write).
  * **never-shrink** (``assert_publish_not_shrunk``) — a publish that would overwrite
    a currently-populated published line with an empty or point-reduced one RAISES
    (the publish analogue of ``lines.ledger.EquityLedger._assert_cache_not_shrunk``,
    which protects the ledger CACHE_KEY the same way). The good published line is
    left intact.

``publish_line`` composes both: it writes ONLY when parity-or-hold passes AND the
write is not a shrink; otherwise it HOLDS. There is no code path in which an
empty/degraded publish reaches the write.

The displayed line is the STORED ledger (``lines/``) — a pure fold, never a
recompute, never ``sim_book_value``. This module reads the ledger + the currently
published dashboard; it does not build the non-line trade/exposure/regime halves of
the full dashboard (those stay in the prod ``publish_artifacts.py`` until the P9
cutover — out of this packet's scope).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from lines.ledger import BUCKET, EquityLedger
from lines.line import load_line_view, read_cache_rows

logger = logging.getLogger(__name__)

# The live displayed-line artifacts (same keys the prod publish path writes).
DASHBOARD_KEY = "dashboard/dashboard.json"
DASHBOARD_DATA_KEY = "dashboard/data/dashboard.json"


class DestructivePublishError(Exception):
    """A publish that would overwrite a currently-populated published line with an
    empty or point-reduced one — the publish-surface analogue of the ledger's
    ``DestructiveCacheError``. Fail-loud: raised, never silently applied."""


def _equity_curve_len(dashboard: Optional[Dict[str, Any]]) -> int:
    if not dashboard:
        return 0
    return len(dashboard.get("equity_curve", []) or [])


def _read_published(s3_client, key: str = DASHBOARD_KEY) -> Optional[Dict[str, Any]]:
    """The currently-published dashboard, or None if absent/unreadable."""
    try:
        raw = s3_client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 — absent dashboard is a normal "not yet" state
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001 — unreadable => treat as nothing to protect
        return None


# --------------------------------------------------------------------------- #
# the line build path (invariant-3 half of build_dashboard_data)
# --------------------------------------------------------------------------- #
def build_line_surface(s3_client) -> Dict[str, Any]:
    """Fold the STORED ledger into the displayed-line surface.

    The invariant-3 half of the prod ``build_dashboard_data``: a pure projection of
    the write-once leaves (``equity_curve`` + drawdowns + monthly returns + terminal
    + line metrics). No recompute, no ``sim_book_value``. If the ledger is unseeded
    the ``equity_curve`` is empty and the parity-or-hold gate holds the publish.
    """
    line = load_line_view(s3_client)
    metrics = line.get("line_metrics") or {}
    return {
        "equity_curve": line["equity_curve"],
        "drawdowns": line["drawdowns"],
        "monthly_returns": line["monthly_returns"],
        "line_metrics": metrics,
        "terminal": line["terminal"],
        # The displayed total is ALWAYS the stored ledger's terminal — NEVER sim_book.
        "total_value": metrics.get("total_value"),
        "canon_source": "ledger",
    }


# --------------------------------------------------------------------------- #
def build_publish_surface(s3_client) -> Dict[str, Any]:
    """The FULL dashboard to publish: the corrected/advancing line surface overlaid
    onto the last-published dashboard.

    The clean-core night owns the three-line surface (``build_line_surface``) but NOT
    the non-line panels (``metrics``, ``holdings``, ``trades``, ``weather``,
    ``timeseries_url`` — the shadow/challenger line loader — etc.). The live front-end
    reads those top-level keys and **white-screens if they are absent**, so a bare
    line-only publish would blank the page. This overlays the authoritative line half
    onto the prior published dashboard so the panels carry forward (they are not
    rebuilt by the clean-core night; they hold their last snapshot) while the line
    stays corrected and advances. ``metrics.total_value`` is re-stamped to the canon
    line terminal (the only published portfolio value — repo single-book invariant).
    On the very first publish (no prior dashboard) this degrades to the line surface.
    """
    line = build_line_surface(s3_client)
    prev = _read_published(s3_client) or {}
    surface = {**prev, **line}
    prev_metrics = prev.get("metrics")
    if isinstance(prev_metrics, dict):
        surface["metrics"] = {**prev_metrics, "total_value": line.get("total_value")}
    return surface


# --------------------------------------------------------------------------- #
# gate 1: parity-or-hold (verbatim logic from _verify_ledger_or_hold)
# --------------------------------------------------------------------------- #
def verify_ledger_or_hold(
    dashboard_data: Dict[str, Any],
    s3_client,
    phase: Optional[str] = None,
    run_date: Optional[str] = None,
) -> Tuple[bool, str]:
    """Parity-or-hold gate (FP-08-3) — the displayed line is the STORED ledger.

    Refuses to publish a dashboard whose line is empty, has drifted from the stored
    leaves (cache-projection parity), or has regressed behind the last published
    terminal date (the "history moved overnight" failure class). Returns
    ``(ok, reason)``; ``ok=False`` means HOLD last-known-good + alarm. Never raises.
    """
    try:
        ec = dashboard_data.get("equity_curve", []) or []
        if not ec:
            return False, "PARITY check failed: equity_curve is empty (ledger not readable / unseeded)"

        rendered_term = ec[-1]
        # 1. CACHE-PROJECTION parity: the rendered terminal must equal the stored
        #    ledger frontier leaf EXACTLY (the line is a pure fold of the leaves).
        front = EquityLedger(s3_client).frontier()
        if front is None:
            return False, "PARITY check failed: ledger has no frontier (unseeded)"
        cache_rows = read_cache_rows(s3_client)
        if not cache_rows:
            return False, "PARITY check failed: ledger cache empty"
        led_term = cache_rows[-1]
        if rendered_term.get("date") != led_term["date"] or rendered_term.get("value") != led_term["value"]:
            return False, (
                f"PARITY check failed: rendered terminal "
                f"{rendered_term.get('date')}={rendered_term.get('value')} != ledger "
                f"{led_term['date']}={led_term['value']} (cache projection drift)"
            )

        # 2. NO-REGRESSION: the terminal date must not move backward vs the last
        #    published dashboard (the exact "history moved overnight" failure class).
        prev = _read_published(s3_client) or {}
        prev_ec = prev.get("equity_curve", []) or []
        prev_term_date = prev_ec[-1]["date"] if prev_ec else None
        if prev_term_date is not None and rendered_term["date"] < prev_term_date:
            return False, (
                f"REGRESSION check failed: new terminal {rendered_term['date']} < "
                f"last published {prev_term_date} (line would move backward)"
            )

        return True, f"ok (ledger terminal {led_term['date']}={led_term['value']})"
    except Exception as e:  # noqa: BLE001 — a raising gate HOLDS as a precaution
        return False, f"gate raised (held as precaution): {e}"


# --------------------------------------------------------------------------- #
# gate 2: never-shrink (publish-surface analogue of _assert_cache_not_shrunk)
# --------------------------------------------------------------------------- #
def assert_publish_not_shrunk(dashboard_data: Dict[str, Any], s3_client) -> None:
    """Fail-loud never-shrink guard for the published line surface.

    Refuse to overwrite a currently-populated published ``equity_curve`` with an
    empty or point-reduced one (a degraded/partial build). This is the publish-side
    mirror of ``lines.ledger.EquityLedger._assert_cache_not_shrunk`` (which protects
    the ledger CACHE_KEY). When the current dashboard is absent or itself unpopulated
    there is nothing to protect. On a destructive write we log an ALARM and RAISE,
    leaving the good published line intact.
    """
    current = _read_published(s3_client)
    cur_n = _equity_curve_len(current)
    new_n = _equity_curve_len(dashboard_data)
    if cur_n > 0 and new_n < cur_n:
        msg = (
            f"ABORT destructive dashboard publish — would overwrite a populated line "
            f"with an empty/point-reduced one ({cur_n}->{new_n} points); good line "
            f"left intact"
        )
        logger.error("ALARM %s", msg)
        raise DestructivePublishError(msg)


# --------------------------------------------------------------------------- #
# the composed non-destructive publish
# --------------------------------------------------------------------------- #
def publish_line(
    dashboard_data: Dict[str, Any],
    s3_client,
    *,
    phase: str = "night",
    run_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-destructively publish the displayed line (framework invariant 3).

    Writes the dashboard ONLY when BOTH guards pass:
      * ``verify_ledger_or_hold`` (parity-or-hold) is ok, AND
      * ``assert_publish_not_shrunk`` (never-shrink) does not fire.
    A degraded / empty / regressed / point-reduced publish HOLDS last-known-good
    (no write) — structurally unable to overwrite a populated line. Returns
    ``{published, held, reason}`` (plus ``terminal`` on a real publish). Never
    silently wipes the good line.
    """
    ok, reason = verify_ledger_or_hold(dashboard_data, s3_client, phase, run_date)
    if not ok:
        logger.error("ALARM dashboard parity-or-hold gate FAILED — %s", reason)
        return {"published": False, "held": True, "reason": reason}

    # Second, independent structural guard: never shrink a populated line.
    try:
        assert_publish_not_shrunk(dashboard_data, s3_client)
    except DestructivePublishError as e:
        return {"published": False, "held": True, "reason": str(e)}

    body = json.dumps(dashboard_data, indent=2, sort_keys=True, allow_nan=False).encode()
    # Write to both locations for compatibility (mirrors the prod publish path).
    s3_client.put_object(Bucket=BUCKET, Key=DASHBOARD_DATA_KEY, Body=body,
                         ContentType="application/json")
    s3_client.put_object(Bucket=BUCKET, Key=DASHBOARD_KEY, Body=body,
                         ContentType="application/json")
    term = (dashboard_data.get("equity_curve") or [{}])[-1]
    return {
        "published": True,
        "held": False,
        "reason": reason,
        "terminal": {"date": term.get("date"), "value": term.get("value")},
    }
