"""Replay-owned refresh for the promoted TILT canonical ledger.

The displayed model lines have one authoritative producer: a full, as-of-day replay
on the settled SPY session grid. The replay engine natively emits the two-stage book
as ``value`` and independent TILT as ``comparison``. The promoted ledger stores the
same replay with those roles intentionally reversed:

* ``value``      = TILT (canonical solid-blue line and all line-derived metrics)
* ``comparison`` = two-stage (dotted-yellow diagnostic comparison)

No portfolio-state dollar value enters this module. Before any write, the replayed
TILT must reproduce the currently stored TILT terminal to the cent. Once the promoted
ledger exists, both stored model lines must reproduce to the cent at its frontier.
Missing real sessions are appended in order; weekends and exchange holidays never
enter the grid because the grid comes from settled SPY bars.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from lines.ledger import BUCKET, EquityLedger

REFERENCE_PREFIX = "canon/equity_ledger_clean_v3/"
ISSUED_BY = "tilt-canon-replay@trader-bot"


def _reset_ephemeral_replay_state() -> Dict[str, Any]:
    """Start every scheduled replay from the same baked seed state.

    Lambda execution environments retain ``/tmp`` across warm invocations. Reusing
    that mutable forecast tree makes replay output depend on which phase happened
    to run in the same container. The production image explicitly configures
    ``BRAIN_STATE_DIR`` under ``/tmp``; refuse to delete anything outside a system
    temp root, then clear that one ephemeral tree before seed hydration.
    """
    configured = os.environ.get("BRAIN_STATE_DIR")
    if not configured:
        raise RuntimeError(
            "BRAIN_STATE_DIR is required for deterministic replay-state isolation"
        )

    from forecast import shadow_lib as SL

    state = SL.STATE.expanduser().resolve()
    configured_state = Path(configured).expanduser().resolve()
    if state != configured_state:
        raise RuntimeError(
            f"cached replay state {state} does not match BRAIN_STATE_DIR "
            f"{configured_state}; refusing an ambiguous reset"
        )
    temp_roots = {
        Path("/tmp").resolve(),
        Path(tempfile.gettempdir()).expanduser().resolve(),
    }
    if not any(state != root and root in state.parents for root in temp_roots):
        raise RuntimeError(
            f"BRAIN_STATE_DIR {state} is not below an approved temp root; "
            "refusing destructive replay-state reset"
        )
    if state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True, exist_ok=True)
    return {"reset": True, "path": str(state)}


def _active_rows(ledger: EquityLedger) -> List[Dict[str, Any]]:
    """Return the supersede-resolved ledger chain in date order."""
    return ledger._ordered_chain_from_leaves()  # ledger's verified restore path


def _by_date(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["date"]): row for row in rows}


def _cents(value: Any) -> Optional[int]:
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError, OverflowError):
        return None


def _cent_check(series: str, date: str, expected: Any, replayed: Any) -> Dict[str, Any]:
    expected_cents = _cents(expected)
    replayed_cents = _cents(replayed)
    return {
        "series": series,
        "date": date,
        "expected": float(expected) if expected_cents is not None else None,
        "replayed": float(replayed) if replayed_cents is not None else None,
        "ok": expected_cents is not None and expected_cents == replayed_cents,
    }


def promoted_values(native_leaf: Mapping[str, Any]) -> Dict[str, float]:
    """Map one native replay leaf into the promoted storage orientation."""
    if native_leaf.get("comparison") is None:
        raise RuntimeError(f"native replay has no TILT value for {native_leaf.get('date')}")
    return {
        "value": float(native_leaf["comparison"]),
        "benchmark": float(native_leaf["benchmark"]),
        "comparison": float(native_leaf["value"]),
    }


def _grid_check(
    target_rows: List[Mapping[str, Any]],
    settled_window: List[str],
    split_date: str,
) -> Dict[str, Any]:
    """Prove an existing promoted frontier contains every real session in order."""
    if not target_rows:
        return {"ok": True, "expected": [], "actual": [], "missing": [], "extra": []}
    frontier = str(target_rows[-1]["date"])
    expected = [d for d in settled_window if split_date < d <= frontier]
    actual = [str(r["date"]) for r in target_rows if split_date < str(r["date"]) <= frontier]
    return {
        "ok": actual == expected,
        "expected": expected,
        "actual": actual,
        "missing": sorted(set(expected) - set(actual)),
        "extra": sorted(set(actual) - set(expected)),
    }


def _recorded_regime_dates(
    settled_window: List[str],
    continuity_date: str,
    reference_frontier: str,
) -> List[str]:
    """Keep reconstructed history recorded and promoted-forward history forward.

    The frozen reference frontier is the lineage boundary. Dates appended after
    promotion used the deterministic as-of-day picker and must keep using it on
    every later replay; switching those dates to a subsequently recorded regime
    changes the book retroactively and breaks replay continuity.
    """
    recorded_through = min(continuity_date, reference_frontier)
    return [d for d in settled_window if d <= recorded_through]


def _history_parity(
    target_rows: List[Mapping[str, Any]],
    native_by_date: Mapping[str, Mapping[str, Any]],
    split_date: str,
) -> Dict[str, Any]:
    """Compare every promoted post-split point with the clean replay."""
    checks: List[Dict[str, Any]] = []
    for row in target_rows:
        date = str(row["date"])
        if date <= split_date:
            continue
        native = native_by_date.get(date)
        if native is None:
            checks.append({
                "series": "all",
                "date": date,
                "expected": None,
                "replayed": None,
                "ok": False,
                "reason": "replay omitted stored date",
            })
            continue
        promoted = promoted_values(native)
        for series, field in (
            ("tilt_canon", "value"),
            ("SPY", "benchmark"),
            ("two_stage_comparison", "comparison"),
        ):
            checks.append(_cent_check(series, date, row.get(field), promoted[field]))
    mismatches = [check for check in checks if not check["ok"]]
    return {
        "ok": not mismatches,
        "points_checked": len({check["date"] for check in checks}),
        "checks_run": len(checks),
        "mismatches": mismatches,
    }


def refresh_promoted_ledger(
    s3,
    *,
    bucket: str = BUCKET,
    commit: bool = False,
) -> Dict[str, Any]:
    """Reconstruct through the latest settled day and optionally append the result.

    Bootstrap uses the current clean-v3 TILT terminal as the independent continuity
    pin and copies only its byte-preserved pre-split champion leaves. Subsequent runs
    pin both promoted lines at the promoted frontier. A failed continuity or grid
    check is a hard no-write result.
    """
    state_reset = _reset_ephemeral_replay_state()

    from replay import seed_canon as SC
    from replay._fake_s3 import FakeS3

    ohlcv = SC.prepare_substrate()
    window = SC.post_split_window(ohlcv)
    if not window:
        raise RuntimeError("settled replay window is empty")
    latest = window[-1]

    target = EquityLedger(s3, bucket)
    reference = EquityLedger(s3, bucket, prefix=REFERENCE_PREFIX)
    target_rows = _active_rows(target)
    reference_rows = _active_rows(reference)
    if not reference_rows:
        raise RuntimeError(
            "frozen reference ledger is empty; cannot determine regime lineage"
        )
    target_by_date = _by_date(target_rows)
    reference_by_date = _by_date(reference_rows)

    base = target if target_rows else reference
    base_by_date = target_by_date if target_rows else reference_by_date
    if SC.SPLIT_DATE not in base_by_date:
        raise RuntimeError(f"replay base has no split anchor {SC.SPLIT_DATE}")

    if target_rows and target_rows[-1]["date"] > latest:
        raise RuntimeError(
            f"promoted frontier {target_rows[-1]['date']} is past settled SPY {latest}"
        )

    # Reconstruct history through the current frontier with recorded regimes; days
    # after it use the deterministic forward picker. That is the same rule used by
    # the existing successful challenger-advance continuity check.
    if target_rows and target_rows[-1]["date"] > SC.SPLIT_DATE:
        continuity_date = str(target_rows[-1]["date"])
        expected_tilt = target_by_date[continuity_date]["value"]
        expected_two_stage = target_by_date[continuity_date].get("comparison")
        check_two_stage = True
    else:
        reference_tilt_dates = sorted(
            d for d, row in reference_by_date.items()
            if d <= latest and row.get("comparison") is not None
        )
        if not reference_tilt_dates:
            raise RuntimeError("reference ledger has no TILT continuity point")
        continuity_date = reference_tilt_dates[-1]
        expected_tilt = reference_by_date[continuity_date]["comparison"]
        expected_two_stage = None
        check_two_stage = False

    scratch = EquityLedger(FakeS3(), prefix="scratch/tilt-canon-refresh/")
    SC.copy_presplit(base, scratch, SC.SPLIT_DATE)
    reference_frontier = str(reference_rows[-1]["date"])
    recorded_dates = _recorded_regime_dates(
        window,
        continuity_date,
        reference_frontier,
    )
    recorded_map = (
        SC.RR.recorded_regime_map(recorded_dates, s3=s3, verify=True)
        if recorded_dates else {}
    )
    SC.seed_canon_by_replay(
        scratch,
        ohlcv,
        d1=latest,
        regime_fn=SC.mixed_regime_fn(recorded_map),
        independent_challenger=True,
        s3=s3,
    )
    native_rows = _active_rows(scratch)
    native_by_date = _by_date(native_rows)
    if continuity_date not in native_by_date:
        raise RuntimeError(f"replay omitted continuity date {continuity_date}")

    checks = [
        _cent_check(
            "tilt_canon",
            continuity_date,
            expected_tilt,
            native_by_date[continuity_date].get("comparison"),
        )
    ]
    if check_two_stage:
        checks.append(
            _cent_check(
                "two_stage_comparison",
                continuity_date,
                expected_two_stage,
                native_by_date[continuity_date].get("value"),
            )
        )
    continuity_ok = all(check["ok"] for check in checks)
    grid = _grid_check(target_rows, window, SC.SPLIT_DATE)
    history_parity = _history_parity(target_rows, native_by_date, SC.SPLIT_DATE)

    frontier_before = str(target_rows[-1]["date"]) if target_rows else SC.SPLIT_DATE
    planned_dates = [d for d in window if d > frontier_before]
    missing_native = [d for d in planned_dates if d not in native_by_date]
    if missing_native:
        raise RuntimeError(f"replay omitted settled session(s): {missing_native}")

    planned = {
        d: {k: round(v, 6) for k, v in promoted_values(native_by_date[d]).items()}
        for d in planned_dates
    }
    report: Dict[str, Any] = {
        "phase": "refresh-tilt-canon",
        "nondestructive": not commit,
        "latest_settled": latest,
        "frontier_before": frontier_before,
        "continuity_date": continuity_date,
        "continuity_checks": checks,
        "continuity_ok": continuity_ok,
        "history_parity_ok": history_parity["ok"],
        "history_points_checked": history_parity["points_checked"],
        "history_mismatches": history_parity["mismatches"],
        "grid_ok": grid["ok"],
        "grid_missing": grid["missing"],
        "grid_extra": grid["extra"],
        "planned_dates": planned_dates,
        "planned_values": planned,
        "canonical_series": "tilt",
        "comparison_series": "two_stage",
        "replay_state": state_reset,
        "recorded_regime_through": recorded_dates[-1] if recorded_dates else None,
        "deterministic_forward_dates": [d for d in window if d not in recorded_map],
    }

    if not commit:
        report["committed"] = False
        return report
    if not continuity_ok or not history_parity["ok"] or not grid["ok"]:
        report["committed"] = False
        report["reason"] = "continuity/history/grid check refused the write"
        return report

    if not target_rows:
        SC.copy_presplit(reference, target, SC.SPLIT_DATE)

    appended: List[str] = []
    for d in planned_dates:
        native = native_by_date[d]
        values = promoted_values(native)
        target.append(
            date=d,
            value=values["value"],
            benchmark=values["benchmark"],
            comparison=values["comparison"],
            segment="new_brain",
            model_id=f"tilt-canon@{native.get('model_id', 'replay')}",
            source="replay_tilt_canon",
            issued_by=ISSUED_BY,
            extra={
                "forward_confirmed": True,
                "canonical_model": "tilt",
                "comparison_model": "two_stage",
            },
        )
        appended.append(d)

    final_rows = _active_rows(target)
    final_grid = _grid_check(final_rows, window, SC.SPLIT_DATE)
    final_frontier = str(final_rows[-1]["date"]) if final_rows else None
    if not final_grid["ok"] or final_frontier != latest:
        raise RuntimeError(
            f"post-write verification failed: frontier={final_frontier}, "
            f"latest={latest}, grid={final_grid}"
        )
    report.update({
        "committed": True,
        "appended_dates": appended,
        "frontier_after": final_frontier,
        "point_count": len(final_rows),
    })
    return report
