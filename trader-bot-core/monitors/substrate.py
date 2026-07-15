"""Substrate-currency watchdog — the CONTENT-currency check the publish-recency
checks structurally cannot make.

``check_substrate_fresh`` is relocated VERBATIM from ``src/brain/monitors.py``
(the local-HEAD version; KEEP-code). It keys on CONTENT: the brain's
``selected_universe`` byte-identical across the most recent ``min_identical``
distinct trading days is the S3-observable fingerprint of a frozen ``mu`` (a
frozen OHLCV substrate behind a fresh-looking publish). The replay keystone's
acceptance test drives it over the replay's own per-day
``brain_selected_universe.json`` leaves: a deliberately frozen-mu sub-window
makes it flag stale (the freeze can't pass silently).

``LocalReplayStore`` is the minimal store shim (``list_daily_dates`` +
``read_json``) over a local ``daily/<D>/`` tree, matching the S3 store interface
``check_substrate_fresh`` expects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def check_substrate_fresh(s3, lookback: int = 6, min_identical: int = 3) -> Dict[str, Any]:
    """SUBSTRATE-CURRENCY check (ISSUE-01/02/13) — the one the publish-recency
    checks STRUCTURALLY cannot make.

    Reads the last ``lookback`` trading-day leaves' ``brain_selected_universe.json``
    and flags the frozen-``mu`` signature — the selected universe byte-identical
    across the most recent ``min_identical`` DISTINCT trading days. That is the
    S3-observable fingerprint of a frozen substrate; the night's fail-loud gate is
    the primary defense, this is the independent watchdog. (VERBATIM from
    src/brain/monitors.py:209.)"""
    status: Dict[str, Any] = {"stale": False, "observable": False, "reason": "", "identical_run": 0,
                              "days_checked": 0, "selected_universe": None,
                              "dates": [], "sources": []}
    try:
        dates = s3.list_daily_dates(max_days=lookback)
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"could not list daily dates: {type(e).__name__}: {e}"
        return status
    if not dates:
        status["reason"] = "no daily leaves found"
        return status

    # newest-first; read each leaf's selected_universe fingerprint
    fps = []  # (date, tuple(selected_universe), source) newest-first
    for d in reversed(dates):
        try:
            bsu = s3.read_json(f"daily/{d}/brain_selected_universe.json") or {}
        except Exception:  # noqa: BLE001
            bsu = {}
        sel = bsu.get("selected_universe")
        source = "brain_selected_universe"
        if not sel:
            try:
                intents = s3.read_json(f"daily/{d}/trade_intents.json") or {}
            except Exception:  # noqa: BLE001
                intents = {}
            sel = (intents.get("expert_metrics") or {}).get("held_symbols")
            source = "trade_intents.expert_metrics.held_symbols"
        if sel:
            fps.append((d, tuple(sel), source))
    status["days_checked"] = len(fps)
    if len(fps) < min_identical:
        status["reason"] = (f"only {len(fps)} leaf universe(s) available "
                            f"(< {min_identical}); cannot judge substrate freshness")
        return status

    status["observable"] = True

    # length of the leading run of byte-identical selected_universe (newest-first)
    head = fps[0][1]
    run = 1
    for _, sel, _ in fps[1:]:
        if sel == head:
            run += 1
        else:
            break
    status["identical_run"] = run
    status["selected_universe"] = list(head)
    status["dates"] = [d for d, _, _ in fps[:run]]
    status["sources"] = [source for _, _, source in fps]
    if run >= min_identical:
        status["stale"] = True
        status["reason"] = (
            f"FROZEN-SUBSTRATE signature: brain selected_universe byte-identical across "
            f"{run} consecutive trading days {status['dates']} — mu is not advancing "
            f"(a frozen OHLCV substrate behind a fresh-looking publish)")
    return status


class LocalReplayStore:
    """Minimal store over a local ``daily/<D>/`` tree matching the interface
    ``check_substrate_fresh`` reads (``list_daily_dates`` + ``read_json``)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def list_daily_dates(self, max_days: Optional[int] = None) -> List[str]:
        d = self.root / "daily"
        if not d.exists():
            return []
        dates = sorted(p.name for p in d.iterdir() if p.is_dir()
                       and len(p.name) == 10 and p.name[4] == "-")
        return dates[-max_days:] if max_days else dates

    def read_json(self, key: str) -> Any:
        p = self.root / key
        if not p.exists():
            return None
        return json.loads(p.read_text())
