"""First-class clean-core portfolio-state loader (P9 cutover seam).

The night forward path (``app.night``) needs the current intent-sizing portfolio
state — the IN-MEMORY return-engine book the settled-leaf mark is computed from.
Through P8 this borrowed ``src.steps.paper_trader.load_portfolio_state``; P9 gives
``trader-bot-core`` its own first-class loader so the clean core no longer reaches
into the old ``src/steps`` chassis for this read.

Contract preserved verbatim from the shared loader so the cutover is behaviour-
identical: read ``daily/latest.json`` → ``daily/<latest_date>/portfolio_state.json``,
restore published sim-book keys to their internal shape, drop one-day accounting
fields when rolling into a new date, and fall back to the fixed 100k seed state
when nothing is on S3. This is pure glue over ``s3.read_json`` — no data-science
compute, no ``src.steps`` dependency.

NOTE ON THE SIM BOOK (see the repo CLAUDE.md banner): the value this loader
carries under ``portfolio_value`` / ``sim_book_value`` is the internal intent-
sizing simulation, **NOT** the line, **NOT** a portfolio, **NOT** a yardstick.
The only source of truth for the line is the canon/replay ledger. This loader
just hands the return-engine book to the marking machinery; it never produces a
displayed value.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

# One-day accounting fields that must NOT roll forward into a new as-of date
# (external cashflow / continuity-bridge markers are settled-day-scoped).
TRANSIENT_ROLLOVER_KEYS = (
    "external_cashflow",
    "external_cashflow_t",
    "net_external_cashflow",
    "net_cashflow",
    "cashflow",
    "cash_flow",
    "continuity_bridge_marker",
)


def _default_state() -> Dict[str, Any]:
    """The fixed seed portfolio state used when nothing is on S3 yet."""
    return {
        "cash": 100000,
        "holdings": [],
        "portfolio_value": 100000,
        "benchmark_value": 100000,
        "benchmark_start_price": None,  # set on first run
        "trades_today": [],
        "last_updated": datetime.now().isoformat(),
    }


def _restore_internal_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Published sim-book shape → internal shape.

    Re-keys ``sim_book_value`` → ``portfolio_value`` and drops the role markers so
    the in-memory book matches what the engine/marking code expects. Historical
    states already keyed ``portfolio_value`` pass through unchanged.
    """
    restored = dict(state)
    if "sim_book_value" in restored:
        restored["portfolio_value"] = restored.pop("sim_book_value")
    restored.pop("book_role", None)
    restored.pop("book_note", None)
    return restored


def _normalize_loaded_portfolio_state(
    state: Dict[str, Any],
    state_date: Optional[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """Drop one-day accounting fields when rolling state into a new date."""
    normalized = dict(state)
    if state_date and state_date != as_of_date:
        for key in TRANSIENT_ROLLOVER_KEYS:
            normalized.pop(key, None)
    return normalized


def load_portfolio_state(s3, as_of: Optional[str] = None) -> Dict[str, Any]:
    """Load the current intent-sizing portfolio state from S3.

    Args:
        s3: an ``S3Client``-shaped object exposing ``read_json(key)``.
        as_of: as-of date (``YYYY-MM-DD``) the state is rolled into; defaults to
            today so behaviour matches the shared loader exactly. Passing an
            explicit date lets the replay driver reuse this loader deterministically.

    Returns:
        The internal portfolio-state dict, or the fixed 100k seed if none exists.
    """
    latest = s3.read_json("daily/latest.json")
    if latest is None:
        return _default_state()

    latest_date = latest.get("date")
    if latest_date:
        state = s3.read_json(f"daily/{latest_date}/portfolio_state.json")
        if state:
            as_of_date = as_of or datetime.now().strftime("%Y-%m-%d")
            state = _restore_internal_keys(state)
            return _normalize_loaded_portfolio_state(state, latest_date, as_of_date)

    return _default_state()
