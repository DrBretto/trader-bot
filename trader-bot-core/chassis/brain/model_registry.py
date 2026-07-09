"""Model socket + registry dispatch (FP-08-6).

The chassis is the base; a model is a plug-in; a swap is a pointer move. Promotion
or rollback = flip ``config/brain.active.json`` ``engine`` to a registered name and
redeploy — never a code change to the dispatch.

The ``Model`` protocol names the seam the dossier (§6) pins at ``select()``:
``allocate()`` is pure chassis (reads NAV/cash, dollar-clean), so a model touches
NO dollar value and the chassis reuses one allocation across every model. A model
is anything with a ``name`` and a ``select(f, theta_sel) -> SelectionResult``.

``resolve_live_engine`` replaces the old hard-coded
``if engine_name == "tilt_adapter"`` branch with a registry lookup validated
against the FREEZE_ORB1 engine hash at cold start (G-REGISTRY-DISPATCH): an
unregistered engine, or a live ``engine_sha`` that does not match the freeze, is a
hard stop (the caller fail-safes to the deterministic incumbent + alarms) — it can
never silently dispatch the wrong model.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Model(Protocol):
    """A pluggable selection model. The ONLY per-model seam (the body of select).

    ``select`` reads forecasts + frozen selection params and returns a
    SelectionResult; it never sees PortfolioState (NAV/cash), so it cannot touch a
    dollar value — the chassis ``allocate()`` owns sizing (assert_no_dollar_surface).
    """

    name: str

    def select(self, f: Any, theta_sel: Any) -> Any:  # -> SelectionResult
        ...


# Engines the chassis can dispatch. A swap is: set brain.active.json.engine to one
# of these + redeploy. Adding a new model type = register its name here + a select().
REGISTERED_ENGINES = ("native_two_stage", "tilt_adapter")


class EngineRegistryError(Exception):
    """The configured engine is unregistered or fails cold-start hash validation."""


def _freeze_engine_sha() -> str:
    """The FREEZE_ORB1 engine hash (the frozen engine identity), or '' if absent."""
    try:
        from chassis.brain.freeze import load_freeze
        return str((load_freeze().get("engine", {}) or {}).get("engine_sha", "") or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("FREEZE_ORB1 engine sha unreadable: %s", e)
        return ""


def resolve_live_engine(cfg: dict) -> str:
    """Validate + return the live engine name (registry + cold-start hash gate).

    Raises EngineRegistryError on an unregistered engine, or when the live
    ``engine_sha`` is declared and does not match the FREEZE_ORB1 engine hash. The
    caller treats a raise as abort-never-degrade (keep the deterministic incumbent).
    """
    engine = str(cfg.get("engine", "")).lower()
    if engine not in REGISTERED_ENGINES:
        raise EngineRegistryError(
            f"engine '{engine}' is not registered {REGISTERED_ENGINES} — refusing to dispatch")
    declared = str(cfg.get("engine_sha", "") or "").strip()
    frozen = _freeze_engine_sha().strip()
    if declared and frozen and declared != frozen:
        raise EngineRegistryError(
            f"engine_sha cold-start mismatch: brain.active={declared[:8]} != "
            f"FREEZE_ORB1={frozen[:8]} (G-REGISTRY-DISPATCH)")
    return engine


class TiltSelectAdapter:
    """Thin ``Model`` conformance for the tilt_adapter (the dotted comparison line).

    The tilt model is intents-first (it nudges the deterministic incumbent rather
    than selecting from forecasts), so it does not natively fit the ``select`` seam.
    This adapter is the only conformance debt the dossier names: it lets the tilt
    register as a ``Model`` (name + select) while its real cutover stays the
    intents-first ``run_tilt_cutover`` path. ``select`` raises if used as a primary
    selector — the tilt is a comparison/shadow, never the dollar-clean primary seam.
    """

    name = "tilt_adapter"

    def select(self, f: Any, theta_sel: Any) -> Any:
        raise NotImplementedError(
            "tilt_adapter is intents-first (a comparison model); it has no forecast "
            "select() seam — its live path is run_tilt_cutover, not select()")
