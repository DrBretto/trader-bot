"""Native two-stage brain (PKT-TB-008+).

The two-stage engine lives under ``src.brain.engine``. It replaces the
incumbent greedy cash-draining selection loop (``src/steps/decision_engine.py``)
with a strict Select-then-Allocate pipeline in which the held symbol *set* is a
pure function of the forecast and selection parameters and is constant under all
sizing parameters (the symbol-set-invariance-under-sizing property).

PKT-TB-012 adds the live-cutover runtime: the frozen brain (FREEZE_ORB1) runs
inside the night Lambda, asserts its shas at cold start, gates on the invariant
self-check, and — when ``config/brain.active.json`` is live — writes the live
``trade_intents.json`` from the two-stage engine (NOT tilt_adapter). Anything
that fails falls back to the incumbent intents (abort-never-degrade).
"""

from .freeze import (
    BrainFreezeError,
    FreezeAssertion,
    assert_cold_start,
    compute_engine_sha,
    compute_model_sha,
    load_freeze,
)
from .runtime import (
    BrainConfigError,
    CutoverResult,
    DEFAULT_FORWARD_BOUNDARY,
    is_live,
    load_brain_config,
    production_forecaster,
    run_cutover,
    theta_from_freeze,
)

__all__ = [
    "BrainFreezeError",
    "FreezeAssertion",
    "assert_cold_start",
    "compute_engine_sha",
    "compute_model_sha",
    "load_freeze",
    "BrainConfigError",
    "CutoverResult",
    "DEFAULT_FORWARD_BOUNDARY",
    "is_live",
    "load_brain_config",
    "production_forecaster",
    "run_cutover",
    "theta_from_freeze",
]
