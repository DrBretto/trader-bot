"""monitors/config_canary.py — CL-708150 empty-but-critical config canary.

Fails LOUD if a load-bearing config table is present-but-EMPTY — the silent-inert
class of failure that let the regime gate sit dead for the whole project. An empty
dict passes a ``.get()`` and collapses every regime multiplier / admissible set to
the identity, so the engine runs indistinguishable from having no gate at all —
with ZERO error and ZERO alarm. This canary makes that structurally impossible:
either the load-bearing tables are non-empty or the canary goes RED.

Two load-bearing tables are checked:

  * ``regime_compatibility`` — the config-bundle chassis table
    (``config/regime_compatibility.json`` merged into ``brain.active``). Empty =>
    every ``regime_score_mult`` is 1.0 => Stage-1 ranks raw mu (chassis INERT).
  * ``theta_sel.regime_admissibility`` — the FREEZE selection gate
    (``brain/FREEZE_ORB1.json``). Empty => ``regime_admissibility.get(label)`` is
    ``None`` => the per-regime admissibility gate is INERT.

``run_config_canary(force_empty=[...])`` supports an INJECTION lever so a reality
probe can prove the canary fires on a deliberately-emptied table WITHOUT mutating
the real config (the un-fakeable acceptance test for CL-708150).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

CRITICAL_CONFIGS = ("regime_compatibility", "regime_admissibility")


def check_critical_configs(config: Mapping[str, Any], freeze: Mapping[str, Any],
                           *, force_empty: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Pure verdict: are the load-bearing tables non-empty? ``force_empty`` names
    tables to treat as empty (injection lever for the reality probe)."""
    fe = set(force_empty or [])
    problems: List[str] = []

    rc = {} if "regime_compatibility" in fe else config.get("regime_compatibility")
    if not rc or not isinstance(rc, Mapping):
        problems.append(
            "regime_compatibility EMPTY/missing — the regime chassis would run INERT "
            "(regime_score_mult=1.0 for every symbol = raw-mu ranking)")

    ts = (freeze.get("theta_sel") or {}).get("values") or {}
    ra = {} if "regime_admissibility" in fe else ts.get("regime_admissibility")
    if not ra or not isinstance(ra, Mapping):
        problems.append(
            "theta_sel.regime_admissibility EMPTY/missing — the per-regime selection "
            "admissibility gate would be INERT (no regime rotation)")

    return {"ok": not problems, "problems": problems,
            "checked": list(CRITICAL_CONFIGS),
            "regime_compatibility_n": len(rc) if isinstance(rc, Mapping) else 0,
            "regime_admissibility_n": len(ra) if isinstance(ra, Mapping) else 0,
            "forced_empty": sorted(fe)}


def _alert_config_canary_red(result: Dict[str, Any]) -> None:
    body = (
        "EMPTY-BUT-CRITICAL CONFIG CANARY FIRED (CL-708150) — a load-bearing config "
        "table is present-but-EMPTY, so a load-bearing gate would run INERT with no "
        "error. This is the silent-inert class that let the regime gate sit dead the "
        "whole project.\n\n"
        "problems:\n  " + "\n  ".join(result.get("problems") or ["(none)"]) + "\n\n"
        f"regime_compatibility entries = {result.get('regime_compatibility_n')}\n"
        f"regime_admissibility entries = {result.get('regime_admissibility_n')}\n"
        f"forced_empty (injection)     = {result.get('forced_empty')}\n")
    try:
        from chassis.utils.sns_alerts import send_alert
        send_alert(subject="[TraderBot] CRITICAL: empty-but-critical config canary RED (CL-708150)",
                   body=body)
    except Exception as e:  # noqa: BLE001 — alerting must never crash the pipeline
        print(f"[config_canary] SNS alert failed (non-fatal): {e}")


def run_config_canary(*, alert: bool = True,
                      config: Optional[Mapping[str, Any]] = None,
                      freeze: Optional[Mapping[str, Any]] = None,
                      force_empty: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load the live config + FREEZE (unless provided) and run the verdict. Fires an
    SNS CRITICAL on RED when ``alert``. ``force_empty`` proves the canary fires on a
    deliberately-emptied table without touching the real config."""
    if config is None:
        from decide.cutover import load_brain_config
        config = load_brain_config()
    if freeze is None:
        from forecast.freeze import load_freeze
        freeze = load_freeze()
    result = check_critical_configs(config, freeze, force_empty=force_empty)
    if not result["ok"] and alert:
        _alert_config_canary_red(result)
    return result
