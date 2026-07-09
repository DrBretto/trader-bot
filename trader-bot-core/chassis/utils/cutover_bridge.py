"""Pure functions for cutover continuity bridge logic.

Computes external_cashflow patches to neutralize portfolio value
discontinuities at broker cutover boundaries, preserving benchmark
continuity and allowing idempotent re-application.
"""

from typing import Any, Dict, Optional


CONTINUITY_MARKER_PREFIX = "continuity-bridge-v1"


def extract_cutover_date_from_marker(marker: Optional[str]) -> Optional[str]:
    """Return the cutover date embedded in a continuity marker, if present."""
    if not marker or not isinstance(marker, str):
        return None

    prefix = f"{CONTINUITY_MARKER_PREFIX}:"
    if not marker.startswith(prefix):
        return None

    cutover_date = marker[len(prefix):].strip()
    return cutover_date or None


def compute_bridge_cashflow(cutover_value: float, previous_value: float) -> float:
    """Compute external_cashflow that neutralizes the cutover-day return.

    The dashboard metrics formula is:
        daily_return = (value_t - value_{t-1} - external_cashflow) / value_{t-1}

    Setting external_cashflow = cutover_value - previous_value makes daily_return = 0.

    Args:
        cutover_value: Portfolio value on the cutover day (broker-reconciled).
        previous_value: Portfolio value on the last pre-cutover day.

    Returns:
        External cashflow amount (negative means value decreased).
    """
    return cutover_value - previous_value


def build_cutover_patch(
    cutover_state: Dict[str, Any],
    previous_state: Dict[str, Any],
    cutover_date: str,
) -> Dict[str, Any]:
    """Build a patch dict to apply to the cutover-day portfolio_state.

    The patch includes:
    - external_cashflow to neutralize the value discontinuity
    - benchmark fields carried forward from previous day if missing
    - a continuity marker for auditability

    Args:
        cutover_state: The cutover-day portfolio_state dict.
        previous_state: The previous trading day's portfolio_state dict.
        cutover_date: YYYY-MM-DD string for the cutover day.

    Returns:
        Dict of fields to merge into cutover_state.
    """
    cutover_value = float(cutover_state.get("portfolio_value", 0.0) or 0.0)
    previous_value = float(previous_state.get("portfolio_value", 0.0) or 0.0)

    patch: Dict[str, Any] = {
        "external_cashflow": compute_bridge_cashflow(cutover_value, previous_value),
        "continuity_bridge_marker": f"{CONTINUITY_MARKER_PREFIX}:{cutover_date}",
    }

    # Preserve benchmark continuity: carry forward from previous day if missing
    for field in ("benchmark_start_price", "benchmark_shares"):
        if not cutover_state.get(field) and previous_state.get(field):
            patch[field] = previous_state[field]

    return patch


def apply_patch(
    state: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge patch into state, idempotent.

    If the state already has a continuity_bridge_marker matching the patch,
    returns the state unchanged (already patched).

    Args:
        state: The portfolio_state dict to patch.
        patch: The patch dict from build_cutover_patch().

    Returns:
        New dict with patch applied (does not mutate input).
    """
    existing_marker = state.get("continuity_bridge_marker")
    patch_marker = patch.get("continuity_bridge_marker")

    if existing_marker and patch_marker and existing_marker == patch_marker:
        # Already patched — idempotent no-op
        return dict(state)

    result = dict(state)
    result.update(patch)
    return result


def describe_patch(
    previous_state: Dict[str, Any],
    cutover_state: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a human-readable summary of the bridge patch for dry-run output.

    Args:
        previous_state: Pre-cutover portfolio state.
        cutover_state: Cutover-day portfolio state.
        patch: The computed patch.

    Returns:
        Dict with summary fields.
    """
    prev_value = float(previous_state.get("portfolio_value", 0.0) or 0.0)
    cut_value = float(cutover_state.get("portfolio_value", 0.0) or 0.0)
    cashflow = patch.get("external_cashflow", 0.0)

    return {
        "previous_date": previous_state.get("date", "unknown"),
        "previous_value": prev_value,
        "cutover_value": cut_value,
        "value_delta": cut_value - prev_value,
        "external_cashflow": cashflow,
        "neutralized_return": 0.0,
        "benchmark_fields_carried": [
            f for f in ("benchmark_start_price", "benchmark_shares")
            if f in patch
        ],
        "marker": patch.get("continuity_bridge_marker"),
        "already_patched": bool(cutover_state.get("continuity_bridge_marker")),
    }
