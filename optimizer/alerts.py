"""Optimizer alert builders.

Phase 4 of the 2026-04-30 optimizer empirical-mutation packet. Builds the
persistent-rejection alert body and forwards to the existing SNS topic
(`investment-system-alerts`) via `src/utils/sns_alerts.py`. No new SNS
topic, no new infrastructure — the operator's existing email filters
already capture `[TraderBot]` subjects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def send_optimizer_alert(subject: str, body: str) -> bool:
    """Forward to the project's existing SNS alerting helper.

    Returns True on successful publish, False on any failure (the helper
    in src/utils/sns_alerts.py is itself non-throwing). Wrapped here so
    the import is local — the optimizer module avoids requiring boto3
    at import time during unit tests.
    """
    try:
        from src.utils.sns_alerts import send_alert
    except ImportError:
        return False
    return bool(send_alert(subject=subject, body=body))


def _format_failed_checks(checks: List[Dict[str, Any]]) -> str:
    if not checks:
        return '(no checks recorded)'
    failed = [c for c in checks if not c.get('passed', False)]
    if not failed:
        return '(no checks failed)'
    lines = []
    for c in failed:
        name = c.get('name', '?')
        value = c.get('value')
        threshold = c.get('threshold')
        lines.append(f'    - {name}: value={value} threshold={threshold}')
    return '\n'.join(lines)


def build_persistent_rejection_alert_body(
    streak_state: Dict[str, Any],
    config: Any,
    latest_guardrail_results: Optional[Dict[str, Any]] = None,
    latest_promotion_payload: Optional[Dict[str, Any]] = None,
    latest_challenger_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Build (subject, body) for the persistent-rejection SNS alert.

    The subject carries the [TraderBot] prefix so the operator's existing
    email filters route it. The body lists the run timestamps for the
    streak, the most recent challenger's guardrail-failure breakdown, and
    a suggested operator action — specifically pointing to the
    `--empirical-mutation-debug` flag for diagnosis.
    """
    streak_count = int(streak_state.get('consecutive_rejections', 0))
    threshold = int(streak_state.get('alert_threshold', 3))
    history = streak_state.get('history', [])[:streak_count]

    subject = (
        f'[TraderBot] Optimizer rejected {streak_count} consecutive cycles '
        f'(threshold {threshold})'
    )

    rejection_lines = []
    for entry in history:
        rejection_lines.append(
            f'    - {entry.get("run_id", "?")} ({entry.get("timestamp", "?")}) '
            f'decision={entry.get("decision", "?")}'
        )

    failed_checks_text = _format_failed_checks(
        (latest_guardrail_results or {}).get('checks', []) if latest_guardrail_results else []
    )

    promo = latest_promotion_payload or {}
    challenger_summary = ''
    if latest_challenger_metrics:
        wf = latest_challenger_metrics.get('wf_objective')
        guardrails = latest_challenger_metrics.get('guardrails', {})
        passed = guardrails.get('passed') if isinstance(guardrails, dict) else None
        calibration_only = guardrails.get('calibration_only') if isinstance(guardrails, dict) else None
        challenger_summary = (
            f'  Challenger wf_objective: {wf}\n'
            f'  Challenger guardrails passed: {passed}\n'
            f'  Calibration-only path used: {calibration_only}\n'
        )

    body = (
        f'OPTIMIZER PERSISTENT-REJECTION ALERT\n'
        f'\n'
        f'The weekly optimizer has rejected all challengers for '
        f'{streak_count} consecutive cycles (alert threshold = {threshold}).\n'
        f'\n'
        f'Recent rejected cycles:\n{chr(10).join(rejection_lines)}\n'
        f'\n'
        f'Most recent cycle:\n'
        f'  run_id: {promo.get("run_id", "?")}\n'
        f'  champion_version_before: {promo.get("champion_version_before", "?")}\n'
        f'  challenger_version: {promo.get("challenger_version", "?")}\n'
        f'  reason_codes: {promo.get("reason_codes", [])}\n'
        f'{challenger_summary}'
        f'  Failed guardrail checks:\n{failed_checks_text}\n'
        f'\n'
        f'Suggested operator action:\n'
        f'  If calibration-class drift is suspected (i.e., normalization\n'
        f'  constants are stale relative to current empirical distribution),\n'
        f'  re-run with --empirical-mutation-debug to inspect what the\n'
        f'  empirical-re-derivation candidate is proposing:\n'
        f'\n'
        f'    .venv/bin/python -m optimizer.cli run --empirical-mutation-debug\n'
        f'\n'
        f'  Output shows the proposed values for every normalization\n'
        f'  constant and whether the verification gate\n'
        f'  (AVG_CORR_MEAN ~ 0.477 +/- 0.02) is satisfied.\n'
        f'\n'
        f'  If the debug output looks right but the GA never proposes the\n'
        f'  empirical candidate, check that\n'
        f'  optimizer.yaml -> enable_empirical_mutation: true.\n'
        f'\n'
        f'  If the debug output is wildly off, check the\n'
        f'  parameter_inventory.json `empirical_statistic` field per\n'
        f'  constant and the underlying signal_row data quality.\n'
        f'\n'
        f'See docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md\n'
        f'for the full operator runbook.\n'
    )
    return subject, body
