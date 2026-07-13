"""Midday check — clean-core entrypoint (CU-05 rebuild).

The thin router routes ``midday-check`` here. Now a first-class clean-core module:
it orchestrates the (preserved, relocated) chassis midday checker DIRECTLY, no
``chassis.handler`` monolith dispatch and zero ``src`` import. It also carries the
CU-04 in-cycle value-revert DETECTION net (the 18:00Z midday runs AFTER the day's
morning publish, so it catches a same-day / Monday revert the 04:00Z watchdog is
blind to). Logic byte-preserved from the prior handler; only the home changed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from chassis.config_loader import load_config_from_s3
from chassis.steps import midday_checker
from chassis.utils.logging_utils import StepTimer
from chassis.utils.market_calendar import is_trading_session, ny_today
from chassis.utils.s3_client import S3Client
from chassis.utils.sns_alerts import (
    send_alert, format_midday_summary, format_error_alert,
)

logger = logging.getLogger("investment_pipeline.midday")


def run_midday(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Midday check: trailing-stop re-eval, VIX circuit breaker, skipped-buy
    re-check + in-cycle value-revert detection (CU-04). Clean-core."""
    start_time = datetime.now()
    run_date = event.get('run_date') or ny_today().isoformat()
    if not is_trading_session(run_date):
        return {'statusCode': 200, 'body': json.dumps({
            'status': 'skipped', 'phase': 'midday-check', 'date': run_date,
            'reason': 'NYSE closed; no simulated intraday action taken',
        })}
    logger.info(f"Midday check started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        with StepTimer("Midday check", logger):
            result = midday_checker.run(bucket, config)

        actions = result['actions_taken']
        check_log = result['check_log']
        circuit_breaker = result['circuit_breaker_active']

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Midday check completed in {duration:.1f}s")

        # Midday does not rebuild the dashboard, so the canon value is read from the
        # last published dashboard.json (best-effort, PKT-TB-001).
        canon_total_value = None
        try:
            _dash = s3_client.read_json('dashboard/dashboard.json') or {}
            _m = _dash.get('metrics', {})
            if _m.get('canon_source') in ('ledger', 'new_brain', 'optimized_champion') and _m.get('total_value'):
                canon_total_value = float(_m['total_value'])
        except Exception as canon_err:
            logger.warning(f"Canon metrics unavailable for midday email: {canon_err}")

        # CU-04 in-cycle value-revert DETECTION (G1 Monday / G2 same-cycle): the
        # 04:00Z watchdog runs BEFORE the day's 13:45Z morning publish, so a revert
        # was blind until the next day's run (and Monday had no run). Midday (18:00Z
        # Mon-Fri) runs AFTER the morning publish, so it catches a same-day revert
        # here — canon + SPY + challenger — and alarms. Detection only (the
        # publish-time guard is the prevention).
        try:
            from monitors.watchdog import check_value_revert
            _vr = check_value_revert(s3_client)
            if _vr.get('reverted'):
                send_alert(
                    subject="[TraderBot] CRITICAL: value-revert detected at midday (in-cycle)",
                    body=("A displayed line reverted to a contaminated / different-source "
                          "value since this morning's publish (canon / SPY / challenger):\n\n"
                          + (_vr.get('reason') or '')
                          + "\n\nThis is the same-cycle (G2) / Monday (G1) detection net; "
                          "the publish-time value-revert guard should also have blocked it "
                          "at publish."),
                    region=region)
                logger.error(f"MIDDAY value-revert detected: {_vr.get('reason')}")
        except Exception as _vr_err:  # noqa: BLE001 — detection must never crash midday
            logger.warning(f"Midday value-revert check failed (non-fatal): {_vr_err}")

        send_alert(
            subject=("[TraderBot] Midday: %d actions%s" % (
                len(actions), " [CIRCUIT BREAKER]" if circuit_breaker else "")),
            body=format_midday_summary(run_date, canon_total_value, actions,
                                       check_log, circuit_breaker, duration),
            region=region)

        return {'statusCode': 200, 'body': json.dumps({
            'status': 'success', 'phase': 'midday-check', 'date': run_date,
            'duration_seconds': duration, 'actions_taken': len(actions),
            'circuit_breaker_active': circuit_breaker,
            'canon_total_value': canon_total_value})}

    except Exception as e:
        logger.error(f"Midday check failed: {e}", exc_info=True)
        send_alert(
            subject="[TraderBot] ALERT: Midday check failed",
            body=format_error_alert('midday-check', run_date, str(e)), region=region)
        return {'statusCode': 500, 'body': json.dumps({
            'status': 'failed', 'phase': 'midday-check', 'date': run_date,
            'error': str(e), 'timestamp': datetime.now().isoformat()})}
