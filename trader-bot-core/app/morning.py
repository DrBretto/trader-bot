"""Morning execution — clean-core entrypoint (CU-05 rebuild).

The thin router routes ``morning-execution`` here. This is now a first-class
clean-core module: it orchestrates the (preserved, relocated) chassis execution
steps DIRECTLY — it no longer dispatches through the ``chassis.handler`` monolith
and carries zero ``src`` import. The morning executor logic and the
``daily/<D>/trade_intents.json`` schema are byte-preserved (the night path emits
the exact schema ``morning_executor`` consumes); only the home and imports changed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from chassis.config_loader import load_config_from_s3
from chassis.steps import morning_executor, publish_artifacts
from chassis.utils.logging_utils import StepTimer
from chassis.utils.market_calendar import latest_settled_session
from chassis.utils.s3_client import S3Client
from chassis.utils.sns_alerts import (
    send_alert, format_morning_summary, format_error_alert,
)

logger = logging.getLogger("investment_pipeline.morning")


def run_morning(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Morning execution phase: validate intents and execute trades at market
    prices, then publish updated portfolio state + dashboard. Clean-core."""
    start_time = datetime.now()
    # Settled NY trading day, not UTC now (PKT-4). Morning fires 14:45 UTC = 09:45 ET.
    run_date = event.get('run_date') or latest_settled_session()
    logger.info(f"Morning execution started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        with StepTimer("Morning execution", logger):
            result = morning_executor.run(bucket, config)

        portfolio_state = result['portfolio_state']
        trades = result['trades']
        validation_log = result.get('validation_log', [])
        morning_prices = result.get('morning_prices')

        # Load night artifacts for dashboard rebuild. Use intents_date (when the
        # night phase last ran), not date (which morning overwrites to today).
        latest = s3_client.read_json('daily/latest.json') or {}
        night_date = latest.get('intents_date', latest.get('date', run_date))
        night_inference = s3_client.read_json(f'daily/{night_date}/inference.json') or {}
        night_decisions = s3_client.read_json(f'daily/{night_date}/decisions.json') or {}
        night_weather = s3_client.read_json(f'daily/{night_date}/weather_blurb.json') or {}

        # Reconstruct expert_signals from stored signals parquet
        expert_signals = None
        try:
            night_signals = s3_client.read_parquet(f'daily/{night_date}/signals.parquet')
            if len(night_signals) > 0:
                row = night_signals.iloc[0]
                expert_signals = {
                    'macro_credit': {
                        'macro_credit_score': float(row.get('macro_credit_score', 0)),
                        'yield_slope_10y_3m': float(row.get('yield_slope_10y_3m', 0)),
                        'hy_spread_proxy': float(row.get('hy_spread_proxy', 0)),
                    },
                    'vol_uncertainty': {
                        'vol_uncertainty_score': float(row.get('vol_uncertainty_score', 0.5)),
                        'vol_regime_label': str(row.get('vol_regime_label', 'calm')),
                        'vix_percentile': float(row.get('vix_percentile', 0.5)),
                        'vvix_percentile': float(row.get('vvix_percentile', 0.5)),
                    },
                    'fragility': {
                        'fragility_score': float(row.get('fragility_score', 0.5)),
                        'avg_correlation': float(row.get('avg_correlation', 0)),
                        'pc1_explained': float(row.get('pc1_explained', 0)),
                    },
                    'entropy_shift': {
                        'entropy_score': float(row.get('entropy_score', 0.5)),
                        'entropy_z_score': float(row.get('entropy_z_score', 0)),
                        'entropy_shift_flag': bool(row.get('entropy_shift_flag', False)),
                    },
                }
        except Exception as e:
            logger.warning(f"Could not load night signals for dashboard: {e}")

        morning_execution_report = {
            'run_date': run_date,
            'intents_date': latest.get('intents_date'),
            'intents_found': result.get('intents_found', False),
            'intents_stale': result.get('intents_stale', False),
            'trades_executed': len(trades),
            'validation_log': validation_log,
            'skipped_buys': result.get('skipped_buys', []),
            'timestamp': datetime.now().isoformat()
        }

        with StepTimer("Publish morning artifacts", logger):
            publish_result = publish_artifacts.publish_morning_artifacts(
                bucket, run_date, portfolio_state, trades,
                morning_execution_report, night_inference, night_decisions,
                night_weather, expert_signals=expert_signals,
                morning_prices=morning_prices,
            )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Morning execution completed in {duration:.1f}s")

        # The morning email reports the CANON book from this run's publish
        # (PKT-TB-001) — never the internal sim book.
        canon_total_value = publish_result.get('canon_total_value')
        canon_subject = (f"${canon_total_value:,.0f}" if canon_total_value is not None
                         else "canon n/a")
        send_alert(
            subject=f"[TraderBot] Morning: {len(trades)} trades, {canon_subject}",
            body=format_morning_summary(run_date, canon_total_value, trades,
                                        validation_log, duration),
            region=region)

        return {'statusCode': 200, 'body': json.dumps({
            'status': 'success', 'phase': 'morning', 'date': run_date,
            'duration_seconds': duration, 'trades_executed': len(trades),
            'canon_total_value': canon_total_value})}

    except Exception as e:
        logger.error(f"Morning phase failed: {e}", exc_info=True)
        send_alert(
            subject="[TraderBot] ALERT: Morning execution failed",
            body=format_error_alert('morning', run_date, str(e)), region=region)
        return {'statusCode': 500, 'body': json.dumps({
            'status': 'failed', 'phase': 'morning', 'date': run_date,
            'error': str(e), 'timestamp': datetime.now().isoformat()})}
