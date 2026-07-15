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
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from chassis.config_loader import load_config_from_s3
from chassis.steps import morning_executor, paper_trader, publish_artifacts
from chassis.utils.logging_utils import StepTimer
from chassis.utils.market_calendar import (
    is_trading_session,
    morning_execution_window_open,
    ny_today,
)
from chassis.utils.s3_client import S3Client
from chassis.utils.sns_alerts import (
    send_alert, format_morning_summary, format_error_alert,
)

logger = logging.getLogger("investment_pipeline.morning")

CHECKPOINT_SCHEMA = "morning_execution_checkpoint.v1"
CHECKPOINT_PREFIX = "ops/morning_checkpoints"


def _checkpoint_key(run_date: str) -> str:
    return f"{CHECKPOINT_PREFIX}/{run_date}.json"


def _with_execution_ids(
    trades: List[Dict[str, Any]],
    run_date: str,
    intents_date: str,
) -> List[Dict[str, Any]]:
    """Stamp stable IDs so a resumed publish cannot append duplicate fills."""
    stamped = []
    for index, raw in enumerate(trades):
        trade = dict(raw)
        identity = "|".join(str(v) for v in (
            run_date,
            intents_date,
            index,
            trade.get('symbol'),
            trade.get('action'),
            trade.get('shares'),
            trade.get('market_price'),
            trade.get('reason'),
        ))
        trade.setdefault(
            'execution_id',
            'morning-' + hashlib.sha256(identity.encode()).hexdigest()[:20],
        )
        stamped.append(trade)
    return stamped


def _build_checkpoint(
    run_date: str,
    intents_date: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    morning_prices = result.get('morning_prices')
    price_rows = []
    if morning_prices is not None and len(morning_prices) > 0:
        price_rows = json.loads(
            morning_prices.to_json(orient='records', date_format='iso')
        )
    trades = _with_execution_ids(result.get('trades', []), run_date, intents_date)
    return {
        'schema': CHECKPOINT_SCHEMA,
        'status': 'prepared',
        'run_date': run_date,
        'intents_date': intents_date,
        'prepared_at': datetime.now().isoformat(),
        'portfolio_state': paper_trader.to_published_state(result['portfolio_state']),
        'trades': trades,
        'morning_prices': price_rows,
        'validation_log': result.get('validation_log', []),
        'skipped_buys': result.get('skipped_buys', []),
        'intents_found': result.get('intents_found', False),
        'intents_stale': result.get('intents_stale', False),
        'execution_mode': result.get('execution_mode', 'simulated'),
    }


def _result_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if checkpoint.get('schema') != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"unsupported morning checkpoint schema: {checkpoint.get('schema')!r}"
        )
    if not isinstance(checkpoint.get('portfolio_state'), dict):
        raise RuntimeError("morning checkpoint has no recoverable portfolio_state")
    if not isinstance(checkpoint.get('trades'), list):
        raise RuntimeError("morning checkpoint has no recoverable trades list")
    return {
        'portfolio_state': paper_trader.from_published_state(
            checkpoint['portfolio_state']
        ),
        'trades': checkpoint['trades'],
        'morning_prices': pd.DataFrame(checkpoint.get('morning_prices') or []),
        'validation_log': checkpoint.get('validation_log', []),
        'skipped_buys': checkpoint.get('skipped_buys', []),
        'intents_found': checkpoint.get('intents_found', False),
        'intents_stale': checkpoint.get('intents_stale', False),
        'execution_mode': checkpoint.get('execution_mode', 'simulated'),
    }


def run_morning(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Morning execution phase: validate intents and execute trades at market
    prices, then publish updated portfolio state + dashboard. Clean-core."""
    start_time = datetime.now()
    # Morning executes only on the actual ET session date. EventBridge's Mon-Fri
    # schedule also fires on exchange holidays, so reject those before fetching a
    # quote or writing a trade.
    run_date = event.get('run_date') or ny_today().isoformat()
    if not is_trading_session(run_date):
        return {'statusCode': 200, 'body': json.dumps({
            'status': 'skipped', 'phase': 'morning', 'date': run_date,
            'reason': 'NYSE closed; no simulated execution or price artifact written',
        })}
    # Two UTC schedules cover EDT and EST. The early one must be a no-op in
    # standard time; the later one is de-duplicated by the checkpoint below.
    if not event.get('run_date') and not morning_execution_window_open():
        return {'statusCode': 200, 'body': json.dumps({
            'status': 'skipped', 'phase': 'morning', 'date': run_date,
            'reason': 'New York morning execution window has not opened yet',
        })}
    logger.info(f"Morning execution started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        checkpoint_key = _checkpoint_key(run_date)
        checkpoint = s3_client.read_json_strict(checkpoint_key)
        if checkpoint and checkpoint.get('status') == 'completed':
            dashboard = s3_client.read_json_strict('dashboard/dashboard.json') or {}
            canon_total = (dashboard.get('metrics') or {}).get('total_value')
            return {'statusCode': 200, 'body': json.dumps({
                'status': 'success', 'phase': 'morning', 'date': run_date,
                'idempotent_replay': True,
                'trades_executed': len(checkpoint.get('trades') or []),
                'canon_total_value': canon_total,
            })}

        checkpoint_replayed = checkpoint is not None
        if checkpoint_replayed:
            result = _result_from_checkpoint(checkpoint)
        else:
            latest = s3_client.read_json_strict('daily/latest.json')
            if not latest or not latest.get('intents_date'):
                raise RuntimeError("daily/latest.json has no intents_date; refusing execution")
            intents_date = str(latest['intents_date'])
            intents = s3_client.read_json_strict(
                f'daily/{intents_date}/trade_intents.json'
            )
            if not intents:
                raise RuntimeError(
                    f"latest pointer names {intents_date}, but trade_intents.json is absent"
                )

            with StepTimer("Load configuration", logger):
                config = load_config_from_s3(s3_client)

            with StepTimer("Morning execution", logger):
                result = morning_executor.run(bucket, config, run_date=run_date)
            if not result.get('intents_found'):
                raise RuntimeError(
                    "preflight found trade intents but executor did not; refusing a false success"
                )

            checkpoint = _build_checkpoint(run_date, intents_date, result)
            if not s3_client.write_json(checkpoint, checkpoint_key):
                raise RuntimeError("failed to persist resumable morning checkpoint")
            result = _result_from_checkpoint(checkpoint)

        portfolio_state = result['portfolio_state']
        trades = result['trades']
        validation_log = result.get('validation_log', [])
        morning_prices = result.get('morning_prices')

        # Load night artifacts for dashboard rebuild. Use intents_date (when the
        # night phase last ran), not date (which morning overwrites to today).
        latest = s3_client.read_json_strict('daily/latest.json') or {}
        night_date = checkpoint.get('intents_date') or latest.get(
            'intents_date', latest.get('date', run_date)
        )
        night_intents = s3_client.read_json_strict(
            f'daily/{night_date}/trade_intents.json'
        ) or {}
        night_inference = s3_client.read_json(f'daily/{night_date}/inference.json') or {}
        night_decisions = s3_client.read_json(f'daily/{night_date}/decisions.json') or {}
        night_weather = s3_client.read_json(f'daily/{night_date}/weather_blurb.json') or {}

        # Clean-core nights intentionally emit the canonical intent artifact, not
        # the retired legacy analysis bundle. Rebuild the current regime/decision
        # surface from that authoritative payload instead of showing an older day.
        if not night_inference and night_intents:
            night_inference = {
                'regime': {'label': night_intents.get('regime', 'unknown')}
            }
        if not night_decisions and night_intents:
            expert_metrics = dict(night_intents.get('expert_metrics') or {})
            expert_metrics.setdefault(
                'final_regime_label', night_intents.get('regime', 'unknown')
            )
            night_decisions = {
                'actions': night_intents.get('actions', []),
                'buy_candidates': night_intents.get('buy_candidates', []),
                'expert_metrics': expert_metrics,
            }

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
            'intents_date': night_date,
            'intents_found': result.get('intents_found', False),
            'intents_stale': result.get('intents_stale', False),
            'trades_executed': len(trades),
            'validation_log': validation_log,
            'skipped_buys': result.get('skipped_buys', []),
            'checkpoint_replayed': checkpoint_replayed,
            'timestamp': checkpoint.get('prepared_at') or datetime.now().isoformat()
        }

        with StepTimer("Publish morning artifacts", logger):
            publish_result = publish_artifacts.publish_morning_artifacts(
                bucket, run_date, portfolio_state, trades,
                morning_execution_report, night_inference, night_decisions,
                night_weather, expert_signals=expert_signals,
                morning_prices=morning_prices,
            )
        if not publish_result.get('success'):
            raise RuntimeError(
                "morning artifact publish incomplete: "
                + ", ".join(publish_result.get('failed') or ['unknown failure'])
            )

        checkpoint['status'] = 'completed'
        checkpoint['completed_at'] = datetime.now().isoformat()
        checkpoint['publish'] = {
            'published': publish_result.get('published', []),
            'canon_total_value': publish_result.get('canon_total_value'),
        }
        if not s3_client.write_json(checkpoint, checkpoint_key):
            raise RuntimeError("morning publish landed but checkpoint completion did not")

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
            'checkpoint_replayed': checkpoint_replayed,
            'canon_total_value': canon_total_value})}

    except Exception as e:
        logger.error(f"Morning phase failed: {e}", exc_info=True)
        send_alert(
            subject="[TraderBot] ALERT: Morning execution failed",
            body=format_error_alert('morning', run_date, str(e)), region=region)
        return {'statusCode': 500, 'body': json.dumps({
            'status': 'failed', 'phase': 'morning', 'date': run_date,
            'error': str(e), 'timestamp': datetime.now().isoformat()})}
