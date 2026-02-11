"""
Main Lambda handler for the daily investment pipeline.

Supports two execution phases:
- Night analysis (default): Full pipeline + generate trade intents
- Morning execution: Validate intents + execute trades at market prices

Routed via event['source']:
- "morning-execution" -> morning phase
- Anything else -> night phase (backward compatible)
"""

import json
import os
import boto3
import pandas as pd
from datetime import datetime

from src.steps import (
    ingest_prices,
    ingest_fred,
    ingest_gdelt,
    validate_data,
    build_features,
    run_inference,
    llm_risk_check,
    decision_engine,
    paper_trader,
    morning_executor,
    llm_weather,
    publish_artifacts
)
from src.signals.compute_signals import run as compute_signals
from src.utils.s3_client import S3Client
from src.utils.logging_utils import setup_logger, log_step, StepTimer
from src.utils.sns_alerts import (
    send_alert, format_night_summary, format_morning_summary, format_error_alert
)


# Initialize logger
logger = setup_logger('investment_pipeline')


def get_secret(secret_name: str, region: str = 'us-east-1') -> str:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            secret = json.loads(response['SecretString'])
            # Handle both key-value and plain string secrets
            if isinstance(secret, dict):
                return secret.get('api_key', secret.get('key', str(secret)))
            return secret
        return ''
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {e}")
        return ''


def load_config_from_s3(s3_client: S3Client) -> dict:
    """Load configuration files from S3."""
    config = {}

    # Load universe
    universe_df = s3_client.read_csv('config/universe.csv')
    if len(universe_df) > 0:
        config['universe'] = universe_df
    else:
        logger.warning("Universe not found in S3, using empty")
        config['universe'] = pd.DataFrame()

    # Load decision params
    decision_params = s3_client.read_json('config/decision_params.json')
    if decision_params:
        config['decision_params'] = decision_params
    else:
        logger.warning("Decision params not found, using defaults")
        config['decision_params'] = {
            'max_positions': 8,
            'max_position_weight': 0.20,
            'buy_score_threshold': 0.65,
            'min_health_buy': 0.60,
            'trailing_stop_base': 0.10,
            'min_order_dollars': 250,
            'initial_portfolio_value': 100000
        }

    # Load regime compatibility
    regime_compat = s3_client.read_json('config/regime_compatibility.json')
    if regime_compat:
        config['regime_compatibility'] = regime_compat
    else:
        config['regime_compatibility'] = {}

    # Load current portfolio state
    portfolio_state = paper_trader.load_portfolio_state(s3_client)
    config['portfolio_state'] = portfolio_state

    return config


def lambda_handler(event: dict, context) -> dict:
    """
    Main Lambda entry point. Routes to night or morning phase.

    Args:
        event: Lambda event with optional 'bucket', 'source', 'region' keys
        context: Lambda context object

    Returns:
        Response dict with statusCode and body
    """
    source = event.get('source', 'eventbridge-scheduled')
    bucket = event.get('bucket', os.environ.get('S3_BUCKET', 'investment-system-data'))
    region = (
        event.get('region')
        or os.environ.get('AWS_REGION')
        or os.environ.get('AWS_REGION_NAME')
        or 'us-east-1'
    )

    if source == 'morning-execution':
        return _run_morning_phase(event, bucket, region)
    else:
        return _run_night_phase(event, bucket, region)


def _run_night_phase(event: dict, bucket: str, region: str) -> dict:
    """
    Night analysis phase: full pipeline + generate trade intents.

    Runs steps 1-12 but does NOT execute trades. Instead, saves trade intents
    for the morning execution phase. Portfolio valuations are updated with
    closing prices so the dashboard shows current values.
    """
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")

    run_date = datetime.now().strftime('%Y-%m-%d')
    s3_client = S3Client(bucket, region)

    try:
        # Load configuration
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        # Get API keys from Secrets Manager
        with StepTimer("Retrieve API keys", logger):
            openai_key = get_secret('investment-system/openai-key', region)
            fred_key = get_secret('investment-system/fred-key', region)
            alphavantage_key = get_secret('investment-system/alphavantage-key', region)
            logger.info("Using Claude Haiku via Bedrock for LLM calls")

        # Extract universe symbols
        universe = config['universe']
        if isinstance(universe, pd.DataFrame) and len(universe) > 0:
            symbols = universe['symbol'].tolist()
        else:
            # Fallback to critical symbols only
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'HYG', 'LQD', 'GLD', 'VIXY']

        # Step 1: Ingest prices
        log_step(1, 12, "Ingesting prices...", logger)
        with StepTimer("Ingest prices", logger):
            prices_df = ingest_prices.run(symbols, alphavantage_key)

        # Step 2: Ingest FRED
        log_step(2, 12, "Ingesting FRED data...", logger)
        with StepTimer("Ingest FRED", logger):
            fred_df = ingest_fred.run(fred_key)

        # Step 3: Ingest vol complex indices (VVIX, SKEW from Stooq)
        log_step(3, 12, "Ingesting vol indices...", logger)
        with StepTimer("Ingest vol indices", logger):
            vvix_data = ingest_prices.fetch_stooq_index('^VVIX')
            skew_data = ingest_prices.fetch_stooq_index('^SKEW')
            vvix_latest = float(vvix_data.sort_values('date')['close'].iloc[-1]) if len(vvix_data) > 0 else None
            skew_latest = float(skew_data.sort_values('date')['close'].iloc[-1]) if len(skew_data) > 0 else None
            logger.info(f"VVIX: {vvix_latest}, SKEW: {skew_latest}")

        # Step 4: Ingest GDELT
        log_step(4, 12, "Ingesting GDELT...", logger)
        with StepTimer("Ingest GDELT", logger):
            gdelt_data = ingest_gdelt.run(run_date)

        # Step 5: Validate data
        log_step(5, 12, "Validating data...", logger)
        with StepTimer("Validate data", logger):
            validation = validate_data.run(prices_df, fred_df, config)

        if validation.get('critical_failure', False):
            raise Exception(f"Critical data validation failure: {validation['issues']}")

        if validation.get('degraded_mode', False):
            logger.warning("Running in degraded mode due to data quality issues")

        # Step 6: Build features
        log_step(6, 12, "Building features...", logger)
        with StepTimer("Build features", logger):
            features_df, context_df = build_features.run(
                prices_df, fred_df, gdelt_data,
                vvix_value=vvix_latest,
                skew_value=skew_latest
            )

        # Step 7: Compute expert signals
        log_step(7, 12, "Computing expert signals...", logger)
        with StepTimer("Compute expert signals", logger):
            expert_signals = compute_signals(
                prices_df, fred_df, context_df,
                vvix_data=vvix_data,
                skew_data=skew_data,
                s3_client=s3_client,
            )

        # Step 8: Run inference
        log_step(8, 12, "Running inference...", logger)
        with StepTimer("Run inference", logger):
            # Add bucket to config for model loading
            config['s3_bucket'] = bucket
            inference_output = run_inference.run(features_df, context_df, config)

        # Step 9: LLM risk check (uses Bedrock/Haiku, falls back to OpenAI)
        log_step(9, 12, "LLM risk check...", logger)
        with StepTimer("LLM risk check", logger):
            config['aws_region'] = region
            llm_risks = llm_risk_check.run(
                inference_output, features_df, context_df, openai_key, config
            )

        # Step 10: Decision engine (with expert signal fusion)
        log_step(10, 12, "Running decision engine...", logger)
        with StepTimer("Decision engine", logger):
            decisions = decision_engine.run(
                inference_output, llm_risks, features_df, config, validation,
                expert_signals=expert_signals
            )

        # Save trade intents to S3 (queued for morning execution)
        trade_intents = {
            'generated_date': run_date,
            'generated_timestamp': datetime.now().isoformat(),
            'regime': decisions.get('regime', 'unknown'),
            'actions': decisions.get('actions', []),
            'buy_candidates': decisions.get('buy_candidates', []),
            'expert_metrics': decisions.get('expert_metrics', {}),
            'expires_after_days': 3
        }
        s3_client.write_json(trade_intents, f'daily/{run_date}/trade_intents.json')
        logger.info(f"Saved {len(trade_intents['actions'])} trade intents for morning execution")

        # Step 11: Portfolio valuation update (NO trade execution)
        log_step(11, 12, "Updating portfolio valuations...", logger)
        with StepTimer("Portfolio valuation", logger):
            portfolio_state = paper_trader.load_portfolio_state(s3_client)
            portfolio_state = paper_trader.update_portfolio_values(portfolio_state, prices_df)
            try:
                portfolio_state = paper_trader.compute_portfolio_stats(portfolio_state, s3_client)
            except Exception as e:
                logger.warning(f"Stats computation failed: {e}")
            portfolio_state['trades_today'] = []
            trades = []

        print(f"  Portfolio value: ${portfolio_state['portfolio_value']:,.2f}")
        print(f"  Cash: ${portfolio_state['cash']:,.2f}")
        print(f"  Holdings: {len(portfolio_state['holdings'])}")
        print(f"  Pending intents: {len(trade_intents['actions'])}")

        # Step 12: LLM weather blurb (uses Bedrock/Haiku, falls back to OpenAI)
        log_step(12, 12, "Generating weather blurb...", logger)
        with StepTimer("LLM weather", logger):
            weather = llm_weather.run(
                inference_output, decisions, portfolio_state, context_df, openai_key, region,
                expert_signals=expert_signals
            )

        # Publish all artifacts to S3
        logger.info("Publishing artifacts to S3...")
        with StepTimer("Publish artifacts", logger):
            publish_result = publish_artifacts.run(
                bucket, run_date,
                prices_df, context_df, features_df,
                inference_output, llm_risks, decisions,
                portfolio_state, trades, weather, validation,
                expert_signals=expert_signals
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Pipeline completed in {duration:.1f}s")

        # Send email alert
        regime_label = decisions.get('expert_metrics', {}).get(
            'final_regime_label',
            inference_output.get('regime', {}).get('label', 'unknown')
        )
        send_alert(
            subject=f"[TraderBot] Night: {regime_label}, {len(trade_intents['actions'])} intents",
            body=format_night_summary(
                run_date, regime_label,
                portfolio_state.get('portfolio_value', 0),
                trade_intents['actions'],
                weather.get('headline', ''),
                duration
            ),
            region=region
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'phase': 'night',
                'date': run_date,
                'duration_seconds': duration,
                'regime': regime_label,
                'intents_count': len(trade_intents['actions']),
                'portfolio_value': portfolio_state.get('portfolio_value', 0)
            })
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

        send_alert(
            subject="[TraderBot] ALERT: Night analysis failed",
            body=format_error_alert('night', run_date, str(e)),
            region=region
        )

        # Write failure report
        failure_report = {
            'status': 'failed',
            'phase': 'night',
            'date': run_date,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

        try:
            s3_client.write_json(failure_report, f'daily/{run_date}/run_report.json')
        except:
            pass

        return {
            'statusCode': 500,
            'body': json.dumps(failure_report)
        }


def _run_morning_phase(event: dict, bucket: str, region: str) -> dict:
    """
    Morning execution phase: validate intents and execute trades at market prices.

    Loads trade intents from the night run, fetches fresh morning prices via
    yfinance, validates each intent, and executes trades. Publishes updated
    portfolio state and dashboard.
    """
    start_time = datetime.now()
    run_date = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Morning execution started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        # Load configuration
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        # Execute morning phase
        with StepTimer("Morning execution", logger):
            result = morning_executor.run(bucket, config)

        portfolio_state = result['portfolio_state']
        trades = result['trades']
        validation_log = result.get('validation_log', [])

        # Load night artifacts for dashboard rebuild
        latest = s3_client.read_json('daily/latest.json') or {}
        night_date = latest.get('date', run_date)
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

        # Build morning execution report
        morning_execution_report = {
            'run_date': run_date,
            'intents_date': latest.get('intents_date'),
            'intents_found': result.get('intents_found', False),
            'intents_stale': result.get('intents_stale', False),
            'trades_executed': len(trades),
            'validation_log': validation_log,
            'timestamp': datetime.now().isoformat()
        }

        # Publish morning artifacts
        with StepTimer("Publish morning artifacts", logger):
            publish_artifacts.publish_morning_artifacts(
                bucket, run_date, portfolio_state, trades,
                morning_execution_report, night_inference, night_decisions,
                night_weather, expert_signals=expert_signals
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Morning execution completed in {duration:.1f}s")

        # Send email alert
        send_alert(
            subject=(
                f"[TraderBot] Morning: {len(trades)} trades, "
                f"${portfolio_state.get('portfolio_value', 0):,.0f}"
            ),
            body=format_morning_summary(
                run_date,
                portfolio_state.get('portfolio_value', 0),
                trades, validation_log, duration
            ),
            region=region
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'phase': 'morning',
                'date': run_date,
                'duration_seconds': duration,
                'trades_executed': len(trades),
                'portfolio_value': portfolio_state.get('portfolio_value', 0)
            })
        }

    except Exception as e:
        logger.error(f"Morning phase failed: {e}", exc_info=True)

        send_alert(
            subject="[TraderBot] ALERT: Morning execution failed",
            body=format_error_alert('morning', run_date, str(e)),
            region=region
        )

        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'failed',
                'phase': 'morning',
                'date': run_date,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }


# For local testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run investment pipeline locally')
    parser.add_argument('--bucket', default='investment-system-data', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--phase', default='night', choices=['night', 'morning'],
                        help='Which phase to run')
    args = parser.parse_args()

    source = 'morning-execution' if args.phase == 'morning' else 'manual'
    result = lambda_handler(
        {'bucket': args.bucket, 'region': args.region, 'source': source}, None
    )
    print(json.dumps(result, indent=2))
