"""
Main Lambda handler for the daily investment pipeline.

This is the entry point for AWS Lambda execution.
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
    llm_weather,
    publish_artifacts
)
from src.utils.s3_client import S3Client
from src.utils.logging_utils import setup_logger, log_step, StepTimer


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
    Main Lambda entry point for daily pipeline.

    Args:
        event: Lambda event with optional 'bucket' and 'source' keys
        context: Lambda context object

    Returns:
        Response dict with statusCode and body
    """
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")

    # Get configuration
    bucket = event.get('bucket', os.environ.get('S3_BUCKET', 'investment-system-data'))
    region = event.get('region', os.environ.get('AWS_REGION', 'us-east-1'))

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
        log_step(1, 10, "Ingesting prices...", logger)
        with StepTimer("Ingest prices", logger):
            prices_df = ingest_prices.run(symbols, alphavantage_key)

        # Step 2: Ingest FRED
        log_step(2, 10, "Ingesting FRED data...", logger)
        with StepTimer("Ingest FRED", logger):
            fred_df = ingest_fred.run(fred_key)

        # Step 3: Ingest GDELT
        log_step(3, 10, "Ingesting GDELT...", logger)
        with StepTimer("Ingest GDELT", logger):
            gdelt_data = ingest_gdelt.run(run_date)

        # Step 4: Validate data
        log_step(4, 10, "Validating data...", logger)
        with StepTimer("Validate data", logger):
            validation = validate_data.run(prices_df, fred_df, config)

        if validation.get('critical_failure', False):
            raise Exception(f"Critical data validation failure: {validation['issues']}")

        if validation.get('degraded_mode', False):
            logger.warning("Running in degraded mode due to data quality issues")

        # Step 5: Build features
        log_step(5, 10, "Building features...", logger)
        with StepTimer("Build features", logger):
            features_df, context_df = build_features.run(prices_df, fred_df, gdelt_data)

        # Step 6: Run inference
        log_step(6, 10, "Running inference...", logger)
        with StepTimer("Run inference", logger):
            # Add bucket to config for model loading
            config['s3_bucket'] = bucket
            inference_output = run_inference.run(features_df, context_df, config)

        # Step 7: LLM risk check (uses Bedrock/Haiku, falls back to OpenAI)
        log_step(7, 10, "LLM risk check...", logger)
        with StepTimer("LLM risk check", logger):
            config['aws_region'] = region
            llm_risks = llm_risk_check.run(
                inference_output, features_df, context_df, openai_key, config
            )

        # Step 8: Decision engine
        log_step(8, 10, "Running decision engine...", logger)
        with StepTimer("Decision engine", logger):
            decisions = decision_engine.run(
                inference_output, llm_risks, features_df, config, validation
            )

        # Step 9: Paper trader
        log_step(9, 10, "Executing paper trades...", logger)
        with StepTimer("Paper trader", logger):
            portfolio_state, trades = paper_trader.run(decisions, prices_df, bucket)

        # Step 10: LLM weather blurb (uses Bedrock/Haiku, falls back to OpenAI)
        log_step(10, 10, "Generating weather blurb...", logger)
        with StepTimer("LLM weather", logger):
            weather = llm_weather.run(
                inference_output, decisions, portfolio_state, context_df, openai_key, region
            )

        # Publish all artifacts to S3
        logger.info("Publishing artifacts to S3...")
        with StepTimer("Publish artifacts", logger):
            publish_result = publish_artifacts.run(
                bucket, run_date,
                prices_df, context_df, features_df,
                inference_output, llm_risks, decisions,
                portfolio_state, trades, weather, validation
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Pipeline completed in {duration:.1f}s")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'date': run_date,
                'duration_seconds': duration,
                'regime': inference_output.get('regime', {}).get('label'),
                'actions_count': len(decisions.get('actions', [])),
                'portfolio_value': portfolio_state.get('portfolio_value', 0)
            })
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

        # Write failure report
        failure_report = {
            'status': 'failed',
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


# For local testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run investment pipeline locally')
    parser.add_argument('--bucket', default='investment-system-data', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()

    result = lambda_handler({'bucket': args.bucket, 'region': args.region}, None)
    print(json.dumps(result, indent=2))
