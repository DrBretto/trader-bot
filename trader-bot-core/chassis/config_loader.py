"""Config loader + Secrets Manager helper — relocated out of the handler monolith
(CU-05) so the clean-core morning/midday entrypoints (``app.morning`` /
``app.midday``) can load the live params bundle WITHOUT importing the legacy
``chassis.handler`` monolith. Logic is byte-preserved from ``chassis.handler``;
only the home changed. ``chassis.handler`` now re-imports from here.
"""
from __future__ import annotations

import json
import logging

import boto3
import pandas as pd

from chassis.steps import paper_trader
from chassis.utils.s3_client import S3Client

logger = logging.getLogger(__name__)


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

    # Load active decision bundle (single live source of truth).
    active_bundle = s3_client.read_json('config/decision_params.active.json')
    if not active_bundle:
        raise RuntimeError(
            "Missing required live params bundle at config/decision_params.active.json"
        )

    config['decision_params'] = active_bundle.get('decision_params', {})
    config['regime_compatibility'] = active_bundle.get('regime_compatibility', {})
    config['signal_overrides'] = active_bundle.get('signals', {})
    config['regime_fusion_overrides'] = active_bundle.get('regime_fusion', {})
    config['decision_engine_overrides'] = active_bundle.get('decision_engine', {})
    config['ensemble_overrides'] = active_bundle.get('ensemble', {})
    config['transaction_cost_overrides'] = active_bundle.get('transaction_costs', {})
    config['active_params_metadata'] = {
        'version_id': active_bundle.get('version_id'),
        'source_run_id': active_bundle.get('source_run_id'),
        'updated_at': active_bundle.get('updated_at'),
    }

    if not config['decision_params'] or not config['regime_compatibility']:
        raise RuntimeError(
            "Invalid config/decision_params.active.json: missing decision_params or regime_compatibility"
        )

    # Load current portfolio state
    portfolio_state = paper_trader.load_portfolio_state(s3_client)
    config['portfolio_state'] = portfolio_state

    return config
