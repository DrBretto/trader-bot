"""
Main Lambda handler for the daily investment pipeline.

Supports three execution phases:
- Night analysis (default): Full pipeline + generate trade intents
- Morning execution: Validate intents + execute trades at market prices
- Midday check: Trailing stop re-eval, VIX circuit breaker, skipped-buy re-check

Routed via event['source']:
- "morning-execution" -> morning phase
- "midday-check" -> midday check phase
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
    midday_checker,
    llm_weather,
    publish_artifacts
)
from src.signals.compute_signals import run as compute_signals
from src.utils.s3_client import S3Client
from src.utils.logging_utils import setup_logger, log_step, StepTimer
from src.brokers.router import get_broker, resolve_broker_mode, BrokerMode
from src.utils.sns_alerts import (
    send_alert, format_night_summary, format_morning_summary,
    format_midday_summary, format_error_alert
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

    # Load active decision bundle (single live source of truth).
    # Expected schema:
    # {
    #   "decision_params": {...},
    #   "regime_compatibility": {...},
    #   "signals": {...},
    #   "regime_fusion": {...},
    #   "decision_engine": {...},
    #   "ensemble": {...},
    #   "transaction_costs": {...}
    # }
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
    config['broker'] = active_bundle.get('broker', {})
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
    elif source == 'midday-check':
        return _run_midday_check(event, bucket, region)
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
            alpaca_key_id = ''
            alpaca_secret_key = ''
            broker_mode = resolve_broker_mode(config)
            if broker_mode in (BrokerMode.ALPACA_PAPER, BrokerMode.ALPACA_LIVE):
                prefix = 'paper' if broker_mode == BrokerMode.ALPACA_PAPER else 'live'
                alpaca_key_id = get_secret(
                    f'investment-system/alpaca-{prefix}-key-id', region
                )
                alpaca_secret_key = get_secret(
                    f'investment-system/alpaca-{prefix}-secret-key', region
                )
            logger.info("Using Claude Haiku via Bedrock for LLM calls")

        # Reconcile portfolio_state against broker truth before the decision
        # engine runs. Without this, the night phase evaluates phantom holdings
        # (e.g. partial-fill leftovers the broker has since cleared to dust),
        # which is what produced the May 4-5 sell-down + dashboard divergence.
        if broker_mode in (BrokerMode.ALPACA_PAPER, BrokerMode.ALPACA_LIVE):
            with StepTimer("Night-phase broker reconcile", logger):
                try:
                    from src.steps.morning_executor import _reconcile_portfolio_from_broker
                    pre_broker = get_broker(
                        config, alpaca_key_id, alpaca_secret_key
                    )
                    if pre_broker is not None:
                        reconciled = _reconcile_portfolio_from_broker(
                            pre_broker, config['portfolio_state']
                        )
                        if reconciled.get('broker_reconciled'):
                            config['portfolio_state'] = reconciled
                            logger.info(
                                "Night reconcile OK: cash=$%.2f, holdings=%d, equity=$%.2f",
                                reconciled.get('cash', 0.0),
                                len(reconciled.get('holdings', [])),
                                reconciled.get('portfolio_value', 0.0),
                            )
                except Exception as exc:
                    logger.warning(
                        "Night-phase broker reconcile failed (non-fatal, continuing on stored state): %s",
                        exc,
                    )

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
            prices_df = ingest_prices.run(
                symbols,
                alphavantage_key=alphavantage_key,
                alpaca_key_id=alpaca_key_id,
                alpaca_secret_key=alpaca_secret_key,
            )

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
                signal_params=config.get('signal_overrides'),
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

        # Step 10: Decision engine (with expert signal fusion + ranking model if active)
        log_step(10, 12, "Running decision engine...", logger)
        with StepTimer("Decision engine", logger):
            active_ranking_blend = config.get('decision_engine_overrides', {}).get('ranking_blend', 0)
            active_ranking_model_dir = config.get('decision_engine_overrides', {}).get('ranking_model_dir', '')
            active_ranking_scores = None

            if active_ranking_blend > 0 and active_ranking_model_dir:
                try:
                    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
                    import torch, json as _json
                    from pathlib import Path
                    import tempfile

                    model_path = Path(active_ranking_model_dir) / 'ranking_mlp.pt'
                    norm_path = Path(active_ranking_model_dir) / 'ranking_normalization.json'

                    if not model_path.exists():
                        logger.info("Active ranking: model not local, downloading from S3...")
                        _tmp = Path(tempfile.mkdtemp()) / 'ranking'
                        _tmp.mkdir(parents=True, exist_ok=True)
                        s3_model = s3_client.download_file(f'{active_ranking_model_dir}/ranking_mlp.pt', str(_tmp / 'ranking_mlp.pt'))
                        s3_norm = s3_client.download_file(f'{active_ranking_model_dir}/ranking_normalization.json', str(_tmp / 'ranking_normalization.json'))
                        if s3_model and s3_norm:
                            model_path = _tmp / 'ranking_mlp.pt'
                            norm_path = _tmp / 'ranking_normalization.json'
                        else:
                            logger.warning("Active ranking: model not available on S3, falling back to health-only")
                            model_path = None

                    if model_path and model_path.exists() and norm_path.exists():
                        with open(norm_path) as _f:
                            ranking_norm = _json.load(_f)
                        ranking_model = RankingMLP(input_dim=len(RANKING_FEATURES))
                        ranking_model.load_state_dict(torch.load(model_path, weights_only=True))
                        ranking_model.eval()

                        active_ranking_scores = {}
                        for _, row in features_df.iterrows():
                            sym = row.get('symbol')
                            if sym:
                                feat_dict = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
                                active_ranking_scores[sym] = ranking_model.predict_scores(feat_dict, ranking_norm)
                        logger.info(f"Active ranking: loaded model, scored {len(active_ranking_scores)} symbols at blend {active_ranking_blend}")
                    else:
                        logger.warning("Active ranking: model files not available, falling back to health-only")
                except Exception as rank_err:
                    logger.warning(f"Active ranking: model load failed, falling back to health-only: {rank_err}")

            decisions = decision_engine.run(
                inference_output, llm_risks, features_df, config, validation,
                expert_signals=expert_signals,
                ranking_scores=active_ranking_scores,
                ranking_blend=active_ranking_blend,
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

        # Shadow challenger computation (non-executing)
        try:
            candidate_bundle = s3_client.read_json('config/decision_params.candidate.json')
            if candidate_bundle and candidate_bundle.get('version_id') != config.get('active_version'):
                ranking_blend = candidate_bundle.get('decision_engine', {}).get('ranking_blend', 0)
                ranking_model_dir = candidate_bundle.get('decision_engine', {}).get('ranking_model_dir', '')

                if ranking_blend > 0 and ranking_model_dir:
                    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
                    import torch, json as _json
                    from pathlib import Path
                    import tempfile

                    model_path = Path(ranking_model_dir) / 'ranking_mlp.pt'
                    norm_path = Path(ranking_model_dir) / 'ranking_normalization.json'

                    # Try local filesystem first; fall back to S3 download
                    if not model_path.exists():
                        logger.info("Shadow: ranking model not local, downloading from S3...")
                        _tmp = Path(tempfile.mkdtemp()) / 'ranking'
                        _tmp.mkdir(parents=True, exist_ok=True)
                        s3_model = s3_client.download_file(f'{ranking_model_dir}/ranking_mlp.pt', str(_tmp / 'ranking_mlp.pt'))
                        s3_norm = s3_client.download_file(f'{ranking_model_dir}/ranking_normalization.json', str(_tmp / 'ranking_normalization.json'))
                        if s3_model and s3_norm:
                            model_path = _tmp / 'ranking_mlp.pt'
                            norm_path = _tmp / 'ranking_normalization.json'
                        else:
                            logger.info("Shadow: ranking model not available on S3 either, skipping")
                            model_path = None

                    if model_path and model_path.exists() and norm_path.exists():
                        with open(norm_path) as _f:
                            ranking_norm = _json.load(_f)
                        ranking_model = RankingMLP(input_dim=len(RANKING_FEATURES))
                        ranking_model.load_state_dict(torch.load(model_path, weights_only=True))
                        ranking_model.eval()

                        # Compute ranking scores
                        ranking_scores = {}
                        for _, row in features_df.iterrows():
                            sym = row.get('symbol')
                            if sym:
                                feat_dict = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
                                ranking_scores[sym] = ranking_model.predict_scores(feat_dict, ranking_norm)

                        # Build shadow config from candidate bundle
                        shadow_config = dict(config)
                        shadow_config['decision_params'] = candidate_bundle.get('decision_params', config.get('decision_params', {}))
                        shadow_config['regime_compatibility'] = candidate_bundle.get('regime_compatibility', config.get('regime_compatibility', {}))
                        shadow_config['regime_fusion_overrides'] = candidate_bundle.get('regime_fusion', {})
                        shadow_config['decision_engine_overrides'] = candidate_bundle.get('decision_engine', {})
                        shadow_config['ensemble_overrides'] = candidate_bundle.get('ensemble', {})

                        shadow_decisions = decision_engine.run(
                            inference_output, llm_risks, features_df, shadow_config, validation,
                            expert_signals=expert_signals,
                            ranking_scores=ranking_scores,
                            ranking_blend=ranking_blend,
                        )

                        shadow_intents = {
                            'generated_date': run_date,
                            'generated_timestamp': datetime.now().isoformat(),
                            'shadow': True,
                            'candidate_version': candidate_bundle.get('version_id', 'unknown'),
                            'ranking_blend': ranking_blend,
                            'regime': shadow_decisions.get('regime', 'unknown'),
                            'actions': shadow_decisions.get('actions', []),
                            'buy_candidates': shadow_decisions.get('buy_candidates', []),
                        }
                        s3_client.write_json(shadow_intents, f'daily/{run_date}/shadow_intents.json')

                        # Write daily comparison
                        inc_syms = set(a.get('symbol', '') for a in trade_intents['actions'])
                        shd_syms = set(a.get('symbol', '') for a in shadow_intents['actions'])
                        comparison = {
                            'date': run_date,
                            'incumbent_version': config.get('active_params_metadata', {}).get('version_id', 'unknown'),
                            'candidate_version': candidate_bundle.get('version_id', 'unknown'),
                            'ranking_blend': ranking_blend,
                            'regime': decisions.get('regime', 'unknown'),
                            'incumbent_actions': len(trade_intents['actions']),
                            'shadow_actions': len(shadow_intents['actions']),
                            'symbol_overlap': sorted(inc_syms & shd_syms),
                            'incumbent_only': sorted(inc_syms - shd_syms),
                            'shadow_only': sorted(shd_syms - inc_syms),
                        }
                        s3_client.write_json(comparison, f'daily/{run_date}/shadow_comparison.json')
                        logger.info(
                            f"Shadow challenger ({candidate_bundle.get('version_id')}): "
                            f"{len(shadow_intents['actions'])} actions "
                            f"(incumbent: {len(trade_intents['actions'])})"
                        )
                    else:
                        logger.info("Shadow: ranking model not available at %s, skipping", model_path)
                else:
                    logger.info("Shadow: candidate has no ranking_blend, skipping")
            else:
                logger.info("Shadow: no candidate bundle or same as active, skipping")
        except Exception as shadow_err:
            logger.warning(f"Shadow challenger computation failed (non-fatal): {shadow_err}")

        # Step 11: Portfolio valuation update (NO trade execution).
        # Use the in-memory portfolio_state that decision_engine just mutated
        # (e.g. peak_price, consecutive_below_health_days). Re-loading from S3
        # at this point would discard those per-day mutations, which is how the
        # sell_health_days persistence counter would silently never increment
        # across runs.
        log_step(11, 12, "Updating portfolio valuations...", logger)
        with StepTimer("Portfolio valuation", logger):
            portfolio_state = config['portfolio_state']
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

        # Set up broker adapter
        with StepTimer("Broker setup", logger):
            broker_mode = resolve_broker_mode(config)
            alpaca_key_id = ''
            alpaca_secret_key = ''
            if broker_mode in (BrokerMode.ALPACA_PAPER, BrokerMode.ALPACA_LIVE):
                prefix = 'paper' if broker_mode == BrokerMode.ALPACA_PAPER else 'live'
                alpaca_key_id = get_secret(
                    f'investment-system/alpaca-{prefix}-key-id', region
                )
                alpaca_secret_key = get_secret(
                    f'investment-system/alpaca-{prefix}-secret-key', region
                )
            broker = get_broker(config, alpaca_key_id, alpaca_secret_key)
            logger.info("Broker mode: %s", broker.mode_label)

        # Execute morning phase
        with StepTimer("Morning execution", logger):
            result = morning_executor.run(bucket, config, broker=broker)

        portfolio_state = result['portfolio_state']
        trades = result['trades']
        validation_log = result.get('validation_log', [])

        # Load night artifacts for dashboard rebuild.
        # Use intents_date (when the night phase last ran), not date (which
        # the morning phase overwrites to today).  Over weekends these diverge.
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

        # Build morning execution report
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


def _run_midday_check(event: dict, bucket: str, region: str) -> dict:
    """
    Midday check phase: trailing stop re-eval, VIX circuit breaker, skipped-buy re-check.

    A lightweight check that reuses existing calibrated parameters. Does not
    run features, signals, inference, or the decision engine.
    """
    start_time = datetime.now()
    run_date = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Midday check started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        # Load configuration
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        # Set up broker adapter
        with StepTimer("Broker setup", logger):
            broker_mode = resolve_broker_mode(config)
            alpaca_key_id = ''
            alpaca_secret_key = ''
            if broker_mode in (BrokerMode.ALPACA_PAPER, BrokerMode.ALPACA_LIVE):
                prefix = 'paper' if broker_mode == BrokerMode.ALPACA_PAPER else 'live'
                alpaca_key_id = get_secret(
                    f'investment-system/alpaca-{prefix}-key-id', region
                )
                alpaca_secret_key = get_secret(
                    f'investment-system/alpaca-{prefix}-secret-key', region
                )
            broker = get_broker(config, alpaca_key_id, alpaca_secret_key)
            logger.info("Midday check broker mode: %s", broker.mode_label)

        # Run midday check
        with StepTimer("Midday check", logger):
            result = midday_checker.run(bucket, config, broker=broker)

        actions = result['actions_taken']
        check_log = result['check_log']
        circuit_breaker = result['circuit_breaker_active']
        portfolio_state = result['portfolio_state']

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Midday check completed in {duration:.1f}s")

        # Send email alert
        send_alert(
            subject=(
                f"[TraderBot] Midday: {len(actions)} actions"
                + (" [CIRCUIT BREAKER]" if circuit_breaker else "")
            ),
            body=format_midday_summary(
                run_date,
                portfolio_state.get('portfolio_value', 0),
                actions, check_log, circuit_breaker, duration
            ),
            region=region
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'phase': 'midday-check',
                'date': run_date,
                'duration_seconds': duration,
                'actions_taken': len(actions),
                'circuit_breaker_active': circuit_breaker,
                'portfolio_value': portfolio_state.get('portfolio_value', 0)
            })
        }

    except Exception as e:
        logger.error(f"Midday check failed: {e}", exc_info=True)

        send_alert(
            subject="[TraderBot] ALERT: Midday check failed",
            body=format_error_alert('midday-check', run_date, str(e)),
            region=region
        )

        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'failed',
                'phase': 'midday-check',
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
    parser.add_argument('--phase', default='night', choices=['night', 'morning', 'midday'],
                        help='Which phase to run')
    args = parser.parse_args()

    source_map = {'morning': 'morning-execution', 'midday': 'midday-check'}
    source = source_map.get(args.phase, 'manual')
    result = lambda_handler(
        {'bucket': args.bucket, 'region': args.region, 'source': source}, None
    )
    print(json.dumps(result, indent=2))
