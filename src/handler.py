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
from src.utils.market_calendar import latest_settled_session, settled_day_from_prices
from src.utils.s3_client import S3Client
from src.utils.logging_utils import setup_logger, log_step, StepTimer
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
    elif source == 'republish-dashboard':
        return _run_republish_dashboard(event, bucket, region)
    elif source == 'shadow-publish':
        return _run_shadow_publish(event, bucket, region)
    elif source == 'healthcheck':
        return _run_healthcheck(event, bucket, region)
    elif source == 'forecast-diag':
        return _run_forecast_diag(event, bucket, region)
    elif source == 'regime-diag':
        return _run_regime_diag(event, bucket, region)
    elif source == 'date-grid-diag':
        return _run_date_grid_diag(event, bucket, region)
    elif source == 'feeds-diag':
        return _run_feeds_diag(event, bucket, region)
    elif source == 'forecast-spine-diag':
        return _run_forecast_spine_diag(event, bucket, region)
    else:
        return _run_night_phase(event, bucket, region)


def _run_feeds_diag(event: dict, bucket: str, region: str) -> dict:
    """Governed, NON-DESTRUCTIVE reality-test of trader-bot-core/feeds/prices.py
    (PKT-TRADER-BOT-FEEDS-YAHOO-PRIMARY, P1).

    Runs the REBUILT Yahoo-v8-primary price feed from the AWS-IP Lambda context
    (the block that motivated the rebuild is AWS-IP-specific) and returns the
    packet's five acceptance criteria as machine-checkable rows. Writes NO S3
    object; the live night path (src/steps/ingest_prices.py) is untouched.

      1. full-64 universe -> non-empty OHLCV bars from Yahoo v8 primary;
      2. a stored settled SPY close cross-checks the Yahoo v8 quote to < 0.5%;
      3. stooq deliberately blocked -> full-64 still non-empty (no freeze), AND a
         forced primary-block probe shows the fallback path taken visibly;
      4. vvix_value / skew_value now nonzero (Yahoo v8 vol-index replacement);
      5. OHLCVBar dtypes conform, and a forced dtype-mismatch RAISES (not swallowed).
    """
    import sys
    from pathlib import Path as _Path

    # Bring the clean-folder module onto sys.path (baked at
    # ${LAMBDA_TASK_ROOT}/trader-bot-core by Dockerfile.lambda). NOT wired into
    # the live pipeline — imported only here for the governed reality-test.
    core = _Path(os.environ.get('LAMBDA_TASK_ROOT', '.')) / 'trader-bot-core'
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from feeds import prices as tbc_prices  # type: ignore
    from feeds import contracts as tbc_contracts  # type: ignore

    # Surface the feed module's INFO logs (Lambda's root logger defaults to
    # WARNING, which hides the handshake diagnostics).
    import logging as _logging
    _logging.getLogger('feeds').setLevel(_logging.INFO)

    s3 = S3Client(bucket, region)
    result: dict = {'phase': 'feeds-diag', 'nondestructive': True}

    # --- (0) surgical Yahoo handshake probe: is the raw v8 endpoint reachable
    # from this AWS IP with a proper cookie+crumb, or is it a hard IP block? ---
    with StepTimer("feeds-diag yahoo handshake probe", logger):
        result['yahoo_handshake_probe'] = tbc_prices.yahoo_handshake_probe()
    logger.info("feeds-diag handshake probe: %s", result['yahoo_handshake_probe'])

    # probe_only: a cheap (~2 Yahoo requests) reachability check. Used to test
    # whether the raw endpoint recovers after a cooldown WITHOUT re-hammering it
    # with a 64-symbol burst. Returns just the handshake probe.
    if event.get('probe_only'):
        result['probe_only'] = True
        return {'statusCode': 200, 'body': json.dumps(
            {'status': 'success', 'phase': 'feeds-diag', 'result': result},
            default=str)}

    # --- Universe (64) ---
    universe_df = s3.read_csv('config/universe.csv')
    symbols = universe_df['symbol'].tolist() if len(universe_df) else []
    result['universe_size'] = len(symbols)

    # --- (1)+(3a) full-64 with stooq DELIBERATELY BLOCKED (proves stooq's AWS-IP
    # block cannot freeze the run: Yahoo v8 primary carries the whole universe) ---
    with StepTimer("feeds-diag full-64 (stooq blocked)", logger):
        full_df, full_rep = tbc_prices.run_with_report(
            symbols, lookback_days=90, blocked_sources=['stooq'])
    n_nonempty = int(full_df['symbol'].nunique()) if len(full_df) else 0
    result['full_universe'] = {
        'requested': len(symbols),
        'symbols_nonempty': n_nonempty,
        'total_rows': int(len(full_df)),
        'served_by_source': full_rep.source_counts,
        'failed_symbols': full_rep.failed_symbols,
        'blocked_sources': full_rep.blocked_sources,
        'pass': bool(len(symbols) > 0 and n_nonempty == len(symbols)),
    }

    # --- (2) stored settled SPY close cross-checks Yahoo v8 to < 0.5% ---
    spy_check = {'pass': False}
    try:
        latest = s3.read_json('daily/latest.json') or {}
        settled = latest.get('intents_date') or latest.get('date')
        stored_spy = None
        if settled:
            stored = s3.read_parquet(f'daily/{settled}/prices.parquet')
            if len(stored):
                srows = stored[stored['symbol'] == 'SPY'].copy()
                if len(srows):
                    srows['date'] = pd.to_datetime(srows['date'])
                    stored_spy = float(srows.sort_values('date')['close'].iloc[-1])
        yv8 = tbc_prices.fetch_yahoo_v8_daily('SPY', lookback_days=15)
        yahoo_spy = None
        if len(yv8):
            yv8 = yv8.copy()
            yv8['date'] = pd.to_datetime(yv8['date'])
            match = yv8[yv8['date'] == pd.to_datetime(settled)] if settled else yv8
            row = match if len(match) else yv8
            yahoo_spy = float(row.sort_values('date')['close'].iloc[-1])
        pct = None
        if stored_spy and yahoo_spy:
            pct = abs(stored_spy - yahoo_spy) / stored_spy * 100.0
        spy_check = {
            'settled_date': settled,
            'stored_spy_close': stored_spy,
            'yahoo_v8_spy_close': yahoo_spy,
            'pct_diff': pct,
            'pass': bool(pct is not None and pct < 0.5),
        }
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow the whole diag
        spy_check = {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
    result['spy_crosscheck'] = spy_check

    # --- (3b) forced primary block -> fallback taken VISIBLY, no empty result ---
    probe = symbols[:5] if symbols else ['SPY', 'QQQ', 'TLT', 'GLD', 'HYG']
    with StepTimer("feeds-diag fallback probe (yahoo_v8 blocked)", logger):
        fb_df, fb_rep = tbc_prices.run_with_report(
            probe, lookback_days=30, blocked_sources=['yahoo_v8'])
    result['fallback_probe'] = {
        'probe_symbols': probe,
        'served_by_source': fb_rep.source_counts,
        'fallback_events': fb_rep.fallback_events,
        'symbols_nonempty': int(fb_df['symbol'].nunique()) if len(fb_df) else 0,
        'no_freeze_nonempty': bool(len(fb_df) > 0),
        'pass': bool(len(fb_df) > 0 and fb_rep.fallback_events),
    }

    # --- (3c) evidence that stooq itself is AWS-IP-blocked from this Lambda ---
    stooq_probe = tbc_prices.fetch_stooq_daily('SPY', lookback_days=10)
    result['stooq_from_aws'] = {
        'spy_rows': int(len(stooq_probe)),
        'blocked_or_empty': bool(len(stooq_probe) == 0),
        'note': 'stooq is a demoted last-resort fallback; empty here confirms the '
                'AWS-IP block that made it unusable as PRIMARY.',
    }

    # --- (4) vvix_value / skew_value now nonzero (Yahoo v8 replacement) ---
    vvix = tbc_prices.latest_vol_index('^VVIX')
    skew = tbc_prices.latest_vol_index('^SKEW')
    result['vol_complex'] = {
        'vvix_value': vvix,
        'skew_value': skew,
        'pass': bool(vvix and vvix > 0 and skew and skew > 0),
    }

    # --- (5) OHLCVBar dtype conformance + forced dtype-mismatch RAISES ---
    dtype_ok = None
    if len(full_df):
        got = {c: str(t) for c, t in full_df.dtypes.items()}
        want = {'date': 'datetime64[ns]', 'symbol': 'object', 'open': 'float64',
                'high': 'float64', 'low': 'float64', 'close': 'float64',
                'volume': 'int64'}
        dtype_ok = all(got.get(k) == v for k, v in want.items())
    forced_raise = False
    forced_detail = ''
    bad = pd.DataFrame([{'date': '2026-07-02', 'symbol': 'SPY', 'open': 1.0,
                         'high': 1.0, 'low': 1.0, 'close': 'not_a_number',
                         'volume': 100}])
    try:
        tbc_contracts.coerce_to_ohlcv(bad)
    except tbc_contracts.OHLCVContractError as exc:
        forced_raise = True
        forced_detail = str(exc)[:120]
    result['dtype_contract'] = {
        'output_dtypes_conform': dtype_ok,
        'forced_mismatch_raises': forced_raise,
        'forced_raise_detail': forced_detail,
        'pass': bool(dtype_ok and forced_raise),
    }

    # --- overall ---
    checks = {
        'full_universe_nonempty': result['full_universe']['pass'],
        'spy_close_within_0.5pct': result['spy_crosscheck']['pass'],
        'stooq_blocked_no_freeze': result['full_universe']['pass'],
        'fallback_visible': result['fallback_probe']['pass'],
        'vvix_skew_nonzero': result['vol_complex']['pass'],
        'dtype_coerce_or_raise': result['dtype_contract']['pass'],
    }
    result['acceptance'] = checks
    result['all_pass'] = all(checks.values())

    logger.info(
        "feeds-diag: full64=%d/%d spy_pct=%s vvix=%s skew=%s fallback=%s "
        "dtype_raise=%s ALL_PASS=%s",
        n_nonempty, len(symbols), spy_check.get('pct_diff'), vvix, skew,
        result['fallback_probe']['served_by_source'],
        forced_raise, result['all_pass'],
    )
    return {'statusCode': 200, 'body': json.dumps(
        {'status': 'success', 'phase': 'feeds-diag', 'result': result},
        default=str)}


def _run_forecast_spine_diag(event: dict, bucket: str, region: str) -> dict:
    """Governed, NON-DESTRUCTIVE reality-test of the RELOCATED forecast spine
    (PKT-TRADER-BOT-FORECAST-SPINE-RELOCATE, P3).

    Exercises the confirmed-correct kernel from its first-class trader-bot-core
    paths (store/forecast/engine/adapter/decide) — the prototype-dir-as-runtime
    trap is gone. Returns the packet's five acceptance criteria as machine-
    checkable rows. Writes NO S3 object; the live night path is untouched.

      1. run_inference(today) runs with NO runs/ on sys.path (imports/executes
         clean from trader-bot-core) + FREEZE shas compute first-class & match.
      2. the parity self-check is green (run_inference ABORTS on parity fail; a
         returned mu ⇒ parity passed).
      3. live mu_sha16 DIFFERS day-over-day on fresh substrate (not frozen).
      4. n_mu == full universe AND invariant_green=True (run_cutover clean path).
      5. an injected freeze-hash fault -> ok=False, the incumbent is byte-
         unchanged, and SNS fired (abort-never-degrade holds after relocation).
    """
    import sys
    import hashlib
    from pathlib import Path as _Path

    # Bring the clean-folder spine onto sys.path (baked at
    # ${LAMBDA_TASK_ROOT}/trader-bot-core by Dockerfile.lambda). NOT wired into
    # the live pipeline — imported only here for the governed reality-test.
    core = _Path(os.environ.get('LAMBDA_TASK_ROOT', '.')) / 'trader-bot-core'
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))

    result: dict = {'phase': 'forecast-spine-diag', 'nondestructive': True}

    # --- (1) first-class imports, no runs/ on sys.path, FREEZE shas match ---
    from decide import cutover as CUT          # noqa: E402 — first-class
    from forecast import freeze as FZ          # noqa: E402
    import pandas as pd
    runs_on_path = [p for p in sys.path if 'runs/pkt_tb' in p or 'runs\\pkt_tb' in p]
    fr = FZ.load_freeze()
    engine_sha = FZ.compute_engine_sha()
    model_sha = FZ.compute_model_sha()
    result['acc1_first_class_imports'] = {
        'no_runs_on_syspath': (runs_on_path == []),
        'runs_on_syspath': runs_on_path,
        'engine_sha': engine_sha,
        'engine_sha_matches_FREEZE': engine_sha == fr.get('engine', {}).get('engine_sha'),
        'model_sha': model_sha,
        'model_sha_matches_FREEZE': model_sha == fr.get('model_artifacts', {}).get('model_sha'),
        'model_root': str(FZ.resolve_model_root()),
    }

    # --- settled trading days (today + prior) for the day-over-day check ---
    settled = CUT._latest_settled_trading_day()
    prior = event.get('prior_date')
    if not prior:
        import datetime as _dt
        d = _dt.date.fromisoformat(settled)
        d -= _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        prior = d.isoformat()

    # --- (2)+(4-n_mu)+(3-today): run the RELOCATED forward inference for today ---
    with StepTimer("forecast-spine-diag run_inference(today)", logger):
        diag_today = CUT.diagnose_forecast_freshness(pending=[settled])
    n_mu = diag_today.get('n_mu', 0)
    mu_sha_today = diag_today.get('mu_sha16')
    result['acc2_parity_self_check'] = {
        'run_inference_returned_mu': n_mu > 0,
        'parity_green': n_mu > 0,   # run_inference ABORTS on parity fail; mu ⇒ green
        'note': 'run_inference raises RuntimeError on any parity divergence vs '
                'the frozen nightly reference; a non-empty mu proves parity passed.',
        'gate': diag_today.get('gate'),
    }

    # --- (3) day-over-day mu_sha16 must differ on fresh substrate ---
    with StepTimer("forecast-spine-diag run_inference(prior)", logger):
        diag_prior = CUT.diagnose_forecast_freshness(pending=[prior])
    mu_sha_prior = diag_prior.get('mu_sha16')
    result['acc3_mu_sha16_day_over_day'] = {
        'settled_date': settled, 'mu_sha16_today': mu_sha_today,
        'prior_date': prior, 'mu_sha16_prior': mu_sha_prior,
        'differs_day_over_day': bool(mu_sha_today and mu_sha_prior
                                     and mu_sha_today != mu_sha_prior),
        'substrate_max_bar': diag_today.get('ohlcv_max_date'),
    }

    # --- universe size for the n_mu == full-universe check ---
    uni_path = CUT._regime_compat_path().parent / 'universe.csv'
    universe_df = pd.read_csv(uni_path)
    universe_n = int(len(universe_df))
    mu_today = {}  # re-fetch full mu map for the run_cutover closure
    # diagnose_forecast_freshness returned only top10; recompute the full mu via
    # the same first-class path (cheap — substrate already extended).
    from forecast import shadow_lib as SL     # noqa: E402
    from forecast import inference as FI       # noqa: E402
    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()
    FI.build_panel([settled])
    recs = FI.run_inference([settled])
    mu_today = dict((recs.get(settled) or {}).get('mu', {}))

    # --- (4) invariant_green via the REAL run_cutover clean path ---
    live_cfg = CUT.load_brain_config()
    live_cfg = dict(live_cfg)
    live_cfg['mode'] = 'live'
    live_cfg['engine'] = 'native_two_stage'   # exercise the chassis-loaded assert
    live_cfg['forward_boundary'] = '2026-06-11'
    pstate = {'cash': 100000.0, 'holdings': []}
    incumbent = {'incumbent': True, 'source': 'forecast-spine-diag-synthetic',
                 'trade_intents': []}
    incumbent_bytes = json.dumps(incumbent, sort_keys=True).encode()
    incumbent_sha_before = hashlib.sha256(incumbent_bytes).hexdigest()[:16]

    def _closure_forecaster(pending):
        return {pending[-1]: {'mu': dict(mu_today)}}

    with StepTimer("forecast-spine-diag run_cutover(clean)", logger):
        # regime_label=None -> run_cutover resolves the as-of-D fused regime via the
        # ONE shared picker forecast.regime.regime(settled) (PKT-TRADER-BOT-REGIME-
        # AS-OF-D). The forward path now uses the real regime, not a constant neutral.
        res_clean = CUT.run_cutover(
            date=settled, features_df=None, regime_label=None,
            universe_df=universe_df, portfolio_state=pstate,
            forecaster=_closure_forecaster, config=live_cfg,
            incumbent_intents=incumbent)
    result['acc4_invariant_and_universe'] = {
        'n_mu': n_mu, 'universe_n': universe_n,
        'n_mu_eq_universe': n_mu == universe_n,
        'run_cutover_ok': bool(res_clean.ok),
        'invariant_green': bool(res_clean.invariant_green),
        'engine': res_clean.engine,
        'reason': res_clean.reason,
        'n_selected': len(res_clean.selected_universe or []),
    }

    # --- (5) injected freeze-hash fault -> ok=False + incumbent unchanged + SNS ---
    tampered = _Path('/tmp/tbc_tampered_engine')
    import shutil as _shutil
    if tampered.exists():
        _shutil.rmtree(tampered)
    _shutil.copytree(core / 'engine', tampered)
    # flip a byte in one engine file so compute_engine_sha != FREEZE.engine_sha
    victim = tampered / 'contracts.py'
    b = bytearray(victim.read_bytes())
    b[0] = (b[0] + 1) % 256
    victim.write_bytes(bytes(b))

    orig_engine_dir = FZ._ENGINE_DIR
    fault_reason = ''
    try:
        FZ._ENGINE_DIR = tampered            # assert_cold_start computes over tampered
        with StepTimer("forecast-spine-diag run_cutover(freeze-fault)", logger):
            res_fault = CUT.run_cutover(
                date=settled, features_df=None, regime_label='neutral',
                universe_df=universe_df, portfolio_state=pstate,
                forecaster=_closure_forecaster, config=live_cfg,
                incumbent_intents=incumbent)
        fault_reason = res_fault.reason
        fault_ok = bool(res_fault.ok)
        # incumbent passed straight through, byte-unchanged (run_cutover NEVER writes)
        inc_after = json.dumps(res_fault.incumbent_intents, sort_keys=True).encode()
        incumbent_sha_after = hashlib.sha256(inc_after).hexdigest()[:16]
    finally:
        FZ._ENGINE_DIR = orig_engine_dir
        if tampered.exists():
            _shutil.rmtree(tampered)

    # abort-never-degrade: fire the SNS CRITICAL the night caller fires on ok=False
    sns_message_id = None
    sns_error = None
    if fault_ok is False:
        try:
            resp = send_alert(
                subject="[TraderBot][FORECAST-SPINE-DIAG] injected FREEZE fault -> "
                        "ABORT to incumbent (reality-test, non-destructive)",
                body=(f"PKT-TRADER-BOT-FORECAST-SPINE-RELOCATE reality-test: an "
                      f"injected freeze-hash fault made run_cutover return "
                      f"ok=False:\n\n{fault_reason}\n\nThe incumbent intents were "
                      f"retained byte-unchanged and NOTHING was written to S3. "
                      f"This is a governed diag invoke, not a live abort."))
            sns_message_id = (resp or {}).get('MessageId') if isinstance(resp, dict) else str(resp)
        except Exception as e:  # noqa: BLE001
            sns_error = f"{type(e).__name__}: {e}"

    result['acc5_freeze_fault_abort'] = {
        'ok_is_false': fault_ok is False,
        'reason': fault_reason,
        'reason_is_freeze': 'FREEZE' in (fault_reason or ''),
        'incumbent_sha_before': incumbent_sha_before,
        'incumbent_sha_after': incumbent_sha_after,
        'incumbent_byte_unchanged': incumbent_sha_before == incumbent_sha_after,
        'sns_fired': sns_message_id is not None,
        'sns_message_id': sns_message_id,
        'sns_error': sns_error,
    }

    # --- overall pass roll-up ---
    result['all_pass'] = bool(
        result['acc1_first_class_imports']['no_runs_on_syspath'] and
        result['acc1_first_class_imports']['engine_sha_matches_FREEZE'] and
        result['acc1_first_class_imports']['model_sha_matches_FREEZE'] and
        result['acc2_parity_self_check']['parity_green'] and
        result['acc3_mu_sha16_day_over_day']['differs_day_over_day'] and
        result['acc4_invariant_and_universe']['n_mu_eq_universe'] and
        result['acc4_invariant_and_universe']['invariant_green'] and
        result['acc5_freeze_fault_abort']['ok_is_false'] and
        result['acc5_freeze_fault_abort']['reason_is_freeze'] and
        result['acc5_freeze_fault_abort']['incumbent_byte_unchanged'] and
        result['acc5_freeze_fault_abort']['sns_fired'])

    logger.info("forecast-spine-diag: acc1=%s acc2=%s acc3=%s acc4=%s acc5=%s all_pass=%s",
                result['acc1_first_class_imports']['no_runs_on_syspath'],
                result['acc2_parity_self_check']['parity_green'],
                result['acc3_mu_sha16_day_over_day']['differs_day_over_day'],
                result['acc4_invariant_and_universe']['invariant_green'],
                result['acc5_freeze_fault_abort']['ok_is_false'],
                result['all_pass'])
    return {'statusCode': 200, 'body': json.dumps(
        {'status': 'success', 'phase': 'forecast-spine-diag', 'result': result},
        default=str)}


def _run_forecast_diag(event: dict, bucket: str, region: str) -> dict:
    """Governed, NON-DESTRUCTIVE forecast-freshness probe (PKT-FORECAST-FRESHNESS-
    GATE). Runs the in-Lambda forward path (extend OHLCV -> panel -> inference)
    and returns the substrate-currency diagnostics WITHOUT writing any S3 object:
    the OHLCV watermark before/after the S3 extend, the exact per-date extend
    errors (the root of the ISSUE-01 swallow), the resulting mu hash / top-10, and
    the staleness-gate verdict. ``{"force_stale": true}`` skips the extend to prove
    the gate fires over a deliberately frozen store."""
    from src.brain import diagnose_forecast_freshness
    pending = event.get('pending')
    if isinstance(pending, str):
        pending = [pending]
    force_stale = bool(event.get('force_stale'))
    with StepTimer("Forecast freshness diagnostic", logger):
        result = diagnose_forecast_freshness(pending=pending, force_stale=force_stale)
    logger.info(f"forecast-diag: settled={result.get('settled_trading_day')} "
                f"ohlcv_max={result.get('ohlcv_max_date')} n_mu={result.get('n_mu')} "
                f"gate_stale={result.get('gate', {}).get('stale')}")
    return {'statusCode': 200, 'body': json.dumps({
        'status': 'success', 'phase': 'forecast-diag', 'result': result},
        default=str)}


def _run_regime_diag(event: dict, bucket: str, region: str) -> dict:
    """Governed, NON-DESTRUCTIVE regime-chassis probe (PKT-3). Proves the chassis
    is LOADED (regime_compat_loaded=True) and observably RE-RANKS a fresh mu
    (regime_score_mult != 1.0; regime-tilted top-10 differs from raw-mu top-10),
    and that the fail-loud startup assertion fires if the table is emptied under
    native_two_stage. Writes NO S3 object."""
    from src.brain import diagnose_regime_chassis
    pending = event.get('pending')
    if isinstance(pending, str):
        pending = [pending]
    with StepTimer("Regime chassis diagnostic", logger):
        result = diagnose_regime_chassis(pending=pending)
    logger.info(f"regime-diag: regime={result.get('regime_label')} "
                f"compat_loaded={result.get('regime_compat_loaded')} "
                f"reranks_top10={result.get('regime_tilt_reranks_top10')} "
                f"mu_sha={result.get('mu_sha16')}")
    return {'statusCode': 200, 'body': json.dumps({
        'status': 'success', 'phase': 'regime-diag', 'result': result},
        default=str)}


def _run_date_grid_diag(event: dict, bucket: str, region: str) -> dict:
    """Governed, NON-DESTRUCTIVE proof of the settled-day forward stamp (PKT-4).

    Writes NOTHING to S3. Reads the LIVE equity-ledger frontier (read-only) and, for
    a suite of simulated ``asof`` instants + settled SPY closes, reports:

      * the OLD buggy key (UTC ``datetime.now()`` calendar date) vs the NEW key (the
        settled NY trading day) — a Friday-evening UTC instant is Saturday under the
        old key, Friday under the new one;
      * whether the equity append would ADD a leaf or NO-OP against the live frontier
        (``run_date <= frontier`` = no-op) — so weekend/holiday runs produce no leaf;
      * that the live frontier leaf carries BOTH the canon ``value`` and the SPY
        ``benchmark`` on ONE date — SPY + canon ride one real-trading-day grid.

    Optional event override ``scenarios``: list of ``{label, asof, spy_max}`` rows
    (``asof`` = ISO-8601 UTC instant; ``spy_max`` = settled SPY close in the fresh
    panel, omit to use the clock proxy).
    """
    from datetime import datetime as _dt, timezone as _tz
    import pandas as _pd
    from src.canon.equity_ledger import EquityLedger
    from src.canon.equity_append import _cache_tail
    from src.utils.market_calendar import (
        ny_today, latest_settled_session, settled_day_from_prices,
    )

    s3 = S3Client(bucket, region)

    # Read-only frontier: the live settled leaf both lines share.
    frontier = None
    front_date = None
    try:
        ledger = EquityLedger(s3.s3, bucket)
        frontier = ledger.frontier()
        tail = _cache_tail(s3.s3, bucket)
        if tail:
            frontier = {
                'date': tail.get('date'),
                'value': tail.get('value'),
                'benchmark': tail.get('benchmark'),
                'shares_one_grid': ('value' in tail and 'benchmark' in tail
                                    and tail.get('date') is not None),
            }
            front_date = tail.get('date')
        elif frontier:
            front_date = frontier.get('date')
    except Exception as e:  # noqa: BLE001 — read-only best-effort
        logger.warning(f"date-grid-diag: frontier read soft-fail: {e}")

    def _utc_calendar_date(asof_iso: str) -> str:
        """The OLD buggy key: the UTC calendar date of the instant."""
        d = _dt.fromisoformat(asof_iso.replace('Z', '+00:00'))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_tz.utc)
        return d.astimezone(_tz.utc).strftime('%Y-%m-%d')

    # Default suite: the exact failure the packet targets (Fri-night UTC = Saturday),
    # a Saturday and Sunday cron mis-fire, and a holiday (SPY close does not advance).
    default_scenarios = [
        {'label': 'Friday-night settle (03:00 UTC Sat)',
         'asof': '2026-06-27T03:00:00Z', 'spy_max': '2026-06-26'},
        {'label': 'Saturday run (no new session)',
         'asof': '2026-06-27T14:00:00Z', 'spy_max': '2026-06-26'},
        {'label': 'Sunday run (no new session)',
         'asof': '2026-06-28T14:00:00Z', 'spy_max': '2026-06-26'},
        {'label': 'Holiday run (SPY close frozen at prior session)',
         'asof': '2026-07-03T22:00:00Z', 'spy_max': '2026-07-02'},
        {'label': 'Normal weeknight (03:00 UTC Wed)',
         'asof': '2026-07-02T03:00:00Z', 'spy_max': '2026-07-01'},
    ]
    scenarios = event.get('scenarios') or default_scenarios

    rows = []
    for sc in scenarios:
        asof = sc['asof']
        asof_dt = _dt.fromisoformat(asof.replace('Z', '+00:00'))
        spy_max = sc.get('spy_max')
        if spy_max:
            panel = _pd.DataFrame([{'symbol': 'SPY', 'date': spy_max}])
            settled = settled_day_from_prices(panel, symbol='SPY', now_utc=asof_dt)
        else:
            settled = latest_settled_session(now_utc=asof_dt)
        old_key = _utc_calendar_date(asof)
        # The append-only frontier guard: run_date <= frontier -> NO leaf.
        if front_date is None:
            decision = 'unknown (ledger unseeded)'
        elif settled > front_date:
            decision = 'APPEND (new settled leaf)'
        else:
            decision = 'NO-OP (no leaf: run_date <= frontier)'
        rows.append({
            'label': sc['label'],
            'asof_utc': asof,
            'ny_calendar_date': ny_today(asof_dt).isoformat(),
            'old_utc_key': old_key,
            'settled_day_key': settled,
            'fixed_drift': bool(old_key != settled),
            'append_decision_vs_frontier': decision,
        })

    # Live "now" resolution for the record.
    now_settled = latest_settled_session()
    result = {
        'phase': 'date-grid-diag',
        'nondestructive': True,
        'live_frontier': frontier,
        'spy_and_canon_one_grid': bool(frontier and frontier.get('shares_one_grid')),
        'now_settled_trading_day': now_settled,
        'now_old_utc_key': datetime.now(_tz.utc).strftime('%Y-%m-%d'),
        'scenarios': rows,
    }
    logger.info(f"date-grid-diag: frontier={front_date} one_grid="
                f"{result['spy_and_canon_one_grid']} now_settled={now_settled}")
    for r in rows:
        logger.info(f"  [DATE-GRID] {r['label']}: old_utc={r['old_utc_key']} -> "
                    f"settled={r['settled_day_key']} => {r['append_decision_vs_frontier']}")
    return {'statusCode': 200, 'body': json.dumps(
        {'status': 'success', 'result': result}, default=str)}


def _run_shadow_publish(event: dict, bucket: str, region: str) -> dict:
    """Autonomous cloud replacement for the laptop shadow launchd job
    (``com.traderbot.shadow.plist``).

    Runs the PKT-TB-007 dual-forward shadow end to end and publishes
    ``dashboard/shadow_timeseries.json`` — the dotted challenger line. ISOLATED
    from the trade pipeline: it appends NO equity leaf and cannot touch the canon
    line. A failure ALERTS (the challenger is one of the three watched lines).
    """
    import sys
    from pathlib import Path as _Path
    run_date = event.get('run_date') or latest_settled_session()  # settled NY day (PKT-4)
    # File logging -> /tmp; SHADOW lives on the read-only /var/task in Lambda.
    os.environ.setdefault('SHADOW_LOG_DIR', '/tmp/shadow_logs')
    # Bootstrap the baked shadow package onto sys.path (same recipe as
    # src.brain.runtime, which already runs the forward inference in-cloud).
    cands = []
    if os.environ.get('BRAIN_RUNTIME_SUBSET'):
        cands.append(os.environ['BRAIN_RUNTIME_SUBSET'])
    cands.append(str(_Path(os.environ.get('LAMBDA_TASK_ROOT', '.'))
                     / 'runs' / 'pkt_tb_007_orthogonal_brain' / 'shadow'))
    for sp in cands:
        if sp and _Path(sp).exists() and sp not in sys.path:
            sys.path.append(sp)
    # Non-destructive route/reachability verification. Proving the shadow-publish
    # path FIRES must never require a destructive live publish (PKT-SHADOW-PUBLISH-
    # SAFETY): a dry-run invoke confirms the branch is reachable and the shadow
    # package loads WITHOUT mutating any S3 object.
    dry_run = bool(event.get('dry_run') or event.get('publish') is False)
    try:
        import shadow_nightly as SN  # type: ignore
        if dry_run:
            logger.info("Shadow publish DRY-RUN: route reachable, shadow module "
                        "loaded, no S3 write performed")
            return {'statusCode': 200, 'body': json.dumps({
                'status': 'success', 'phase': 'shadow-publish',
                'dry_run': True, 'route_ok': True, 'published': False})}
        with StepTimer("Shadow publish (cloud)", logger):
            ctx = SN.build_production_ctx(publish=True)
            summary = SN.run_night(ctx)
        s3 = S3Client(bucket, region)
        shadow = s3.read_json('dashboard/shadow_timeseries.json') or {}
        as_of = (shadow.get('as_of') or '')[:19]
        sa = shadow.get('shadow_A') or []
        last_sa = sa[-1][0] if sa else None
        logger.info(f"Shadow publish OK: as_of={as_of} last_shadow_A={last_sa}")
        return {'statusCode': 200, 'body': json.dumps({
            'status': 'success', 'phase': 'shadow-publish',
            'as_of': as_of, 'last_shadow_A': last_sa,
            'summary': summary if isinstance(summary, dict) else str(summary)},
            default=str)}
    except Exception as e:
        logger.error(f"Shadow publish FAILED: {e}", exc_info=True)
        try:
            send_alert(
                subject="[TraderBot] CRITICAL: challenger (shadow) publish FAILED",
                body=(f"The cloud shadow-publish run failed for {run_date}.\n\n"
                      f"{type(e).__name__}: {e}\n\n"
                      f"The dotted challenger line will not advance until this is "
                      f"fixed. Check CloudWatch."))
        except Exception:  # noqa: BLE001
            pass
        return {'statusCode': 500, 'body': json.dumps(
            {'status': 'error', 'phase': 'shadow-publish', 'error': str(e)})}


def _run_healthcheck(event: dict, bucket: str, region: str) -> dict:
    """Independent daily three-line health report (the anti-betrayal watchdog).

    Reads the actual published S3 surfaces and emails a ✓/✗ status for the canon,
    SPY-benchmark, and challenger lines — so a silently-frozen line is caught the
    SAME day instead of by eyeballing the chart on Friday.
    """
    from src.brain import monitors
    s3 = S3Client(bucket, region)
    status = monitors.run_daily_health_check(s3)
    return {'statusCode': 200, 'body': json.dumps({
        'status': 'success', 'phase': 'healthcheck',
        'ok': status.get('ok'), 'lines': status.get('lines'),
        'chassis': status.get('chassis'), 'subject': status.get('subject')},
        default=str)}


def _run_night_phase(event: dict, bucket: str, region: str) -> dict:
    """
    Night analysis phase: full pipeline + generate trade intents.

    Runs steps 1-12 but does NOT execute trades. Instead, saves trade intents
    for the morning execution phase. Portfolio valuations are updated with
    closing prices so the dashboard shows current values.
    """
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")

    # Forward stamp = the settled NY trading day, NOT UTC datetime.now() (PKT-4).
    # The night fires at 03:00 UTC = the prior ET evening, so a UTC stamp lands one
    # calendar day ahead of the settled market day (Friday's settle -> a Saturday
    # leaf). Seed from the ET session; the true key is the freshly-ingested SPY
    # close set right after price ingest below.
    run_date = event.get('run_date') or latest_settled_session()
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
            prices_df = ingest_prices.run(
                symbols,
                alphavantage_key=alphavantage_key,
            )

        # Key every leaf/decision/folder to the SETTLED NY trading day — the actual
        # close date the book is marked at (max SPY bar in the fresh panel), not the
        # UTC wall clock (PKT-4). Weekends/holidays have no bar, so a non-trading-day
        # run resolves to the last settled session and the append-only frontier guard
        # (run_date <= frontier) makes it a no-op = no leaf.
        settled_run_date = settled_day_from_prices(prices_df, symbol='SPY')
        if settled_run_date != run_date:
            logger.info(
                f"run_date re-keyed to settled NY trading day: {run_date} -> "
                f"{settled_run_date} (settled SPY close, not UTC now)"
            )
            run_date = settled_run_date

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

        # Single-book invariant (PKT-TB-001): the internal sim book's dollar
        # values stay out of log lines — only counts are logged here.
        print(f"  Holdings: {len(portfolio_state['holdings'])}")
        print(f"  Pending intents: {len(trade_intents['actions'])}")

        # Step 12: LLM weather blurb (uses Bedrock/Haiku, falls back to OpenAI).
        # The prompt's portfolio posture comes from the CANON line (last
        # published dashboard.json) — never from the internal sim book
        # (PKT-TB-001). Best-effort: when unavailable, the prompt omits it.
        canon_metrics = None
        try:
            _dash = s3_client.read_json('dashboard/dashboard.json') or {}
            _m = _dash.get('metrics', {})
            if _m.get('canon_source') in ('ledger', 'new_brain', 'optimized_champion') and _m.get('total_value'):
                canon_metrics = {
                    'total_value': float(_m['total_value']),
                    'cash_pct': float(_m.get('cash_pct', 0.0) or 0.0),
                    'num_positions': len(_dash.get('holdings', [])),
                }
        except Exception as canon_err:
            logger.warning(f"Canon metrics unavailable for weather prompt: {canon_err}")

        log_step(12, 12, "Generating weather blurb...", logger)
        with StepTimer("LLM weather", logger):
            weather = llm_weather.run(
                inference_output, decisions, portfolio_state, context_df, openai_key, region,
                expert_signals=expert_signals,
                canon_metrics=canon_metrics,
            )

        # Publish all artifacts to S3
        logger.info("Publishing artifacts to S3...")
        with StepTimer("Publish artifacts", logger):
            publish_result = publish_artifacts.run(
                bucket, run_date,
                prices_df, context_df, features_df,
                inference_output, llm_risks, decisions,
                portfolio_state, trades, weather, validation,
                expert_signals=expert_signals,
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Pipeline completed in {duration:.1f}s")

        # Send email alert. The reported value is the CANON line from this
        # run's publish (None when the advance guard held the dashboard).
        canon_total_value = publish_result.get('canon_total_value')
        regime_label = decisions.get('expert_metrics', {}).get(
            'final_regime_label',
            inference_output.get('regime', {}).get('label', 'unknown')
        )
        send_alert(
            subject=f"[TraderBot] Night: {regime_label}, {len(trade_intents['actions'])} intents",
            body=format_night_summary(
                run_date, regime_label,
                canon_total_value,
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
                'canon_total_value': canon_total_value
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
    # Settled NY trading day, not UTC now (PKT-4). Morning fires 14:45 UTC = 09:45
    # ET, so the ET session date is the correct grid key.
    run_date = event.get('run_date') or latest_settled_session()
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
        morning_prices = result.get('morning_prices')

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
            publish_result = publish_artifacts.publish_morning_artifacts(
                bucket, run_date, portfolio_state, trades,
                morning_execution_report, night_inference, night_decisions,
                night_weather, expert_signals=expert_signals,
                morning_prices=morning_prices,
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Morning execution completed in {duration:.1f}s")

        # Send email alert. The morning email reports the CANON book from this
        # run's publish (PKT-TB-001) — never the internal sim book.
        canon_total_value = publish_result.get('canon_total_value')
        canon_subject = (
            f"${canon_total_value:,.0f}" if canon_total_value is not None
            else "canon n/a"
        )
        send_alert(
            subject=(
                f"[TraderBot] Morning: {len(trades)} trades, {canon_subject}"
            ),
            body=format_morning_summary(
                run_date,
                canon_total_value,
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
                'canon_total_value': canon_total_value
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
    # Settled NY trading day, not UTC now (PKT-4). Midday fires 18:00 UTC = 13:00 ET.
    run_date = event.get('run_date') or latest_settled_session()
    logger.info(f"Midday check started at {start_time}")

    s3_client = S3Client(bucket, region)

    try:
        # Load configuration
        with StepTimer("Load configuration", logger):
            config = load_config_from_s3(s3_client)

        # Run midday check
        with StepTimer("Midday check", logger):
            result = midday_checker.run(bucket, config)

        actions = result['actions_taken']
        check_log = result['check_log']
        circuit_breaker = result['circuit_breaker_active']
        portfolio_state = result['portfolio_state']

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Midday check completed in {duration:.1f}s")

        # Midday does not rebuild the dashboard, so the canon value is read
        # from the last published dashboard.json (best-effort, PKT-TB-001).
        canon_total_value = None
        try:
            _dash = s3_client.read_json('dashboard/dashboard.json') or {}
            _m = _dash.get('metrics', {})
            if _m.get('canon_source') in ('ledger', 'new_brain', 'optimized_champion') and _m.get('total_value'):
                canon_total_value = float(_m['total_value'])
        except Exception as canon_err:
            logger.warning(f"Canon metrics unavailable for midday email: {canon_err}")

        # Send email alert
        send_alert(
            subject=(
                f"[TraderBot] Midday: {len(actions)} actions"
                + (" [CIRCUIT BREAKER]" if circuit_breaker else "")
            ),
            body=format_midday_summary(
                run_date,
                canon_total_value,
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
                'canon_total_value': canon_total_value
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


def _run_republish_dashboard(event: dict, bucket: str, region: str) -> dict:
    """PURE RE-SERVE of the stored equity ledger (clean core, G-REPUBLISH-PURE).

    No trading, no new day, NO recompute, NO re-anchor. Rebuilds dashboard.json by
    reading the STORED ledger line (build_dashboard_data folds equity_history.jsonl)
    and re-serving it behind the parity-or-hold gate. It does NOT append a leaf (the
    night owns the one settled append/day) — so a manual re-serve is byte-identical
    on every call by construction: the leaves are immutable, the fold is pure.
    """
    from src.steps.publish_artifacts import (
        build_dashboard_data, _build_snapshot_meta, _can_publish_dashboard,
        _verify_ledger_or_hold,
    )
    from src.utils.dashboard_metrics import attach_new_brain_surface, sanitize_nan_for_json

    run_date = event.get('run_date') or latest_settled_session()  # settled NY day (PKT-4)
    s3 = S3Client(bucket, region)

    portfolio_state = paper_trader.load_portfolio_state(s3)
    latest = s3.read_json('daily/latest.json') or {}
    night_date = latest.get('intents_date', latest.get('date', run_date))
    inference = s3.read_json(f'daily/{night_date}/inference.json') or {}
    decisions = s3.read_json(f'daily/{night_date}/decisions.json') or {}
    weather = s3.read_json(f'daily/{night_date}/weather_blurb.json') or {}

    # Reconstruct expert_signals from the stored signals parquet (handler logic).
    expert_signals = None
    try:
        sig = s3.read_parquet(f'daily/{night_date}/signals.parquet')
        if len(sig) > 0:
            row = sig.iloc[0]
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
        logger.warning(f"republish: could not reconstruct expert_signals: {e}")

    if not _can_publish_dashboard(expert_signals):
        return {'statusCode': 409, 'body': json.dumps(
            {'status': 'skipped', 'phase': 'republish-dashboard',
             'reason': 'expert_signals null/incomplete'})}

    snapshot_meta = _build_snapshot_meta(run_date, 'morning', portfolio_state)
    dash = build_dashboard_data(portfolio_state, inference, decisions, weather, s3,
                                expert_signals=expert_signals, snapshot_meta=snapshot_meta)
    try:
        shadow = s3.read_json('dashboard/shadow_timeseries.json')
    except Exception:
        shadow = None
    dash = attach_new_brain_surface(dash, shadow)
    dash = sanitize_nan_for_json(dash)

    ok, reason = _verify_ledger_or_hold(dash, s3, 'morning', run_date)
    if not ok:
        return {'statusCode': 409, 'body': json.dumps(
            {'status': 'held', 'phase': 'republish-dashboard', 'reason': reason})}

    s3.write_json(dash, 'dashboard/data/dashboard.json')
    s3.write_json(dash, 'dashboard/dashboard.json')

    m = dash.get('metrics', {})
    tail = [{'date': r['date'], 'value': r['value']}
            for r in dash.get('equity_curve', []) if r['date'] >= '2026-06-16']
    return {'statusCode': 200, 'body': json.dumps({
        'status': 'success', 'phase': 'republish-dashboard', 'run_date': run_date,
        'max_drawdown': m.get('max_drawdown'), 'total_value': m.get('total_value'),
        'tail_06_16_onward': tail,
    })}


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
