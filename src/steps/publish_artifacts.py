"""Publish artifacts to S3 for dashboard consumption."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.utils.s3_client import S3Client
from src.utils.dashboard_metrics import compute_canonical_dashboard_metrics


def _invalidate_dashboard_cache() -> None:
    distribution_id = os.environ.get('DASHBOARD_CLOUDFRONT_DISTRIBUTION_ID')
    if not distribution_id:
        return
    try:
        import boto3
        client = boto3.client('cloudfront')
        caller_reference = f"publish-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        resp = client.create_invalidation(
            DistributionId=distribution_id,
            InvalidationBatch={
                'Paths': {'Quantity': 1, 'Items': ['/*']},
                'CallerReference': caller_reference,
            },
        )
        print(f"  CloudFront invalidation queued: {resp['Invalidation']['Id']}")
    except Exception as e:
        print(f"  CloudFront invalidation failed (non-fatal): {e}")


def _load_chart_markers() -> List[Dict[str, Any]]:
    """Load timeline markers from config/chart_markers.json.

    Markers are vertical event labels rendered on time-series dashboard charts
    (equity curve, drawdowns, regime strip). The config file is small and
    editable so operators can add new markers (model upgrades, bug fixes,
    incidents) without changing code.
    """
    env_override = os.environ.get('CHART_MARKERS_PATH')
    candidates: List[Path] = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path('config/chart_markers.json'))
    candidates.append(Path(__file__).resolve().parents[2] / 'config' / 'chart_markers.json')
    for p in candidates:
        if p.is_file():
            try:
                with p.open() as f:
                    data = json.load(f)
                markers = data.get('markers', [])
                return sorted(markers, key=lambda m: m.get('date', ''))
            except (OSError, json.JSONDecodeError):
                return []
    return []

DUST_SHARE_EPSILON = 0.001
DUST_VALUE_EPSILON = 0.01


def _build_snapshot_meta(
    run_date: str,
    phase: str,
    portfolio_state: Dict[str, Any],
) -> Dict[str, str]:
    """Build a stable snapshot identifier shared by all dashboard panels."""
    timestamp = (
        portfolio_state.get('last_updated')
        or datetime.now().isoformat()
    )
    snapshot_id = f"{run_date}:{phase}:{timestamp}"
    return {
        'id': snapshot_id,
        'date': run_date,
        'phase': phase,
        'timestamp': timestamp,
    }


def build_dashboard_data(
    portfolio_state: Dict[str, Any],
    inference_output: Dict[str, Any],
    decisions: Dict[str, Any],
    weather: Dict[str, Any],
    s3: S3Client,
    expert_signals: Optional[Dict[str, Any]] = None,
    snapshot_meta: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the dashboard.json data structure for the frontend."""
    snapshot = snapshot_meta or _build_snapshot_meta(
        run_date=datetime.now().strftime('%Y-%m-%d'),
        phase='night',
        portfolio_state=portfolio_state,
    )
    timestamp = snapshot['timestamp']
    snapshot_date = snapshot['date']
    snapshot_id = snapshot['id']

    canonical = compute_canonical_dashboard_metrics(
        s3=s3,
        portfolio_state=portfolio_state,
        snapshot_date=snapshot_date,
        current_state=portfolio_state,
        max_days=730,
        initial_value=100000.0,
        risk_free_rate_annual=0.0,
        min_sharpe_observations=60,
    )
    canonical_metrics = canonical['metrics']
    continuity_curve = canonical.get('equity_curve', [])
    continuity_total_value = (
        continuity_curve[-1]['value']
        if continuity_curve
        else portfolio_state.get('portfolio_value', 100000)
    )
    broker_total_value = portfolio_state.get('portfolio_value', continuity_total_value)

    # Keep invested aligned with current holdings if upstream field is missing.
    invested = portfolio_state.get('invested')
    if invested is None:
        invested = sum(float(h.get('market_value', 0.0) or 0.0) for h in portfolio_state.get('holdings', []))

    # Build metrics
    metrics = {
        # Continuity-adjusted value for dashboard presentation.
        'total_value': continuity_total_value,
        # Raw broker/account-reconciled value for auditability.
        'broker_total_value': broker_total_value,
        'cash': portfolio_state.get('cash', 100000),
        'invested': invested,
        'ytd_return': canonical_metrics['ytd_return'],
        'mtd_return': canonical_metrics['mtd_return'],
        'sharpe_ratio': canonical_metrics['sharpe_ratio'],
        'sharpe_observations': canonical_metrics['sharpe_observations'],
        'sharpe_min_observations': canonical_metrics['sharpe_min_observations'],
        'max_drawdown': canonical_metrics['max_drawdown'],
        'current_drawdown': canonical_metrics['current_drawdown'],
        'win_rate': canonical_metrics['win_rate'],
        'total_trades': canonical_metrics['total_trades'],
        'wins': canonical_metrics['wins'],
        'losses': canonical_metrics['losses'],
        'breakeven_trades': canonical_metrics['breakeven_trades'],
        'realized_round_trips': canonical_metrics['realized_round_trips'],
        'total_fills': canonical_metrics['total_fills'],
        'cumulative_transaction_costs': canonical_metrics['cumulative_transaction_costs'],
        'cash_pct': canonical_metrics['cash_pct'],
        'gross_exposure': canonical_metrics['gross_exposure'],
        'net_exposure': canonical_metrics['net_exposure'],
        'top_position_pct': canonical_metrics['top_position_pct'],
        'beta_proxy': canonical_metrics['beta_proxy'],
        'snapshot_id': snapshot_id,
        'timestamp': timestamp,
    }

    # Build holdings
    holdings = []
    for h in portfolio_state.get('holdings', []):
        shares = float(h.get('shares', 0) or 0)
        market_value = float(h.get('market_value', 0) or 0)
        if abs(shares) < DUST_SHARE_EPSILON or abs(market_value) < DUST_VALUE_EPSILON:
            continue
        holdings.append({
            'symbol': h.get('symbol', ''),
            'shares': shares,
            'entry_price': h.get('entry_price', 0),
            'current_price': h.get('current_price', 0),
            'market_value': market_value,
            'unrealized_pnl': h.get('unrealized_pnl', 0),
            'unrealized_pnl_pct': h.get('unrealized_pnl_pct', 0),
            'health_score': h.get('health_score', 0.5),
            'vol_bucket': h.get('vol_bucket', 'med'),
            'days_held': h.get('days_held', 0)
        })

    # Build candidates from decisions
    candidates = []
    for candidate in decisions.get('buy_candidates', []):
        candidates.append({
            'symbol': candidate.get('symbol', ''),
            'score': candidate.get('score', 0),
            'health_score': candidate.get('health_score', 0.5),
            'vol_bucket': candidate.get('vol_bucket', 'med'),
            'behavior': candidate.get('behavior', 'mixed'),
            'return_21d': candidate.get('return_21d', 0),
            'return_63d': candidate.get('return_63d', 0),
            'suggested_size': candidate.get('suggested_size', 0)
        })

    # Canonical timeseries from the same snapshot context.
    equity_curve = canonical['equity_curve']
    drawdowns = canonical['drawdowns']
    monthly_returns = canonical['monthly_returns']

    # Build regime info (use fused regime if available)
    regime_data = inference_output.get('regime', {})
    expert_metrics = decisions.get('expert_metrics', {})
    regime_label = expert_metrics.get('final_regime_label', regime_data.get('label', 'unknown'))

    risk_level_map = {
        'calm_uptrend': 'low',
        'risk_on_trend': 'low',
        'choppy': 'medium',
        'risk_off_trend': 'high',
        'high_vol_panic': 'extreme'
    }

    # Build ensemble metrics if available
    model_versions = inference_output.get('model_versions', {})
    is_ensemble = model_versions.get('ensemble', False)

    ensemble_metrics = {
        'confidence': regime_data.get('confidence', 1.0),
        'disagreement': regime_data.get('disagreement', 0.0),
        'agreement': regime_data.get('agreement', 1.0),
        'position_size_multiplier': regime_data.get('position_size_multiplier', 1.0),
        'is_ensemble': is_ensemble,
    }

    # Add individual model predictions if ensemble
    if is_ensemble:
        gru_pred = regime_data.get('gru_prediction', {})
        trans_pred = regime_data.get('transformer_prediction', {})
        ensemble_metrics['gru_prediction'] = {
            'label': gru_pred.get('label', ''),
            'confidence': gru_pred.get('confidence', 0),
            'probs': gru_pred.get('probs', {})
        }
        ensemble_metrics['transformer_prediction'] = {
            'label': trans_pred.get('label', ''),
            'confidence': trans_pred.get('confidence', 0),
            'probs': trans_pred.get('probs', {})
        }

    regime_info = {
        'regime': regime_label,
        'description': regime_data.get('description', ''),
        'risk_level': risk_level_map.get(regime_label, 'medium'),
        'probs': regime_data.get('probs', {}),
        'ensemble': ensemble_metrics
    }

    # Build weather report
    weather_report = {
        'headline': weather.get('headline', 'Market Update'),
        'summary': weather.get('summary', weather.get('blurb', '')),
        'regime': regime_info,
        'outlook': weather.get('outlook', ''),
        'risks': weather.get('risks', []),
        'timestamp': timestamp,
    }

    # Canonical fill history and round-trip summary from same active segment.
    fills = canonical.get('fills', [])
    trades_history = []
    for fill in fills:
        trade = {k: v for k, v in fill.items() if not k.startswith('_')}
        trades_history.append(trade)
    trades_history.sort(key=lambda t: t.get('timestamp', ''), reverse=True)

    result = {
        'snapshot': snapshot,
        'panel_snapshot_ids': {
            'metrics': snapshot_id,
            'equity_curve': snapshot_id,
            'drawdowns': snapshot_id,
            'monthly_returns': snapshot_id,
            'trade_log': snapshot_id,
            'regime': snapshot_id,
        },
        'metrics': metrics,
        'holdings': holdings,
        'candidates': candidates,
        'equity_curve': equity_curve,
        'drawdowns': drawdowns,
        'monthly_returns': monthly_returns,
        'chart_markers': _load_chart_markers(),
        'weather': weather_report,
        'trades': trades_history,
        'trade_summary': canonical.get('trade_summary', {}),
        'round_trips': canonical.get('round_trips', []),
        'reset_boundary': canonical.get('reset_boundary'),
    }

    # Add expert signals if available
    if expert_signals is not None:
        macro = expert_signals.get('macro_credit', {})
        vol = expert_signals.get('vol_uncertainty', {})
        frag = expert_signals.get('fragility', {})
        ent = expert_signals.get('entropy_shift', {})
        result['expert_signals'] = {
            'macro_credit_score': macro.get('macro_credit_score', 0.0),
            'yield_slope_10y_3m': macro.get('yield_slope_10y_3m', 0.0),
            'hy_spread_proxy': macro.get('hy_spread_proxy', 0.0),
            'vol_uncertainty_score': vol.get('vol_uncertainty_score', 0.5),
            'vol_regime_label': vol.get('vol_regime_label', 'calm'),
            'vix_percentile': vol.get('vix_percentile', 0.5),
            'vvix_percentile': vol.get('vvix_percentile', 0.5),
            'fragility_score': frag.get('fragility_score', 0.5),
            'avg_correlation': frag.get('avg_correlation', 0.0),
            'pc1_explained': frag.get('pc1_explained', 0.0),
            'entropy_score': ent.get('entropy_score', 0.5),
            'entropy_z_score': ent.get('entropy_z_score', 0.0),
            'entropy_shift_flag': ent.get('entropy_shift_flag', False),
            'final_regime_label': expert_metrics.get('final_regime_label', regime_label),
            'regime_confidence': expert_metrics.get('regime_confidence', 1.0),
            'position_size_modifier': expert_metrics.get('position_size_modifier', 1.0),
            'risk_throttle_factor': expert_metrics.get('risk_throttle_factor', 0.0),
            'override_reason': expert_metrics.get('override_reason'),
            'target_gross_exposure': expert_metrics.get('target_gross_exposure'),
            'effective_exposure_multiplier': expert_metrics.get('effective_exposure_multiplier'),
            'throttle_mapping': expert_metrics.get('throttle_mapping'),
            'fusion_rules': expert_metrics.get('fusion_rules', []),
            'ensemble_regime_label': regime_data.get('label', 'unknown'),
            'panic_prob': regime_data.get('probs', {}).get('high_vol_panic', 0.0),
            'ensemble_disagreement': regime_data.get('disagreement', 0.0),
            'ensemble_multiplier': regime_data.get('position_size_multiplier', 1.0),
        }
        result['timeseries_url'] = 'timeseries.json'

    return result


def _can_publish_dashboard(expert_signals: Optional[Dict[str, Any]]) -> bool:
    """Return whether the current snapshot is safe to expose as live dashboard truth."""
    return expert_signals is not None


def load_recent_trades(s3: S3Client, max_days: int = 90) -> List[Dict]:
    """Load recent trades from daily trades.jsonl files."""
    dates = s3.list_daily_dates(max_days=max_days)
    all_trades = []
    for date_str in dates:
        day_trades = s3.read_jsonl(f'daily/{date_str}/trades.jsonl')
        all_trades.extend(day_trades)
    # Sort newest first
    all_trades.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
    return all_trades


def _build_equity_curve_from_daily(s3: S3Client, max_days: int = 365) -> List[Dict]:
    """Build equity curve from daily portfolio_state.json artifacts.
    Returns list of {date, value, benchmark} for frontend EquityCurvePoint.
    Only includes dates from when trading started (portfolio value changed).
    """
    dates = s3.list_daily_dates(max_days=max_days)
    if not dates:
        return []

    # Build full curve first
    full_curve = []
    for date_str in dates:
        state = s3.read_json(f'daily/{date_str}/portfolio_state.json')
        if state is not None and 'portfolio_value' in state:
            pv = state['portfolio_value']
            full_curve.append({
                'date': date_str,
                'value': pv,
                'benchmark': state.get('benchmark_value', pv),
            })

    # Find first date where portfolio value changed from initial (trading started)
    # Use tolerance for floating point comparison
    initial_value = 100000
    tolerance = 1.0  # $1 tolerance
    first_trade_idx = 0
    for i, point in enumerate(full_curve):
        if abs(point['value'] - initial_value) > tolerance:
            # Include one day before first trade for context (the starting point)
            first_trade_idx = max(0, i - 1)
            break

    return full_curve[first_trade_idx:]


def load_historical_equity(s3: S3Client) -> List[Dict]:
    """Load historical equity curve from portfolio history or build from daily artifacts."""
    try:
        history = s3.read_json('portfolio/equity_history.json')
        if history and isinstance(history, list):
            return history[-365:]
    except Exception:
        pass
    return _build_equity_curve_from_daily(s3, max_days=365)


def load_historical_drawdowns(s3: S3Client) -> List[Dict]:
    """Load historical drawdowns from portfolio history or build from daily artifacts."""
    try:
        history = s3.read_json('portfolio/drawdown_history.json')
        if history and isinstance(history, list):
            return history[-365:]
    except Exception:
        pass
    curve = _build_equity_curve_from_daily(s3, max_days=365)
    if not curve:
        return []
    sorted_curve = sorted(curve, key=lambda x: x['date'])
    peak = sorted_curve[0]['value']
    drawdowns = []
    for point in sorted_curve:
        if point['value'] > peak:
            peak = point['value']
        dd = (point['value'] - peak) / peak if peak > 0 else 0.0
        drawdowns.append({'date': point['date'], 'drawdown': dd})
    return drawdowns


def load_monthly_returns(s3: S3Client) -> List[Dict]:
    """Load monthly returns from portfolio history or build from daily artifacts."""
    try:
        returns = s3.read_json('portfolio/monthly_returns.json')
        if returns and isinstance(returns, list):
            return returns
    except Exception:
        pass
    curve = _build_equity_curve_from_daily(s3, max_days=730)
    if not curve:
        return []
    by_month = {}
    for point in curve:
        date_str = point['date']
        ym = date_str[:7]
        if ym not in by_month:
            by_month[ym] = []
        by_month[ym].append(point['value'])
    monthly = []
    for ym in sorted(by_month.keys()):
        vals = by_month[ym]
        if len(vals) >= 2:
            ret = (vals[-1] / vals[0]) - 1.0
        else:
            ret = 0.0
        year, month = int(ym[:4]), int(ym[5:7])
        monthly.append({'year': year, 'month': month, 'return_pct': ret})
    return monthly


def _build_timeseries_row(
    run_date: str,
    expert_signals: Dict[str, Any],
    inference_output: Dict[str, Any],
    decisions: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    context_df: pd.DataFrame
) -> Dict[str, Any]:
    """Build a single row for the rolling timeseries dataset."""
    macro = expert_signals.get('macro_credit', {})
    vol = expert_signals.get('vol_uncertainty', {})
    frag = expert_signals.get('fragility', {})
    ent = expert_signals.get('entropy_shift', {})
    expert_metrics = decisions.get('expert_metrics', {})
    regime_data = inference_output.get('regime', {})
    ctx = context_df.iloc[0] if len(context_df) > 0 else {}

    # F-6: propagate signal-block status flags into the timeseries row so a
    # neutral fallback value (e.g. fragility_score=0.5 from _neutral_result)
    # can be distinguished from a real computation. Without these markers,
    # 168 days of fallback-VIX (F-1) looked byte-identical to live signal.
    def _signal_status(d: Dict[str, Any]) -> str:
        if d.get('degraded_reason'):
            return f"degraded:{d['degraded_reason'][:60]}"
        inputs_degraded = d.get('inputs_degraded')
        if inputs_degraded:
            return 'partial:' + ','.join(inputs_degraded)
        return 'ok'

    return {
        'date': run_date,
        'final_regime_label': expert_metrics.get('final_regime_label',
                                                  regime_data.get('label', 'unknown')),
        'regime_confidence': expert_metrics.get('regime_confidence',
                                                 regime_data.get('confidence', 1.0)),
        'trend_risk_on_prob': regime_data.get('probs', {}).get('risk_on_trend', 0.0),
        'panic_prob': regime_data.get('probs', {}).get('high_vol_panic', 0.0),
        'macro_credit_score': macro.get('macro_credit_score', 0.0),
        'macro_credit_status': _signal_status(macro),
        'yield_slope_10y_3m': macro.get('yield_slope_10y_3m', 0.0),
        'hy_spread_proxy': macro.get('hy_spread_proxy', 0.0),
        'vol_uncertainty_score': vol.get('vol_uncertainty_score', 0.5),
        'vol_uncertainty_status': _signal_status(vol),
        'vol_regime_label': vol.get('vol_regime_label', 'calm'),
        'vix_percentile': vol.get('vix_percentile', 0.5),
        'vvix_percentile': vol.get('vvix_percentile', 0.5),
        'skew_value': vol.get('skew_value', 0.0),
        'fragility_score': frag.get('fragility_score', 0.5),
        'fragility_status': _signal_status(frag),
        'avg_correlation': frag.get('avg_correlation', 0.0),
        'pc1_explained': frag.get('pc1_explained', 0.0),
        'entropy_score': ent.get('entropy_score', 0.5),
        'entropy_status': _signal_status(ent),
        'entropy_z_score': ent.get('entropy_z_score', 0.0),
        'entropy_shift_flag': ent.get('entropy_shift_flag', False),
        'entropy_consecutive_days': ent.get('entropy_consecutive_days', 0),
        'entropy_above_threshold': ent.get('entropy_above_threshold', False),
        'position_size_modifier': expert_metrics.get('position_size_modifier', 1.0),
        'risk_throttle_factor': expert_metrics.get('risk_throttle_factor', 0.0),
        'spy_close': float(ctx.get('spy_return_1d', 0)) if hasattr(ctx, 'get') else 0.0,
        'portfolio_value': portfolio_state.get('portfolio_value', 100000),
    }


def run(
    bucket: str,
    run_date: str,
    prices_df: pd.DataFrame,
    context_df: pd.DataFrame,
    features_df: pd.DataFrame,
    inference_output: Dict[str, Any],
    llm_risks: Dict[str, Dict],
    decisions: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    trades: List[Dict],
    weather: Dict[str, Any],
    validation: Dict[str, Any],
    expert_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Publish all pipeline artifacts to S3.

    Artifacts are stored in:
    - daily/{date}/  - Day-specific artifacts
    - daily/latest.json - Pointer to current day

    Args:
        bucket: S3 bucket name
        run_date: Date string (YYYY-MM-DD)
        prices_df: Price data
        context_df: Market context
        features_df: Asset features
        inference_output: Model inference results
        llm_risks: LLM risk assessments
        decisions: Decision engine output
        portfolio_state: Current portfolio state
        trades: Trade records
        weather: Weather blurb
        validation: Data validation results
        expert_signals: Expert signal outputs (None = skip signal artifacts)

    Returns:
        Dict with publish status
    """
    print("Publishing artifacts to S3...")

    s3 = S3Client(bucket)
    base_path = f"daily/{run_date}"
    snapshot_meta = _build_snapshot_meta(run_date, 'night', portfolio_state)
    portfolio_state = dict(portfolio_state)
    portfolio_state['date'] = run_date

    published = []
    failed = []

    # 1. Prices parquet
    try:
        if len(prices_df) > 0:
            s3.write_parquet(prices_df, f"{base_path}/prices.parquet")
            published.append("prices.parquet")
    except Exception as e:
        print(f"Failed to publish prices.parquet: {e}")
        failed.append("prices.parquet")

    # 2. Context parquet
    try:
        if len(context_df) > 0:
            s3.write_parquet(context_df, f"{base_path}/context.parquet")
            published.append("context.parquet")
    except Exception as e:
        print(f"Failed to publish context.parquet: {e}")
        failed.append("context.parquet")

    # 3. Features parquet
    try:
        if len(features_df) > 0:
            # Only keep latest date per symbol for artifact
            latest_date = features_df['date'].max()
            latest_features = features_df[features_df['date'] == latest_date]
            s3.write_parquet(latest_features, f"{base_path}/features.parquet")
            published.append("features.parquet")
    except Exception as e:
        print(f"Failed to publish features.parquet: {e}")
        failed.append("features.parquet")

    # 4. Inference JSON
    try:
        s3.write_json(inference_output, f"{base_path}/inference.json")
        published.append("inference.json")
    except Exception as e:
        print(f"Failed to publish inference.json: {e}")
        failed.append("inference.json")

    # 5. LLM Risk JSON
    try:
        llm_risk_output = {
            'date': run_date,
            'status': 'ok',
            'calls_made': len(llm_risks),
            'risks': llm_risks
        }
        s3.write_json(llm_risk_output, f"{base_path}/llm_risk.json")
        published.append("llm_risk.json")
    except Exception as e:
        print(f"Failed to publish llm_risk.json: {e}")
        failed.append("llm_risk.json")

    # 6. Decisions JSON
    try:
        s3.write_json(decisions, f"{base_path}/decisions.json")
        published.append("decisions.json")
    except Exception as e:
        print(f"Failed to publish decisions.json: {e}")
        failed.append("decisions.json")

    # 7. Portfolio State JSON
    try:
        s3.write_json(portfolio_state, f"{base_path}/portfolio_state.json")
        published.append("portfolio_state.json")
    except Exception as e:
        print(f"Failed to publish portfolio_state.json: {e}")
        failed.append("portfolio_state.json")

    # 8. Trades JSONL (append to historical)
    try:
        for trade in trades:
            s3.append_jsonl(trade, f"{base_path}/trades.jsonl")
        if trades:
            published.append("trades.jsonl")
    except Exception as e:
        print(f"Failed to publish trades.jsonl: {e}")
        failed.append("trades.jsonl")

    # 9. Weather Blurb JSON
    try:
        s3.write_json(weather, f"{base_path}/weather_blurb.json")
        published.append("weather_blurb.json")
    except Exception as e:
        print(f"Failed to publish weather_blurb.json: {e}")
        failed.append("weather_blurb.json")

    # 10. Run Report JSON
    try:
        run_report = {
            'status': 'success' if len(failed) == 0 else 'partial',
            'date': run_date,
            'timestamp': datetime.now().isoformat(),
            'validation': validation,
            'actions_count': len(decisions.get('actions', [])),
            'trades_count': len(trades),
            'artifacts_published': published,
            'artifacts_failed': failed
        }
        s3.write_json(run_report, f"{base_path}/run_report.json")
        published.append("run_report.json")
    except Exception as e:
        print(f"Failed to publish run_report.json: {e}")
        failed.append("run_report.json")

    # 11. Expert signals parquet (if available)
    if expert_signals is not None:
        try:
            signals_row = _build_timeseries_row(
                run_date, expert_signals, inference_output,
                decisions, portfolio_state, context_df
            )
            signals_df = pd.DataFrame([signals_row])
            s3.write_parquet(signals_df, f"{base_path}/signals.parquet")
            published.append("signals.parquet")
        except Exception as e:
            print(f"Failed to publish signals.parquet: {e}")
            failed.append("signals.parquet")

    # 12. Rolling timeseries (append today, trim to 400 days)
    if expert_signals is not None:
        try:
            ts_row = _build_timeseries_row(
                run_date, expert_signals, inference_output,
                decisions, portfolio_state, context_df
            )

            # Read existing timeseries
            existing_ts = s3.read_parquet('dashboard/timeseries.parquet')
            if len(existing_ts) > 0:
                # Remove any existing row for today (idempotent re-runs)
                existing_ts['date'] = existing_ts['date'].astype(str)
                existing_ts = existing_ts[existing_ts['date'] != run_date]
                ts_df = pd.concat([existing_ts, pd.DataFrame([ts_row])], ignore_index=True)
            else:
                ts_df = pd.DataFrame([ts_row])

            # Trim to last 400 rows
            ts_df = ts_df.tail(400).reset_index(drop=True)

            s3.write_parquet(ts_df, 'dashboard/timeseries.parquet')
            published.append("timeseries.parquet")

            # Also write JSON version for frontend
            # Sanitize pandas NaN → null before JSON serialization.
            # DataFrame.to_dict() converts NaN to float('nan'), which
            # Python's json module serializes as the non-standard token
            # NaN, breaking browser JSON.parse().  DataFrame.to_json()
            # correctly emits null for NaN, so round-trip through it.
            ts_json = json.loads(ts_df.to_json(orient='records'))
            s3.write_json(ts_json, 'dashboard/data/timeseries.json')
            s3.write_json(ts_json, 'dashboard/timeseries.json')
            published.append("timeseries.json")
        except Exception as e:
            print(f"Failed to publish timeseries: {e}")
            failed.append("timeseries.parquet")

    # 13. Generate dashboard.json for frontend
    dashboard_publishable = _can_publish_dashboard(expert_signals)
    try:
        dashboard_data = build_dashboard_data(
            portfolio_state, inference_output, decisions, weather, s3,
            expert_signals=expert_signals,
            snapshot_meta=snapshot_meta,
        )
        # Apply the three-line replay extension (optimized-champion canon line).
        # MUST be wired here: without a committed call site the corrected line
        # is produced only by working-tree code baked into the Lambda image,
        # and a fresh checkout + rebuild would silently drop it. extend_dashboard
        # is internally defensive (returns dashboard_data unchanged on any
        # failure), so this degrades gracefully to the raw canonical line.
        from src.utils.three_line_replay.extender import extend_dashboard
        dashboard_data = extend_dashboard(s3.s3, dashboard_data)
        # Publish guard: do not overwrite a valid dashboard with broken data.
        # When expert_signals is None the frontend shows "unknown" posture and
        # hides Today's Story.  Preserving the last known good dashboard.json
        # is strictly better than publishing a degraded snapshot.
        if not dashboard_publishable:
            print("  WARNING: Skipping dashboard.json publish — expert_signals is null. "
                  "Preserving last known good dashboard state.")
            failed.append("dashboard.json (skipped: null signals)")
        else:
            # Write to both locations for compatibility
            s3.write_json(dashboard_data, "dashboard/data/dashboard.json")
            s3.write_json(dashboard_data, "dashboard/dashboard.json")
            published.append("dashboard.json")
    except Exception as e:
        print(f"Failed to publish dashboard.json: {e}")
        failed.append("dashboard.json")

    # 14. Update latest.json pointer only when the dashboard snapshot is valid.
    try:
        if dashboard_publishable:
            fused_regime = decisions.get('expert_metrics', {}).get(
                'final_regime_label',
                inference_output.get('regime', {}).get('label', 'unknown')
            )
            latest = {
                'date': run_date,
                'intents_date': run_date,
                'timestamp': snapshot_meta['timestamp'],
                'snapshot_id': snapshot_meta['id'],
                'regime': fused_regime,
                'portfolio_value': portfolio_state.get('portfolio_value', 0),
                'positions_count': len(portfolio_state.get('holdings', [])),
                'actions_count': len(decisions.get('actions', [])),
                'phase': 'night'
            }
            s3.write_json(latest, "daily/latest.json")
            published.append("latest.json")
        else:
            print("  WARNING: Skipping latest.json update — dashboard snapshot was not publishable.")
            failed.append("latest.json (skipped: invalid dashboard snapshot)")
    except Exception as e:
        print(f"Failed to update latest.json: {e}")
        failed.append("latest.json")

    print(f"  Published: {len(published)} artifacts")
    if failed:
        print(f"  Failed: {len(failed)} artifacts - {failed}")

    _invalidate_dashboard_cache()

    return {
        'success': len(failed) == 0,
        'published': published,
        'failed': failed,
        'base_path': base_path
    }


def publish_morning_artifacts(
    bucket: str,
    run_date: str,
    portfolio_state: Dict[str, Any],
    trades: List[Dict],
    morning_execution: Dict[str, Any],
    night_inference: Dict[str, Any],
    night_decisions: Dict[str, Any],
    night_weather: Dict[str, Any],
    expert_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Publish morning execution artifacts (lightweight subset).

    Only publishes portfolio state, trades, execution report, dashboard.json,
    and latest.json. All analysis artifacts (prices, features, signals, etc.)
    were already published by the night run.
    """
    print("Publishing morning artifacts to S3...")

    s3 = S3Client(bucket)
    base_path = f"daily/{run_date}"
    snapshot_meta = _build_snapshot_meta(run_date, 'morning', portfolio_state)
    portfolio_state = dict(portfolio_state)
    portfolio_state['date'] = run_date
    published = []
    failed = []

    # 1. Portfolio state (overwrite night's valuation-only snapshot)
    try:
        s3.write_json(portfolio_state, f"{base_path}/portfolio_state.json")
        published.append("portfolio_state.json")
    except Exception as e:
        print(f"Failed to publish portfolio_state.json: {e}")
        failed.append("portfolio_state.json")

    # 2. Trades
    try:
        for trade in trades:
            s3.append_jsonl(trade, f"{base_path}/trades.jsonl")
        if trades:
            published.append("trades.jsonl")
    except Exception as e:
        print(f"Failed to publish trades.jsonl: {e}")
        failed.append("trades.jsonl")

    # 3. Morning execution report
    try:
        s3.write_json(morning_execution, f"{base_path}/morning_execution.json")
        published.append("morning_execution.json")
    except Exception as e:
        print(f"Failed to publish morning_execution.json: {e}")
        failed.append("morning_execution.json")

    # 4. Update latest.json only if the dashboard snapshot is publishable.
    try:
        if _can_publish_dashboard(expert_signals):
            latest = s3.read_json('daily/latest.json') or {}
            latest.update({
                'date': run_date,
                'portfolio_value': portfolio_state.get('portfolio_value', 0),
                'positions_count': len(portfolio_state.get('holdings', [])),
                'morning_executed': True,
                'trades_count': len(trades),
                'phase': 'morning',
                'timestamp': snapshot_meta['timestamp'],
                'snapshot_id': snapshot_meta['id'],
            })
            s3.write_json(latest, 'daily/latest.json')
            published.append("latest.json")
        else:
            print("  WARNING: Skipping morning latest.json update — dashboard snapshot was not publishable.")
            failed.append("latest.json (skipped: invalid dashboard snapshot)")
    except Exception as e:
        print(f"Failed to update latest.json: {e}")
        failed.append("latest.json")

    # 5. Rebuild and publish dashboard.json with post-trade portfolio
    try:
        dashboard_publishable = _can_publish_dashboard(expert_signals)
        dashboard_data = build_dashboard_data(
            portfolio_state, night_inference, night_decisions, night_weather, s3,
            expert_signals=expert_signals,
            snapshot_meta=snapshot_meta,
        )
        # Apply the three-line replay extension (optimized-champion canon line).
        # See the night-path note above: committed call site required for
        # durability; extend_dashboard degrades gracefully on failure.
        from src.utils.three_line_replay.extender import extend_dashboard
        dashboard_data = extend_dashboard(s3.s3, dashboard_data)
        # Publish guard: do not overwrite a valid dashboard with broken data.
        if not dashboard_publishable:
            print("  WARNING: Skipping morning dashboard.json publish — expert_signals is null. "
                  "Preserving last known good dashboard state.")
            failed.append("dashboard.json (skipped: null signals)")
        else:
            s3.write_json(dashboard_data, "dashboard/data/dashboard.json")
            s3.write_json(dashboard_data, "dashboard/dashboard.json")
            published.append("dashboard.json")
    except Exception as e:
        print(f"Failed to publish dashboard.json: {e}")
        failed.append("dashboard.json")

    print(f"  Published: {len(published)} morning artifacts")
    if failed:
        print(f"  Failed: {len(failed)} artifacts - {failed}")

    _invalidate_dashboard_cache()

    return {
        'success': len(failed) == 0,
        'published': published,
        'failed': failed,
        'base_path': base_path
    }
