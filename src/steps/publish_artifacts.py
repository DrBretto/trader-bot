"""Publish artifacts to S3 for dashboard consumption."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src.utils.s3_client import S3Client


def build_dashboard_data(
    portfolio_state: Dict[str, Any],
    inference_output: Dict[str, Any],
    decisions: Dict[str, Any],
    weather: Dict[str, Any],
    s3: S3Client,
    expert_signals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build the dashboard.json data structure for the frontend."""
    timestamp = datetime.now().isoformat()

    # Build metrics
    metrics = {
        'total_value': portfolio_state.get('portfolio_value', 100000),
        'cash': portfolio_state.get('cash', 100000),
        'invested': portfolio_state.get('invested', 0),
        'ytd_return': portfolio_state.get('ytd_return', 0),
        'mtd_return': portfolio_state.get('mtd_return', 0),
        'sharpe_ratio': portfolio_state.get('sharpe_ratio', 0),
        'max_drawdown': portfolio_state.get('max_drawdown', 0),
        'current_drawdown': portfolio_state.get('current_drawdown', 0),
        'win_rate': portfolio_state.get('win_rate', 0),
        'total_trades': portfolio_state.get('total_trades', 0),
        'timestamp': timestamp
    }

    # Build holdings
    holdings = []
    for h in portfolio_state.get('holdings', []):
        holdings.append({
            'symbol': h.get('symbol', ''),
            'shares': h.get('shares', 0),
            'entry_price': h.get('entry_price', 0),
            'current_price': h.get('current_price', 0),
            'market_value': h.get('market_value', 0),
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

    # Load historical equity curve if available
    equity_curve = load_historical_equity(s3)
    drawdowns = load_historical_drawdowns(s3)
    monthly_returns = load_monthly_returns(s3)

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
        'timestamp': timestamp
    }

    result = {
        'metrics': metrics,
        'holdings': holdings,
        'candidates': candidates,
        'equity_curve': equity_curve,
        'drawdowns': drawdowns,
        'monthly_returns': monthly_returns,
        'weather': weather_report
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
        }
        result['timeseries_url'] = 'timeseries.json'

    return result


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

    return {
        'date': run_date,
        'final_regime_label': expert_metrics.get('final_regime_label',
                                                  regime_data.get('label', 'unknown')),
        'regime_confidence': expert_metrics.get('regime_confidence',
                                                 regime_data.get('confidence', 1.0)),
        'trend_risk_on_prob': regime_data.get('probs', {}).get('risk_on_trend', 0.0),
        'panic_prob': regime_data.get('probs', {}).get('high_vol_panic', 0.0),
        'macro_credit_score': macro.get('macro_credit_score', 0.0),
        'yield_slope_10y_3m': macro.get('yield_slope_10y_3m', 0.0),
        'hy_spread_proxy': macro.get('hy_spread_proxy', 0.0),
        'vol_uncertainty_score': vol.get('vol_uncertainty_score', 0.5),
        'vol_regime_label': vol.get('vol_regime_label', 'calm'),
        'vix_percentile': vol.get('vix_percentile', 0.5),
        'vvix_percentile': vol.get('vvix_percentile', 0.5),
        'skew_value': vol.get('skew_value', 0.0),
        'fragility_score': frag.get('fragility_score', 0.5),
        'avg_correlation': frag.get('avg_correlation', 0.0),
        'pc1_explained': frag.get('pc1_explained', 0.0),
        'entropy_score': ent.get('entropy_score', 0.5),
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
    expert_signals: Optional[Dict[str, Any]] = None
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

    # 11. Update latest.json pointer
    try:
        fused_regime = decisions.get('expert_metrics', {}).get(
            'final_regime_label',
            inference_output.get('regime', {}).get('label', 'unknown')
        )
        latest = {
            'date': run_date,
            'timestamp': datetime.now().isoformat(),
            'regime': fused_regime,
            'portfolio_value': portfolio_state.get('portfolio_value', 0),
            'positions_count': len(portfolio_state.get('holdings', [])),
            'actions_count': len(decisions.get('actions', []))
        }
        s3.write_json(latest, "daily/latest.json")
        published.append("latest.json")
    except Exception as e:
        print(f"Failed to update latest.json: {e}")
        failed.append("latest.json")

    # 12. Expert signals parquet (if available)
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

    # 13. Rolling timeseries (append today, trim to 400 days)
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
            ts_json = ts_df.to_dict(orient='records')
            s3.write_json(ts_json, 'dashboard/data/timeseries.json')
            s3.write_json(ts_json, 'dashboard/timeseries.json')
            published.append("timeseries.json")
        except Exception as e:
            print(f"Failed to publish timeseries: {e}")
            failed.append("timeseries.parquet")

    # 14. Generate dashboard.json for frontend
    try:
        dashboard_data = build_dashboard_data(
            portfolio_state, inference_output, decisions, weather, s3,
            expert_signals=expert_signals
        )
        # Write to both locations for compatibility
        s3.write_json(dashboard_data, "dashboard/data/dashboard.json")
        s3.write_json(dashboard_data, "dashboard/dashboard.json")
        published.append("dashboard.json")
    except Exception as e:
        print(f"Failed to publish dashboard.json: {e}")
        failed.append("dashboard.json")

    print(f"  Published: {len(published)} artifacts")
    if failed:
        print(f"  Failed: {len(failed)} artifacts - {failed}")

    return {
        'success': len(failed) == 0,
        'published': published,
        'failed': failed,
        'base_path': base_path
    }
