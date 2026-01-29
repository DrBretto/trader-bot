"""Publish artifacts to S3 for dashboard consumption."""

import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

from src.utils.s3_client import S3Client


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
    validation: Dict[str, Any]
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
        latest = {
            'date': run_date,
            'timestamp': datetime.now().isoformat(),
            'regime': inference_output.get('regime', {}).get('label', 'unknown'),
            'portfolio_value': portfolio_state.get('portfolio_value', 0),
            'positions_count': len(portfolio_state.get('holdings', [])),
            'actions_count': len(decisions.get('actions', []))
        }
        s3.write_json(latest, "daily/latest.json")
        published.append("latest.json")
    except Exception as e:
        print(f"Failed to update latest.json: {e}")
        failed.append("latest.json")

    print(f"  Published: {len(published)} artifacts")
    if failed:
        print(f"  Failed: {len(failed)} artifacts - {failed}")

    return {
        'success': len(failed) == 0,
        'published': published,
        'failed': failed,
        'base_path': base_path
    }
