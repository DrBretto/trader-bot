"""Backfill expert signals for historical dates.

Reads daily/{date}/ artifacts from S3, runs all 4 expert signal modules
and regime fusion, then writes the full timeseries to dashboard/.

Usage:
    .venv/bin/python training/scripts/backfill_signals.py \
        --bucket investment-system-data --region us-east-1
"""

import argparse
import json
import sys
import os
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.s3_client import S3Client
from src.signals.macro_credit import compute_macro_credit
from src.signals.vol_uncertainty import compute_vol_uncertainty
from src.signals.fragility import compute_fragility
from src.signals.entropy_shift import compute_entropy_shift
from src.signals.regime_fusion import decide_regime_v3
from src.steps.ingest_fred import get_latest_values


def backfill_date(
    date_str: str,
    s3: S3Client,
    prev_entropy_state: dict,
) -> dict | None:
    """Compute expert signals for a single historical date.

    Returns a timeseries row dict, or None if critical data is missing.
    """
    base = f"daily/{date_str}"

    # Load prices
    prices_df = s3.read_parquet(f"{base}/prices.parquet")
    if prices_df is None or len(prices_df) == 0:
        print(f"  {date_str}: no prices, skipping")
        return None

    # Load FRED data
    fred_df = s3.read_parquet(f"{base}/fred.parquet")
    if fred_df is None:
        fred_df = pd.DataFrame(columns=['date', 'series_id', 'value'])

    # Load inference for ensemble outputs
    inference = s3.read_json(f"{base}/inference.json")
    if inference is None:
        print(f"  {date_str}: no inference, skipping")
        return None

    # Load portfolio state for portfolio_value
    portfolio_state = s3.read_json(f"{base}/portfolio_state.json")
    portfolio_value = portfolio_state.get('portfolio_value', 100000) if portfolio_state else 100000

    # --- Compute expert signals ---

    # 1. Macro/Credit
    try:
        fred_latest = get_latest_values(fred_df) if len(fred_df) > 0 else {}
        hyg_data = prices_df[prices_df['symbol'] == 'HYG'].sort_values('date')
        ief_data = prices_df[prices_df['symbol'] == 'IEF'].sort_values('date')
        hyg_closes = hyg_data.set_index('date')['close'] if len(hyg_data) > 0 else None
        ief_closes = ief_data.set_index('date')['close'] if len(ief_data) > 0 else None

        macro = compute_macro_credit(
            rate_10y=fred_latest.get('DGS10', 0),
            rate_3m=fred_latest.get('DGS3MO', 0),
            hyg_prices=hyg_closes,
            ief_prices=ief_closes,
            rate_2y=fred_latest.get('DGS2', 0),
        )
    except Exception:
        macro = {'macro_credit_score': 0.0, 'yield_slope_10y_3m': 0.0, 'hy_spread_proxy': 0.0}

    # 2. Vol Uncertainty (no VVIX/SKEW historically)
    try:
        vix_data = fred_df[fred_df['series_id'] == 'VIXCLS'].sort_values('date') if len(fred_df) > 0 else pd.DataFrame()
        vix_value = float(vix_data['value'].iloc[-1]) if len(vix_data) > 0 else 0
        vix_history = vix_data['value'] if len(vix_data) >= 60 else None

        vol = compute_vol_uncertainty(
            vix=vix_value,
            vvix=None,
            skew=None,
            vix_history=vix_history,
            vvix_history=None,
        )
    except Exception:
        vol = {'vol_uncertainty_score': 0.5, 'vol_regime_label': 'calm',
               'vix_percentile': 0.5, 'vvix_percentile': 0.5, 'skew_value': 0.0}

    # 3. Fragility
    try:
        frag = compute_fragility(prices_df)
    except Exception:
        frag = {'fragility_score': 0.5, 'avg_correlation': 0.0, 'pc1_explained': 0.0}

    # 4. Entropy Shift
    try:
        spy_data = prices_df[prices_df['symbol'] == 'SPY'].sort_values('date')
        spy_closes = spy_data.set_index('date')['close'] if len(spy_data) > 0 else pd.Series(dtype=float)
        spy_returns = spy_closes.pct_change().dropna() if len(spy_closes) > 1 else pd.Series(dtype=float)

        ent = compute_entropy_shift(
            spy_returns=spy_returns,
            prev_consecutive_days=prev_entropy_state.get('prev_consecutive_days', 0),
            prev_above_threshold=prev_entropy_state.get('prev_above_threshold', False),
        )
    except Exception:
        ent = {'entropy_score': 0.5, 'entropy_z_score': 0.0, 'entropy_shift_flag': False,
               'entropy_consecutive_days': 0, 'entropy_above_threshold': False}

    # --- Regime Fusion ---
    regime_data = inference.get('regime', {})
    probs = regime_data.get('probs', {})

    try:
        fusion = decide_regime_v3(
            ensemble_regime_label=regime_data.get('label', 'choppy'),
            trend_risk_on_prob=probs.get('risk_on_trend', 0.0),
            panic_prob=probs.get('high_vol_panic', 0.0),
            ensemble_disagreement=regime_data.get('disagreement', 0.0),
            ensemble_multiplier=regime_data.get('position_size_multiplier', 1.0),
            macro_credit_score=macro.get('macro_credit_score', 0.0),
            vol_uncertainty_score=vol.get('vol_uncertainty_score', 0.5),
            vol_regime_label=vol.get('vol_regime_label', 'calm'),
            fragility_score=frag.get('fragility_score', 0.5),
            entropy_score=ent.get('entropy_score', 0.5),
            entropy_shift_flag=ent.get('entropy_shift_flag', False),
        )
    except Exception:
        fusion = {
            'final_regime_label': regime_data.get('label', 'choppy'),
            'regime_confidence': regime_data.get('confidence', 1.0),
            'position_size_modifier': 1.0,
            'risk_throttle_factor': 0.0,
            'override_reason': None,
        }

    # Get SPY close for the day
    spy_today = prices_df[prices_df['symbol'] == 'SPY'].sort_values('date')
    spy_close = float(spy_today['close'].iloc[-1]) if len(spy_today) > 0 else 0.0

    return {
        'date': date_str,
        'final_regime_label': fusion.get('final_regime_label', 'choppy'),
        'regime_confidence': fusion.get('regime_confidence', 1.0),
        'trend_risk_on_prob': probs.get('risk_on_trend', 0.0),
        'panic_prob': probs.get('high_vol_panic', 0.0),
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
        'position_size_modifier': fusion.get('position_size_modifier', 1.0),
        'risk_throttle_factor': fusion.get('risk_throttle_factor', 0.0),
        'override_reason': fusion.get('override_reason'),
        'spy_close': spy_close,
        'portfolio_value': portfolio_value,
    }


def main():
    parser = argparse.ArgumentParser(description='Backfill expert signals from historical data')
    parser.add_argument('--bucket', default='investment-system-data')
    parser.add_argument('--region', default='us-east-1')
    args = parser.parse_args()

    os.environ.setdefault('AWS_DEFAULT_REGION', args.region)
    s3 = S3Client(args.bucket)

    # List all daily dates
    dates = s3.list_daily_dates(max_days=500)
    dates = sorted(dates)
    print(f"Found {len(dates)} dates to backfill")

    # Process sequentially (entropy needs previous day's state)
    rows = []
    prev_entropy_state = {'prev_consecutive_days': 0, 'prev_above_threshold': False}

    for i, date_str in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] Processing {date_str}...")
        row = backfill_date(date_str, s3, prev_entropy_state)
        if row is not None:
            rows.append(row)
            # Update entropy state for next day
            prev_entropy_state = {
                'prev_consecutive_days': row.get('entropy_consecutive_days', 0),
                'prev_above_threshold': row.get('entropy_above_threshold', False),
            }

    if not rows:
        print("No rows produced, exiting")
        return

    print(f"\nBuilt {len(rows)} timeseries rows")

    # Write to S3
    ts_df = pd.DataFrame(rows)
    ts_df = ts_df.tail(400).reset_index(drop=True)

    s3.write_parquet(ts_df, 'dashboard/timeseries.parquet')
    print("Wrote dashboard/timeseries.parquet")

    ts_json = ts_df.to_dict(orient='records')
    s3.write_json(ts_json, 'dashboard/data/timeseries.json')
    s3.write_json(ts_json, 'dashboard/timeseries.json')
    print("Wrote dashboard/timeseries.json")

    print(f"\nDone! {len(rows)} days backfilled.")
    print(f"Date range: {rows[0]['date']} to {rows[-1]['date']}")


if __name__ == '__main__':
    main()
