#!/usr/bin/env python3
"""One-shot: rebuild + republish dashboard.json from current S3 state, mirroring
publish_artifacts.publish_morning_artifacts step 5 exactly (build_dashboard_data
-> extend_dashboard -> attach_new_brain_surface -> sanitize -> publish).

Use after a manual portfolio_state repair to push the corrected equity-curve /
benchmark line immediately, without waiting for or triggering a trading run. The
nightly Lambda produces the same output on its next scheduled run.

Backs up both live dashboard keys before overwriting (S3 versioning Suspended).

Env: AWS_PROFILE=personal. Run from repo root with the project venv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.s3_client import S3Client
from src.steps import paper_trader, publish_artifacts
from src.steps.publish_artifacts import build_dashboard_data, _build_snapshot_meta, _can_publish_dashboard, _verify_extension_or_alarm
from src.utils.three_line_replay.extender import extend_dashboard
from src.utils.dashboard_metrics import attach_new_brain_surface, sanitize_nan_for_json

BUCKET = "investment-system-data"
KEYS = ["dashboard/data/dashboard.json", "dashboard/dashboard.json"]


def main(run_date: str, stamp: str) -> int:
    s3 = S3Client(BUCKET)

    portfolio_state = paper_trader.load_portfolio_state(s3)
    print(f"loaded book: pv={portfolio_state.get('portfolio_value')} "
          f"holdings={len(portfolio_state.get('holdings', []))} "
          f"benchmark={portfolio_state.get('benchmark_value')}")

    latest = s3.read_json('daily/latest.json') or {}
    night_date = latest.get('intents_date', latest.get('date', run_date))
    night_inference = s3.read_json(f'daily/{night_date}/inference.json') or {}
    night_decisions = s3.read_json(f'daily/{night_date}/decisions.json') or {}
    night_weather = s3.read_json(f'daily/{night_date}/weather_blurb.json') or {}

    # Reconstruct expert_signals from stored signals parquet (handler logic).
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
        print(f"WARN could not reconstruct expert_signals: {e}")

    if not _can_publish_dashboard(expert_signals):
        sys.exit("ABORT: expert_signals null/incomplete — would skip publish")

    snapshot_meta = _build_snapshot_meta(run_date, 'morning', portfolio_state)
    dash = build_dashboard_data(
        portfolio_state, night_inference, night_decisions, night_weather, s3,
        expert_signals=expert_signals, snapshot_meta=snapshot_meta,
    )
    dash = extend_dashboard(s3.s3, dash)
    shadow = None
    try:
        shadow = s3.read_json('dashboard/shadow_timeseries.json')
    except Exception:
        pass
    dash = attach_new_brain_surface(dash, shadow)
    dash = sanitize_nan_for_json(dash)

    ok, reason = _verify_extension_or_alarm(dash, s3, 'morning', run_date)
    if not ok:
        sys.exit(f"ABORT: advance guard failed — {reason}")

    ec = dash.get('equity_curve', [])
    print("rebuilt curve tail:")
    for p in ec[-3:]:
        print("  ", p.get('date'), "value=", round(p.get('value', 0), 1),
              "benchmark=", round(p.get('benchmark', 0), 1))

    # Back up then publish.
    for k in KEYS:
        cur = s3.read_json(k)
        if cur is not None:
            s3.write_json(cur, f"dashboard/backups/{k.replace('/', '_')}.{stamp}.json")
    for k in KEYS:
        s3.write_json(dash, k)
        print("wrote", k)
    return 0


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "2026-06-18"
    st = sys.argv[2] if len(sys.argv) > 2 else "manual"
    raise SystemExit(main(rd, st))
