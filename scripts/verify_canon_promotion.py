"""Post-deploy verification: read the live dashboard.json and check that
the optimized-canon promotion is intact and the trading pipeline is fresh.

Exits non-zero on failure so this can be wired into a deploy guard.

Usage:
    python scripts/verify_canon_promotion.py [--bucket investment-system-data] [--region us-east-1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta


def _err(label: str, msg: str) -> None:
    print(f"  FAIL  {label}: {msg}", file=sys.stderr)


def _ok(label: str, msg: str = "") -> None:
    print(f"  OK    {label}{(': ' + msg) if msg else ''}")


def verify(bucket: str, region: str) -> int:
    import boto3
    s3 = boto3.client('s3', region_name=region)

    dash_raw = s3.get_object(Bucket=bucket, Key='dashboard/dashboard.json')['Body'].read()
    dash = json.loads(dash_raw)
    latest_raw = s3.get_object(Bucket=bucket, Key='daily/latest.json')['Body'].read()
    latest = json.loads(latest_raw)

    failures = 0

    metrics = dash.get('metrics', {})
    ec = dash.get('equity_curve', [])
    tc = dash.get('timeline_correction', {})

    # --- Canon-source stamp ---
    if metrics.get('canon_source') == 'optimized_champion':
        _ok('canon_source', "metrics.canon_source == 'optimized_champion'")
    else:
        _err('canon_source', f"got {metrics.get('canon_source')!r}, want 'optimized_champion'")
        failures += 1

    if tc.get('main_line_label') == 'Portfolio (optimized champion)':
        _ok('main_line_label', tc['main_line_label'])
    else:
        _err('main_line_label', f"got {tc.get('main_line_label')!r}")
        failures += 1

    # --- value field carries optimized ---
    if not ec:
        _err('equity_curve', 'empty')
        failures += 1
    else:
        last = ec[-1]
        if last.get('optimized_value') is None:
            _err('last_optimized_value', 'missing on last row')
            failures += 1
        elif last.get('value') == last.get('optimized_value'):
            _ok('value_eq_optimized_last_row',
                f"{last['value']} == optimized_value (date {last['date']})")
        else:
            _err('value_eq_optimized_last_row',
                 f"value={last.get('value')} != optimized_value={last.get('optimized_value')} on {last['date']}")
            failures += 1

        # hybrid_value preserved
        rows_with_hybrid = sum(1 for r in ec if 'hybrid_value' in r)
        if rows_with_hybrid >= 10:
            _ok('hybrid_value_count', f"{rows_with_hybrid} rows carry hybrid_value")
        else:
            _err('hybrid_value_count', f"only {rows_with_hybrid} rows carry hybrid_value")
            failures += 1

    # --- Trading pipeline freshness ---
    intents_date = latest.get('intents_date')
    if intents_date:
        gen_dt = datetime.strptime(intents_date, '%Y-%m-%d')
        age = (datetime.now() - gen_dt).days
        # MAX_INTENT_AGE_DAYS = 3 in morning_executor; allow 4 to absorb
        # weekend gaps (Friday-night intent → Monday morning is 3 days).
        if age <= 4:
            _ok('intents_freshness', f"intents_date={intents_date}, age {age}d")
        else:
            _err('intents_freshness', f"intents_date={intents_date}, age {age}d (>{4}d)")
            failures += 1
    else:
        _err('intents_freshness', 'no intents_date in latest.json')
        failures += 1

    if failures == 0:
        print("\n  All canon + freshness checks PASS")
    else:
        print(f"\n  {failures} check(s) FAILED")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default=os.environ.get('S3_BUCKET', 'investment-system-data'))
    parser.add_argument('--region', default=os.environ.get('AWS_REGION', 'us-east-1'))
    args = parser.parse_args()
    return verify(args.bucket, args.region)


if __name__ == '__main__':
    sys.exit(main())
