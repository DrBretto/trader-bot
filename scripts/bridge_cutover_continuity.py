#!/usr/bin/env python3
"""Bridge cutover continuity by patching external_cashflow on cutover-day state.

Usage:
    # Dry-run (default)
    python scripts/bridge_cutover_continuity.py --cutover-date 2026-03-12

    # Apply
    python scripts/bridge_cutover_continuity.py --cutover-date 2026-03-12 --apply
"""

import argparse
import sys
from datetime import datetime

# Allow running from repo root
sys.path.insert(0, ".")

from src.utils.s3_client import S3Client
from src.utils.cutover_bridge import build_cutover_patch, apply_patch, describe_patch


def find_previous_trading_date(s3: S3Client, cutover_date: str) -> str:
    """Find the most recent date before cutover_date that has a portfolio_state."""
    dates = s3.list_daily_dates(max_days=30)
    candidates = [d for d in sorted(dates, reverse=True) if d < cutover_date]
    for d in candidates:
        state = s3.read_json(f"daily/{d}/portfolio_state.json")
        if state and state.get("portfolio_value"):
            return d
    raise ValueError(
        f"No pre-cutover portfolio_state found before {cutover_date}. "
        f"Available dates: {dates}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bridge cutover continuity metrics"
    )
    parser.add_argument(
        "--bucket", default="investment-system-data", help="S3 bucket"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="AWS region"
    )
    parser.add_argument(
        "--cutover-date", required=True,
        help="Cutover date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Print plan without writing (default)"
    )
    parser.add_argument(
        "--apply", action="store_true", dest="apply_mode",
        help="Apply the bridge patch to S3"
    )
    parser.add_argument(
        "--profile", default=None,
        help="AWS profile to use"
    )
    args = parser.parse_args()

    if args.profile:
        import boto3
        from botocore.exceptions import ProfileNotFound

        try:
            boto3.setup_default_session(profile_name=args.profile)
        except ProfileNotFound:
            print(
                f"ERROR: AWS profile '{args.profile}' not found. "
                "Pass a valid --profile or omit it to use environment/instance credentials."
            )
            sys.exit(2)

    s3 = S3Client(bucket=args.bucket, region=args.region)
    cutover_date = args.cutover_date
    today = datetime.now().strftime("%Y-%m-%d")

    # Load states
    print(f"Loading cutover-day state: daily/{cutover_date}/portfolio_state.json")
    cutover_state = s3.read_json(f"daily/{cutover_date}/portfolio_state.json")
    if not cutover_state:
        print(f"ERROR: No portfolio_state.json found for {cutover_date}")
        sys.exit(1)

    previous_date = find_previous_trading_date(s3, cutover_date)
    print(f"Loading previous-day state: daily/{previous_date}/portfolio_state.json")
    previous_state = s3.read_json(f"daily/{previous_date}/portfolio_state.json")
    if not previous_state:
        print(f"ERROR: No portfolio_state.json found for {previous_date}")
        sys.exit(1)

    # Compute patch
    patch = build_cutover_patch(cutover_state, previous_state, cutover_date)
    summary = describe_patch(previous_state, cutover_state, patch)

    print("\n=== Cutover Continuity Bridge ===")
    print(f"Previous date:       {summary['previous_date']}")
    print(f"Previous value:      ${summary['previous_value']:,.2f}")
    print(f"Cutover value:       ${summary['cutover_value']:,.2f}")
    print(f"Value delta:         ${summary['value_delta']:,.2f}")
    print(f"External cashflow:   ${summary['external_cashflow']:,.2f}")
    print(f"Neutralized return:  {summary['neutralized_return']:.4%}")
    print(f"Benchmark carried:   {summary['benchmark_fields_carried']}")
    print(f"Marker:              {summary['marker']}")
    print(f"Already patched:     {summary['already_patched']}")

    if summary["already_patched"]:
        print("\nState already has continuity bridge marker. No changes needed.")
        return

    if args.apply_mode:
        # Apply mode
        patched_state = apply_patch(cutover_state, patch)
        key = f"daily/{cutover_date}/portfolio_state.json"
        print(f"\nWriting patched state to s3://{args.bucket}/{key}")
        success = s3.write_json(patched_state, key)
        if not success:
            print("ERROR: Failed to write patched state")
            sys.exit(1)

        # Write result artifact
        result = {
            "action": "apply",
            "cutover_date": cutover_date,
            "previous_date": previous_date,
            "summary": summary,
            "patch_applied": patch,
            "timestamp": datetime.now().isoformat(),
        }
        result_key = f"daily/{today}/cutover_bridge_result.json"
        s3.write_json(result, result_key)
        print(f"Result artifact: s3://{args.bucket}/{result_key}")
        print("Bridge applied successfully.")
    else:
        # Dry-run mode
        plan = {
            "action": "dry_run",
            "cutover_date": cutover_date,
            "previous_date": previous_date,
            "summary": summary,
            "proposed_patch": patch,
            "timestamp": datetime.now().isoformat(),
        }
        plan_key = f"daily/{today}/cutover_bridge_plan.json"
        s3.write_json(plan, plan_key)
        print(f"\nDry-run plan: s3://{args.bucket}/{plan_key}")
        print("Re-run with --apply to write the patch.")


if __name__ == "__main__":
    main()
