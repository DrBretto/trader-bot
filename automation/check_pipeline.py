#!/usr/bin/env python3
"""
Pipeline health check script.

Checks if the daily pipeline ran successfully and alerts on failures.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import subprocess

try:
    import boto3
except ImportError:
    print("boto3 not installed. Install with: pip install boto3")
    sys.exit(1)


def get_latest_run(bucket: str, region: str = 'us-east-1') -> Optional[Dict]:
    """
    Get the latest pipeline run report.

    Args:
        bucket: S3 bucket name
        region: AWS region

    Returns:
        Run report dictionary or None
    """
    s3 = boto3.client('s3', region_name=region)

    try:
        response = s3.get_object(Bucket=bucket, Key='daily/latest.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error reading latest.json: {e}")
        return None


def get_run_report(bucket: str, date: str, region: str = 'us-east-1') -> Optional[Dict]:
    """
    Get run report for a specific date.

    Args:
        bucket: S3 bucket name
        date: Date string (YYYY-MM-DD)
        region: AWS region

    Returns:
        Run report dictionary or None
    """
    s3 = boto3.client('s3', region_name=region)

    try:
        response = s3.get_object(Bucket=bucket, Key=f'daily/{date}/run_report.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return None


def check_lambda_errors(function_name: str, hours: int = 24, region: str = 'us-east-1') -> Dict:
    """
    Check CloudWatch for Lambda errors.

    Args:
        function_name: Lambda function name
        hours: Hours to look back
        region: AWS region

    Returns:
        Dictionary with error info
    """
    logs = boto3.client('logs', region_name=region)
    log_group = f'/aws/lambda/{function_name}'

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

    try:
        response = logs.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            filterPattern='ERROR'
        )

        errors = []
        for event in response.get('events', []):
            errors.append({
                'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
                'message': event['message'][:200]
            })

        return {
            'error_count': len(errors),
            'errors': errors[:10]  # Limit to 10
        }

    except Exception as e:
        return {'error_count': 0, 'errors': [], 'check_error': str(e)}


def send_alert(title: str, message: str):
    """Send alert using the alert script (resolved relative to this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'send_alert.sh')
    try:
        subprocess.run([script_path, title, message], check=True)
    except Exception as e:
        print(f"Failed to send alert: {e}")


def main():
    parser = argparse.ArgumentParser(description='Check pipeline health')
    parser.add_argument('--bucket', type=str, default='investment-system-data')
    parser.add_argument('--function', type=str, default='investment-system-daily-pipeline')
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--max-age-hours', type=int, default=36,
                        help='Alert if last run is older than this')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    print(f"Checking pipeline health...")

    issues = []

    # Check latest run
    latest = get_latest_run(args.bucket, args.region)

    if latest is None:
        issues.append("Could not read latest.json")
    else:
        last_date = latest.get('date', '')
        last_timestamp = latest.get('timestamp', '')

        # Check if run is recent
        if last_timestamp:
            try:
                run_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                age_hours = (datetime.now(run_time.tzinfo) - run_time).total_seconds() / 3600
                if age_hours > args.max_age_hours:
                    issues.append(f"Last run was {age_hours:.1f} hours ago (max: {args.max_age_hours})")
            except:
                pass

        # Check run report
        if last_date:
            report = get_run_report(args.bucket, last_date, args.region)
            if report:
                if report.get('status') != 'success':
                    issues.append(f"Last run status: {report.get('status')}")
                if report.get('artifacts_failed'):
                    issues.append(f"Failed artifacts: {report.get('artifacts_failed')}")

    # Check Lambda errors
    lambda_errors = check_lambda_errors(args.function, 24, args.region)
    if lambda_errors['error_count'] > 0:
        issues.append(f"Lambda errors in last 24h: {lambda_errors['error_count']}")

    # Determine status
    status = 'healthy' if len(issues) == 0 else 'unhealthy'

    result = {
        'status': status,
        'issues': issues,
        'latest_run': latest,
        'lambda_errors': lambda_errors,
        'checked_at': datetime.now().isoformat()
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nPipeline Status: {status.upper()}")
        if latest:
            print(f"Last run: {latest.get('date')} at {latest.get('timestamp', 'unknown')}")
            print(f"Portfolio value: ${latest.get('portfolio_value', 0):,.2f}")
            print(f"Regime: {latest.get('regime', 'unknown')}")

        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"  - {issue}")

        if lambda_errors['error_count'] > 0:
            print(f"\nRecent Lambda errors: {lambda_errors['error_count']}")

    # Send alert if unhealthy
    if status == 'unhealthy':
        send_alert(
            "Pipeline Health Check Failed",
            f"Issues found:\n" + "\n".join(f"- {i}" for i in issues)
        )

    return 0 if status == 'healthy' else 1


if __name__ == '__main__':
    sys.exit(main())
