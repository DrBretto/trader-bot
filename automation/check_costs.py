#!/usr/bin/env python3
"""
AWS Cost monitoring script.

Checks current month's AWS costs and alerts if exceeding budget.
"""

import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import subprocess
import sys

try:
    import boto3
except ImportError:
    print("boto3 not installed. Install with: pip install boto3")
    sys.exit(1)


def get_current_month_costs(region: str = 'us-east-1') -> Dict:
    """
    Get current month's AWS costs using Cost Explorer.

    Args:
        region: AWS region

    Returns:
        Dictionary with cost breakdown
    """
    ce = boto3.client('ce', region_name=region)

    # Get first day of current month
    today = datetime.now()
    start_date = today.replace(day=1).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    try:
        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        )

        costs = {}
        total = 0.0

        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                costs[service] = amount
                total += amount

        return {
            'total': total,
            'by_service': costs,
            'period': {
                'start': start_date,
                'end': end_date
            }
        }

    except Exception as e:
        print(f"Error getting costs: {e}")
        return {'total': 0, 'by_service': {}, 'error': str(e)}


def get_forecasted_month_end_cost(region: str = 'us-east-1') -> Optional[float]:
    """
    Get forecasted end-of-month cost.

    Args:
        region: AWS region

    Returns:
        Forecasted cost or None if unavailable
    """
    ce = boto3.client('ce', region_name=region)

    today = datetime.now()
    start_date = today.strftime('%Y-%m-%d')

    # Get last day of month
    if today.month == 12:
        end_date = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end_date = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    end_date = end_date.strftime('%Y-%m-%d')

    try:
        response = ce.get_cost_forecast(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Metric='UNBLENDED_COST',
            Granularity='MONTHLY'
        )

        return float(response.get('Total', {}).get('Amount', 0))

    except Exception as e:
        # Forecast may not be available with limited data
        return None


def send_alert(title: str, message: str, script_path: str = None):
    """Send alert using the alert script."""
    if script_path is None:
        script_path = '/Users/drbretto/Desktop/Projects/trader-bot/automation/send_alert.sh'

    try:
        subprocess.run([script_path, title, message], check=True)
    except Exception as e:
        print(f"Failed to send alert: {e}")


def main():
    parser = argparse.ArgumentParser(description='Check AWS costs')
    parser.add_argument('--budget', type=float, default=20.0, help='Monthly budget in USD')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--alert-threshold', type=float, default=0.8,
                        help='Alert when costs exceed this fraction of budget')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    print(f"Checking AWS costs (budget: ${args.budget:.2f})...")

    # Get current costs
    costs = get_current_month_costs(args.region)

    # Get forecast
    forecast = get_forecasted_month_end_cost(args.region)

    # Determine status
    current = costs['total']
    threshold = args.budget * args.alert_threshold

    status = 'ok'
    if current > args.budget:
        status = 'over_budget'
    elif current > threshold:
        status = 'warning'
    elif forecast and forecast > args.budget:
        status = 'forecast_warning'

    result = {
        'current_cost': current,
        'budget': args.budget,
        'forecast': forecast,
        'status': status,
        'by_service': costs['by_service'],
        'period': costs.get('period', {}),
        'checked_at': datetime.now().isoformat()
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nCurrent month costs: ${current:.2f}")
        print(f"Budget: ${args.budget:.2f}")
        if forecast:
            print(f"Forecast: ${forecast:.2f}")
        print(f"Status: {status.upper()}")

        if costs['by_service']:
            print("\nBy service:")
            for service, amount in sorted(costs['by_service'].items(), key=lambda x: -x[1]):
                if amount > 0.01:
                    print(f"  {service}: ${amount:.2f}")

    # Send alert if needed
    if status == 'over_budget':
        send_alert(
            "AWS Budget Exceeded",
            f"Current costs: ${current:.2f} (budget: ${args.budget:.2f})"
        )
    elif status == 'warning':
        send_alert(
            "AWS Costs Warning",
            f"Current costs: ${current:.2f} approaching budget of ${args.budget:.2f}"
        )
    elif status == 'forecast_warning':
        send_alert(
            "AWS Cost Forecast Warning",
            f"Forecasted end-of-month cost: ${forecast:.2f} exceeds budget of ${args.budget:.2f}"
        )

    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
