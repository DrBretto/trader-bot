"""SNS email alerting for pipeline runs."""

import boto3
from typing import Dict, Any, List, Optional


def get_sns_topic_arn(region: str = 'us-east-1') -> str:
    """Get the SNS topic ARN, discovering account ID dynamically."""
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']
    return f"arn:aws:sns:{region}:{account_id}:investment-system-alerts"


def send_alert(subject: str, body: str, region: str = 'us-east-1') -> bool:
    """Publish a message to the SNS alerts topic. Never raises."""
    try:
        sns = boto3.client('sns', region_name=region)
        topic_arn = get_sns_topic_arn(region)
        sns.publish(
            TopicArn=topic_arn,
            Subject=subject[:100],
            Message=body
        )
        return True
    except Exception as e:
        print(f"SNS alert failed (non-fatal): {e}")
        return False


def format_night_summary(
    run_date: str,
    regime: str,
    portfolio_value: float,
    intents: List[Dict[str, Any]],
    weather_headline: str,
    duration_seconds: float
) -> str:
    """Format the night run summary email."""
    buys = [a for a in intents if a.get('action') == 'BUY']
    sells = [a for a in intents if a.get('action') in ('SELL', 'REDUCE')]

    lines = [
        f"Night Analysis Complete - {run_date}",
        f"Duration: {duration_seconds:.0f}s",
        "",
        f"Regime: {regime}",
        f"Portfolio Value: ${portfolio_value:,.2f}",
        "",
        "Trade Intents for Morning Execution:",
        f"  Buys:  {len(buys)}",
    ]
    for b in buys:
        lines.append(
            f"    {b['symbol']}: {b.get('shares', '?')} shares "
            f"@ ~${b.get('price', 0):.2f} (${b.get('dollars', 0):,.0f})"
        )
    lines.append(f"  Sells: {len(sells)}")
    for s in sells:
        lines.append(f"    {s['symbol']}: {s.get('shares', '?')} shares ({s.get('reason', '')})")

    if not buys and not sells:
        lines.append("  (none)")

    lines.extend(["", f"Weather: {weather_headline}"])
    return "\n".join(lines)


def format_morning_summary(
    run_date: str,
    portfolio_value: float,
    trades: List[Dict[str, Any]],
    validation_log: List[str],
    duration_seconds: float
) -> str:
    """Format the morning execution summary email."""
    lines = [
        f"Morning Execution Complete - {run_date}",
        f"Duration: {duration_seconds:.0f}s",
        "",
        f"Portfolio Value: ${portfolio_value:,.2f}",
        f"Trades Executed: {len(trades)}",
    ]
    for t in trades:
        pnl_str = ""
        if 'pnl' in t:
            pnl_str = f" (P&L: ${t['pnl']:+,.2f})"
        lines.append(
            f"  {t.get('action', '?')} {t.get('shares', '?')} "
            f"{t.get('symbol', '?')} @ ${t.get('price', 0):.2f}{pnl_str}"
        )

    if not trades:
        lines.append("  (no trades)")

    lines.extend(["", "Validation Log:"])
    for entry in validation_log:
        lines.append(f"  {entry}")

    return "\n".join(lines)


def format_error_alert(run_phase: str, run_date: str, error: str) -> str:
    """Format an error alert email."""
    return (
        f"PIPELINE ERROR - {run_phase} phase - {run_date}\n\n"
        f"Error: {error}\n\n"
        f"Check CloudWatch logs for details:\n"
        f"  aws logs tail /aws/lambda/investment-system-daily-pipeline "
        f"--region us-east-1 --since 30m"
    )
