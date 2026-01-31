#!/bin/bash
#
# Simple alerting script for investment system
#
# Supports multiple notification methods:
# - macOS notifications (default)
# - Slack webhook (if configured)
# - Email via AWS SES (if configured)
#
# Usage: send_alert.sh "Title" "Message"
#

TITLE="${1:-Alert}"
MESSAGE="${2:-No message provided}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Configuration (set these environment variables or edit here)
SLACK_WEBHOOK_URL="${INVESTMENT_SLACK_WEBHOOK:-}"
SES_EMAIL="${INVESTMENT_ALERT_EMAIL:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Log the alert
echo "[$TIMESTAMP] ALERT: $TITLE - $MESSAGE" >> /tmp/investment-alerts.log

# Method 1: macOS Notification (always available)
if command -v osascript &> /dev/null; then
    osascript -e "display notification \"$MESSAGE\" with title \"Investment System\" subtitle \"$TITLE\""
fi

# Method 2: Slack webhook (if configured)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"*$TITLE*\n$MESSAGE\n_$TIMESTAMP_\"}" \
        "$SLACK_WEBHOOK_URL" > /dev/null 2>&1
fi

# Method 3: AWS SES Email (if configured)
if [ -n "$SES_EMAIL" ] && command -v aws &> /dev/null; then
    aws ses send-email \
        --from "$SES_EMAIL" \
        --to "$SES_EMAIL" \
        --subject "Investment System: $TITLE" \
        --text "$MESSAGE\n\nTimestamp: $TIMESTAMP" \
        --region "$AWS_REGION" > /dev/null 2>&1 || true
fi

# Method 4: Terminal bell (for interactive sessions)
if [ -t 1 ]; then
    echo -e "\a"
fi

echo "Alert sent: $TITLE"
