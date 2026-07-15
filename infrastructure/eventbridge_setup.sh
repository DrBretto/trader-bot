#!/bin/bash
# EventBridge setup script for scheduling the daily pipeline

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
RULE_NAME="investment-system-daily-trigger"

# Backward-compatible argument parsing:
# - Old form: eventbridge_setup.sh <function> <region>
# - New form: eventbridge_setup.sh <function> <bucket> <region>
ARG2="${2:-}"
ARG3="${3:-}"
BUCKET_NAME="investment-system-data"
REGION="us-east-1"

if [ -n "$ARG3" ]; then
    BUCKET_NAME="$ARG2"
    REGION="$ARG3"
elif [[ "$ARG2" =~ ^[a-z]{2}-[a-z]+-[0-9]+$ ]]; then
    REGION="$ARG2"
elif [ -n "$ARG2" ]; then
    BUCKET_NAME="$ARG2"
fi

echo "Setting up EventBridge rule: $RULE_NAME"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"

# Create the EventBridge rule
# Schedule: 3 AM UTC = 10 PM ET (previous day)
# Run Tue-Sat to catch Mon-Fri market data
echo "Creating EventBridge rule..."
aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "cron(0 3 ? * TUE-SAT *)" \
    --state ENABLED \
    --description "Triggers the daily investment pipeline at 10 PM ET on weeknights" \
    --region "$REGION"

# Add Lambda permission for EventBridge to invoke
echo "Adding Lambda permission..."
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME" \
    --region "$REGION" 2>/dev/null || true

# Add the Lambda as a target
echo "Adding Lambda target..."
aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "[{
        \"Id\": \"investment-system-lambda\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"eventbridge-scheduled\\\"}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 7200}
    }]" \
    --region "$REGION"

# --- Morning execution rule ---
MORNING_RULE_NAME="investment-system-morning-trigger"

echo ""
echo "Setting up morning execution rule: $MORNING_RULE_NAME"

# Primary schedule: 13:45 UTC = 9:45 AM during EDT. A second 14:45 UTC
# standard-time rule is installed below. The app rejects pre-09:40 ET invokes,
# and its daily checkpoint makes the later summer invoke an idempotent no-op.
aws events put-rule \
    --name "$MORNING_RULE_NAME" \
    --schedule-expression "cron(45 13 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Primary morning execution at 09:45 New York during EDT" \
    --region "$REGION"

# Add Lambda permission for morning rule
echo "Adding Lambda permission for morning rule..."
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeMorningInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MORNING_RULE_NAME" \
    --region "$REGION" 2>/dev/null || true

# Add Lambda as target for morning rule
echo "Adding Lambda target for morning rule..."
aws events put-targets \
    --rule "$MORNING_RULE_NAME" \
    --targets "[{
        \"Id\": \"investment-system-lambda-morning\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"morning-execution\\\"}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 7200}
    }]" \
    --region "$REGION"

# Standard-time companion: 14:45 UTC = 09:45 EST. In EDT this is a harmless
# idempotent replay of the already-completed morning checkpoint.
MORNING_EST_RULE_NAME="investment-system-morning-est-trigger"
aws events put-rule \
    --name "$MORNING_EST_RULE_NAME" \
    --schedule-expression "cron(45 14 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Standard-time companion for 09:45 New York morning execution" \
    --region "$REGION"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeMorningESTInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MORNING_EST_RULE_NAME" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$MORNING_EST_RULE_NAME" \
    --targets "[{
        \"Id\": \"investment-system-lambda-morning-est\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"morning-execution\\\"}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 7200}
    }]" \
    --region "$REGION"

# --- Midday check rule ---
MIDDAY_RULE_NAME="investment-system-midday-trigger"

echo ""
echo "Setting up midday check rule: $MIDDAY_RULE_NAME"

# Schedule: 18:00 UTC = 1:00 PM ET, Mon-Fri
aws events put-rule \
    --name "$MIDDAY_RULE_NAME" \
    --schedule-expression "cron(0 18 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Triggers midday check (trailing stops, VIX breaker, skipped buys) at 1:00 PM ET on weekdays" \
    --region "$REGION"

# Add Lambda permission for midday rule
echo "Adding Lambda permission for midday rule..."
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeMiddayInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MIDDAY_RULE_NAME" \
    --region "$REGION" 2>/dev/null || true

# Add Lambda as target for midday rule
echo "Adding Lambda target for midday rule..."
aws events put-targets \
    --rule "$MIDDAY_RULE_NAME" \
    --targets "[{
        \"Id\": \"investment-system-lambda-midday\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"midday-check\\\"}\"
    }]" \
    --region "$REGION"

echo ""
echo "EventBridge rules created successfully!"
echo ""
echo "Night schedule:   Every weeknight at 10 PM ET (3 AM UTC next day, Tue-Sat)"
echo "Morning schedule: DST-safe 9:45 AM New York pair (13:45 + 14:45 UTC)"
echo "Midday schedule:  Every weekday at 1:00 PM ET (18:00 UTC, Mon-Fri)"
echo ""
echo "Night rule ARN:   arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME"
echo "Morning rule ARN: arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MORNING_RULE_NAME"
echo "Midday rule ARN:  arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MIDDAY_RULE_NAME"
echo ""
echo "To disable:"
echo "  aws events disable-rule --name $RULE_NAME --region $REGION"
echo "  aws events disable-rule --name $MORNING_RULE_NAME --region $REGION"
echo "  aws events disable-rule --name $MIDDAY_RULE_NAME --region $REGION"
echo ""
echo "To test:"
echo "  Night:   aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\", \"source\": \"manual\"}' /tmp/response.json"
echo "  Morning: aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\", \"source\": \"morning-execution\"}' --invocation-type Event /tmp/response.json"
echo "  Midday:  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\", \"source\": \"midday-check\"}' --invocation-type Event /tmp/response.json"
