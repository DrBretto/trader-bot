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
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"eventbridge-scheduled\\\"}\"
    }]" \
    --region "$REGION"

# --- Morning execution rule ---
MORNING_RULE_NAME="investment-system-morning-trigger"

echo ""
echo "Setting up morning execution rule: $MORNING_RULE_NAME"

# Schedule: 14:45 UTC = 9:45 AM ET, Mon-Fri
aws events put-rule \
    --name "$MORNING_RULE_NAME" \
    --schedule-expression "cron(45 14 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Triggers morning trade execution at 9:45 AM ET on weekdays" \
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
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"morning-execution\\\"}\"
    }]" \
    --region "$REGION"

echo ""
echo "EventBridge rules created successfully!"
echo ""
echo "Night schedule:   Every weeknight at 10 PM ET (3 AM UTC next day, Tue-Sat)"
echo "Morning schedule: Every weekday at 9:45 AM ET (14:45 UTC, Mon-Fri)"
echo ""
echo "Night rule ARN:   arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME"
echo "Morning rule ARN: arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MORNING_RULE_NAME"
echo ""
echo "To disable:"
echo "  aws events disable-rule --name $RULE_NAME --region $REGION"
echo "  aws events disable-rule --name $MORNING_RULE_NAME --region $REGION"
echo ""
echo "To test:"
echo "  Night:   aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\", \"source\": \"manual\"}' /tmp/response.json"
echo "  Morning: aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\", \"source\": \"morning-execution\"}' --invocation-type Event /tmp/response.json"
