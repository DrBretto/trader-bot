#!/bin/bash
# EventBridge setup script for scheduling the daily pipeline

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
RULE_NAME="investment-system-daily-trigger"
REGION="${2:-us-east-1}"

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
        \"Input\": \"{\\\"bucket\\\": \\\"investment-system-data\\\", \\\"source\\\": \\\"eventbridge-scheduled\\\"}\"
    }]" \
    --region "$REGION"

echo ""
echo "EventBridge rule created successfully!"
echo ""
echo "Schedule: Every weeknight at 10 PM ET (3 AM UTC next day)"
echo "Rule ARN: arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME"
echo ""
echo "To disable the schedule:"
echo "  aws events disable-rule --name $RULE_NAME --region $REGION"
echo ""
echo "To test immediately:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"investment-system-data\", \"source\": \"manual\"}' /tmp/response.json"
