#!/bin/bash
# Reproducible schedule for the sole displayed-model-line writer.

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
BUCKET_NAME="${2:-investment-system-data}"
REGION="${3:-us-east-1}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"
RULE_NAME="investment-system-advance-challenger-trigger"

aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "cron(30 4 ? * TUE-SAT *)" \
    --state ENABLED \
    --description "Sole line writer: replay TILT canon and two-stage comparison at 04:30 UTC Tue-Sat" \
    --region "$REGION"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgePromotedReplayInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "[{
        \"Id\": \"advance-challenger-target\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"advance-challenger\\\", \\\"commit\\\": true}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 3600}
    }]" \
    --region "$REGION"

EXTRA_TARGET_IDS=$(aws events list-targets-by-rule \
    --rule "$RULE_NAME" \
    --query "Targets[?Id!='advance-challenger-target'].Id" \
    --output text \
    --region "$REGION")
if [ -n "$EXTRA_TARGET_IDS" ] && [ "$EXTRA_TARGET_IDS" != "None" ]; then
    # EventBridge target IDs cannot contain whitespace; split the text result.
    # shellcheck disable=SC2086
    aws events remove-targets \
        --rule "$RULE_NAME" \
        --ids $EXTRA_TARGET_IDS \
        --region "$REGION"
fi

TARGET_COUNT=$(aws events list-targets-by-rule \
    --rule "$RULE_NAME" \
    --query "length(Targets)" \
    --output text \
    --region "$REGION")
if [ "$TARGET_COUNT" -ne 1 ]; then
    echo "ERROR: $RULE_NAME has $TARGET_COUNT targets after reconciliation" >&2
    exit 1
fi

# The promoted replay already publishes shadow_timeseries.json. Leaving the old
# 03:30 mirror enabled creates a transient stale write before the authoritative
# 04:30 refresh and adds no independent recovery value.
if aws events describe-rule \
    --name "investment-system-shadow-trigger" \
    --region "$REGION" >/dev/null 2>&1; then
    aws events disable-rule \
        --name "investment-system-shadow-trigger" \
        --region "$REGION"
fi

echo "Promoted replay line writer live at 04:30 UTC Tue-Sat; retired shadow cron disabled."
