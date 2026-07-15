#!/bin/bash
# EventBridge setup for the three-line daily health watchdog (Lambda-native).
#
# This provisions the P9 LIVE, in-prod watchdog cron: an EventBridge rule that
# invokes the daily-pipeline Lambda with {"source":"healthcheck"}, which the clean
# thin router (trader-bot-core/app/handler.py) routes to
# monitors.watchdog.run_daily_health_check — the three-line ✓/✗ email + missed-run
# /stale alarm.
#
# The post-replay check fires 05:15 UTC every day, after the 04:30 replay writer.
# Daily cadence provides a real heartbeat on weekends/holidays too; the watchdog's
# NYSE calendar resolves the expected settled session without a grace-day loophole.
# A second 15:15 UTC weekday check verifies that morning execution refreshed the
# public operational snapshot and wrote its receipt.
#
# Reproducibility (P9): this script is the committed source-of-truth for the LIVE
# EventBridge watchdog rule + its Lambda invoke permission. It is idempotent — a
# fresh clone/redeploy re-creates the exact live state. The GitHub Actions check
# independently verifies the public CloudFront view without AWS credentials.
#
# NOTE: the {"source":"healthcheck"} route only reaches the three-line watchdog
# once prod runs the clean thin handler — apply/verify this AFTER the P9 clean-core
# image is deployed to the Lambda.

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
BUCKET_NAME="${2:-investment-system-data}"
REGION="${3:-us-east-1}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"

# --- Post-replay health rule ---
HEALTH_RULE="investment-system-healthcheck-trigger"
echo "Setting up daily-health watchdog rule: $HEALTH_RULE"
aws events put-rule \
    --name "$HEALTH_RULE" \
    --schedule-expression "cron(15 5 ? * * *)" \
    --state ENABLED \
    --description "Daily line/dashboard health at 05:15 UTC after the 04:30 promoted replay writer" \
    --region "$REGION"

# Idempotent: AddPermission errors if the statement id already exists — ignore it.
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeHealthcheckInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$HEALTH_RULE" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$HEALTH_RULE" \
    --targets "[{
        \"Id\": \"health-target\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"healthcheck\\\"}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 3600}
    }]" \
    --region "$REGION"

# --- Post-morning public snapshot/receipt check ---
MORNING_HEALTH_RULE="investment-system-post-morning-healthcheck-trigger"
echo "Setting up post-morning watchdog rule: $MORNING_HEALTH_RULE"
aws events put-rule \
    --name "$MORNING_HEALTH_RULE" \
    --schedule-expression "cron(15 15 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Weekday post-morning check for current public snapshot and execution receipt" \
    --region "$REGION"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgePostMorningHealthcheckInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$MORNING_HEALTH_RULE" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$MORNING_HEALTH_RULE" \
    --targets "[{
        \"Id\": \"post-morning-health-target\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"healthcheck\\\", \\\"require_morning\\\": true}\",
        \"RetryPolicy\": {\"MaximumRetryAttempts\": 2, \"MaximumEventAgeInSeconds\": 3600}
    }]" \
    --region "$REGION"

echo "Health watchdogs live (05:15 UTC daily post-replay; 15:15 UTC weekdays post-morning)."
