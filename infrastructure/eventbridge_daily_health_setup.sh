#!/bin/bash
# EventBridge setup for the P8 three-line daily health watchdog (Lambda-native).
#
# This is the P9-READY schedule surface: it points an EventBridge rule at the
# Lambda with {"source":"daily-health"}, which the clean thin router
# (trader-bot-core/app/handler.py) routes to monitors.watchdog.run_daily_health_check
# — the three-line ✓/✗ email + missed-run/stale alarm.
#
# It fires NIGHTLY (after the night pipeline settles) AND POST-PIPELINE (after
# morning execution), mirroring the existing eventbridge_setup.sh rule shape.
#
# NOTE: apply this only AFTER P9 cuts the prod Lambda over to the clean handler
# (src.handler still owns prod during P8). Until then, the GitHub Actions
# workflow .github/workflows/daily-health-watchdog.yml runs the SAME clean-core
# health check against live S3 from CI — no prod cutover required.

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
BUCKET_NAME="${2:-investment-system-data}"
REGION="${3:-us-east-1}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"

# --- Nightly health rule (after the night pipeline settles) ---
NIGHT_RULE="investment-system-daily-health-nightly"
echo "Setting up daily-health rule: $NIGHT_RULE"
aws events put-rule \
    --name "$NIGHT_RULE" \
    --schedule-expression "cron(35 8 * * ? *)" \
    --state ENABLED \
    --description "Three-line daily health email nightly, after the night pipeline settles" \
    --region "$REGION"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeInvokeDailyHealthNightly" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$NIGHT_RULE" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$NIGHT_RULE" \
    --targets "[{
        \"Id\": \"investment-system-daily-health-nightly\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"daily-health\\\"}\"
    }]" \
    --region "$REGION"

# --- Post-pipeline health rule (after morning execution, weekdays) ---
POST_RULE="investment-system-daily-health-postpipeline"
echo "Setting up daily-health rule: $POST_RULE"
aws events put-rule \
    --name "$POST_RULE" \
    --schedule-expression "cron(15 15 ? * MON-FRI *)" \
    --state ENABLED \
    --description "Three-line daily health email post-pipeline, after morning execution" \
    --region "$REGION"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeInvokeDailyHealthPost" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "arn:aws:events:$REGION:$ACCOUNT_ID:rule/$POST_RULE" \
    --region "$REGION" 2>/dev/null || true

aws events put-targets \
    --rule "$POST_RULE" \
    --targets "[{
        \"Id\": \"investment-system-daily-health-postpipeline\",
        \"Arn\": \"$LAMBDA_ARN\",
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"daily-health\\\"}\"
    }]" \
    --region "$REGION"

echo "Daily-health EventBridge rules configured (nightly + post-pipeline)."
