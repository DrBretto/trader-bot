#!/bin/bash
# EventBridge setup for the three-line daily health watchdog (Lambda-native).
#
# This provisions the P9 LIVE, in-prod watchdog cron: an EventBridge rule that
# invokes the daily-pipeline Lambda with {"source":"healthcheck"}, which the clean
# thin router (trader-bot-core/app/handler.py) routes to
# monitors.watchdog.run_daily_health_check — the three-line ✓/✗ email + missed-run
# /stale alarm.
#
# It fires 04:00 UTC Tue-Sat — right after the night pipeline settles (the night
# rule is cron(0 3 ? * TUE-SAT)). The watchdog computes the EXPECTED settled day
# as the latest weekday on-or-before today and tolerates a one-trading-day lag, so
# a weekend/holiday does NOT false-alarm; TUE-SAT keeps the cron aligned to real
# trading nights with no redundant weekend fire.
#
# Reproducibility (P9): this script is the committed source-of-truth for the LIVE
# EventBridge watchdog rule + its Lambda invoke permission. It is idempotent — a
# fresh clone/redeploy re-creates the exact live state. The 08:35/15:15 daily
# health checks are ALREADY covered belt-and-suspenders by the GitHub Actions
# workflow .github/workflows/daily-health-watchdog.yml (which runs the SAME
# clean-core check against live S3 from CI); this EventBridge cron is the in-prod
# Lambda-native path that no longer depends on CI once prod is cut over.
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

# --- Post-night health rule (fires right after the night pipeline settles) ---
HEALTH_RULE="investment-system-healthcheck-trigger"
echo "Setting up daily-health watchdog rule: $HEALTH_RULE"
aws events put-rule \
    --name "$HEALTH_RULE" \
    --schedule-expression "cron(0 4 ? * TUE-SAT *)" \
    --state ENABLED \
    --description "Three-line daily health email, 04:00 UTC Tue-Sat after the night pipeline settles" \
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
        \"Input\": \"{\\\"bucket\\\": \\\"$BUCKET_NAME\\\", \\\"source\\\": \\\"healthcheck\\\"}\"
    }]" \
    --region "$REGION"

echo "Daily-health watchdog EventBridge rule live (04:00 UTC Tue-Sat, source=healthcheck)."
