#!/bin/bash
# Idempotent production alarms for pipeline execution and schedule delivery.

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
REGION="${2:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TOPIC_ARN="arn:aws:sns:$REGION:$ACCOUNT_ID:investment-system-alerts"

aws cloudwatch put-metric-alarm \
    --alarm-name "trader-bot-lambda-errors" \
    --alarm-description "Trader Bot Lambda returned an unhandled execution error" \
    --namespace "AWS/Lambda" \
    --metric-name "Errors" \
    --dimensions "Name=FunctionName,Value=$FUNCTION_NAME" \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --datapoints-to-alarm 1 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"

aws cloudwatch put-metric-alarm \
    --alarm-name "trader-bot-lambda-throttles" \
    --alarm-description "Trader Bot Lambda invocation was throttled" \
    --namespace "AWS/Lambda" \
    --metric-name "Throttles" \
    --dimensions "Name=FunctionName,Value=$FUNCTION_NAME" \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --datapoints-to-alarm 1 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"

aws cloudwatch put-metric-alarm \
    --alarm-name "trader-bot-daily-health-red" \
    --alarm-description "Post-replay or post-morning health verdict was red" \
    --namespace "TraderBot/Brain" \
    --metric-name "DailyHealthOK" \
    --statistic Minimum \
    --period 300 \
    --evaluation-periods 1 \
    --datapoints-to-alarm 1 \
    --threshold 1 \
    --comparison-operator LessThanThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"

# A red verdict and an absent verdict are different failures. The short-period
# alarm above catches an explicit zero; this daily SampleCount alarm catches a
# disabled/deleted health schedule or a Lambda that never reaches the watchdog.
aws cloudwatch put-metric-alarm \
    --alarm-name "trader-bot-daily-health-missed" \
    --alarm-description "No Trader Bot health verdict was emitted during a full UTC day" \
    --namespace "TraderBot/Brain" \
    --metric-name "DailyHealthOK" \
    --statistic SampleCount \
    --period 86400 \
    --evaluation-periods 1 \
    --datapoints-to-alarm 1 \
    --threshold 1 \
    --comparison-operator LessThanThreshold \
    --treat-missing-data breaching \
    --alarm-actions "$TOPIC_ARN" \
    --region "$REGION"

RULES=(
    investment-system-daily-trigger
    investment-system-morning-trigger
    investment-system-morning-est-trigger
    investment-system-midday-trigger
    investment-system-advance-challenger-trigger
    investment-system-healthcheck-trigger
    investment-system-post-morning-healthcheck-trigger
)

for rule in "${RULES[@]}"; do
    aws cloudwatch put-metric-alarm \
        --alarm-name "trader-bot-eventbridge-failed-${rule#investment-system-}" \
        --alarm-description "EventBridge failed to deliver $rule" \
        --namespace "AWS/Events" \
        --metric-name "FailedInvocations" \
        --dimensions "Name=RuleName,Value=$rule" \
        --statistic Sum \
        --period 300 \
        --evaluation-periods 1 \
        --datapoints-to-alarm 1 \
        --threshold 1 \
        --comparison-operator GreaterThanOrEqualToThreshold \
        --treat-missing-data notBreaching \
        --alarm-actions "$TOPIC_ARN" \
        --region "$REGION"
done

echo "Trader Bot reliability alarms reconciled."
