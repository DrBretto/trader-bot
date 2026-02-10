#!/bin/bash
# SNS topic setup for pipeline email alerts

set -e

REGION="${1:-us-east-1}"
EMAIL="${2:-drbretto82@gmail.com}"
TOPIC_NAME="investment-system-alerts"

echo "Setting up SNS alerts topic: $TOPIC_NAME"

# Create topic
TOPIC_ARN=$(aws sns create-topic \
    --name "$TOPIC_NAME" \
    --region "$REGION" \
    --query TopicArn \
    --output text)

echo "Topic ARN: $TOPIC_ARN"

# Subscribe email
echo "Subscribing $EMAIL..."
aws sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$EMAIL" \
    --region "$REGION"

echo ""
echo "SNS topic created successfully!"
echo ""
echo "IMPORTANT: Check your email ($EMAIL) and confirm the subscription."
echo "You will not receive alerts until the subscription is confirmed."
echo ""
echo "To test:"
echo "  aws sns publish --topic-arn $TOPIC_ARN --subject 'Test' --message 'Test alert' --region $REGION"
