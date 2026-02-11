#!/bin/bash
# Lambda deployment script for the investment system

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
BUCKET_NAME="${2:-investment-system-data}"
REGION="${3:-us-east-1}"
ROLE_NAME="investment-system-lambda-role"

echo "Deploying Lambda function: $FUNCTION_NAME"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create deployment package directory
DEPLOY_DIR="/tmp/lambda_deploy_$$"
mkdir -p "$DEPLOY_DIR"

echo "Creating deployment package..."

# Copy Lambda code
cp -r src "$DEPLOY_DIR/"
# Use slim requirements (numpy/pandas/fastparquet come from Lambda layer)
REQ_FILE="requirements-lambda.txt"
if [ -f "$PROJECT_ROOT/$REQ_FILE" ]; then
    cp "$PROJECT_ROOT/$REQ_FILE" "$DEPLOY_DIR/requirements.txt"
else
    cp "$PROJECT_ROOT/requirements.txt" "$DEPLOY_DIR/"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r "$DEPLOY_DIR/requirements.txt" -t "$DEPLOY_DIR/" --quiet

# Remove unnecessary files
find "$DEPLOY_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DEPLOY_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$DEPLOY_DIR" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find "$DEPLOY_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true

# Create zip file
DEPLOY_ZIP="/tmp/lambda_deploy_$$.zip"
cd "$DEPLOY_DIR"
zip -r "$DEPLOY_ZIP" . -x "*.pyc" -x "__pycache__/*" -q
cd "$PROJECT_ROOT"

echo "Deployment package size: $(du -h "$DEPLOY_ZIP" | cut -f1)"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"

# Use existing pandas layer if present (see DEPLOY.md)
LAYER_ARN=$(aws lambda list-layer-versions --layer-name investment-system-pandas-x86 --region "$REGION" --query 'LayerVersions[0].LayerVersionArn' --output text 2>/dev/null || true)
if [ -z "$LAYER_ARN" ] || [ "$LAYER_ARN" = "None" ]; then
    LAYER_ARN=""
fi

# Check if role exists, create if not
if ! aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "Creating IAM role..."

    # Create trust policy
    cat > /tmp/trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --description "Role for investment system Lambda function"

    # Attach policies
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "investment-system-policy" \
        --policy-document file://infrastructure/iam_policies.json

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

    # Wait for role to propagate
    echo "Waiting for IAM role to propagate..."
    sleep 10

    rm /tmp/trust-policy.json
fi

# Check if function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null; then
    echo "Updating existing function..."
    ZIP_SIZE=$(stat -f%z "$DEPLOY_ZIP" 2>/dev/null || stat -c%s "$DEPLOY_ZIP" 2>/dev/null)
    if [ "$ZIP_SIZE" -gt 52428800 ]; then
        echo "Zip exceeds 50MB ($((ZIP_SIZE / 1048576))MB), uploading via S3..."
        S3_KEY="lambda/deploy-$(date +%Y%m%d-%H%M).zip"
        aws s3 cp "$DEPLOY_ZIP" "s3://${BUCKET_NAME}/${S3_KEY}" --region "$REGION"
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --s3-bucket "$BUCKET_NAME" \
            --s3-key "$S3_KEY" \
            --region "$REGION"
    else
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --zip-file "fileb://$DEPLOY_ZIP" \
            --region "$REGION"
    fi

    # Wait for update to complete
    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"

    # Update configuration
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout 900 \
        --memory-size 3008 \
        --environment "Variables={S3_BUCKET=$BUCKET_NAME,AWS_REGION=$REGION,AWS_REGION_NAME=$REGION}" \
        --region "$REGION"
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime python3.11 \
        --role "$ROLE_ARN" \
        --handler src.handler.lambda_handler \
        --zip-file "fileb://$DEPLOY_ZIP" \
        --timeout 900 \
        --memory-size 3008 \
        --environment "Variables={S3_BUCKET=$BUCKET_NAME,AWS_REGION=$REGION,AWS_REGION_NAME=$REGION}" \
        --region "$REGION"
fi

# Re-add EventBridge invoke permission (idempotent; needed after delete+recreate)
RULE_NAME="investment-system-daily-trigger"
RULE_ARN="arn:aws:events:$REGION:$ACCOUNT_ID:rule/$RULE_NAME"
echo "Ensuring EventBridge invoke permission..."
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "EventBridgeInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" \
    --region "$REGION" 2>/dev/null || true

# Cleanup
rm -rf "$DEPLOY_DIR"
rm -f "$DEPLOY_ZIP"

echo ""
echo "Lambda function deployed successfully!"
echo ""
echo "Function ARN: arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"
echo ""
echo "To test the function:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\"}' /tmp/response.json && cat /tmp/response.json"
echo ""
echo "To set up the EventBridge schedule:"
echo "  ./infrastructure/eventbridge_setup.sh"
