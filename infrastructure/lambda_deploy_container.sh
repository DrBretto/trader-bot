#!/bin/bash
# Lambda container deployment script for the investment system
# Uses container image to include PyTorch for ensemble model inference

set -e

FUNCTION_NAME="${1:-investment-system-daily-pipeline}"
BUCKET_NAME="${2:-investment-system-data}"
REGION="${3:-us-east-1}"
ROLE_NAME="investment-system-lambda-role"
ECR_REPO_NAME="investment-system-pipeline"

echo "Deploying Lambda container: $FUNCTION_NAME"

ensure_docker_ready() {
    if docker info >/dev/null 2>&1; then
        return 0
    fi

    # Previously this auto-started Docker Desktop via `open -gja Docker`.
    # Even with the -g (no foreground) / -j (launch hidden) flags, Docker
    # Desktop still surfaces its UI window during init, which the operator
    # does not want during a deploy. We now require Docker to already be
    # running and bail out with a clear message otherwise.
    echo "Docker is not ready. Start Docker Desktop manually (whale icon" >&2
    echo "stops animating in menu bar), then rerun this deploy." >&2
    return 1
}

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
IMAGE_URI="$ECR_URI/$ECR_REPO_NAME:latest"
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"

echo "Account: $ACCOUNT_ID"
echo "ECR URI: $IMAGE_URI"

ensure_docker_ready

build_environment_arg() {
    local function_name="$1"
    local region="$2"
    local bucket="$3"

    python3 - "$function_name" "$region" "$bucket" <<'PY'
import json
import os
import subprocess
import sys

function_name, region, bucket = sys.argv[1:]

env = {}
try:
    existing = subprocess.check_output(
        [
            "aws", "lambda", "get-function-configuration",
            "--function-name", function_name,
            "--region", region,
            "--query", "Environment.Variables",
            "--output", "json",
        ],
        text=True,
    ).strip()
    if existing and existing != "null":
        env = json.loads(existing)
except Exception:
    env = {}

env["S3_BUCKET"] = bucket
env["AWS_REGION_NAME"] = region

for key in ("BROKER_MODE", "BROKER_TRADING_ENABLED"):
    value = os.environ.get(key)
    if value:
        env[key] = value

parts = ",".join(f"{key}={value}" for key, value in env.items())
print(f"Variables={{{parts}}}")
PY
}

# Create ECR repository if it doesn't exist
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null; then
    echo "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=false
fi

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"

# Build the container image (x86_64 for Lambda) and push directly
# Use buildx with --push and --provenance=false to avoid multi-arch manifest issues
echo "Building and pushing container image..."
docker buildx build --platform linux/amd64 \
    -f Dockerfile.lambda \
    -t "$IMAGE_URI" \
    --provenance=false \
    --push \
    .

# Get the image digest for deterministic deployment
IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name "$ECR_REPO_NAME" \
    --image-ids imageTag=latest \
    --region "$REGION" \
    --query 'imageDetails[0].imageDigest' \
    --output text)
IMAGE_WITH_DIGEST="$ECR_URI/$ECR_REPO_NAME@$IMAGE_DIGEST"
echo "Image digest: $IMAGE_DIGEST"

# Check if role exists, create if not
if ! aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "Creating IAM role..."

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

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "investment-system-policy" \
        --policy-document file://infrastructure/iam_policies.json

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

    echo "Waiting for IAM role to propagate..."
    sleep 10

    rm /tmp/trust-policy.json
fi

# Check if function exists and its package type
EXISTING_FUNCTION=$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null || true)

if [ -n "$EXISTING_FUNCTION" ]; then
    PACKAGE_TYPE=$(echo "$EXISTING_FUNCTION" | python3 -c "import sys,json; print(json.load(sys.stdin)['Configuration']['PackageType'])")
    ENV_ARG=$(build_environment_arg "$FUNCTION_NAME" "$REGION" "$BUCKET_NAME")

    if [ "$PACKAGE_TYPE" = "Image" ]; then
        echo "Updating existing container function..."
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --image-uri "$IMAGE_WITH_DIGEST" \
            --region "$REGION"
    else
        echo "Existing function is Zip-based. Deleting and recreating as container..."
        aws lambda delete-function --function-name "$FUNCTION_NAME" --region "$REGION"
        echo "Waiting for deletion to complete..."
        sleep 5

        echo "Creating new container function..."
        aws lambda create-function \
            --function-name "$FUNCTION_NAME" \
            --package-type Image \
            --code ImageUri="$IMAGE_WITH_DIGEST" \
            --role "$ROLE_ARN" \
            --timeout 900 \
            --memory-size 3008 \
            --environment "$ENV_ARG" \
            --region "$REGION"
    fi

    # Wait for code update to complete before updating configuration.
    echo "Waiting for function code update..."
    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"

    # Keep runtime settings in sync on updates as well as creates.
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout 900 \
        --memory-size 3008 \
        --environment "$ENV_ARG" \
        --region "$REGION"

    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"
else
    echo "Creating new function with container image..."
    ENV_ARG=$(build_environment_arg "$FUNCTION_NAME" "$REGION" "$BUCKET_NAME")
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --package-type Image \
        --code ImageUri="$IMAGE_WITH_DIGEST" \
        --role "$ROLE_ARN" \
        --timeout 900 \
        --memory-size 3008 \
        --environment "$ENV_ARG" \
        --region "$REGION"

    echo "Waiting for function to be active..."
    aws lambda wait function-active \
        --function-name "$FUNCTION_NAME" \
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

echo ""
echo "Lambda container deployed successfully!"
echo ""
echo "Function ARN: arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"
echo "Image: $IMAGE_WITH_DIGEST"
echo ""
echo "To test:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"bucket\": \"$BUCKET_NAME\"}' --region $REGION /tmp/response.json && cat /tmp/response.json"
