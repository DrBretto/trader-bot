#!/bin/bash
# S3 bucket setup script for the investment system

set -e

BUCKET_NAME="${1:-investment-system-data}"
REGION="${2:-us-east-1}"

echo "Setting up S3 bucket: $BUCKET_NAME in $REGION"

# Create bucket
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket $BUCKET_NAME already exists"
else
    echo "Creating bucket $BUCKET_NAME..."
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    else
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
fi

# Block public access
echo "Configuring public access settings..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Set lifecycle policy to clean up old data after 1 year
echo "Setting lifecycle policy..."
cat > /tmp/lifecycle.json << 'EOF'
{
    "Rules": [
        {
            "ID": "CleanupOldDailyData",
            "Status": "Enabled",
            "Expiration": {
                "Days": 365
            },
            "Filter": {
                "Prefix": "daily/"
            }
        }
    ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket "$BUCKET_NAME" \
    --lifecycle-configuration file:///tmp/lifecycle.json

rm /tmp/lifecycle.json

# Create initial folder structure
echo "Creating folder structure..."
echo "" | aws s3 cp - "s3://$BUCKET_NAME/config/.keep"
echo "" | aws s3 cp - "s3://$BUCKET_NAME/daily/.keep"
echo "" | aws s3 cp - "s3://$BUCKET_NAME/models/.keep"
echo "" | aws s3 cp - "s3://$BUCKET_NAME/templates/.keep"
echo "" | aws s3 cp - "s3://$BUCKET_NAME/backtests/.keep"

# Upload config files if they exist locally
echo "Uploading config files (if present)..."

if [ -f "config/universe.csv" ]; then
    aws s3 cp config/universe.csv "s3://$BUCKET_NAME/config/universe.csv"
fi

# Canonical live config bundle required by src/handler.py
if [ -f "config/decision_params.active.json" ]; then
    aws s3 cp config/decision_params.active.json "s3://$BUCKET_NAME/config/decision_params.active.json"
else
    echo "WARNING: config/decision_params.active.json not found."
    echo "         Upload it before running Lambda (required live config bundle)."
fi

# Legacy files are still useful for audit/history and backward references
if [ -f "config/decision_params.json" ]; then
    aws s3 cp config/decision_params.json "s3://$BUCKET_NAME/config/decision_params.json"
fi
if [ -f "config/regime_compatibility.json" ]; then
    aws s3 cp config/regime_compatibility.json "s3://$BUCKET_NAME/config/regime_compatibility.json"
fi

if [ -f "config/aws_config.json" ]; then
    aws s3 cp config/aws_config.json "s3://$BUCKET_NAME/config/aws_config.json"
fi
if [ -f "config/data_sources.json" ]; then
    aws s3 cp config/data_sources.json "s3://$BUCKET_NAME/config/data_sources.json"
fi

echo "S3 bucket setup complete!"
echo ""
echo "Next steps:"
echo "1. Run secrets_setup.sh to store API keys"
echo "2. Run lambda_deploy.sh to deploy the Lambda function"
