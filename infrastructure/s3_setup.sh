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

# Enable versioning
echo "Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Set lifecycle policy to clean up old versions after 30 days
echo "Setting lifecycle policy..."
cat > /tmp/lifecycle.json << 'EOF'
{
    "Rules": [
        {
            "ID": "CleanupOldVersions",
            "Status": "Enabled",
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            },
            "Filter": {
                "Prefix": ""
            }
        },
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
if [ -f "config/universe.csv" ]; then
    echo "Uploading config files..."
    aws s3 cp config/universe.csv "s3://$BUCKET_NAME/config/universe.csv"
    aws s3 cp config/decision_params.json "s3://$BUCKET_NAME/config/decision_params.json"
    aws s3 cp config/regime_compatibility.json "s3://$BUCKET_NAME/config/regime_compatibility.json"
    aws s3 cp config/aws_config.json "s3://$BUCKET_NAME/config/aws_config.json"
    aws s3 cp config/data_sources.json "s3://$BUCKET_NAME/config/data_sources.json"
fi

echo "S3 bucket setup complete!"
echo ""
echo "Next steps:"
echo "1. Run secrets_setup.sh to store API keys"
echo "2. Run lambda_deploy.sh to deploy the Lambda function"
