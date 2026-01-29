# Deployment Guide

This document covers deployment procedures for the investment system. Cursor is responsible for running these steps.

## Prerequisites

- AWS CLI configured with `personal` profile
- Docker installed (for building Lambda layers)
- API keys stored in Secrets Manager (one-time setup)

## Environment

- **Region**: us-east-1
- **S3 Bucket**: investment-system-data
- **Lambda Function**: investment-system-daily-pipeline
- **EventBridge Rule**: investment-system-daily-trigger

---

## First-Time Setup

Run these once when setting up a new environment:

```bash
# 1. Create S3 bucket (no versioning!)
./infrastructure/s3_setup.sh investment-system-data us-east-1

# 2. Store API keys in Secrets Manager
./infrastructure/secrets_setup.sh us-east-1
# Then update secrets with actual keys via AWS Console or CLI
```

---

## Deploy Lambda Function

After code changes, deploy the updated Lambda:

```bash
# Build and deploy
./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1
```

**Note**: If numpy/pandas dependencies change, you may need to rebuild the Lambda layer:

```bash
# Build layer with Docker (x86_64 for Lambda)
rm -rf /tmp/layer-build/*
docker run --rm --platform linux/amd64 -v /tmp/layer-build:/output public.ecr.aws/sam/build-python3.11:latest \
  pip install numpy pandas fastparquet -t /output/python --only-binary=:all: --quiet

# Zip and upload
cd /tmp/layer-build && zip -r /tmp/layer.zip python -q
aws s3 cp /tmp/layer.zip s3://investment-system-data/layers/pandas-layer.zip --region us-east-1

# Publish new layer version
aws lambda publish-layer-version \
  --layer-name investment-system-pandas-x86 \
  --content S3Bucket=investment-system-data,S3Key=layers/pandas-layer.zip \
  --compatible-runtimes python3.11 \
  --compatible-architectures x86_64 \
  --region us-east-1

# Update Lambda to use new layer (get ARN from previous command)
aws lambda update-function-configuration \
  --function-name investment-system-daily-pipeline \
  --layers "arn:aws:lambda:us-east-1:767398003959:layer:investment-system-pandas-x86:VERSION" \
  --region us-east-1
```

---

## Set Up Schedule

Enable the daily EventBridge trigger (10 PM ET weeknights):

```bash
./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline us-east-1
```

To disable the schedule:
```bash
aws events disable-rule --name investment-system-daily-trigger --region us-east-1
```

---

## Verify Deployment

### Manual Test
```bash
aws lambda invoke \
  --function-name investment-system-daily-pipeline \
  --cli-binary-format raw-in-base64-out \
  --payload '{"bucket": "investment-system-data", "source": "manual"}' \
  --invocation-type Event \
  --region us-east-1 \
  /tmp/response.json
```

### Check Logs
```bash
aws logs tail /aws/lambda/investment-system-daily-pipeline --region us-east-1 --follow
```

### Check Artifacts
```bash
aws s3 ls s3://investment-system-data/daily/ --region us-east-1
```

---

## Rollback

To rollback to a previous Lambda version:

```bash
# List versions
aws lambda list-versions-by-function --function-name investment-system-daily-pipeline --region us-east-1

# Update alias or invoke specific version
aws lambda update-alias \
  --function-name investment-system-daily-pipeline \
  --name live \
  --function-version VERSION_NUMBER \
  --region us-east-1
```

---

## Cost Notes

- **NEVER enable S3 versioning**
- Lambda: ~$0.20/day at current usage (84s execution, 3GB memory)
- S3: Minimal (lifecycle deletes data after 365 days)
- Total estimated: < $10/month
