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

## Deploy Lambda Function (Container)

The Lambda uses a container image to include PyTorch for ensemble model inference.

```bash
# Build and deploy container image
./infrastructure/lambda_deploy_container.sh investment-system-daily-pipeline investment-system-data us-east-1
```

This script:
1. Builds a Docker image with PyTorch (CPU-only, ~1.2GB)
2. Pushes to ECR (`investment-system-pipeline` repository)
3. Updates the Lambda function to use the new image

**ECR costs**: ~$0.10/GB/month storage (~$0.12/month for the image)

### Legacy Zip Deployment (no PyTorch)

For deployments without PyTorch (baseline models only):

```bash
./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1
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

## Deploy Frontend Dashboard

Build and upload the React dashboard to S3 for static hosting.

**Production build:** Set `VITE_DATA_URL=dashboard.json` so the deployed app at `.../dashboard/` fetches data from `.../dashboard/dashboard.json` (same prefix). Without this, the app uses `./data/dashboard.json` (local dev).

```bash
# 1. Build the frontend (production: load data from same origin)
cd frontend
npm install
VITE_DATA_URL=dashboard.json npm run build

# 2. Upload to S3
aws s3 sync dist/ s3://investment-system-data/dashboard/ --delete --region us-east-1
```

**First-time only** (enables public read for the dashboard prefix only):

```bash
# Allow a bucket policy to grant public access (dashboard prefix only)
aws s3api put-public-access-block --bucket investment-system-data --region us-east-1 \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# Attach policy: public GetObject for dashboard/*
aws s3api put-bucket-policy --bucket investment-system-data --region us-east-1 \
  --policy file://infrastructure/dashboard_bucket_policy.json

# Enable static website hosting
aws s3 website s3://investment-system-data --index-document index.html --region us-east-1
```

Access the dashboard at: `http://investment-system-data.s3-website-us-east-1.amazonaws.com/dashboard/`

---

## Go-Live Order

Use this sequence for a full first-time go-live:

1. **S3 bucket** – From repo root: `./infrastructure/s3_setup.sh investment-system-data us-east-1`
2. **Secrets** – `./infrastructure/secrets_setup.sh us-east-1` (enter OpenAI, FRED, Alpha Vantage keys)
3. **Lambda** – `./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1`
4. **EventBridge** – `./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline us-east-1`
5. **Dashboard (optional)** – First-time bucket policy and static website per "Deploy Frontend Dashboard" above; then `VITE_DATA_URL=dashboard.json npm run build` and `aws s3 sync dist/ s3://investment-system-data/dashboard/ --delete --region us-east-1`
6. **Verify daily pipeline** – Invoke Lambda once (see "Verify Deployment" above); check `daily/latest.json`, `daily/<date>/*`, and `dashboard/dashboard.json` in S3.
7. **After enough daily data (e.g. 30+ days)** – Run training: `python training/train.py --bucket investment-system-data --region us-east-1`; then evolution: `python evolution/evolve.py --bucket investment-system-data --generations 25`.
8. **Monthly automation** – Edit the launchd plist path to point to your repo’s `automation/run_training.sh`, then run `./automation/install_launchd.sh`.

---

## Cost Notes

- **NEVER enable S3 versioning**
- Lambda: ~$0.20/day at current usage (84s execution, 3GB memory)
- S3: Minimal (lifecycle deletes data after 365 days)
- Total estimated: < $10/month
