# Deployment Guide

This document covers deployment procedures for the investment system.

## Prerequisites

- AWS CLI configured with `personal` profile
- Docker installed (for building Lambda layers)
- API keys stored in Secrets Manager (one-time setup)

## Environment

- **Region**: us-east-1
- **S3 Bucket**: investment-system-data
- **Lambda Function**: investment-system-daily-pipeline
- **EventBridge Rules**: investment-system-daily-trigger (night), investment-system-morning-trigger (morning)
- **SNS Topic**: investment-system-alerts

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

The pipeline runs in two phases: night analysis (10 PM ET) and morning execution (9:45 AM ET). Both are set up by a single script:

```bash
./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline investment-system-data us-east-1
```

This creates two rules:
- **Night** (`investment-system-daily-trigger`): 3 AM UTC Tue-Sat (10 PM ET Mon-Fri)
- **Morning** (`investment-system-morning-trigger`): 14:45 UTC Mon-Fri (9:45 AM ET Mon-Fri)

To disable both:
```bash
aws events disable-rule --name investment-system-daily-trigger --region us-east-1
aws events disable-rule --name investment-system-morning-trigger --region us-east-1
```

---

## Set Up Email Alerts

```bash
./infrastructure/sns_setup.sh us-east-1 drbretto82@gmail.com
```

**Important**: Confirm the subscription by clicking the link in the email you receive. Alerts are sent on every pipeline run (night summary, morning execution report) and on errors.

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

# 2. Upload to S3 (--exclude protects data files written by the pipeline)
aws s3 sync dist/ s3://investment-system-data/dashboard/ \
  --exclude "dashboard.json" \
  --exclude "timeseries.json" \
  --exclude "timeseries.parquet" \
  --exclude "data/*" \
  --delete --region us-east-1
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
4. **EventBridge** – `./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline investment-system-data us-east-1` (creates both night + morning rules)
5. **SNS Alerts** – `./infrastructure/sns_setup.sh us-east-1 drbretto82@gmail.com` (confirm subscription via email)
6. **Dashboard (optional)** – First-time bucket policy and static website per "Deploy Frontend Dashboard" above; then build and sync per the commands in that section (includes `--exclude` flags to protect data files).
7. **Verify daily pipeline** – Invoke Lambda once (see "Verify Deployment" above); check `daily/latest.json`, `daily/<date>/*`, and `dashboard/dashboard.json` in S3.
8. **After enough daily data (e.g. 30+ days)** – Run training: `python training/train.py --bucket investment-system-data --region us-east-1`; then evolution: `python evolution/evolve.py --bucket investment-system-data --generations 25`.
9. **Monthly automation** – Edit the launchd plist path to point to your repo's `automation/run_training.sh`, then run `./automation/install_launchd.sh`.

---

## Cost Notes

- **NEVER enable S3 versioning**
- Lambda night run: ~$0.20/day (110s execution, 3GB memory)
- Lambda morning run: ~$0.02/day (~10-15s execution, 3GB memory)
- S3: Minimal (lifecycle deletes data after 365 days)
- SNS: Free tier (1M publishes/month free)
- Total estimated: < $10/month
