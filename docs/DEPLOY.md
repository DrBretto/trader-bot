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

# 2. Store API keys in Secrets Manager (OpenAI, FRED, Alpha Vantage)
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

# 2. Upload to S3 (--exclude protects data files written by the pipeline;
#    shadow_timeseries.json is written nightly by the dual forward shadow job)
aws s3 sync dist/ s3://investment-system-data/dashboard/ \
  --exclude "dashboard.json" \
  --exclude "timeseries.json" \
  --exclude "timeseries.parquet" \
  --exclude "shadow_timeseries.json" \
  --exclude "data/*" \
  --delete --region us-east-1

# 3. MANDATORY: invalidate the CloudFront edge cache for index.html.
# Without this the deploy is silently broken — see the caution below.
aws cloudfront create-invalidation --distribution-id E10EHVNQ0CELM2 \
  --paths "/*" --profile personal
```

> ⚠️ **CAUTION — you MUST run the CloudFront invalidation (step 3) on every
> frontend redeploy.** This is not optional housekeeping; skipping it produces a
> blank white page for users.
>
> **Why:** Vite emits content-hashed asset filenames (`assets/index-<hash>.js`),
> and the S3 sync above runs with `--delete`, so the *previous* build's bundle is
> removed from the bucket. But `index.html` has a fixed name and is served under
> CloudFront's default cache behavior (`Managed-CachingOptimized`, 24h edge TTL).
> After a new build, the edge keeps serving the **old** `index.html` for up to 24h
> — and that old HTML points at a JS hash that no longer exists in S3. The browser
> requests the missing bundle, S3/CloudFront returns the SPA fallback `index.html`
> with `Content-Type: text/html`, the browser refuses it ("Expected a
> JavaScript-or-Wasm module script…"), `#root` stays empty, and the page is blank.
> Because it's an *edge-cache* effect it is intermittent and per-edge-node, so it
> can look fine from one machine and broken from another.
>
> **Fix / prevention:** always invalidate after `s3 sync` (step 3). `/*` is safe
> because asset names are content-hashed; the only thing that actually needs
> dropping is the stale `index.html`. To verify a deploy is healthy, load the site
> and confirm `#root` is non-empty with no console MIME errors (see
> `frontend/diag-runtime.mjs`). Incident: 2026-06-07 (model-promotion redeploy left
> stale HTML cached → blank page). See `docs/POSTMORTEMS.md`.

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

## CloudFront cache policy split (live data vs static assets)

The public dashboard is served via CloudFront distribution `E10EHVNQ0CELM2` (alias `trader-bot.infotrope.io`). The distribution has two cache behaviors with different policies:

- **Default behavior** — hashed Vite static assets (JS/CSS bundles, images). Uses AWS managed `Managed-CachingOptimized` (`658327ea-f89d-4fab-a63d-7e88639e58f6`). 24h edge TTL is fine because the asset filenames change on every build.
- **Ordered behavior `*.json`** — live pipeline-written data files (`/dashboard.json`, `/data/dashboard.json`, `/timeseries.json`, `/data/timeseries.json`). Uses AWS managed `Managed-CachingDisabled` (`4135ea2d-6df8-44a3-9df3-4b5a84be39ad`). MinTTL=MaxTTL=DefaultTTL=0; query strings are not part of the cache key but each request goes to origin.

This split exists because the frontend's `useDashboardData.ts` cache-busts requests with `?t=Date.now()`, but `Managed-CachingOptimized` ignores query strings, so the cache-bust did not reach the edge. With the `*.json` behavior on `Managed-CachingDisabled`, every fetch reaches origin and serves the current snapshot.

If you redeploy or recreate the distribution, recreate this split. To verify it is in place:

```bash
aws cloudfront get-distribution-config --id E10EHVNQ0CELM2 --profile personal \
  --query 'DistributionConfig.CacheBehaviors.Items[?PathPattern==`*.json`].CachePolicyId' --output text
# expect: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
```

After any pipeline run that produces a fresh `dashboard.json`, normal browser refresh on `https://trader-bot.infotrope.io` should load the current snapshot — no manual invalidation required because the live data behavior is on `Managed-CachingDisabled`.

If a new live data path is introduced that does not match `*.json` (e.g. `/data/something.parquet`), either rename it to `.json` or add another ordered cache behavior under the same disabled-cache policy.

---

## Go-Live Order

Use this sequence for a full first-time go-live:

1. **S3 bucket** – From repo root: `./infrastructure/s3_setup.sh investment-system-data us-east-1`
2. **Secrets** – `./infrastructure/secrets_setup.sh us-east-1` (enter OpenAI, FRED, Alpha Vantage)
3. **Lambda** – `./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1`
4. **EventBridge** – `./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline investment-system-data us-east-1` (creates both night + morning rules)
5. **SNS Alerts** – `./infrastructure/sns_setup.sh us-east-1 drbretto82@gmail.com` (confirm subscription via email)
6. **Dashboard (optional)** – First-time bucket policy and static website per "Deploy Frontend Dashboard" above; then build and sync per the commands in that section (includes `--exclude` flags to protect data files).
7. **Verify daily pipeline** – Invoke Lambda once (see "Verify Deployment" above); check `daily/latest.json`, `daily/<date>/*`, and `dashboard/dashboard.json` in S3.
8. **After enough daily data (e.g. 30+ days)** – Run training: `python training/train.py --bucket investment-system-data --region us-east-1`; then evolution: `python evolution/evolve.py --bucket investment-system-data --generations 25`.
9. **Monthly automation** – Edit the launchd plist path to point to your repo's `automation/run_training.sh`, then run `./automation/install_launchd.sh`.

---

## Remove Broker Env Vars (Alpaca removal, 2026-06-08)

This system is now a pure simulation. The live Lambda `investment-system-daily-pipeline` must have its broker environment variables **removed**. As of this removal, the live function still carried:

- `BROKER_MODE=alpaca_paper`
- `BROKER_TRADING_ENABLED=true`

Remove both from the Lambda environment configuration. With them gone, the pipeline runs simulated-only (the simulated fill at the open is the new forward point on the line).

```bash
# Inspect current env vars
aws lambda get-function-configuration \
  --function-name investment-system-daily-pipeline --region us-east-1 \
  --query 'Environment.Variables'

# Update the environment to drop BROKER_MODE and BROKER_TRADING_ENABLED,
# preserving the remaining keys (S3_BUCKET, AWS_REGION, alert vars, etc.).
```

### How to restore Alpaca

The two Secrets Manager secrets are **PRESERVED, not deleted** — they are the saved copy:

- `investment-system/alpaca-paper-key-id`
- `investment-system/alpaca-paper-secret-key`

To revisit Alpaca later:

1. Re-add the Lambda env vars `BROKER_MODE=alpaca_paper` and `BROKER_TRADING_ENABLED=true` (exact values recorded above).
2. Re-add the broker code from git history — restore from this packet's removal commit on branch `ai/alpaca-removal-continuous-line`.

---

## Cost Notes

- **NEVER enable S3 versioning**
- Lambda night run: ~$0.20/day (110s execution, 3GB memory)
- Lambda morning run: ~$0.02/day (~10-15s execution, 3GB memory)
- S3: Minimal (lifecycle deletes data after 365 days)
- SNS: Free tier (1M publishes/month free)
- Total estimated: < $10/month
