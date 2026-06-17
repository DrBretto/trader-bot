#!/bin/bash
#
# Entry script for AWS Batch monthly training container.
# Runs ML training + evolutionary search, then exits (Batch terminates the instance).
#
set -euo pipefail

S3_BUCKET="${S3_BUCKET:-investment-system-data}"
AWS_REGION="${AWS_REGION_NAME:-us-east-1}"

echo "========================================"
echo "[$(date -u)] Monthly training started"
echo "  Bucket: $S3_BUCKET"
echo "  Region: $AWS_REGION"
echo "========================================"

# Preflight: frozen ORB-1 New Brain isolation (PKT-TB-014 / D-AUTO-20260616).
# Refuse to run if any training output path overlaps the frozen brain weights.
# (The Batch image also does not ship runs/ or brain/, so the frozen artifacts
# are absent in this container; this guard catches any future regression.)
echo "[$(date -u)] Preflight: frozen ORB-1 brain isolation check..."
python -c "from training.frozen_brain_guard import assert_frozen_brain_isolated; assert_frozen_brain_isolated('/tmp/models', '/tmp/evolution')"

# Step 1: ML model training (~30-60 min)
echo "[$(date -u)] Step 1/3: ML model training..."
python training/train.py \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION" \
    --max-days 365 \
    --epochs 100

echo "[$(date -u)] ML training complete"

# Step 2: Evolutionary search (~60-120 min, non-critical)
echo "[$(date -u)] Step 2/3: Evolutionary parameter search..."
python evolution/evolve.py \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION" \
    --population 30 \
    --generations 25 \
    --max-days 365 || echo "[$(date -u)] Evolution failed (non-critical, continuing)"

# Step 3: Cost check
echo "[$(date -u)] Step 3/3: AWS cost check..."
python automation/check_costs.py --budget 20 || true

echo "========================================"
echo "[$(date -u)] Monthly training complete"
echo "========================================"
