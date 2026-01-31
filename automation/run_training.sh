#!/bin/bash
#
# Monthly training wrapper script for launchd
#
# Runs ML model training and evolutionary search, then uploads to S3.
# Called by launchd on the 1st of each month at 2am.
#

set -e

# Derive project root from script location (portable)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_FILE="/tmp/investment-training-$(date +%Y%m%d).log"
S3_BUCKET="investment-system-data"
AWS_REGION="us-east-1"

# Ensure we're using the correct Python
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "========================================" >> "$LOG_FILE"
echo "Training started at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$PROJECT_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Set PYTHONPATH so modules can be found
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Step 1: Run ML model training
echo "[$(date)] Starting ML model training..." >> "$LOG_FILE"
python training/train.py \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION" \
    --max-days 365 \
    --epochs 100 \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] ML training completed successfully" >> "$LOG_FILE"
else
    echo "[$(date)] ML training failed!" >> "$LOG_FILE"
    # Send alert
    "$SCRIPT_DIR/send_alert.sh" "ML Training Failed" "Check logs at $LOG_FILE"
    exit 1
fi

# Step 2: Run evolutionary search (optional, skip if no historical data)
echo "[$(date)] Starting evolutionary search..." >> "$LOG_FILE"
python evolution/evolve.py \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION" \
    --population 30 \
    --generations 25 \
    --max-days 365 \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] Evolution completed successfully" >> "$LOG_FILE"
else
    echo "[$(date)] Evolution failed (non-critical)" >> "$LOG_FILE"
    # Evolution failure is non-critical, continue
fi

# Step 3: Check costs
echo "[$(date)] Checking AWS costs..." >> "$LOG_FILE"
python "$SCRIPT_DIR/check_costs.py" --budget 20 >> "$LOG_FILE" 2>&1

echo "========================================" >> "$LOG_FILE"
echo "Training completed at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Send success notification
"$SCRIPT_DIR/send_alert.sh" "Monthly Training Complete" "Training finished successfully. Check dashboard for updates."

exit 0
