#!/bin/bash
# Install weekly launchd schedule for local offline optimizer runs.
#
# RETIRED 2026-06-17 (PKT-TB-014 / D-AUTO-20260616 decision 5): the weekly
# optimizer tuned the now-retired, entanglement-contaminated champion config.
# The live algorithm is the native two-stage New Brain engine, which is frozen
# (no EA optimization). Re-installing this scheduler is blocked on purpose.
# To genuinely revive it, remove this guard in a fresh decision.
echo "RETIRED: the weekly optimizer was retired per PKT-TB-014 (D-AUTO-20260616)." >&2
echo "The live New Brain engine is frozen and not EA-optimized. Install blocked." >&2
exit 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PLIST_NAME="com.traderbot.optimizer.plist"
PLIST_SRC="$SCRIPT_DIR/launchd/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
LOG_DIR="$HOME/Library/Logs/traderbot"
AWS_PROFILE_VALUE="${AWS_PROFILE:-}"

if [ -z "$AWS_PROFILE_VALUE" ]; then
  echo "ERROR: AWS_PROFILE is not set."
  echo "Set AWS_PROFILE to a valid configured profile before installing launchd."
  echo "Example: AWS_PROFILE=personal ./scripts/install_optimizer_launchd.sh"
  exit 1
fi

if command -v aws >/dev/null 2>&1; then
  if ! aws configure list-profiles 2>/dev/null | grep -qx "$AWS_PROFILE_VALUE"; then
    echo "ERROR: AWS profile '$AWS_PROFILE_VALUE' not found in local AWS config."
    echo "Run 'aws configure list-profiles' and retry with a valid profile."
    exit 1
  fi
fi

if [ ! -f "$VENV_PYTHON" ]; then
  echo "ERROR: Python virtual environment not found at $VENV_PYTHON"
  echo "Create it first: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PLIST_DST")"

if launchctl list | grep -q "com.traderbot.optimizer"; then
  echo "Unloading existing com.traderbot.optimizer job"
  launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

sed \
  -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
  -e "s|__VENV_PYTHON__|$VENV_PYTHON|g" \
  -e "s|__AWS_PROFILE__|$AWS_PROFILE_VALUE|g" \
  -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$PLIST_SRC" > "$PLIST_DST"

launchctl load "$PLIST_DST"

echo "Installed launchd job: com.traderbot.optimizer"
echo "Schedule: Weekly Sunday at 03:30 local time"
echo "AWS profile: $AWS_PROFILE_VALUE"
echo "Run now: launchctl start com.traderbot.optimizer"
echo "Logs: $LOG_DIR/optimizer.log and $LOG_DIR/optimizer.error.log"
