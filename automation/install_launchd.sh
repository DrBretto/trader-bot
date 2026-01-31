#!/bin/bash
#
# Install launchd job for monthly training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.investment-system.monthly-training.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
LOG_DIR="$HOME/Library/Logs/investment-system"

echo "Installing launchd job for investment system training..."
echo "Project directory: $PROJECT_DIR"

# Create log directory
echo "Creating log directory at $LOG_DIR..."
mkdir -p "$LOG_DIR"

# Unload existing job if present
if launchctl list | grep -q "com.investment-system.monthly-training"; then
    echo "Unloading existing job..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy plist to LaunchAgents with path substitution
echo "Copying plist to $PLIST_DST..."
sed -e "s|/Users/drbretto/Desktop/Projects/trader-bot|$PROJECT_DIR|g" \
    -e "s|/Users/drbretto/Library/Logs|$HOME/Library/Logs|g" \
    "$PLIST_SRC" > "$PLIST_DST"

# Load the job
echo "Loading job..."
launchctl load "$PLIST_DST"

# Verify
if launchctl list | grep -q "com.investment-system.monthly-training"; then
    echo "Job installed successfully!"
    echo ""
    echo "The training will run on the 1st of each month at 2:00 AM."
    echo ""
    echo "To run manually:"
    echo "  launchctl start com.investment-system.monthly-training"
    echo ""
    echo "To uninstall:"
    echo "  launchctl unload $PLIST_DST"
    echo "  rm $PLIST_DST"
else
    echo "Warning: Job may not have loaded correctly."
    echo "Check: launchctl list | grep investment"
fi
