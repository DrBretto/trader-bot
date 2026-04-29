#!/bin/bash
# Uninstall weekly launchd schedule for local offline optimizer runs.

set -euo pipefail

PLIST_NAME="com.traderbot.optimizer.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

if launchctl list | grep -q "com.traderbot.optimizer"; then
  launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

if [ -f "$PLIST_DST" ]; then
  rm "$PLIST_DST"
fi

echo "Uninstalled launchd job: com.traderbot.optimizer"
