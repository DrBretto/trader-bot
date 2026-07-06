"""monitors/ — substrate + three-line watchdogs (clean spine).

- ``substrate`` — the content-currency freeze-signature check (P4, VERBATIM from
  ``src/brain/monitors.py``).
- ``canary_gate`` — the post-pipeline reality-canary hook (P7).
- ``watchdog`` — the three-line watchdog + daily health email + missed-run
  heartbeat (P8): canon + SPY + challenger each advanced, populated, not stale;
  ALWAYS emails a ✓/✗ status; SNS on a missed-run/stale line.
"""
from .substrate import LocalReplayStore, check_substrate_fresh  # noqa: F401
from .canary_gate import run_post_pipeline_canaries  # noqa: F401
from .watchdog import (  # noqa: F401
    run_daily_health_check,
    check_canon_line,
    check_spy_line,
    check_challenger_line,
    emit_run_heartbeat,
    daily_health_handler,
    STALE_TRADING_DAYS,
)
