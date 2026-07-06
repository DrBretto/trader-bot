# CU-04 — publish-time value-revert guard + watchdog challenger/SPY coverage + timing gaps

## Context

The publish gate (`src/steps/publish_artifacts.py:_verify_ledger_or_hold`) is
date-only / parity-against-self: it compares the rendered line to the SAME ledger it
reads and only checks the terminal DATE for regression. So the morning path silently
reverted the canon line to the contaminated `121147.52` (same date 2026-07-02, changed
value) with no block and no alarm. The watchdog (`monitors/watchdog.py`) is
detection-only, lagging (fires 04:00Z, before the day's 13:45Z morning + 18:00Z midday
publishes → a revert is blind ~24h; Monday has no run), and covered only canon `value`
— not challenger (G3) or SPY `benchmark` (G4).

## Plan

- **Prevention — publish-time guard** (`publish_artifacts.py`): `guard_publish_not_reverted`
  checks the rendered terminal's canon `value` + SPY `benchmark` against the corrected
  clean_v2 ledger (hard-pinned via `monitors.watchdog._ledger_terminal`) — refuse
  (a) a known-contaminated value or (b) a same-date divergence. Wired into
  `_verify_ledger_or_hold` AFTER the existing parity/date checks (adds on top; the gate
  returning `(False, …)` uses the existing hold+SNS path). Covers morning + night (both
  route through this gate). Midday does not publish the line, so it has no publish gate.
- **Detection — watchdog** (`watchdog.py`): `check_value_revert` refactored into a pure
  `evaluate_value_revert(dash, shadow, terminal)` covering canon + SPY + challenger; the
  04:00Z health email now shows per-line revert status.
- **Timing gaps** (`src/handler.py:_run_midday_check`): run `check_value_revert`
  in-cycle at midday (18:00Z Mon-Fri, AFTER the morning publish) and alarm on a revert —
  closes G2 (same-cycle) and G1 (Monday).
- **Reality-test** (`app/ops_probes.py`): `publish-revert-diag` probe — non-destructive,
  runs the REAL guard/gate against the REAL clean_v2 on S3 with injected
  contaminated/reverted values, fires a marked `[REALITY-TEST]` SNS on block, and proves
  the watchdog flags challenger + SPY reverts.

## Execution Log

- Implemented all 5 edits; local logic validation against real clean_v2 passed
  (guard blocks contaminated/reverted canon+SPY; watchdog flags SPY+challenger).
- Deployed container image; ran `publish-revert-diag` on the live Lambda. (see run receipt)

## Follow-ups

- Challenger publish-time revert is prevented by CU-02's shadow-publish (sources shadow_A
  from clean_v2 `comparison` + contaminated guard); CU-04 adds challenger DETECTION.
- The 04:00Z healthcheck cron stays TUE-SAT (CU-01 owns it); Monday coverage is via the
  in-cycle midday check, not a cron change.
