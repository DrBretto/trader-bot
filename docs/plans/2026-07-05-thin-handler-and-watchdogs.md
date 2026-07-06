# P8 — Thin handler + three-line watchdogs

## Context
Clean-rebuild step P8 (dossier `05_CLEAN_REBUILD_ARCHITECTURE.md` / `06_SEQUENCED_PACKET_PLAN.md`).
Give `trader-bot-core/` a clean forward-running entrypoint and the always-on
watchdogs that make a freeze/degrade impossible to miss — the check that was
missing during the two-week silent freeze. Does NOT cut production over (P9 owns
cutover + removal).

## Plan
- `trader-bot-core/app/handler.py` — a THIN router (pure dispatch, no pandas/torch
  compute inline): night / morning / midday / ops-probe / canary / daily-health.
- `trader-bot-core/app/night.py` — the night forward path the router dispatches to
  (`run_cutover(production_forecaster)` → intents → `lines.append` → `publish_line`
  → post-pipeline canary + health + heartbeat). Emits `trade_intents.json` in the
  exact schema the morning executor consumes (`engine.build_trade_intents`).
- `trader-bot-core/app/morning.py` / `midday.py` — PRESERVED paths (delegate to the
  prod phases; zero morning-path change).
- `trader-bot-core/app/ops_probes.py` — governed, non-destructive diag branches
  (forecast-diag, freshness-diag, regime-diag, canary, watchdog-diag).
- `trader-bot-core/monitors/watchdog.py` — the three-line watchdog + daily health
  email (canon + SPY + challenger each advanced, populated, not stale) + missed-run
  heartbeat. Substrate freeze-signature check reused verbatim from `monitors.substrate`.
- Schedule: `.github/workflows/daily-health-watchdog.yml` (nightly + post-pipeline,
  runs the clean check against live S3 from CI — no prod cutover) and
  `infrastructure/eventbridge_daily_health_setup.sh` (P9-ready Lambda-native rule).

## Execution Log
- Built all modules; import-verified; handler thinness confirmed by grep.
- Reality-test (governed invoke, live S3, NOT mocked): the daily health email shows
  ✓ for all three lines (canon, SPY, challenger) at the real settled day 2026-07-02;
  a deliberately-skipped-night injection trips the missed-run/stale alarm (✗ email +
  real SNS `MessageId 82c7cfbf-df53-5309-90d7-ad7ad84fd34b`).
- trade_intents schema confirmed unchanged (all 7 keys); morning reads
  generated_date + actions.

## Follow-ups
- P9: point the prod Lambda at `app.handler`, observe a full live night+morning,
  apply `eventbridge_daily_health_setup.sh`, then execute the removal manifest.
- P9: land a first-class clean-core portfolio loader (night.py uses the shared
  `src.steps.paper_trader` loader as a documented seam until then).
