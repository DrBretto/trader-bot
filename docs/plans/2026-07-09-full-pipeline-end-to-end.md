# 2026-07-09 — Full pipeline end-to-end consolidation

## Context
Consolidation drive: make the app work all the way through on clean-core, no legacy
`src` path, all guards live, holding across real daily cycles.

## Plan
1. Substrate freshness (CL-708001) — already restored (feeds/produce.py); verify live.
2. No legacy path (CU-05): `git mv src -> trader-bot-core/chassis`, rewrite every
   `from src`/`import src` -> `chassis` repo-wide, rebuild `app/morning`+`app/midday`
   as first-class clean-core entrypoints (config loader relocated to
   `chassis/config_loader.py`), fix `parents[N]` repo-root depth (+1), drop
   `COPY src/` from the Dockerfile, rebake (chassis-clean), `from src == 0` on image.
3. Guards: CU-04 publish-time value-revert guard (`guard_publish_not_reverted` /
   `_verify_ledger_or_hold`) already wired on night/morning/republish + midday
   in-cycle detection; CU-03 `correct()` + manifest rebuild present; watchdog
   scheduled (healthcheck-trigger cron 0 4). Verify live.
4. IAM: `cloudwatch:PutMetricData` (namespace `TraderBot/*`) already granted; verify
   `BrainNightOK` lands, no AccessDenied.
5. Recurrence canary (CL-708150): new `monitors/config_canary.py` fails loud if
   `regime_compatibility` or `theta_sel.regime_admissibility` is empty-but-critical;
   wired into the night post-pipeline + an ops probe `config-canary` with an
   `inject_empty` lever to prove it fires.

## Execution Log
- `src` relocated to `trader-bot-core/chassis` (git mv), 129 import stmts rewritten
  across 54 files; 0 residual `from src`. parents[2]->parents[3] on 8 repo-root
  computations + bake `_REPO`. app/morning + app/midday rebuilt clean-core;
  config_loader relocated. Dockerfile drops `COPY src/`; deploy BAKE_PATHS updated.
  Rebake chassis-clean (0 from src). Local: all 20 entrypoints import; config canary
  live-ok + fires on injection; correct()+manifest present; tier1 suite 29 passed.
- Deploy + full-cycle proof: see run receipt
  `Infotropy Book/book-factory/runs/20260709_trader-bot-full-pipeline-end-to-end/`.

## Follow-ups
- Yahoo v8 429 from AWS IP (yfinance fallback carries) — CL-708201, feed hardening.
