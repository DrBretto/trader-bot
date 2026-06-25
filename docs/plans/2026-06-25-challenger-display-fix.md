# 2026-06-25 — Challenger comparison line fix + provisional-frontier correction

Governed by `book-factory/runs/20260625_trader-bot-challenger-display-fix/PKT-TRADER-BOT-CHALLENGER-DISPLAY-FIX-V1.md`
(Infotropy governance spine). This doc is the subject-repo mirror.

## Context
The clean-core implementation (3e50686) made the main equity line a stored
append-only ledger (correct, stays) but deleted
`src/utils/three_line_replay/replay_engine.py`, which the challenger/shadow
producer (`runs/pkt_tb_007_orthogonal_brain/shadow/*`) still imports. The shadow
launchd job (`com.traderbot.shadow.plist`) was therefore failing with ImportError,
so the dotted-blue challenger line on the comparison charts was frozen at
2026-06-24. Separately, the ledger seed had included the un-settled 2026-06-25
provisional point ($116,126.12) as the frontier; it should be the last SETTLED
day (2026-06-24).

## Plan
- Restore `replay_engine.py` from history as the **challenger-isolated** re-sim
  engine. Prove it is NOT imported by any main-line path (`src/handler.py`,
  `src/steps/publish_artifacts.py`, `src/canon/*`).
- Re-run the shadow producer so `dashboard/shadow_timeseries.json` advances.
- Make the bottom comparison chart flexible: render the first available
  challenger series (never assume exactly `shadow_A` + `shadow_B`); no crash when
  a series is absent or N changes.
- Drop the un-settled 2026-06-25 provisional ledger leaf so the frontier is
  2026-06-24; the priors stay byte-identical; the 06-25 point lands later via the
  normal settled append.

## Execution Log
- Restored `src/utils/three_line_replay/replay_engine.py` (`git show
  3e50686^:...`). Import clean; `Position/Portfolio/_ohlc_for_date/_mark_to_close/
  _apply_cluster_cap/_latest_close_per_symbol` all present. Static check: no `src/`
  module imports it (only the `runs/` shadow + prototype paths do).
- Ran the shadow producer (`shadow_nightly.py`, AWS_PROFILE=personal). Published
  11 S3 objects; `shadow_A` advanced 7→8 points, last = 2026-06-25 (provisional).
- Added `pickChallengerSeries()` to `frontend/src/hooks/useShadowData.ts` and wired
  `PerformanceChart.tsx`, `PerformanceLenses.tsx`, `mobile/MobileChart.tsx` to it.
  `tsc` + `vite build` green.
- Frontier fix: dropped `canon/equity_ledger/points/2026-06-25/627c297275f5a239.json`
  (backed up first), rebuilt manifest+cache → frontier 2026-06-24, 247 leaves, 247
  priors byte-identical. Append-without-revert proven in-memory (247→248, priors
  byte-identical). Republished `dashboard.json` from the corrected ledger
  (cache-projection parity held; the deployed no-regression gate is intentionally
  overridden for this one authorized backward correction).
- Gated infra untouched (champion freeze 114772.39, universe.csv — no writes).

## Follow-ups
- The shipped night-append uses `run_date = datetime.now()` (UTC). The next
  scheduled night run will append a leaf dated by its own run date; if the operator
  wants the settled 2026-06-25 point to land specifically as 06-25, run a forced
  night run with `run_date=2026-06-25` before the next scheduled run. (Surfaced as
  an executor concern in the governance routing receipt.)
