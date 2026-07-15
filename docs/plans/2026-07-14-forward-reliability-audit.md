# 2026-07-14 - Forward reliability audit

## Context

The production bot has repeatedly appeared healthy while one link in the daily chain
was stale, disconnected, or publishing from the wrong source. The July 13 repair
promoted replayed TILT to canon, reconstructed the two-stage comparison, separated
operational pointers from chart publication, and restored holiday-aware execution.
This audit must now prove the whole autonomous chain will continue working after that
repair, rather than accepting one successful cycle as sufficient evidence.

## Plan

- [x] Reconcile production dates and values across settled prices, intents, morning
  execution, portfolio state, promoted ledger, dashboard JSON, and shadow JSON.
- [x] Verify every EventBridge rule, target payload, Lambda deployment setting,
  recent invocation, CloudWatch alarm, heartbeat, and SNS/watchdog path.
- [x] Audit the code for swallowed failures, stale hard-coded prefixes, calendar
  mistakes, pointer coupling, duplicate writers, and checks that can pass against
  their own contaminated source.
- [x] Run focused and broad automated checks; add regression coverage and repair any
  forward-reliability defects found.
- [ ] Commit and push the audit evidence and fixes. If code changes are required,
  deploy after the documented checkpoint and verify the live next-run path.

## Execution Log

- 2026-07-14: Reloaded `CLAUDE.md`, the roadmap, recent line/freshness/watchdog
  plans, deployment guide, and relevant postmortems. Unrelated worktree changes are
  present and will remain untouched.
- 2026-07-14: Confirmed all public line mirrors agree through July 13: solid-blue
  TILT `$114,983.92395947811`, dotted-yellow two-stage `$114,310.89774002078`, and
  SPY `$108,605.49389097832`. The ledger, dashboard line overlay, shadow mirror,
  and headline value agree; no contaminated-value reversion is present.
- 2026-07-14: Found the public page was only partially current. Its model lines had
  advanced, but the operational snapshot, holdings, regime, and trades were still
  dated July 8 because morning publishing required a retired `signals.parquet`
  payload. Clean nights intentionally stopped producing that artifact.
- 2026-07-14: Found the health job ran before the sole 04:30 replay writer and
  allowed a one-session grace period, making a missed line append structurally
  green. It also treated an absent substrate fingerprint history as green and could
  substitute unrelated attribution series when the yellow comparison was absent.
- 2026-07-14: Found morning retries could execute the same simulated intents twice,
  failed S3 writes were often ignored, caught phase failures returned successful
  Lambda invocations, the morning UTC schedule was wrong during standard time, and
  the deployed schedule/IAM state was not fully reproducible from the repository.
- 2026-07-14: Added resumable morning checkpoints, stable execution IDs plus legacy
  fill deduplication, strict read/write failures, current intent-derived dashboard
  fields, true Lambda failure propagation, DST-safe morning scheduling, a single
  reproducible replay schedule, post-replay and post-morning health checks, and
  CloudWatch alarms including a missing-health-heartbeat alarm.
- 2026-07-14: Applied the same prepared/completed checkpoint contract to midday.
  A total quote outage is no longer reported as a zero-action success, every state
  write is checked, and a retry resumes the exact computed portfolio instead of
  executing a skipped-buy reattempt twice.
- 2026-07-14: Found a second line defect during the required no-write replay. July
  13 was appended using the deterministic forward regime, but the next replay
  reclassified it with a later recorded regime. Blue was unchanged by coincidence;
  yellow moved by `$36.229888916015625`, so the next append would have been refused.
- 2026-07-14: Fixed regime lineage at the frozen reference frontier (July 10):
  reconstructed history remains recorded, while July 13 and every later promoted
  session permanently use the deterministic forward picker. Each run also clears
  warm Lambda replay state and now verifies every stored blue, SPY, and yellow point
  to the cent, not only the terminal.
- 2026-07-14: A fresh isolated no-write replay verified all 20 post-split sessions
  and all 60 stored series values with zero mismatches. Both July 13 terminals now
  reproduce exactly; the real-session grid is complete and no write was planned.
- 2026-07-14: Automated verification currently passes: 38 core tier-1 tests plus
  one expected xfail, 32 focused ledger/calendar/publish tests, and the commit
  canary tier (34 passed, one expected xfail). Shell syntax, Python compilation,
  workflow/IAM parsing, and diff checks are
  clean. The external checker correctly remains red until the pending live repair:
  lines stop at July 13 after today's close and the operational snapshot is July 8.

## Follow-ups

- Commit and push only the owned audit slice, then deploy the image and reconcile
  schedules/alarms at the documented live-change checkpoint.
- After deployment, advance the July 14 settled replay, republish the already-run
  July 14 morning state through a prepared checkpoint without re-executing fills,
  and require the public checker plus both internal health modes to return green.
- The public GitHub watchdog workflow becomes scheduled only when these changes
  reach the default branch; AWS health checks and alarms are the immediate live
  independent path.
