# 2026-07-13 - Promote TILT to canon and repair autonomous line execution

## Context

The live yellow two-stage line is not trustworthy at its July 9-10 frontier. July 9
was calculated from the July 8 portfolio state divided by itself, and July 10 used
the stale July 8 state divided by July 9, producing the exact displayed -1.8748597%
artifact. The simulator also executed July 2 intents on the July 3 market holiday
using a July 2 price bar relabeled as July 3.

The existing blue TILT line is independently reconstructed on the settled SPY
trading-day grid and has reproduced its stored frontier under replay. It must become
the canonical solid line and the source for total value, returns, drawdown, Sharpe,
and SPY-relative performance. The repaired former canon remains visible only as a
dotted yellow comparison.

## Plan

- [x] Add one replay-owned promoted ledger where `value` is TILT and `comparison`
  is the repaired two-stage line; bootstrap non-destructively from the current line
  and require cent-level continuity before any write.
- [x] Stop the night job from appending portfolio-state ratios; make the scheduled
  replay advance/repair job the only writer of both displayed model lines.
- [x] Route every night/morning/midday/dashboard/watchdog read through the same
  promoted ledger so no legacy duplicate can revert the site.
- [x] Make the NYSE calendar holiday-aware and skip morning execution on closed
  sessions; measure intent freshness in trading sessions.
- [x] Render canon as solid blue and the former canon as dotted yellow on desktop,
  mobile, and the model-comparison lens.
- [ ] Add regression coverage, run backend tests and the production frontend build,
  seed the promoted ledger, deploy, and verify the public JSON and page.

## Execution Log

- 2026-07-13: Reproduced the yellow July 10 error exactly from stale state:
  `97749.38444251234 / 99617.06465644056 - 1 = -1.8748597144%`.
- 2026-07-13: Confirmed July 3 was a market holiday but the morning simulator wrote
  14 fills using the July 2 bar under a July 3 label.
- 2026-07-13: Confirmed the scheduled replay used the real settled-session grid
  through July 10 and passed continuity before publishing blue TILT at $114,966.55.
- 2026-07-13: Seeded `canon/equity_ledger_tilt_canon_v1/` non-destructively: 255
  verified points through July 10. A second dry-run reproduced both the TILT and
  two-stage terminals exactly, found a complete grid, and planned zero writes.
- 2026-07-13: First deployed canary refused before replay because local `.claude`
  metadata had entered the baked seed cache. Seed hydration now excludes workspace
  metadata; no ledger or dashboard write occurred on the failed canary.

## Follow-ups

- None. This task is not complete until the live line and metrics are verified from
  the promoted ledger and the recurring schedules remain active.
