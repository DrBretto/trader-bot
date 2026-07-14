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
- [x] Add regression coverage, run backend tests and the production frontend build,
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
- 2026-07-13: Focused backend coverage passed (13 tests), the TypeScript check and
  production Vite build passed, and Lambda image digest
  `sha256:9e047d3e1fa857be2454c794f1ae452663a256cca79f4b095c40a608c61b5b79`
  deployed successfully.
- 2026-07-13: Published frontend bundle `assets/index-BFt4SGXI.js`; CloudFront
  invalidation `IZ116AWM0AQ6GRS6G2IFI84HX` completed for `/*`. Browser inspection
  confirmed TILT as an undashed blue line (`#3b82f6`) and two-stage as dotted yellow
  (`#f59e0b`, `3 4`), with no browser warnings or errors.
- 2026-07-14: The enabled `advance-challenger` schedule autonomously appended the
  settled July 13 session. The immutable linked leaf is
  `cda1731b4c5ce5a9ce5ffaf0d21bac4809c016d738f14d8b72ec960a83e6d6de`:
  TILT `$114,983.92395947811`, two-stage `$114,310.89774002078`, SPY
  `$108,605.49389097832`. The manifest frontier, public dashboard, comparison
  payload, and rendered page all agree on July 13.
- 2026-07-14: Repaired `daily/latest.json` from stale July 8 state to the verified
  July 13 midday portfolio artifact. The deployed night/morning code now advances
  state and intent pointers independently of chart publication.
- 2026-07-14: The next autonomous morning cycle then advanced the pointer to July 14,
  loaded July 13 intents with `intents_stale=false`, and executed 11 trades. The
  chart correctly remained at the latest settled session (July 13), proving line
  publication and operational state advancement are now decoupled.
- 2026-07-14: The documented frontend `--delete` sync removed three historical
  `.bak` objects because the recipe protected active data but not backups. Active
  production data was not affected. S3 versioning was suspended and the exact
  backups were not recoverable; the deploy recipe now excludes `*.bak*`.

## Follow-ups

- The focused repair tests and production build pass. Several older broad tests
  still fail during collection because they import the already-removed legacy
  `src` package; that pre-existing relocation gap is outside this line repair.
