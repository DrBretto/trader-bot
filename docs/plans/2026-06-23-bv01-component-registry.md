# BV-01 — Component registry + no-hard-coding CI

## Context

Brain-visualizer FP-08 build packet 1 of 7 (PKT-TB-BV-01), authored from the
Infotropy `20260623_trader-bot-brain-visualizer-committee` DESIGN_DOSSIER. The
brain was retired-and-restored several times on 2026-06-23; `RentLedger.tsx`
hard-codes its 4 rungs (`LADDER_RUNGS` + `COMPONENT_LABEL`) — exactly the
brittleness to design out. Build the declarative **component registry** (the
single source of truth for what components the brain has and how each is
measured) + the **no-hard-coding CI**, render `RentLedger.tsx` off the registry,
generalize `migrate_state_books`, and fix the inverted `tilt_live.py` docstring.

NO measured numbers live in the registry — existence + how-it-would-be-measured
only; numbers join from the contribution ledger (BV-03) on the stable `id`.
This packet ships the backbone only — NOT the visualizer surface/cards/hooks
(BV-02..07).

## Plan

- [ ] `frontend/src/registry/component_registry.json` — declarative registry,
      every §1 component (Groups A–H incl. live-but-dark, comparison, retired),
      honest expected/instrumentation status, no measured numbers; + `ladder_books`
      describing the I,R,F,E,U ladder (parent + history_of) for the migration.
- [ ] `frontend/src/registry/componentRegistry.ts` — typed loader + helpers
      (`componentsInOrder`, `byId`, `ladderRungs`, `FORECAST_RUNG_ID`).
- [ ] `RentLedger.tsx` — render off the registry; delete `LADDER_RUNGS` +
      `COMPONENT_LABEL`; iterate `ladderRungs()`, join the ledger on stable `id`,
      honest `awaiting` fallback default.
- [ ] `useShadowData.ts` — `component` wire type → `string` (registry is the
      source of valid ids; no id enumeration in the frontend).
- [ ] `PerformanceChart.tsx` — replace the lone `'forecast'` literal with the
      registry-exported `FORECAST_RUNG_ID`.
- [ ] `src/brain/component_registry.py` — Python loader + `ladder_migration_plan`
      (derives from→to/seed map from `history_of`/`parent`).
- [ ] `shadow_lib.py` — `migrate_state_books` + `_migrate_ledgers` read the map
      from the registry (data-described) instead of the hand-written table; keep
      idempotent / forward-only / refuse-to-guess.
- [ ] `tilt_live.py` — correct the inverted docstring (two-stage = live since
      `ec9561d`; tilt = comparison line).
- [ ] `tests/test_component_registry_no_hardcoding.py` — the two CI rules:
      (a) no hard-coded component-id enumeration in frontend render modules;
      (b) writer-methods ⊆ registry, live-ladder-components ⊆ registry, registry
      carries no measured numbers, migration map matches the registry.
- [ ] Deploy (frontend S3 sync + invalidation) and run tests.

## Execution Log

- 2026-06-23: plan created; read-first list consumed (DESIGN_DOSSIER §1/§5,
  RentLedger, shadow_lib, tilt_live, DEPLOY, contracts). Confirmed via
  `runtime.run_cutover` + git `ec9561d` that the two-stage engine is live and
  the tilt is the comparison line (tilt_live docstring is inverted).

## Follow-ups

- BV-02..07 build the visualizer surface, the contribution ledger, hooks, cards,
  mute/solo, and aesthetics. BV-01 ships the registry + CI backbone only.
- `src/steps/publish_artifacts.py` carries a parallel stale comment (says tilt is
  live / two-stage retired) — out of this packet's named scope; surfaced as an
  executor_concern, not fixed here.
