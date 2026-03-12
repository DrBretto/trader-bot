# Optimizer Discovery (Wise Offline Evolution)

## Context

Need a discovery-only artifact set for designing a champion-challenger, walk-forward evolutionary optimizer with hard promotion gates. Scope is strictly repo fact-finding: map live decision path, enumerate tunable parameters in-path, inventory historical data and offline evaluation reality, identify guardrails/scheduling/integration points.

## Plan

- [x] Re-orient on project docs (`docs/PLAN.md`, latest plan doc, `docs/POSTMORTEMS.md`, `CLAUDE.md`)
- [x] Build exact decision pipeline map (entrypoints, function-level flow, key I/O)
- [x] Build complete decision-path parameter inventory (excluding NN weights)
- [x] Inventory historical data assets and local time coverage
- [x] Audit offline evaluation harness/backtest reality and runnable commands
- [x] Audit guardrails/promotion safety/scheduling mechanisms
- [x] Audit UI/API integration points and define minimal additions needed
- [x] Write deliverables:
  - [x] `docs/OPTIMIZER_DISCOVERY.md`
  - [x] `optimizer/discovery/param_inventory.json`
  - [x] `optimizer/discovery/data_inventory.json`
  - [x] `optimizer/discovery/pipeline_map.json`

## Execution Log

- Created branch `codex/feat/optimizer-discovery`.
- Confirmed pre-existing unrelated local modification in `.claude/settings.local.json` (left untouched).
- Completed session orientation pass per `CLAUDE.md` requirements.
- Mapped live decision pipeline and call graph from `src/handler.py` into `optimizer/discovery/pipeline_map.json`.
- Built complete decision-path parameter inventory (253 entries) including config keys, fusion/signal thresholds, ensemble uncertainty knobs, and cost-model assumptions in `optimizer/discovery/param_inventory.json`.
- Profiled local historical datasets under `training/data/*.parquet` and captured runtime S3 dataset expectations in `optimizer/discovery/data_inventory.json`.
- Documented offline evaluation commands, current guardrails, missing safety mechanisms, and minimal UI/artifact integration plan in `docs/OPTIMIZER_DISCOVERY.md`.

## Follow-ups

- Implement optimizer only after user confirms discovery artifacts are sufficient.
