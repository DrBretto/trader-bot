## Context
Define a concrete implementation plan for a new offline-only champion–challenger optimizer that uses the live decision path (`src/steps/decision_engine.py:429` + `src/signals/regime_fusion.py:19`) and only optimizes parameters marked `safe_to_optimize=true` in discovery artifacts.

## Plan
- [x] Re-read discovery artifacts and verify live decision/config load/promote paths in source files.
- [x] Extract exact optimization scope from `optimizer/discovery/param_inventory.json` using `safe_to_optimize` and `used_live` flags.
- [x] Write `docs/OPTIMIZER_PLAN.md` with exact modules to add, deterministic walk-forward evaluation design, fitness/guardrails/promotion, persistence, UI/API surfaces, scheduling, rollback.
- [x] Write `optimizer/contracts/live_params_contract.md` defining active param loading and atomic champion promotion artifact semantics.
- [x] Write `optimizer/contracts/run_artifacts_contract.md` defining run outputs, schemas, statuses, lineage, and retention.
- [x] Validate the three deliverable files exist and summarize findings in console response.

## Execution Log
- Read `docs/PLAN.md`, latest `docs/plans/`, and `docs/POSTMORTEMS.md` for project workflow alignment.
- Parsed `optimizer/discovery/pipeline_map.json`, `optimizer/discovery/data_inventory.json`, and `optimizer/discovery/param_inventory.json`.
- Confirmed in code that live config currently loads from `config/decision_params.json` and `config/regime_compatibility.json` in `src/handler.py:76-96`.
- Confirmed current evolution promotion writes those files directly (`evolution/evolve.py:185-188`, `evolution/promotion.py:234-238`), which motivates an active-pointer contract.
- Wrote deliverables:
  - `docs/OPTIMIZER_PLAN.md`
  - `optimizer/contracts/live_params_contract.md`
  - `optimizer/contracts/run_artifacts_contract.md`
- Verified deliverable files exist and contain concrete implementation contracts/plan details.

## Follow-ups
- Implement optimizer modules and CLI only after this plan/contracts are approved.
- Add schema validation for optimizer artifacts and active param pointer in implementation phase.
