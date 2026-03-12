## Context
Implement the offline "wise" evolutionary optimizer to run as a local one-shot job on macOS launchd. It must use the live decision path (`decision_engine.run` + `regime_fusion.decide_regime_v3`), enforce champion-challenger promotion safety with walk-forward + separate gate segment, provide atomic promotion/rollback plumbing, and expose run history to the frontend via persisted artifacts.

## Plan
- [ ] Read relevant existing codepaths before edits: live config loading, decision path, transaction costs, paper trader, publish/dashboard hooks.
- [ ] Add optimizer config and core modules (param scope from discovery, data loading, deterministic replay, walk-forward folds, fitness, guardrails, champion-challenger orchestration).
- [ ] Add optimizer CLI (`python -m optimizer.cli run --config config/optimizer.yaml` + rollback).
- [ ] Wire live loader to `config/decision_params.active.json` and implement candidate/active/history promotion semantics.
- [ ] Add persistence under `runs/optimizer/<run_id>/...`, index, lineage, and dashboard JSON mirrors.
- [ ] Add frontend Optimizer view/panel + hooks to display summary/history/detail/diffs.
- [ ] Add launchd scripts/plist for weekly one-shot scheduling.
- [ ] Add runbook docs + minimal validation tests/checks.
- [ ] Run checks/tests and summarize install/uninstall/run commands + artifact locations.

## Execution Log
- Branch created: `codex/feat/offline-ea-optimizer-local`.

## Follow-ups
- If any production behavior conflict appears (e.g., strict active-only load), document migration path in runbook.
