# OPTIMIZER PLAN

## Scope and Fixed Decisions
- This optimizer is offline-only and local CPU-only.
- Canonical live decision path for all candidate evaluations is fixed to:
  - `src/steps/decision_engine.py:429` (`run`)
  - `src/signals/regime_fusion.py:19` (`decide_regime_v3`)
- No neural-network weights are optimized.
- Only parameters with `safe_to_optimize=true` from `optimizer/discovery/param_inventory.json` are eligible.
- Promotion model is champion-challenger only: live params change only after hard guardrails pass and objective improvement is confirmed.

## Exact Files/Modules to Add

### Optimizer core (new)
- `optimizer/__init__.py`
  - Package marker.
- `optimizer/config.py`
  - Load optimizer runtime config, seeds, fold settings, and bucket/region.
- `optimizer/param_space.py`
  - Build candidate gene space directly from `optimizer/discovery/param_inventory.json` (`safe_to_optimize=true`, `used_live=true`).
  - Enforce bounds/types/dependencies from inventory.
- `optimizer/data_access.py`
  - Load historical daily artifacts from S3 with local cache:
    - `daily/<date>/features.parquet`
    - `daily/<date>/inference.json`
    - `daily/<date>/signals.parquet`
    - `daily/<next_date>/prices.parquet`
    - `config/universe.csv`
  - Build deterministic date index with only complete days.
- `optimizer/replay.py`
  - Deterministic historical replay runner.
  - Calls `decision_engine.run` once per day with reconstructed expert signal payload.
  - Executes resulting actions through `paper_trader.execute_trade` and marks portfolio daily with `paper_trader.update_portfolio_values`.
- `optimizer/walk_forward.py`
  - Construct fixed walk-forward folds and gate segment.
  - Run champion and challenger across same folds.
- `optimizer/fitness.py`
  - Compute fold metrics, aggregate objective, and stability penalty.
- `optimizer/guardrails.py`
  - Evaluate hard constraints for each fold and promotion gate segment.
- `optimizer/champion_challenger.py`
  - Evolution loop + champion baseline comparison + promotion decision.
- `optimizer/persistence.py`
  - Write run artifacts, run index, and lineage outputs.
- `optimizer/promote.py`
  - Atomic promotion writer for versioned params + active pointer swap.
- `optimizer/rollback.py`
  - Atomic rollback by pointer revert.
- `optimizer/cli.py`
  - Commands: `run`, `evaluate`, `promote`, `rollback`, `status`.

### Contracts/docs (new)
- `optimizer/contracts/live_params_contract.md`
- `optimizer/contracts/run_artifacts_contract.md`

### Automation (new)
- `automation/run_optimizer.sh`
  - Launchd-safe wrapper to run optimizer CLI.
- `automation/com.investment-system.weekly-optimizer.plist`
  - Weekly scheduled optimizer run.
- `automation/install_optimizer_launchd.sh`
  - Install/uninstall helper for optimizer launchd job.

### Monitoring surfaces (new)
- `frontend/src/hooks/useOptimizerData.ts`
  - Fetch optimizer index/detail/lineage static JSON.
- `frontend/src/components/OptimizerStatus.tsx`
  - Current active version + last run result + promotion state.
- `frontend/src/components/OptimizerRunDetail.tsx`
  - Fold metrics, gate checks, parameter deltas.
- `frontend/src/App.tsx`
  - Add Optimizer section.
- `frontend/src/types/index.ts`
  - Optimizer artifact types.

### Existing files to modify
- `src/handler.py`
  - Update config loading to read `config/decision_params.active.json` pointer first.
- `src/utils/transaction_costs.py`
  - Add deterministic RNG injection support for replay.
- `src/steps/paper_trader.py`
  - Accept replay timestamp/RNG inputs so offline replay is deterministic.
- `src/steps/publish_artifacts.py`
  - Publish optimizer index/detail/lineage static JSON for dashboard.

## Exact Optimization Scope (Include / Exclude)

### Include (all with `safe_to_optimize=true` and `used_live=true`)
Total included parameters: **236**.

Included categories (from inventory components):
- `candidate_scoring_regime_multiplier` (90)
- `cost_model` (57)
- `expert_signal_computation` (37)
- `regime_fusion_rules` (18)
- `position_sizing` (9)
- `decision_engine` (9)
- `ensemble_uncertainty` (8)
- `cash_reserve_gate` (5)
- `leveraged_risk_controls` (1)
- `dashboard_watchlist` (1)
- `buy_filter_gate` (1)

Included prefixes:
- `decision_params.*` (safe subset)
- `regime_compatibility.*`
- `regime_fusion.*`
- `macro_credit.*`
- `vol_uncertainty.*`
- `fragility.*` (safe subset)
- `entropy_shift.*`
- `ensemble.*`
- `decision_engine.*`
- `transaction_costs.*`

### Exclude (all with `safe_to_optimize=false`)
Total excluded parameters: **17**.

Excluded names:
- `baseline_health.health_weights.inverse_risk`
- `baseline_health.health_weights.momentum`
- `baseline_health.health_weights.rel_strength`
- `baseline_regime.vol_p40`
- `baseline_regime.vol_p50`
- `baseline_regime.vol_p85`
- `decision_params.leveraged_constraints.max_weight`
- `decision_params.leveraged_constraints.stop_multiplier`
- `decision_params.max_sector_weight`
- `decision_params.reduce_health_drop`
- `decision_params.sell_health_days`
- `fragility.PANEL_SYMBOLS`
- `llm_risk.max_symbols_checked`
- `llm_risk.prompt_confidence_adjustment_max`
- `llm_risk.top_candidate_count`
- `morning_execution.buy_price_gap_threshold`
- `morning_execution.max_intent_age_days`

## Exact Evaluation Dataset and Deterministic Walk-Forward Replay

### Dataset source
Use runtime daily artifacts from S3 (not local training parquet) because replay requires live decision inputs and execution prices.

Required per-date artifacts:
- `daily/<date>/features.parquet`
- `daily/<date>/inference.json`
- `daily/<date>/signals.parquet`
- `daily/<next_date>/prices.parquet` (for next-session execution)
- `config/universe.csv`

Date inclusion rule:
- Keep only dates that have all required files and a valid next trading day with prices.
- Sort strictly ascending.

### Replay path (deterministic)
For each simulation date `d`:
1. Reconstruct `expert_signals` payload from `daily/<d>/signals.parquet`.
2. Load `inference_output` from `daily/<d>/inference.json`.
3. Load `features_df` from `daily/<d>/features.parquet`.
4. Build config object with candidate params + candidate regime compatibility + simulated portfolio state + universe.
5. Call `src/steps/decision_engine.py:429` (`run`) with reconstructed `expert_signals` (so fusion uses `decide_regime_v3`).
6. Execute produced actions at next-day open prices from `daily/<d+1>/prices.parquet` using `paper_trader.execute_trade`.
7. Mark portfolio to next-day close with `paper_trader.update_portfolio_values`.
8. Append simulated `portfolio_state` and fill log for metric computation.

Determinism controls:
- Fixed run seed from CLI.
- Deterministic RNG object passed into transaction cost model.
- Deterministic timestamps in replay (derived from simulation date, not `datetime.now()`).
- Deterministic symbol/action ordering before execution.

### Walk-forward and gate segmentation
- Use last `max_days=900` complete trading days.
- Reserve newest `gate_days=63` as promotion holdout gate segment (never used for challenger ranking).
- On remaining data, build folds with:
  - `train_days=252`
  - `test_days=63`
  - `step_days=63`
- For each fold, replay full train+test chronologically; score only test segment.

## Exact Fitness, Guardrails, and Promotion Rule

### Fold metrics
Each fold test segment computes:
- Annualized return
- Sharpe ratio
- Max drawdown
- Win rate (realized round-trips only)
- Realized round-trip count
- Cost ratio = `cumulative_transaction_costs / traded_notional`

### Fold score
Normalize per fold:
- `sharpe_norm = (clip(sharpe, -1, 3) + 1) / 4`
- `calmar = annualized_return / max(abs(max_drawdown), 0.01)`
- `calmar_norm = (clip(calmar, -1, 4) + 1) / 5`
- `ret_norm = (clip(annualized_return, -0.50, 0.80) + 0.50) / 1.30`
- `win_norm = clip(win_rate, 0, 1)`

`fold_score = 0.35*sharpe_norm + 0.30*calmar_norm + 0.20*ret_norm + 0.15*win_norm`

### Candidate objective (walk-forward)
- `wf_mean = mean(fold_score)`
- `wf_stability_penalty = 0.20 * stdev(fold_score)`
- `wf_objective = wf_mean - wf_stability_penalty`

This is the required stability penalty to control overfitting.

### Hard guardrails (must pass all)
Applied to challenger and champion on each fold and on gate segment:
- `max_drawdown >= -0.25`
- `cost_ratio <= 0.015`
- Aggregate realized round-trips across walk-forward tests `>= 20`
- Gate-segment realized round-trips `>= 5`
- Gate-segment win rate `>= 0.45`
- No fold may have annualized return `< -0.20`

### Promotion rule (champion-challenger)
Promote challenger only if all are true:
1. Challenger passes all hard guardrails on walk-forward and gate segment.
2. `challenger_wf_objective >= champion_wf_objective + 0.01`.
3. `challenger_gate_score >= champion_gate_score + 0.01`.
4. `challenger_gate_max_drawdown >= champion_gate_max_drawdown - 0.02`.

Else: no promotion; champion remains active.

## Exact Persistence Layout (Runs + Lineage)

Local run root:
- `optimizer/runs/<run_id>/`

Required files:
- `run_manifest.json`
- `param_space_snapshot.json`
- `walk_forward_folds.json`
- `champion_metrics.json`
- `challenger_metrics.json`
- `gate_segment_metrics.json`
- `guardrail_results.json`
- `promotion_decision.json`
- `candidate_params_bundle.json`
- `generation_log.jsonl`

S3 mirror (same structure):
- `optimizer/runs/<run_id>/...`

Global indexes:
- `optimizer/runs/index.json`
- `optimizer/lineage/active_params_lineage.json`

Dashboard-consumed copies:
- `dashboard/data/optimizer_runs_index.json`
- `dashboard/data/optimizer_runs/<run_id>.json`
- `dashboard/data/active_params_lineage.json`

## Exact Backend Endpoints + Frontend Optimizer Page

Because this repo uses static JSON artifacts (no API server), endpoints are static object URLs:
- `GET /data/optimizer_runs_index.json`
- `GET /data/optimizer_runs/<run_id>.json`
- `GET /data/active_params_lineage.json`

Frontend additions:
- `useOptimizerData` hook fetches index/detail/lineage.
- `OptimizerStatus` shows active version, last run, promotion status, guardrail pass/fail.
- `OptimizerRunDetail` shows fold-by-fold metrics, stability penalty, guardrail checks, and challenger-vs-champion deltas.
- Add a dedicated Optimizer section in `App.tsx` below existing strategy panels.

## Exact Scheduling Approach (Always-On Local)
- Add weekly launchd job via:
  - `automation/com.investment-system.weekly-optimizer.plist`
  - `automation/run_optimizer.sh`
  - `automation/install_optimizer_launchd.sh`
- Schedule: Sundays at 03:30 local time.
- Command executed by wrapper:
  - `.venv/bin/python -m optimizer.cli run --bucket investment-system-data --region us-east-1 --population 40 --generations 30 --max-days 900 --train-days 252 --test-days 63 --step-days 63 --gate-days 63 --seed 20260217`
- Logs:
  - `~/Library/Logs/investment-system/optimizer.log`
  - `~/Library/Logs/investment-system/optimizer.error.log`

## Exact Rollback Workflow
1. Operator selects prior `active_version` from `optimizer/lineage/active_params_lineage.json`.
2. Run:
   - `.venv/bin/python -m optimizer.cli rollback --bucket investment-system-data --region us-east-1 --to-version <version_id> --reason "manual rollback"`
3. Rollback writes only `config/decision_params.active.json` (single-pointer swap).
4. Append rollback event to lineage and run index.
5. Trigger next night pipeline run; live loader picks up reverted active version automatically.

## Live-Bot Contract Summary
- Current loader uses `config/decision_params.json` + `config/regime_compatibility.json` in `src/handler.py:76-96`.
- Planned loader contract: read `config/decision_params.active.json` pointer first; fallback to legacy files if pointer missing/invalid.
- Optimizer promotion writes immutable version bundle first, then atomically swaps active pointer.
