# 1) Executive Summary (what exists vs missing)

## What exists

- Live decision flow is centralized in `src/handler.py` night pipeline, with decision output produced by `src/steps/decision_engine.py:429` (`run`).
- Regime fusion (post-model expert overrides/gates) is explicit in `src/signals/regime_fusion.py:19` (`decide_regime_v3`) and returns final regime, throttle, and effective exposure multiplier.
- Canonical dashboard performance/risk/trade lifecycle metrics are already consolidated in `src/utils/dashboard_metrics.py` and published from `src/steps/publish_artifacts.py` with snapshot IDs.
- Existing offline evolution tooling exists (`evolution/evolve.py`, `evolution/fitness.py`, `evolution/genome.py`, `evolution/promotion.py`).
- Existing local scheduler pattern exists via macOS `launchd` (`automation/com.investment-system.monthly-training.plist`, `automation/run_training.sh`).

## What is missing for the "wise" optimizer design target

- No pipeline-faithful offline replay harness that reuses the live decision path end-to-end (data -> inference -> expert signals -> fusion -> decisions -> costs/P&L) with walk-forward windows.
- Existing evolution backtester in `evolution/fitness.py` is a parallel simulator with parameters that do not match the current live decision config schema.
- Promotion gates are incomplete: `TemplatePromoter` defines max drawdown/min trades thresholds in constructor but current `validate()` only enforces fitness + sharpe + non-negative calmar.
- No atomic active-config swap mechanism; promotion currently writes directly to `config/decision_params.json` and `config/regime_compatibility.json` keys in S3.
- No optimizer monitoring artifacts or UI surfaces yet (current dashboard is strategy-state oriented only).

# 2) Decision Pipeline Map (with file/function pointers)

Primary machine-readable map: `optimizer/discovery/pipeline_map.json`.

## Live entrypoints

- Lambda router: `src/handler.py:105` -> `lambda_handler`
- Night path: `src/handler.py:131` -> `_run_night_phase`
- Morning path: `src/handler.py:355` -> `_run_morning_phase`
- Decision output entrypoint: `src/steps/decision_engine.py:429` -> `run`

## Night call graph (ordered)

1. Config + state load
- `src/handler.py:147` -> `load_config_from_s3` (`src/handler.py:63`)
- Loads: universe, decision params, regime compatibility, latest portfolio state

2. Data ingestion + validation
- Prices: `src/handler.py:168` -> `src/steps/ingest_prices.py:161` (`run`)
- FRED: `src/handler.py:173` -> `src/steps/ingest_fred.py:142` (`run`)
- Vol indices (VVIX/SKEW): `src/handler.py:178` -> `src/steps/ingest_prices.py:62` (`fetch_stooq_index`)
- GDELT: `src/handler.py:187` -> `src/steps/ingest_gdelt.py:88` (`run`)
- Validation: `src/handler.py:192` -> `src/steps/validate_data.py:9` (`run`)

3. Feature + expert signal generation
- Features/context: `src/handler.py:203` -> `src/steps/build_features.py:15` (`run`)
- Expert signals: `src/handler.py:212` -> `src/signals/compute_signals.py:62` (`run`)
  - Macro/Credit: `src/signals/macro_credit.py:24` (`compute_macro_credit`)
  - Vol uncertainty: `src/signals/vol_uncertainty.py:36` (`compute_vol_uncertainty`)
  - Fragility: `src/signals/fragility.py:26` (`compute_fragility`)
  - Entropy shift: `src/signals/entropy_shift.py:13` (`compute_entropy_shift`)

4. Model outputs
- Inference step: `src/handler.py:224` -> `src/steps/run_inference.py:13` (`run`)
- Model loader: `src/models/loader.py:214` (`load_models`), `src/models/loader.py:365` (`predict_regime`), `src/models/loader.py:470` (`predict_health`)
- Ensemble uncertainty/multiplier: `src/models/ensemble_regime.py:15` (`EnsembleRegimeModel`)

5. LLM risk overlay
- `src/handler.py:230` -> `src/steps/llm_risk_check.py:207` (`run`)
- Produces per-symbol `structural_risk_veto` and `confidence_adjustment` used by decision engine filters/sizing

6. Fusion + decision + order generation
- Decision engine: `src/handler.py:237` -> `src/steps/decision_engine.py:429` (`run`)
- Fusion: `src/steps/decision_engine.py:481` -> `src/signals/regime_fusion.py:19` (`decide_regime_v3`)
- Candidate scoring: `src/steps/decision_engine.py:19` (`score_candidates`)
- Buy filtering: `src/steps/decision_engine.py:86` (`filter_buy_candidates`)
- Sell evaluation: `src/steps/decision_engine.py:140` (`evaluate_holdings`)
- Sizing: `src/steps/decision_engine.py:259` (`compute_position_size`)
- Outputs: `decisions.regime`, `decisions.actions`, `decisions.expert_metrics`

7. Persistence + dashboard publish
- Trade intents: `src/handler.py:252` -> `daily/<date>/trade_intents.json`
- Portfolio valuation/stats: `src/handler.py:258` -> `src/steps/paper_trader.py`
- Artifact publish: `src/handler.py:283` -> `src/steps/publish_artifacts.py:417` (`run`)
  - Snapshot metadata: `src/steps/publish_artifacts.py:12`
  - Dashboard snapshot/timeseries: `src/steps/publish_artifacts.py:648`, `src/steps/publish_artifacts.py:628`

## Morning execution path (post-intent)

- Entrypoint: `src/steps/morning_executor.py:163` (`run`)
- Freshness/gap/revalidation: `src/steps/morning_executor.py:33`, `:44`, `:61`
- Trade execution: `paper_trader.execute_trade` (`src/steps/paper_trader.py:52`)
- Morning dashboard republish: `src/steps/publish_artifacts.py:667`

# 3) Parameter Inventory Summary (counts by category; link to JSON)

Full inventory: `optimizer/discovery/param_inventory.json`.

## Inventory totals

- Total parameters discovered in decision path scope: **253**
- `used_live=true`: **248**
- `used_live=false`: **5** (present in config but unused in live path)
- `safe_to_optimize=true`: **236**
- `safe_to_optimize=false`: **17** (ops/safety policy knobs or fallback-only behavior)

## Counts by component category

- `candidate_scoring_regime_multiplier`: 90
- `cost_model`: 57
- `expert_signal_computation`: 38
- `regime_fusion_rules`: 18
- `decision_engine`: 12
- `position_sizing`: 9
- `ensemble_uncertainty`: 8
- `model_fallback`: 6
- `cash_reserve_gate`: 5
- `leveraged_risk_controls`: 3
- `llm_risk_overlay`: 3
- `execution_validation`: 2
- `buy_filter_gate`: 1
- `dashboard_watchlist`: 1

## Unused config keys (explicitly marked `used_live=false`)

- `decision_params.max_sector_weight`
- `decision_params.sell_health_days`
- `decision_params.reduce_health_drop`
- `decision_params.leveraged_constraints.max_weight`
- `decision_params.leveraged_constraints.stop_multiplier`

# 4) Offline Evaluation Capability (exact commands; gaps)

## What can run today

1. Existing evolution loop (S3-backed historical build-from-daily)
```bash
source .venv/bin/activate
python evolution/evolve.py \
  --bucket investment-system-data \
  --region us-east-1 \
  --population 30 \
  --generations 25 \
  --max-days 365
```
Source refs: `evolution/evolve.py:272`, `evolution/evolve.py:84`, `training/utils/data_loader.py:75`.

2. Existing evolution local dummy smoke test
```bash
source .venv/bin/activate
python evolution/evolve.py --local-test --population 30 --generations 25
```
Source ref: `evolution/evolve.py:285`.

3. Full live pipeline manual invocation (single-date operational run, not historical replay)
```bash
source .venv/bin/activate
python src/handler.py --bucket investment-system-data --region us-east-1 --phase night
python src/handler.py --bucket investment-system-data --region us-east-1 --phase morning
```
Source ref: `src/handler.py:492`.

## Gaps vs required "wise offline optimizer"

- No walk-forward harness exists that repeatedly runs live decision logic across historical dates.
- Existing `evolution/fitness.py` backtester does not match live decision/fusion stack and uses a different parameter schema (`health_collapse_threshold`, `momentum_weight`, `profit_take_threshold`, etc.).
- No built-in command that guarantees canonical live metric definitions (cashflow-adjusted returns, live fill-cost model, live fusion ordering) during optimizer evaluation.

# 5) Data Inventory (time coverage; gaps)

Full inventory: `optimizer/discovery/data_inventory.json`.

## Local historical datasets present in workspace

1. `training/data/historical_combined.parquet`
- Coverage: 2014-12-10 to 2026-02-03 (2803 rows)
- Contains context + GDELT aggregate fields

2. `training/data/historical_context.parquet`
- Coverage: 2014-12-10 to 2026-02-03 (2803 rows)
- Contains context inputs (SPY returns/vol, rates, slope, credit/risk-off proxies, sentiment tone)

3. `training/data/historical_gdelt.parquet`
- Coverage: 2015-02-18 to 2026-02-04 (4005 rows)
- Contains GDELT aggregate sentiment features + availability flag

## Runtime datasets expected (S3-backed, not present locally in this workspace snapshot)

- `daily/<date>/prices.parquet`
- `daily/<date>/context.parquet`
- `daily/<date>/features.parquet`
- `daily/<date>/inference.json`
- `daily/<date>/decisions.json`
- `daily/<date>/trade_intents.json`
- `daily/<date>/trades.jsonl`
- `daily/<date>/portfolio_state.json`
- `daily/latest.json`
- `dashboard/dashboard.json` / `dashboard/data/dashboard.json`
- `dashboard/timeseries.parquet` / `dashboard/data/timeseries.json`

## Data gaps for optimizer design

- No local daily artifact history under `daily/` in repo checkout.
- No local dashboard snapshot/timeseries JSON in workspace for offline UI-level validation.
- Fill/order/portfolio longitudinal history is runtime S3 data, not committed local fixtures.

# 6) Safety/Guardrails (what’s present; what’s absent)

## Present

- Data quality guardrails with degraded mode + critical fail stop:
  - `src/steps/validate_data.py:37`, `:62`, `:66`
  - `src/utils/data_validation.py`
- Decision constraints:
  - max positions, min cash reserve, min order, trailing stops, leveraged hold cap in `src/steps/decision_engine.py`
- Regime safety overlays:
  - hard panic/unstable overrides + fragility/entropy gates in `src/signals/regime_fusion.py`
- Morning execution validation:
  - stale intents, buy gap threshold, trailing-stop re-check in `src/steps/morning_executor.py`
- Canonical metric robustness:
  - cashflow-adjusted returns + reset-boundary handling + sharpe minimum observations in `src/utils/dashboard_metrics.py`
- Existing reconciliation tests:
  - `tests/test_dashboard_metrics.py`

## Absent or incomplete

- Config schema validation for `decision_params` / `regime_compatibility`: **absent**
- Atomic config swap + rollback pointer for active strategy params: **absent**
- Promotion gate enforcement for max drawdown/min trades in current promoter `validate()`: **absent/incomplete**
  - thresholds exist in constructor (`evolution/promotion.py:44-45`) but are not applied in validation logic (`evolution/promotion.py:65-96`)
- Pipeline-faithful offline walk-forward evaluator: **absent**
- Explicit kill-switch circuit breaker in live decision engine: **absent**

# 7) Scheduling Plan (how to run offline on always-on PC)

## Current pattern

- Existing local scheduler uses macOS `launchd`:
  - plist: `automation/com.investment-system.monthly-training.plist`
  - installer: `automation/install_launchd.sh`
  - runner: `automation/run_training.sh`

## Simplest consistent addition for optimizer job

1. Add `automation/run_optimizer.sh` (wrapper command for offline optimizer run).
2. Add `automation/com.investment-system.optimizer.plist` with desired cadence.
3. Reuse install pattern from `automation/install_launchd.sh` (path substitution + `launchctl load`).
4. Write logs to `~/Library/Logs/investment-system/optimizer.log` and persist run artifacts to S3 dashboard data prefix.

Rationale: this matches existing always-on local scheduling conventions already used by monthly training.

# 8) UI/API Integration Plan (where to add; minimal endpoints)

## Existing backend/frontend pattern

- No backend HTTP API service/router is present; dashboard is static data from S3 JSON/parquet.
- Frontend data hooks:
  - `frontend/src/hooks/useDashboardData.ts`
  - `frontend/src/hooks/useTimeseriesData.ts`
- Existing monitoring panels/components:
  - `frontend/src/components/HeroMetrics.tsx`
  - `frontend/src/components/EnsembleStatus.tsx`
  - `frontend/src/components/RegimeStrip.tsx`
  - `frontend/src/components/TradeLog.tsx`

## Minimal new surfaces needed for optimizer observability

Because the project uses static dashboard artifacts, "endpoints" should be additional published JSON artifacts:

1. Optimizer run index
- `dashboard/data/optimizer_runs_index.json`
- Contents: run_id, timestamp, challenger/champion ids, status, promotion outcome, summary metrics

2. Optimizer run detail
- `dashboard/data/optimizer_runs/<run_id>.json`
- Contents: walk-forward folds, OOS metrics, gate checks, rejected reasons, parameter deltas

3. Active parameter lineage
- `dashboard/data/active_params_lineage.json`
- Contents: current active params, parent version, promotion history, rollback links

## Minimal UI placement

- Add one new dashboard section (single page app, no router currently required):
  - `Optimizer Status` card in `frontend/src/App.tsx`
- Optional split into components:
  - `frontend/src/components/OptimizerStatus.tsx` (index summary)
  - `frontend/src/components/OptimizerRunDetail.tsx` (selected run details)

# 9) Open Questions (only items truly unknown after repo scan)

1. What is the actual retained depth/coverage of `daily/<date>/...` artifacts in production S3 for walk-forward windows (not inferable from local workspace alone)?
2. Should optimizer scheduling run daily, weekly, or monthly on the always-on PC (current repo only encodes monthly training cadence)?
3. Should promotion be automatic on gate pass or require manual confirmation in this repo’s operating model (current evolution supports auto-promote but lacks full hard-gate set)?
