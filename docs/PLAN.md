# Investment System - Implementation Plan

## Goals

Build a fully autonomous daily investment decision system that:

- Ingests market data from free APIs (Stooq, FRED, GDELT)
- Generates buy/sell signals using AI models + qualitative risk assessment
- Executes trades with full transparency (simulated by default; broker mode optional)
- Displays performance in an impressive React dashboard
- Runs completely hands-off with monthly model retraining
- Expands into a multi-expert market intelligence engine (see `docs/PHASE3_PROPOSAL.md`)

## Constraints

- Daily resolution only (no intraday)
- Simulated trading by default; broker-connected execution is opt-in
- AWS cost < $20/month
- Local training (MacBook, monthly, automated)
- Prefer free/cheap data sources; paid sources require explicit approval
- Avoid brittle scrapers in the daily Lambda path unless strongly guarded (timeouts, retries, neutral fallbacks)

## Architecture

```text
Night (10 PM ET, Mon-Fri):
  EventBridge → Lambda {"source": "night-analysis"}
    → [Ingest → Features → Inference → Signals → Regime Fusion → Decisions → LLM Weather]
    → Save trade_intents.json (queued, not executed)
    → Update portfolio valuations at closing prices
    → Publish all artifacts + SNS email alert

Morning (9:45 AM ET, Mon-Fri):
  EventBridge → Lambda {"source": "morning-execution"}
    → Load trade_intents.json
    → Fetch morning prices via yfinance
    → Validate: freshness check, price gap check, re-evaluate stops
    → Execute validated trades at morning prices
    → Update portfolio + re-publish dashboard
    → SNS email alert

Monthly (1st, 2am):
  launchd → train.py → Models uploaded to S3

Anytime:
  User visits dashboard → S3 static site → Reads artifacts → Shows charts
```

---

## Market Intelligence Expansion – Readiness Decisions (Recorded)

This repo already ships a working daily pipeline + dashboard (Phases 1–5). The Phase 3 proposal
(`docs/PHASE3_PROPOSAL.md`) adds 4 new orthogonal “experts” and upgrades regime fusion.

To prevent ambiguous implementation and dashboard drift, this expansion will follow these decisions:

### Decision A — Fusion semantics: **Path B (replace final regime)**

- The system will compute a new **final regime** from:
  - existing regime model outputs (ensemble probabilities, disagreement)
  - new expert signals (macro/credit, vol uncertainty, fragility, entropy/shift)
- The **final regime label** becomes the single source of truth for:
  - decision filters and position sizing
  - paper trader trade records (for post-mortems)
  - dashboard “regime strip”
  - weather report context

### Decision B — What gets stored (learning-first)

To maximize curiosity/learnability, we will store both:

- **Expert outputs** (scores, labels, flags)
- **Expert raw inputs** used to compute them (so charts can show “why”)

This implies storing:

- yields (10Y, 3M, slope), credit proxy/spread inputs
- vol complex inputs (VIX, VVIX, SKEW, term structure metrics)
- cross-asset correlation / PCA diagnostics (avg corr, PC1/PC2 explained variance)
- entropy diagnostics (raw entropy, z-score, consecutive-days counter)

### Decision C — Time-series storage strategy (visualization without high Lambda cost)

We will avoid rebuilding long time series by scanning many `daily/<date>/...` keys every run.

Artifacts will be:

- **Per-day** (already exists): `daily/<date>/context.parquet`, `daily/<date>/inference.json`, etc.
- **Rolling time-series** (new): a compact “history table” updated daily with **one read + one write**:
  - `dashboard/timeseries.parquet` (last ~400 trading days)
  - This table powers all Phase 3 dashboard panels (regime strip, expert lines, thresholds, markers).

Rationale:

- Keeps Lambda I/O bounded (single small parquet read/write).
- Keeps S3 storage tiny (hundreds of rows × dozens of columns).
- Makes frontend development easy (single dataset for charts).

### Decision D — Monthly macro series handling

Monthly indicators (PMI, unemployment, CPI surprises) cannot use the existing 5-day forward-fill logic.

Plan:

- Introduce frequency-aware handling for macro series:
  - **Daily** series: forward-fill short gaps (weekends/holidays).
  - **Monthly** series: “as-of” carry-forward until next release, with a staleness guard (e.g., 45–60 days).

---

## Milestones

### Phase 1: Core Pipeline ✅ COMPLETE

- [x] Project structure and config files
- [x] Data ingestion (Stooq, FRED, GDELT)
- [x] Data validation
- [x] Feature engineering
- [x] Baseline regime model (rule-based)
- [x] Baseline health model (rank-based)
- [x] Decision engine
- [x] Paper trader
- [x] LLM integration (risk + weather)
- [x] Lambda handler
- [x] AWS infrastructure scripts
- [x] Unit tests (see `README.md` for current count)
- [x] Deploy to AWS and test end-to-end

### Phase 2: ML Models ✅ COMPLETE

- [x] Regime model (GRU + Transformer architectures)
- [x] Health model (Autoencoder + VAE architectures)
- [x] Training pipeline (train.py orchestrator)
- [x] Training data loaders
- [x] Model versioning (latest.json pointer)
- [x] Lambda integration (model loader with fallback)

### Phase 3: Frontend Dashboard ✅ COMPLETE

- [x] React + Vite setup
- [x] Hero metrics
- [x] Equity curve chart
- [x] Drawdown chart
- [x] Monthly returns heatmap
- [x] Weather report
- [x] Portfolio/candidates tables

### Phase 4: Evolutionary Search ✅ COMPLETE

- [x] PolicyGenome class
- [x] Fitness evaluation
- [x] Genetic algorithm
- [x] Template promotion

### Phase 5: Automation ✅ COMPLETE

- [x] launchd for monthly training
- [x] Cost monitoring
- [x] Error alerting
- [x] Documentation

### Phase 6: Market Intelligence Expansion (Expert Regime Engine) ✅ COMPLETE

Goal: expand from the current ensemble regime model into a **6-expert regime engine** with rich
time-series visualization, while preserving low operational cost and preserving/increasing prediction quality.

This milestone corresponds to `docs/PHASE3_PROPOSAL.md` (the “Phase 3 expansion” proposal document).

#### 6.1 Data ingestion expansion (daily)

- Add daily inputs required by the four new experts:
  - **Macro/credit**: 10Y yield (exists), **3M yield (new)**, HY credit proxy inputs (HYG/IEF/LQD exist as prices)
  - **Volatility complex**: VIX (exists via FRED), **VVIX (new)**, **SKEW (new)**, **VIX9D and/or VIX3M (new)** as available
  - **Cross-asset panel**: SPY/QQQ/IWM/TLT/HYG/GLD/EFA/EEM (already in universe)
- Source strategy (must be decided explicitly during implementation):
  - Prefer **FRED** when series exist and are stable.
  - Otherwise use a stable public market data source compatible with daily cadence.
  - All new fetches must be guarded (timeouts, neutral fallbacks, and validation).

#### 6.2 Expert signal modules (daily)

Add four expert computations producing one scalar output per day (plus diagnostics):

- `macro_credit_score ∈ [-1, +1]`
- `vol_uncertainty_score ∈ [0, 1]` and `vol_regime_label ∈ {calm, unstable_calm, panic}`
- `fragility_score ∈ [0, 1]` (corr + PCA absorption diagnostics)
- `entropy_score` + `entropy_shift_flag` (and entropy z-score diagnostics)

#### 6.3 Regime fusion v3 (Path B)

Create a `decide_regime_v3(...)`-style fusion layer that outputs:

- `final_regime_label`
- `regime_confidence`
- `position_size_modifier`
- `risk_throttle_factor`

Design constraints:

- **Hard overrides** for true panic conditions (panic probability and/or vol regime panic).
- **Caution gates** for fragility and entropy shift (cap exposure / reduce trust).
- Preserve existing “ensemble disagreement reduces sizing” concept (still valuable as uncertainty).

#### 6.4 Storage / schema (S3 artifacts)

Update artifacts to include:

- New daily fields in a stable location (preferred: `context.parquet` and/or a new `signals.parquet`)
- New rolling dataset:
  - `dashboard/timeseries.parquet` (append today’s row; keep last ~400 trading days)
  - Contains: final regime, per-expert scores, key raw inputs, and diagnostics/flags

#### 6.5 Dashboard expansion (learning + “show-off”)

Add panels driven by `dashboard/timeseries.parquet`:

- Regime strip + expert score mini-charts
- Macro/credit panel (score + yield slope + spread)
- Volatility complex panel (VIX/VVIX + label shading + score)
- Fragility panel (score + thresholds; optional heatmap on hover)
- Entropy panel (score line + event markers + SPY overlay)

Implementation note:

- The frontend currently consumes a single `dashboard.json`. Phase 6 will extend the dashboard data contract
  to include the time-series dataset (or a URL to it) without breaking local dev defaults.

#### 6.6 Backtest + reporting (avoid “toy backtest” drift)

Current state:

- Evolution has a backtester (`evolution/fitness.py`) that does **not** exactly match the live decision engine.

Phase 6 goal:

- Add a pipeline-faithful evaluation path for:
  - fragility gating impact
  - macro backdrop impact
  - entropy warnings vs drawdowns
  - regime accuracy vs forward SPY returns (report + confusion matrix where applicable)

Reports should be produced as artifacts so the dashboard can visualize them.

### Phase 7: Two-Phase Pipeline + Email Alerts ✅ COMPLETE

Goal: decouple night analysis from morning trade execution for realistic P&L, and add email alerting for pipeline monitoring.

#### 7.1 Two-Phase Lambda Routing

- [x] Night phase: full 12-step analysis pipeline, saves `trade_intents.json` instead of executing trades
- [x] Morning phase: loads intents, fetches morning prices via yfinance, validates, executes trades
- [x] Single Lambda function routed by `event.source` field
- [x] Backward-compatible: existing `"eventbridge-scheduled"` and `"manual"` sources run night phase

#### 7.2 Morning Execution Logic (`src/steps/morning_executor.py`)

- [x] Intent freshness validation (max 3 calendar days, handles weekends)
- [x] BUY validation: skip if price gap >5% from intent price
- [x] SELL validation: health collapse/panic/LLM veto always execute; trailing stops re-checked with morning price
- [x] BUY share counts recomputed at morning price (same dollar amount as intent)
- [x] Portfolio valuations always updated even when no intents found

#### 7.3 Morning Price Fetching (`ingest_prices.fetch_morning_quotes()`)

- [x] Uses yfinance for ~15 symbols (held positions + intent symbols + SPY)
- [x] Graceful per-symbol fallback (logs warning, skips failures)
- [x] Added `yfinance==0.2.36` to `requirements-lambda.txt`

#### 7.4 Email Alerts via SNS (`src/utils/sns_alerts.py`)

- [x] `[TraderBot]` prefix on all email subjects for Gmail filtering
- [x] Night summary: regime, intents queued, weather blurb
- [x] Morning summary: trades executed, validation log, portfolio value
- [x] Error alerts: phase, error message, CloudWatch log command
- [x] Never crashes the pipeline (all sends wrapped in try/except)

#### 7.5 Publishing Updates

- [x] Night run saves `trade_intents.json` and includes `intents_date` + `phase` in `latest.json`
- [x] Morning run publishes: portfolio_state, trades, morning_execution report, updated dashboard.json
- [x] Morning run does NOT re-publish: prices, features, inference, signals, timeseries, weather

#### 7.6 Infrastructure

- [x] SNS topic: `investment-system-alerts` with email subscription
- [x] EventBridge morning rule: `investment-system-morning-trigger` at 9:45 AM ET Mon-Fri
- [x] Lambda deploy scripts re-add EventBridge permissions (prevents silent permission loss)

---

### Phase 8: Alpaca Broker Integration ✅ COMPLETE

Goal: Add broker-connected execution (paper first, live later) with fractional/notional trading support.

- [x] Broker abstraction layer (`src/brokers/`) with simulated, alpaca_paper, and alpaca_live modes
- [x] Alpaca client adapter with notional buys, qty sells, account/position queries
- [x] Safety rails: kill switch, per-order notional cap, symbol allowlist, idempotent order IDs
- [x] Morning executor routes through broker adapter when enabled
- [x] Post-execution portfolio reconciliation from broker positions
- [x] Fractional-aware sizing (dollars as canonical, no whole-share floor for broker mode)
- [x] Smoke test script (`scripts/alpaca_paper_smoke_test.py`)
- [x] 40 new tests (broker router, Alpaca adapter, morning execution integration)
- [x] Updated secrets setup, deployment docs, operations guide

### Phase 9: Diagnostic + Recalibration (2026-04-29) — IN PROGRESS

Goal: Diagnose the algorithm's 9-point lag to SPY since 2026-04-02 (after 7 months of beating SPY by 7 points). Find every bug in the daily decision pipeline, fix with rollback safety, recalibrate, re-backtest. End state: algorithm fixed, backtested, golden, ready to ship.

Packet: `docs/plans/2026-04-29-deep-diagnostic-and-calibration-packet.md`

#### 9.1 Phase 1 — Bug audit ✅
- 16 findings (F-1 through F-16). Output: `docs/plans/2026-04-29-phase1-bug-audit-findings.md`.
- F-1, F-3, F-4 confirmed; F-2 reframed (tanh saturation, not ratchet).
- F-5 through F-10 surfaced from adjacent decision-pipeline code review.
- F-11 through F-16 surfaced from Phase-2 / Phase-3 cross-cutting analysis.

#### 9.2 Phase 2 — 26-field signal diagnostic ✅
- Output: `docs/plans/2026-04-29-phase2-signal-diagnostic.md`.
- 5 signals dead all-time, 3 dead until 2026-04-02, 12 stuck pre-pivot (F-11), 1 mislabeled (F-14), 1 effectively binary (F-15).
- Headline discovery: pre-2026-01-31 timeseries rows are structurally inhomogeneous from post-pivot rows. The bot held 100% cash for the first 125 of 161 "pre-hybrid" days. Prior audit's 7-point outperformance attribution was misleading.

#### 9.3 Phase 3 — P&L reconciliation ✅
- Output: `docs/plans/2026-04-29-phase3-pnl-reconciliation.md`.
- Headline metrics reconcile to the cent today. The $10k continuity gap is traced to the cutover bridge (`cumulative_external_cashflow = -10,077.83`).
- F-4 VUG bug was transient — present in 2026-04-28 snapshot, absent in 2026-04-29 snapshot — direct empirical confirmation of F-5 (two divergent valuation paths).
- Direct broker poll blocked: Alpaca paper API is not in this session's sandbox network allowlist; broker-truth comparisons require operator-side execution (3 calls itemized in Phase 3 §F).

#### 9.4 Phase 4 — Fix branches ✅ (on `ai/diagnostic-fix-batch`)
- 4 commits, each with focused tests:
  - `b591ccd` — F-1/F-3/F-6/F-8/F-9 vol_uncertainty degraded surfacing
  - `d195437` — F-4/F-5 unified cost basis in broker reconciliation
  - `73d3791` — F-7 regime-conditional fragility relax (opt-in, default off)
  - `f3e490a` — F-6 per-signal status flags in timeseries
- All Phase-4 tests pass. Pre-existing failure in `test_buy_uses_notional` (broker_status 'accepted' vs 'filled') is unrelated.

#### 9.5 Phase 5 — Calibration sweep ⛔ BLOCKED
- Output: `docs/plans/2026-04-29-phase5-calibration-recommendation.md`.
- Walk-forward 900-day verification cannot run: S3 has only 200 daily snapshots, of which 125 are F-11 backfill. The remaining 63 post-pivot days are too short for the harness's gate windows.
- Required: dataset rebuild via signal-replay against historical FRED/price/GDELT data (separate scope).
- First-cut calibration recommendation drafted (F-7 fragility-relax), pending the dataset rebuild + sweep.

#### 9.6 Phase 6 — Pre-deploy + shadow day ⛔ BLOCKED on Phase 5
- Operator-gated; cannot run shadow day until calibration sweep produces a recommendation. Per packet rule: "Phase 5 walk-forward gate must pass before Phase 6 begins. If it doesn't pass, that is the finding."

#### 9.7 Phase 8 — Postmortems ✅
- Three new entries in `docs/POSTMORTEMS.md`:
  - silent-fallback-signals class (the F-1, F-3, F-13 pattern)
  - one-way-ratchet class (the F-2 saturation pattern)
  - first-pass-audit-framing-trap (the prior-audit failure mode)

#### 9.8 Phase 10 — RETURN doc
- Output: `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md`.
- Does NOT yet end with the literal stop-condition line. Per packet's stop-condition arithmetic, the line is gated on Phase 5 walk-forward verification + Phase 6 shadow day. Both blocked. The RETURN doc surfaces the gap honestly and recommends the dataset-rebuild path forward.

### Phase 10: Fragility + Throttle Calibration Audit (2026-04-30)

Goal: re-validate the throttle pipeline's calibration after Phase 9 left F-2 fragility tanh-saturation as a still-open knob defaulted off. Question the metric, not just the gate parameters. Recalibrate normalizer constants against current empirical distribution. Default off; ship behind feature gate; operator-gated cutover.

Packet: `docs/plans/2026-04-30-fragility-and-throttle-calibration-audit-packet.md`
Branch: `ai/recal-fragility-norms-20260430`

#### 10.1 Phase 1 — Throttle distribution diagnostic ✅
- 1272-day empirical distribution of every multiplicative throttle and its inputs.
- Output: `docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md`
- Headline: `fragility_score` saturated (median 0.905, last 42d median 0.965); `avg_correlation` and `pc1_explained` misnormalized (current constants treat empirical median as +1.8σ).

#### 10.2 Phase 2 — Fragility baseline audit ✅
- Source-traced `AVG_CORR_MEAN=0.30`, `AVG_CORR_STD=0.15`, `PC1_MEAN=0.45`, `PC1_STD=0.12` to commit `b9317e1` (2026-02-06) with no documented source dataset.
- Empirically re-derived from 900d production-equivalent panel: `AVG_CORR_MEAN=0.4774`, `AVG_CORR_STD=0.0979`, `PC1_MEAN=0.6007`, `PC1_STD=0.0658`.
- Output: `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md`

#### 10.3 Phase 3 — Compound throttle cost ✅
- Three representative-day walk-throughs (rally 2026-04-23, panic 2026-03-25, choppy 2026-04-08).
- Output: `docs/plans/2026-04-30-phase3-compound-throttle-cost.md`
- Headline: recalibration alone does not produce rally relief because the current rally has correlations at historical extremes; **must be paired with F-7 relax** for the rally-window improvement to materialize.
- Independent finding: **F-2030-A** — `ensemble_multiplier` is double-applied in the position-size chain (`regime_fusion.py:262` × `decision_engine.py:440`). Estimated 15-25% over-throttle on every v3 trade. Logged for operator review; not silently fixed.

#### 10.4 Phase 4 — Recalibration recommendation + walk-forward sweep ✅
- Output: `docs/plans/2026-04-30-phase4-recalibration-recommendation.md`
- 7 variants tested. **`recalibrated_plus_F7_relax_on` strictly dominates `current_production` on both broad gate (-14.95% vs -17.97%) and rally (-31.21% vs -98.28%) windows**; panic return improved (+23.81% vs +13.26%), rally drawdown improved (-2.41% vs -9.98%), panic drawdown slightly worse (-1.54% vs -0.82%, still very small).
- Recalibration-only variant: +9pp gate improvement, no rally change. Operator-safe half-step.
- Code: `src/signals/fragility.py` adds `RECALIBRATED_2026_04_30` constant block with provenance fields; defaults preserved.
- Bundle: `config/decision_params.recalibrated_2026_04_30.json` shadow-only, not promoted.
- Test: `tests/test_fragility_calibration.py` (5 passing tests; 39 prior signal tests still green).
- Sweep harness: `scripts/run_recalibration_sweep_20260430.py` → `runs/recalibration_sweep_20260430.json`.

#### 10.5 Postmortems ✅
- One new entry: **stale-historical-norm class** (normalizer constants embedded once, never re-validated against current data).

#### 10.6 RETURN doc ✅
- Output: `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` ending with the literal stop-condition line.
- Operator action gated: promote `decision_params.recalibrated_2026_04_30.json` to active and (for full rally relief) flip `regime_fusion.fragility_relax_in_risk_on=True` after one-day shadow.

### Phase 11: Ensemble multiplier double-application investigation (2026-04-30)

Goal: implement F-2030-A from the 2026-04-30 fragility audit (claimed `ensemble_multiplier` is double-applied in the position-size chain). Default off; ship behind feature gate; operator-gated cutover.

Packet: `docs/plans/2026-04-30-ensemble-multiplier-double-application-fix-packet.md`
Branch: `ai/fix-ensemble-double-apply-20260430`

#### 11.1 Bug verification ✅
- Read `regime_fusion.py:262` (folds ensemble into `position_size_mod`) and `decision_engine.py:440-454` (applies `ensemble_adj` and `expert_adj`).
- **Reading the call sites** at `decision_engine.py:798` and `:828`: both pass `ensemble_multiplier=1.0` whenever `expert_signals is not None` — the v3 production defense.
- **Conclusion**: the function is mathematically susceptible to double-application but no production caller invokes the unsafe pattern. F-2030-A was a false positive from function-in-isolation analysis.

#### 11.2 Fix shipped as defense-in-depth ✅
- Branch: `ai/fix-ensemble-double-apply-20260430`. One focused commit.
- `src/steps/decision_engine.py`: added `ensemble_multiplier_already_applied` gate (default False) at lines 440-470. When True, forces `ensemble_adj=1.0` regardless of passed value. Comment rewritten to document the caller defense pattern.
- `tests/test_decision_engine.py`: 6 new tests under `TestEnsembleMultiplierCallerPatterns` lock both production caller patterns (v3 and v2), the unsafe pattern's susceptibility, and the gate's behavior.
- `scripts/run_ensemble_double_fix_sweep_20260430.py` + `runs/ensemble_double_fix_sweep_20260430.json`: 2x2 walk-forward sweep.

#### 11.3 2x2 walk-forward result ✅
- Output: `docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md`.
- All four cells (recal off/on × fix off/on) produce **byte-identical** trading results. The double-fix gate is a no-op under correct callers — empirical confirmation that the bug does not fire in production.

#### 11.4 Postmortem ✅
- New entry: **known-bug-documented-as-feature class** (and false-positive findings from function-in-isolation analysis). Captures: (a) "backward compat" comments are flags not closures; (b) function-in-isolation analysis is incomplete — every claim about runtime behavior must enumerate call sites; (c) a susceptibility is not a bug; (d) defense-in-depth from a false-positive finding is still useful when shipped honestly.

#### 11.5 RETURN doc ✅
- Output: `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` ending with the literal stop-condition line.
- **No operator action required.** The recalibration packet's recommendation (promote `decision_params.recalibrated_2026_04_30.json` ± F-7 relax) is unchanged. The double-fix gate stays off because turning it on changes nothing.

### Phase 12: Optimizer empirical-mutation fix (2026-04-30)

Goal: prevent future stale-historical-norm incidents from requiring an operator-driven audit. The 2026-04-30 fragility audit had to be done manually because the weekly optimizer (launchd Sundays 03:30) couldn't find empirical-distribution-based improvements. Four-part fix to the optimizer itself, gated, default off.

Packet: `docs/plans/2026-04-30-optimizer-empirical-mutation-packet.md`
Branch: `ai/optimizer-empirical-mutation-20260430`

#### 12.1 Phase 1 — Inventory bounds derivation + tagging ✅
- 8 fragility / macro normalization constants tagged `parameter_class: "normalization_constant"` with `empirical_statistic` mappings; bounds re-derived from p1 / p99 of the underlying input distribution.
- 12 vol_uncertainty percentile bins tagged but with `empirical_statistic: null` (raw VIX/VVIX/SKEW not in the optimizer's signal_row); bounds derived from historical knowledge.
- Output: `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md`.
- Commit `1715fc1`.

#### 12.2 Phase 2 — Empirical re-derivation candidate operator ✅
- Once per cycle, computes the live-data statistic for every tagged normalization constant and injects as one candidate genome. Competes under existing fitness + guardrail pipeline.
- Feature flag: `OptimizerConfig.enable_empirical_mutation`, default False.
- New CLI flag `--empirical-mutation-debug` runs the dry-run path: prints what the candidate would propose, exits without GA.
- 17 new tests in `tests/test_optimizer_empirical_mutation.py`.
- Commit `472c21c`.

#### 12.3 Phase 3 — Calibration-only guardrail path ✅
- When every gene that differs between champion and challenger is `parameter_class="normalization_constant"`, drop `min_round_trips_total`, `min_gate_round_trips`, `min_gate_win_rate`. Outcome-quality gates (max_drawdown, cost_ratio, min_fold_ann_return) stay strict.
- Resolves the cause of recent weekly runs rejecting 100% of challengers with `min_round_trips_total = 0` even when the calibration was directionally correct.
- 3 new tests in `tests/test_optimizer_guardrails.py`.
- Commit `672e999`.

#### 12.4 Phase 4 — Persistent-rejection alert ✅
- Counter in `runs/optimizer/rejection_streak.json`; SNS alert via existing `[TraderBot]` topic on 3rd consecutive rejection. Resets on promotion. Fires exactly once per streak.
- Alert body includes failed-guardrail breakdown and runbook directing operator to `--empirical-mutation-debug`.
- 8 new tests in `tests/test_optimizer_rejection_streak.py`.
- Commit `a1c6e39`.

#### 12.5 Verification gate ✅
- Dry-run output proposed `fragility.AVG_CORR_MEAN = 0.4847` (target 0.477 ± 0.02). PASSED.
- 31 new + existing optimizer tests green; no regressions in 39 signal tests, 16 decision-engine tests.

#### 12.6 RETURN doc ✅
- Output: `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` ending with the literal stop-condition line.
- Operator action gated: flip `enable_empirical_mutation: true` in `config/optimizer.yaml` after reviewing the dry-run output. The next weekly run will then propose the empirical candidate and (under the calibration-only guardrail path) promote it if it dominates.

## Non-Goals

- Intraday trading (daily resolution only)
- Complex derivative strategies
- Multi-account management

---

## Deployment Instructions

### Prerequisites

- AWS CLI configured with credentials
- Python 3.11+
- API keys: OpenAI, FRED, Alpha Vantage (all have free tiers)

### Deploy Phase 1

Use `docs/DEPLOY.md` as the canonical, up-to-date deployment procedure.

Notes:

- Prefer the **container-based Lambda deploy** when running PyTorch-backed inference (ensemble models).
- The legacy zip deploy is appropriate for baseline-only operation.

## Deviations

**Directory Rename**: Changed `lambda/` to `src/` because `lambda` is a Python reserved keyword, causing import failures. Documented in `docs/DEVIATIONS.md`.
