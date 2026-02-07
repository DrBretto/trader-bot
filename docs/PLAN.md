# Investment System - Implementation Plan

## Goals

Build a fully autonomous daily investment decision system that:

- Ingests market data from free APIs (Stooq, FRED, GDELT)
- Generates buy/sell signals using AI models + qualitative risk assessment
- Executes paper trades with full transparency
- Displays performance in an impressive React dashboard
- Runs completely hands-off with monthly model retraining
- Expands into a multi-expert market intelligence engine (see `docs/PHASE3_PROPOSAL.md`)

## Constraints

- Daily resolution only (no intraday)
- Paper trading only (no real money)
- AWS cost < $20/month
- Local training (MacBook, monthly, automated)
- Prefer free/cheap data sources; paid sources require explicit approval
- Avoid brittle scrapers in the daily Lambda path unless strongly guarded (timeouts, retries, neutral fallbacks)

## Architecture

```text
Daily (10pm ET):
  EventBridge → Lambda → [Ingest → Features → Inference → Decisions → LLM] → S3 Artifacts

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

### Phase 6: Market Intelligence Expansion (Expert Regime Engine) 🚧 PLANNED

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

---

## Non-Goals

- Real money trading (paper only)
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
