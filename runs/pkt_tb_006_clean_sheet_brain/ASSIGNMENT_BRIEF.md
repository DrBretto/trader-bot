# ASSIGNMENT BRIEF — PKT-TB-006 Clean-Sheet Trader's Brain (Phase 0)

**Author:** Analyst (panel role 1) — 2026-06-10
**Audience:** Phase A designers working BLIND to the incumbent strategy (sealed-incumbent rule).
**Status of this document:** the ONLY repo-derived document you get. Everything here is data,
infrastructure, harness contract, and evidence rules. Nothing here describes the incumbent's
strategy logic, models-as-strategy, signals, or parameters — by design. Do not go looking.

All S3 facts below were verified live on 2026-06-10 against `s3://investment-system-data`
(us-east-1) unless marked "per docs". All local file facts were verified by reading the cited
file.

---

## 1. The assignment (condensed from the packet)

Design, build, and prove a NEW trading system — a "trader's brain." The operator's actual
assignment **outranks raw P&L**: an ML showcase in which each organ demonstrably carries
weight. Six mandatory design items (every one load-bearing; measurable marginal contribution
required from each):

1. **An ensemble of genuinely different ML model types** — different architectures doing
   different jobs, not two networks voting on one label. At least one **transformer** doing
   real work.
2. **An evolutionary algorithm as the balancing organ** — evolution tunes/evolves how the
   brain weighs its parts (allocator parameters, model trust weights, ensemble composition —
   the panel decides where evolution bites hardest).
3. **An LLM doing real sentiment analysis** — reading actual text (news, GDELT-fed or
   otherwise sourced free) and emitting structured signal the brain consumes nightly. "Not a
   veto garnish, not a narrative paragraph: an input organ whose removal measurably hurts."
4. **GDELT as a load-bearing differentiated data source** — event structure, actors, themes,
   tone dynamics — used in a way generic retail bots are not. Honest evidence-backed
   no-signal is acceptable; silent omission is not.
5. **A learned meta-evaluator — the brain's executive** — takes all model outputs and signals
   and makes the final decision-grade calls: position sizing, capital deployment, how much to
   trust which model today. Trained on decision-grade targets (realized forward utility of
   choices), NOT on regime labels. "The intelligence being showcased is allocation, not
   labeling."
6. **Infotropy angle examined** — a panel role attempts a concrete, computable mechanism from
   the Infotropy canon. Honest no-transfer is acceptable; forced mysticism is not.

**The named failure mode this packet exists to break:** every prior attempt collapsed into a
report about the existing system. A design that reads as a critique, refactor, or incremental
extension of the incumbent is a FAILED return regardless of quality. You are inventing, not
renovating. The incumbent appears in your world exactly once: as a score line in the eventual
bake-off.

**Operating envelope (hard):** ≤$10/month AWS target, hard fail above ~$15/month, itemized
cost worksheet mandatory. Daily cadence (nightly batch + morning execution, or justify a
deviation inside the envelope). Training local-monthly on the operator's Mac (launchd
pattern); Lambda handlers stay thin; free data only; network allowlist applies; no new
persistent paid infrastructure.

---

## 2. Tradable universe

Source: `config/universe.csv` (verified by direct read; identical copy lives at
`s3://investment-system-data/config/universe.csv`). Columns:
`symbol,asset_class,sector,leverage_flag,eligible`. **64 symbols** (note: the packet says 65;
the verified count in the file is 64 — all `leverage_flag=0`, all `eligible=1`). All are
US-listed ETFs.

| Asset class | Count | Symbols (sector/category as given in the file) |
|---|---|---|
| equity — broad/global | 9 | SPY, QQQ, IWM, DIA, RSP, VTI, VOO, IVV (broad); VT (global) |
| equity — international/country/region | 9 | VEA, EFA (international_dev); VWO, EEM (international_em); EWJ (japan); EWZ (brazil); FXI (china); INDA (india); VGK (europe) |
| equity — theme | 1 | ARKK (theme_innovation) |
| equity — sectors | 13 | XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLC, VNQ, IYR (reit ×2), — |
| equity — industries | 8 | SOXX, SMH (semis); XBI, IBB (biotech); KRE (regional_banks); XRT (retail); ITA (aerospace_defense); IYT (transport) |
| equity — factor/style | 8 | MTUM, QUAL, VLUE, USMV, VIG, SCHD (factors); VUG, VTV (style) |
| bond | 9 | TLT, IEF, SHY, TIP (treasuries); LQD, HYG (credit); AGG, BND (aggregate); MUB (muni) |
| commodity | 5 | GLD, SLV, DBC, USO, UNG |
| fx | 2 | UUP (usd), FXE (eur) |
| vol | 1 | VIXY (volatility) |

The `sector` strings are the keys the transaction-cost model prices against (§5).

Operational quirk you must handle: **VUG had a 6:1 split effective 2026-04-20** (the replay
harness applies it; your data handling must not double-apply it).

---

## 3. Data inventory with historical depth

### 3.1 Daily price snapshots in S3 (the replay substrate)

- **Where:** `s3://investment-system-data/daily/<YYYY-MM-DD>/prices.parquet`.
- **What:** per-snapshot rolling window of daily OHLCV. Verified schema:
  `date, symbol, open, high, low, close, volume`. Snapshot `daily/2026-06-10/prices.parquet`
  = 16,000 rows = **250 trading dates (2025-06-11 → 2026-06-09) × 64 symbols**, ~496 KB.
- **Snapshot range (verified):** first dir `daily/2025-08-04/`, last `daily/2026-06-10/`;
  **235 date dirs, 211 contain prices.parquet**.
- **CRITICAL DEPTH CAVEAT (verified):** snapshots **before 2026-01-31 are single-day files**
  (~5.6 KB, 1 date, 27–64 symbols; e.g. `daily/2025-08-04/prices.parquet` = 27 symbols ×
  1 date). The full rolling-~250-day window format starts **2026-01-31**. So S3 alone gives
  you a continuous reconstructable daily OHLCV series from roughly 2025-02 (via the
  2026-01-31 window) onward, plus single-day points back to 2025-08-04.
- **No-look-ahead convention:** `daily/D/` artifacts contain data **through D-1 close**. The
  file containing date D's OHLC is `daily/D+1/prices.parquet`. (Verified: the 2026-06-10
  snapshot's max date is 2026-06-09.)
- Lifecycle policy deletes S3 data after 365 days (per `docs/DEPLOY.md`); S3 is NOT your
  deep-history archive.

### 3.2 Deep price history (local + refetchable)

- **Local:** `training/data/asset_features_history.parquet` (verified): 188,249 rows,
  **64 symbols, 2014-08-29 → 2026-06-05, 2,959 trading dates** (~11.8 years). Columns:
  `date, symbol, close, return_1d, return_5d, return_21d, return_63d, vol_21d, vol_63d,
  drawdown_21d, drawdown_63d, rel_strength_21d, rel_strength_63d`. Note: closes only, not
  full OHLCV.
- **Refetchable free APIs** (per `src/steps/ingest_prices.py`, verified docstrings/URLs):
  - **Stooq** (primary): `https://stooq.com/q/d/l/?s=<SYM>.US&i=d` — full-history daily OHLCV
    CSV, no key. Also index series via `^` symbols (e.g. ^VVIX, ^SKEW).
  - **yfinance** (fallback; `yfinance>=0.2.37` in `requirements-lambda.txt`) — used by
    `training/scripts/backfill_historical.py` for multi-year backfills.
  - **Alpha Vantage** (fallback for critical symbols; API key in Secrets Manager).
  Caveat: some universe ETFs are younger than 10y (e.g. XLC inception 2018, INDA 2012) —
  verify per-symbol inception when you state your sample budget.

### 3.3 Macro / auxiliary series

- **FRED** (`src/steps/ingest_fred.py`, key in Secrets Manager; host on allowlist). Live
  series fetched nightly: `DGS2, DGS3MO, DGS10, VIXCLS, DCOILWTICO, DEXUSEU` (default 365d
  lookback; full multi-decade history available from the same free API on demand).
- **Volatility-structure indices** via Stooq: ^VVIX, ^SKEW (live `context.parquet` carries
  `vix_term_slope, vvix_value, skew_value`).
- **Engineered context series, deep history (local, verified):**
  `training/data/historical_context.parquet` / `historical_combined.parquet`: 2,803 dates,
  **2014-12-10 → 2026-02-03**. Columns: `spy_return_1d, spy_return_21d, spy_vol_21d, rate_2y,
  rate_10y, yield_slope, credit_spread_proxy, risk_off_proxy, vixy_return_21d` (+ GDELT cols
  in combined). Rebuildable via `training/scripts/backfill_historical.py` (yfinance + FRED).

### 3.4 GDELT — what is ingested today vs what GDELT offers

**Ingested today (live nightly, `src/steps/ingest_gdelt.py`):** a single daily aggregate from
the GKG **counts** file (`http://data.gdeltproject.org/gdeltv2/<YYYYMMDD>.gkgcounts.csv.zip`),
emitting `gdelt_doc_count, gdelt_avg_tone, gdelt_tone_std, gdelt_neg_tone_share` — and the
tone fields are **hard-coded placeholders (0.0)** in the live code; only doc_count was ever
real.

**Verified live status (2026-06-10): GDELT is dark in the live pipeline.** The daily
gkgcounts URL returns **HTTP 404** (verified for 20260608), and the most recent
`daily/2026-06-09/context.parquet` shows `gdelt_doc_count=0` and all tone fields 0.0 — the
fail-soft placeholder path. Meanwhile the 15-minute GKG file
(`http://data.gdeltproject.org/gdeltv2/20260608120000.gkg.csv.zip`) returns **HTTP 200**
(verified). Implication: the daily-counts endpoint the repo uses appears retired/stale, but
GDELT v2's 15-minute files are alive and reachable from this sandbox. Any GDELT organ you
design should build on the 15-minute v2 files (or masterfile lists), not the dead daily URL.

**Historical GDELT already collected (local, verified):**
`training/data/historical_gdelt.parquet` — 4,005 dates, **2015-02-18 → 2026-02-04**, with
real tone metrics (`gdelt_avg_tone, gdelt_tone_std, gdelt_neg_tone_share, gdelt_doc_count`),
built by `training/scripts/backfill_gdelt.py`, which samples **4 GKG files/day (00/06/12/18
UTC)** and parses the V2 TONE field (column 15). That script is your working template for
deeper/wider GDELT pulls.

**What GDELT offers free (public record):**
- **GDELT 2.0** (events + mentions + GKG, 15-minute cadence): from **2015-02-18** — which is
  exactly where the local parquet starts. GKG 2.0 records carry tone (V2TONE: tone,
  positive/negative score, polarity, activity/self-group density), themes (V2THEMES),
  actors/orgs/persons, locations, source URLs.
- **GDELT 1.0 events**: daily files back to **1979** (reduced schema, no GKG richness).
- Event records carry CAMEO actor/event codes, Goldstein scale, AvgTone, geo.
- All free over HTTP from `data.gdeltproject.org` (on the sandbox allowlist; verified
  reachable). Volume warning: a full 15-min GKG day is hundreds of MB compressed — sampling
  /filtering strategy is a design decision with a cost line.

### 3.5 News/sentiment text beyond GDELT

No other news/text artifacts exist in S3 today (no news prefix in the bucket). GDELT GKG
records include source-article URLs, which is the existing free path to actual text if your
LLM organ wants documents rather than aggregates. Free-only constraint applies; network
allowlist currently includes `data.gdeltproject.org`, `stooq.com`, `api.stlouisfed.org`
(FRED), `query1/query2.finance.yahoo.com`, `www.alphavantage.co`, plus AWS endpoints
(S3/Bedrock/Secrets Manager). New hosts need an operator allowlist change — flag, don't
assume.

### 3.6 The brain's own live record (~10 months — a sample-size fact)

`daily/<date>/` dirs from **2025-08-04 → 2026-06-10** (235 dirs ≈ 10.2 months of trading
days) contain the live pipeline's per-day artifact set. Verified file names in a recent day
(2026-06-09): `prices.parquet, morning_prices.parquet, context.parquet, features.parquet,
signals.parquet, inference.json, decisions.json, trade_intents.json, trades.jsonl,
morning_execution.json, midday_check_report.json, portfolio_state.json, llm_risk.json,
weather_blurb.json, run_report.json`. Plus `daily/latest.json` (pointer: date, snapshot id,
portfolio_value, phase). The decision-bearing files (`decisions.json, trade_intents.json,
signals.parquet, inference.json, llm_risk.json` and the parameter bundles under
`s3://.../config/`) are **under seal until design registration** — their existence and count
is what you may use (e.g. ~10 months ≈ ~210 trading days of realized decision/fill/state
records available later for fine-tuning or evaluation), not their contents.

Data-layer files you MAY rely on now: `prices.parquet` (§3.1), `morning_prices.parquet`
(single-date open-price file written by the morning run), `context.parquet` (verified live
schema: `date, spy_return_1d, spy_return_21d, spy_vol_21d, rate_2y, rate_3m, rate_10y,
yield_slope, yield_slope_10y_3m, credit_spread_proxy, risk_off_proxy, vixy_return_21d,
vixy_return_5d, vix_term_slope, vvix_value, skew_value, gdelt_*`), and `features.parquet`
(per-symbol engineered features; same column family as §3.2).

### 3.7 Model artifact storage (names only)

`s3://investment-system-data/models/` — 28 objects, ~6.5 MB total: `latest.json` (pointer
with version/metrics fields) plus versioned `.pkl` files (name families: `health_*`,
`regime_*`, `regime_gru_*`, `regime_transformer_*`, ~230–443 KB each). Cited only as the
storage convention a new brain's artifacts would follow (versioned files + `latest.json`
pointer; Lambda hot-loads from S3, no redeploy needed — per `docs/TRAINING.md`). No
characterization of these models is provided or permitted here.

### 3.8 Other bucket prefixes (shape only)

Top-level S3 prefixes (verified): `backtests/` (empty placeholder), `config/`, `daily/`,
`dashboard/` (public frontend JSON), `frontend/`, `lambda/`, `layers/`, `models/`,
`templates/`.

---

## 4. Compute + deployment envelope

- **AWS account/region:** us-east-1; bucket `investment-system-data`; profile `personal`.
- **Lambda:** one container-image function `investment-system-daily-pipeline` —
  **3008 MB memory, 900 s timeout** (verified in `infrastructure/lambda_deploy*.sh`). Image
  ≈1.2 GB with CPU-only PyTorch, in ECR repo `investment-system-pipeline` (~$0.12/mo storage).
  Night run ≈110 s; morning run ≈10–15 s (per `docs/DEPLOY.md`).
- **Cadence (EventBridge, verified in docs/DEPLOY.md):** night analysis **3:00 UTC Tue–Sat
  (10 PM ET Mon–Fri)** → writes the day's analysis + queued intents; morning execution
  **14:45 UTC Mon–Fri (9:45 AM ET)** → simulates fills at the real market open, writes
  `morning_prices.parquet`; that night's run settles to close. Pure simulation — **no
  broker** (Alpaca removed 2026-06-08; per `docs/DEPLOY.md`). Fail-loud guard: if the line
  can't honestly advance, the pipeline holds and alerts via SNS
  (`investment-system-alerts`) rather than inventing a point (per `docs/OPERATIONS.md`).
- **Training:** local on the operator's Mac, **monthly via launchd** (template
  `infrastructure/launchd_plist/com.investsys.train.plist`: 1st of month, 02:00), 1–2 h
  runs, torch==2.1.2 CPU, uploads artifacts to `models/` + pointer update; Lambda picks them
  up without redeploy. A containerized AWS Batch/EC2-Spot training path also exists
  (`Dockerfile.training`, `infrastructure/batch_training_setup.sh`) but the operating
  pattern of record is local-Mac monthly.
- **Runtimes/deps:** Lambda: python + pandas layer + `boto3, requests, openai==1.6.1,
  httpx<0.28, python-dateutil, yfinance` (`requirements-lambda.txt`). Training:
  `torch==2.1.2, scikit-learn==1.3.2, pandas==2.1.4, numpy==1.24.3, boto3, tqdm`
  (`requirements-training.txt`). Local venv `.venv` has pyarrow.
- **LLM access available today:**
  - **AWS Bedrock**: IAM allows `bedrock:InvokeModel` on exactly one model:
    `anthropic.claude-3-haiku-20240307-v1:0` (verified `infrastructure/iam_policies.json`;
    same ID in `src/handler.py` / `src/steps/llm_*.py`). Widening the model list = IAM
    change, flag it.
  - **OpenAI API**: key in Secrets Manager; `gpt-4o-mini` is the model string present in the
    LLM step code. (Model IDs cited as account capabilities only.)
- **Secrets Manager:** keys for OpenAI, FRED, Alpha Vantage (+ preserved, unused Alpaca
  paper keys).
- **Alerting:** SNS topic `investment-system-alerts` → operator email; night summary +
  morning report + errors.
- **Dashboard:** React frontend served from `dashboard/` prefix via CloudFront
  `E10EHVNQ0CELM2` (`trader-bot.infotrope.io`). Not on your critical path; pipeline writes
  `dashboard/dashboard.json` + `timeseries.json`.
- **Current cost of record:** ~$9/month (Lambda ~$6, S3 ~$1, Secrets ~$1, CloudWatch ~$1 —
  `docs/OPERATIONS.md`); DEPLOY.md states "<$10/month total". Hard rules: **never enable S3
  versioning**; lifecycle deletes after 365 d; no new persistent paid infra; Lambda handlers
  thin (no heavy DS stacks in request path — container image is how PyTorch gets in).
- **Your budget:** the new brain must fit ≈ the same envelope: ≤$10/mo target, ~$15/mo hard
  fail, itemized worksheet (Lambda GB-s, Bedrock tokens/night, S3, ECR, transfer) required.

---

## 5. Harness + cost-model interfaces (the bake-off contract)

### 5.1 Replay engine — `src/utils/three_line_replay/replay_engine.py` (signatures verified)

The mandated bake-off harness. Replays a strategy day-by-day over the S3 `daily/<date>/`
snapshots with no look-ahead. Core contract:

- **Clock and data discipline:** for decision date `D`, only `daily/D/*` artifacts are
  readable (they contain data through D-1 close). **Fill price = next-session OPEN** from
  `daily/D+1/prices.parquet`; positions are **marked to D's CLOSE**. VUG 6:1 split applied
  at `current_date >= 2026-04-20`.
- **Key public surface (class/method signatures):**
  - `class S3Cache(s3_client, bucket='investment-system-data')` — `.get(key)`,
    `.get_json(key)`, `.get_parquet(key)`, `.get_csv(key)`, `.list_daily_dates()`. One
    in-memory cache per replay run.
  - `class Position`; `class Portfolio` — `.value_at_marks(marks) -> (value, issues)`,
    `.to_state_dict_with_marks(marks)`, `.position_map()`.
  - `seed_portfolio(cache) -> Portfolio` — initial portfolio state.
  - `run_variant(cache, variant: VariantConfig, strategy: Optional[Strategy],
    trading_dates: List[str], universe_df: pd.DataFrame) -> Dict[str, Any]` — runs one
    variant forward through `trading_dates[-2]` (the last date with next-day prices
    available) and returns the result record.
  - `_execute_intents(portfolio, intents, ohlc, decision_params, date, ...)` — intents are
    dicts; actions are `BUY` / `SELL` / `REDUCE`; fills at next-open with transaction costs
    applied (§5.2). A harness-level cluster cap (`_apply_cluster_cap`) clamps BUY intents so
    no correlated sector cluster exceeds a max weight of portfolio value; sells pass
    through.
- **What a pluggable brain must therefore do:** given a decision date D and the
  `daily/D/` artifact set (prices window, context, features — plus whatever artifacts your
  own nightly steps add), emit a list of **trade-intent dicts** (symbol, action
  BUY/SELL/REDUCE, sizing) to be filled at the next open. The bake-off requirement is the
  **identical harness**: same replay engine, same fill rule, same cost model, same data
  snapshots, same holdout discipline. A `Strategy` hook abstraction exists
  (`src/utils/three_line_replay/strategies.py` — not read; under seal pre-registration); the
  practical integration plan is settled post-registration with the orchestrator.
- Designed to run inside the Lambda nightly; per-run S3 reads are cached (cheap).

### 5.2 Transaction-cost model — `src/utils/transaction_costs.py` (verified, full interface)

```python
get_cost_config_snapshot() -> Dict[str, Any]
    # {'spread_bps': {...}, 'asset_class_default_bps': {...}, 'slippage_range_bps': 2.0}

get_half_spread_bps(sector: str, asset_class: str = 'equity') -> float

apply_transaction_costs(
    price: float,
    action: str,                      # 'BUY' | 'SELL' ('REDUCE' treated as SELL)
    sector: str = 'broad',            # the universe.csv sector string
    asset_class: str = 'equity',
    rng: Optional[random.Random] = None,   # seedable — determinism rules apply
    cost_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]              # (fill_price, total_cost_bps)
```

Mechanics: per-sector **half-spread table in bps** (e.g. broad 1.0, sector_tech 2.0,
biotech 4.0, natural_gas 8.0, volatility 8.0; asset-class fallbacks equity 3.0 / bond 2.5 /
commodity 4.0 / fx 3.0 / vol 8.0) **plus uniform random slippage in ±2.0 bps** (seedable via
`rng`). BUY pays spread+slippage above price; SELL receives spread−slippage below. Every
simulated fill in the bake-off goes through this. Price your turnover honestly: a
daily-rebalance design pays ~3–10 bps per side per trade in the niche names.

---

## 6. Evidence rules (condensed from `committee/EVIDENCE_PROTOCOL.md` — read the original)

- **Grades:** E0 hypothesis (never ships) → E1 in-sample counterfactual (full-history
  replay, ON vs OFF, identical data/seeds/fill model, same code revision) → **E2 holdout
  counterfactual = the adoption bar** → E3 forward shadow (optional, never a timed gate).
- **E2 holdout:** verdict read ONLY from data after **`holdout_start: 2026-03-11`** (defined
  in `config/optimizer.committee_20260606.json`). **No tuning against the holdout**: one
  pre-registered configuration per candidate, declared in the run dir BEFORE the holdout
  replay executes. PKT-TB-006 pre-registers the full bake-off criteria in `TOURNAMENT.md`
  before any building.
- **Required reporting per comparison:** one JSON + one MD: deltas (total return, CAGR,
  Sharpe, max DD, win rate, round trips, cumulative costs, avg gross exposure) full-period
  AND holdout-only; **paired daily-difference stats** (mean, sd, t-stat on identical dates)
  — endpoint deltas alone are inadmissible; **run manifest** (code SHA, params hash, seeds,
  S3 `daily/<date>/prices.parquet` snapshot range, wall-clock, exact command).
- **Every variant logged.** N configurations tried ⇒ all N in the run dir with results.
  Search-found winners need the E2 read to count at all. Panels must state how many looks
  the holdout has absorbed (a +0.1 Sharpe winner among 40 candidates is noise).
- **Determinism:** fixed seeds everywhere, recorded; replays pin to S3 snapshots (no live
  re-fetch mid-experiment); cost-model version recorded in the manifest.

---

## 7. Sample-size reality (justify learnability or simplify)

The honest numbers, verified:

- **Public price history:** ~2,959 trading days × 64 symbols ≈ **189k symbol-days**
  (2014-08-29 → 2026-06-05), daily bars only, closes in the local features file (full OHLCV
  refetchable from Stooq/yfinance). Cross-sectionally correlated ETFs — the effective sample
  is far smaller than 189k independent draws; ~11.7 years spans only a handful of distinct
  market regimes (2015-16 chop, 2018 vol events, 2020 crash, 2022 bear, 2023-25 bull...).
- **Market-level context series:** ~2,803 daily rows (one per day, not per symbol).
- **GDELT with real tone:** ~4,005 days (2015-02-18 →), one aggregate row/day as collected
  so far; richer per-event/per-theme structure available but must be backfilled within
  cost.
- **The brain's own live record:** ~10.2 months ≈ ~210 trading days of decision-grade
  artifacts. This is the ONLY data that reflects the live pipeline's actual
  morning-fill/cost regime; it supports fine-tuning/calibration at best, never primary
  training.
- **Holdout:** 2026-03-11 → present ≈ **~62 trading days** at brief time. One quarter of
  daily data — paired daily stats will have wide error bars; design your claimed effect
  sizes accordingly.

What this supports: models with modest parameter counts trained on pooled cross-sections,
strong regularization, walk-forward validation; transformers only with aggressive
weight-sharing/small dims or pretraining tricks. What it cannot support: per-symbol deep
nets, wide hyperparameter sweeps read against the holdout, or RL with long credit
assignment horizons trained from scratch on 2.9k daily steps. Every learnable component in
your design must state its parameter count vs. its effective sample and why it won't just
memorize (packet requirement; Training Realist will audit).

---

## 8. Seal incidents

**None.** Forbidden files were not opened. Sources read: the packet,
`committee/EVIDENCE_PROTOCOL.md`, `config/universe.csv`, `docs/DEPLOY.md`,
`docs/OPERATIONS.md`, `docs/TRAINING.md` (infra/cadence only; model-description content not
transcribed), `src/utils/transaction_costs.py`, `replay_engine.py` signatures/docstrings via
AST extraction (strategy internals not transcribed; `strategies.py` not opened),
`src/steps/ingest_{prices,gdelt,fred}.py`, `training/` data-pipeline scripts + parquet
metadata, `requirements*.txt`, `Dockerfile.training`, `infrastructure/*`, plus live S3
listings/HEADs and GDELT HTTP checks. One borderline note for the record: S3 `models/`
artifact file names and `daily/<date>/` artifact file names are listed in §3.6–3.7 as
permitted name-only inventory; no contents of sealed files were read or described.
