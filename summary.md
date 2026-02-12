# TraderBot — Autonomous ML Trading System

## What It Is

TraderBot is a fully autonomous daily investment decision system built on AWS. It ingests market data every evening, runs machine learning models and expert signal analysis to assess market conditions, generates trade intents overnight, then validates and executes those trades at market open the next morning. A React dashboard provides full observability into the system's reasoning, and email alerts keep the operator informed without needing to check the dashboard.

The entire system runs for approximately $9/month on AWS.

---

## How It Works — The Two-Phase Pipeline

### Night Analysis (10 PM ET, Mon–Fri)

An AWS EventBridge schedule triggers a Lambda function that runs a 12-step pipeline:

1. **Data ingestion** — Pulls daily OHLCV prices for 65 symbols from Stooq (with Alpha Vantage fallback), macroeconomic indicators from the FRED API (Treasury yields, VIX, oil prices, USD/EUR), volatility indices (VVIX, SKEW), and news sentiment from GDELT.
2. **Feature engineering** — Computes per-asset features (multi-horizon returns, rolling volatility, drawdowns, trend direction, relative strength vs. SPY) and market-wide context features (yield curve slope, credit spread proxies, risk-off indicators).
3. **Expert signal computation** — Four independent signal modules analyze macro/credit conditions, volatility complexity, cross-asset fragility, and return distribution entropy.
4. **ML inference** — An ensemble of GRU and Transformer neural networks classifies the current market regime. A separate autoencoder/VAE model scores individual asset health.
5. **Regime fusion** — Combines the ML ensemble output with expert signals through a priority-ordered override system to produce a final regime label and position-sizing modifiers.
6. **Decision engine** — Scores and ranks buy/sell candidates, applies layered position sizing (regime, volatility, model confidence, expert throttles), enforces risk constraints, and generates trade intents.
7. **LLM risk check** — Claude Haiku (via AWS Bedrock) reviews asset-specific risks and can veto trades or reduce position sizes.
8. **Weather report** — An LLM generates a plain-English daily market narrative for the dashboard.
9. **Artifact publishing** — All outputs (features, signals, decisions, portfolio state, dashboard data) are written to S3.
10. **Email alert** — An SNS notification summarizes the regime, trade intents, and market outlook.

### Morning Execution (9:45 AM ET, Mon–Fri)

A second EventBridge trigger runs the execution phase:

1. Loads the overnight trade intents.
2. Fetches live morning quotes via yfinance.
3. Validates each intent — checks freshness (max 3 days old), price gaps (skip buys if price moved >5% overnight), and re-evaluates trailing stops at morning prices.
4. Executes validated trades at current market prices.
5. Updates portfolio state, recomputes performance metrics, and republishes the dashboard.
6. Sends a morning summary email with executed trades and portfolio value.

This two-phase design ensures that analysis happens on closing prices (stable, complete data) while execution happens at actual market prices (realistic P&L).

---

## Machine Learning Models

### Market Regime Classification (Ensemble)

Two neural network architectures are trained to classify the market into one of five regimes:

- **GRU (Gated Recurrent Unit)** — Processes 21-day sequences of 10 context features through a 2-layer, 64-dimensional GRU with an attention mechanism for sequence aggregation.
- **Transformer** — Uses multi-head self-attention (4 heads, 2 encoder layers) with positional encoding and a CLS token for classification.

Both models output probabilities over five regime classes:

| Regime | Interpretation | Favored Assets |
|--------|---------------|----------------|
| `calm_uptrend` | Low volatility, positive momentum | Equities |
| `risk_on_trend` | Moderately bullish | Balanced equity allocation |
| `risk_off_trend` | Negative returns or credit stress | Bonds |
| `choppy` | Low returns, elevated volatility | Reduced sizing across the board |
| `high_vol_panic` | VIX spike, significant drawdown | Bonds and commodities only |

In production, both models run and their predictions are averaged (equal weights). When the models disagree significantly, position sizes are automatically reduced — this ensemble disagreement signal captures model uncertainty and triggers caution.

### Asset Health Scoring (Autoencoder / VAE)

An autoencoder (with an optional variational variant) learns to reconstruct asset feature vectors through a compressed latent space. From the latent representation, three task-specific heads produce:

- **Health score** (0–1): Overall asset quality, trained to correlate with forward returns.
- **Volatility bucket** (low/medium/high): Classification used for position sizing adjustments.
- **Behavior type** (momentum/mean-reversion/mixed): Characterizes the asset's current dynamics.

The health score directly feeds into the decision engine's buy/sell thresholds.

### Monthly Retraining

On the 1st of each month, an automated script:
1. Downloads accumulated daily data from S3.
2. Retrains both regime models and the health model.
3. Uploads versioned model artifacts to S3.
4. Optionally runs an **evolutionary search** (genetic algorithm, 30 population × 25 generations) to optimize the decision engine's hyperparameters (thresholds, sizing weights, constraints) against a composite fitness of Sharpe ratio, Calmar ratio, win rate, and consistency.

---

## Expert Signal System

Four independent signal modules provide a production-only analysis layer that modulates the ML models' outputs. These are deliberately kept separate from training features to avoid overfitting.

### 1. Macro/Credit Score
Combines the Treasury yield curve slope (10Y – 3M spread) with a high-yield credit spread proxy (HYG vs. IEF differential). A negative score indicates yield curve inversion plus widening credit spreads — classic recession warning signals. Can downgrade or upgrade the regime classification.

### 2. Volatility Complexity Score
Synthesizes three volatility measures — VIX (implied volatility), VVIX (volatility of volatility), and SKEW (tail risk pricing) — into a composite stress score. Notably detects "unstable calm" conditions where VVIX is elevated but VIX remains low, which historically precedes volatility spikes. Can hard-override the regime to defensive postures.

### 3. Fragility Score
Computes a 60-day rolling correlation matrix across 8 cross-asset symbols and runs PCA to measure how tightly coupled markets have become. High average correlation plus dominant first principal component indicates fragile, herding conditions where diversification breaks down. Caps position sizes when fragility is elevated.

### 4. Entropy Shift
Measures the Shannon entropy of SPY's return distribution over rolling 60-day windows. When entropy deviates significantly from its historical norm for 3+ consecutive days, it signals that the statistical character of the market is changing — either compressing (trending) or dispersing (regime transition). Triggers across-the-board position size reductions.

### Regime Fusion (v3)

A priority-ordered fusion engine combines expert signals with the ML ensemble output:

1. **Hard overrides** — Panic conditions from the vol complexity module force maximum defensiveness.
2. **Macro modulation** — Credit/macro signals can upgrade or downgrade the regime by one level.
3. **Caution gates** — Fragility and entropy shift apply multiplicative position-sizing caps.
4. **Ensemble disagreement** — Model uncertainty reduces position sizes proportionally.

The result is a final regime label, a position-size modifier (0.25–1.0×), and a risk throttle factor — all passed to the decision engine.

---

## Decision Engine & Risk Management

### Buy Logic
Assets are scored based on ML health scores, adjusted by regime compatibility (which asset classes suit the current regime), and filtered through minimum thresholds. Position sizes cascade through multiple adjustment layers:

**Base weight (20%)** → volatility adjustment → regime adjustment → LLM confidence adjustment → ensemble disagreement adjustment → expert signal modifier → risk throttle

### Sell Triggers
Five independent triggers can force a sell:
1. **Trailing stop hit** — Price drops 10% below its peak (6% for leveraged ETFs).
2. **Health collapse** — ML health score drops below 0.35.
3. **Regime panic** — Equities are force-sold in panic mode (only bonds/commodities retained).
4. **LLM structural risk veto** — The language model identifies fundamental risks.
5. **Leverage hold cap** — Leveraged ETFs are sold after 10 days (volatility decay protection).

### Portfolio Constraints
- Maximum 8 concurrent positions.
- Cash reserves scaled by regime (10% in calm markets, up to 40% in panic).
- Maximum 20% weight per position (before adjustments).
- Leveraged ETFs capped at 10% weight with mandatory 10-day exit.
- Minimum order size of $250.

---

## Investment Universe

65 symbols across:
- **Equities**: Broad market (SPY, QQQ, IWM), sectors (XLK, XLF, XLE, XLV, etc.), factors (MTUM, QUAL, VLUE), international (EFA, EEM, VWO)
- **Fixed Income**: Treasuries (TLT, IEF, SHY), corporate/high-yield (LQD, HYG), TIPS
- **Commodities**: Gold (GLD), silver (SLV), broad commodities (DBC)
- **Volatility**: VIXY (VIX ETF)

---

## Dashboard

A React + Vite + TypeScript frontend hosted on S3 as a static website. It provides 13+ panels covering:

- **Performance**: Total value, YTD/MTD returns, Sharpe ratio, max drawdown, win rate, equity curve vs. SPY benchmark, drawdown chart, monthly returns heatmap.
- **Portfolio**: Current holdings with entry prices and P&L, buy candidates with scores and suggested sizes, trade log with per-trade P&L.
- **Market Intelligence**: LLM-generated daily market narrative, regime timeline strip, ensemble model status with disagreement meter.
- **Expert Signals**: Dedicated panels for macro/credit conditions, volatility complexity, cross-asset fragility, and entropy shift — each with historical charts and current readings.

The dashboard reads from a JSON snapshot and a Parquet time-series file (400 days of rolling history), both updated by the pipeline.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Compute | AWS Lambda (containerized, Docker) |
| Storage | S3 (single bucket for all artifacts + frontend) |
| Scheduling | EventBridge (two cron rules) |
| ML Framework | PyTorch (GRU, Transformer, Autoencoder, VAE) |
| ML Training | Local (monthly via launchd), evolutionary search for hyperparameters |
| LLM Integration | AWS Bedrock (Claude Haiku) for risk analysis and market narratives |
| Data Sources | Stooq, FRED API, Alpha Vantage, yfinance, GDELT |
| Alerts | SNS email notifications |
| Frontend | React + Vite + TypeScript, S3 static hosting |
| Data Format | Parquet (time-series), JSON (snapshots and config) |
| Language | Python (backend), TypeScript (frontend) |
| Infra Cost | ~$9/month |

---

## Key Design Decisions

1. **Two-phase pipeline** separates analysis (closing prices) from execution (market prices) for realistic P&L tracking.
2. **Expert signals as production-only overlay** — not used as training features — prevents overfitting while adding real-time market intelligence.
3. **Ensemble disagreement as a risk signal** — when the GRU and Transformer disagree, the system automatically becomes more cautious.
4. **Multi-layer position sizing** — no single model or signal has unchecked authority; sizing passes through 6+ independent adjustment stages.
5. **Morning trade validation** — stale intents, price gaps, and stop re-checks prevent executing on outdated analysis.
6. **Evolutionary hyperparameter optimization** — a genetic algorithm tunes the decision engine's parameters against a multi-objective fitness function.
7. **Cost-optimized architecture** — single S3 bucket (no versioning), containerized Lambda, local training, lifecycle policies. Runs for under $10/month.
