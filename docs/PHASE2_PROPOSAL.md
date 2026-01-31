# Phase 2 Proposal: Enhanced Models, Data, and Visualization

**Author**: Claude Opus 4.5
**Date**: 2026-01-31
**Status**: Proposal for Review

---

## Executive Summary

This proposal outlines enhancements to transform the investment system from a functional trading bot into a comprehensive learning and decision-transparency platform. The focus is on:

1. **New predictive models** that reveal different aspects of market behavior
2. **Additional data sources** that provide orthogonal signals (not duplicating existing data)
3. **Rich visualizations** that expose the full decision-making process

**Guiding Principles**:
- Each addition must provide a *new angle*, not duplicate existing signals
- Prefer free/cheap data sources
- Training happens offline; inference is cheap
- Transparency over black-box predictions
- Learning value alongside profit potential

---

## Part 1: Model Additions

### 1.1 Temporal Fusion Transformer (TFT) for Price Forecasting

**What it adds**: Multi-horizon price predictions with interpretable attention weights showing *why* the model made each prediction.

**Why TFT specifically**:
- Designed for multi-horizon forecasting (predict 1, 5, 21 days ahead simultaneously)
- Built-in interpretability: shows which features and time steps drove the prediction
- Handles mixed inputs: static (sector, asset type), known future (day of week, holidays), observed (prices, volume)
- State-of-the-art performance on financial time series

**Architecture**:
```
Inputs:
├── Static: sector, asset_class, market_cap_bucket
├── Known Future: day_of_week, month, is_earnings_week
└── Observed: price, volume, returns, volatility, sentiment

Outputs:
├── Price predictions: [+1d, +5d, +21d]
├── Prediction intervals: [10%, 50%, 90% quantiles]
└── Attention weights: which features/timesteps mattered
```

**Visualization**: Interactive prediction cones showing expected price range with confidence bands, plus attention heatmaps showing what drove each prediction.

---

### 1.2 Ensemble Regime Classification

**What it adds**: Robustness through model disagreement detection.

**Implementation**:
- Run both RegimeGRU and RegimeTransformer (already built)
- When models agree: high confidence regime
- When models disagree: "uncertain" regime → reduce position sizes

**New insight**: Model disagreement is itself a signal. When sophisticated models can't agree on the market state, that's valuable information.

**Visualization**: Regime timeline with confidence shading. Dark = both models agree. Light = models disagree.

---

### 1.3 Anomaly Detection via Isolation Forest

**What it adds**: Early warning system for unusual market conditions not captured by regime model.

**Why this approach**:
- Unsupervised: doesn't need labeled "crisis" data
- Fast inference
- Catches "unknown unknowns" - conditions unlike anything in training

**Features to monitor**:
- Cross-asset correlation spikes
- Volume anomalies
- Volatility term structure inversions
- Sentiment extremes

**Visualization**: Anomaly score gauge (0-100) with historical context. "Current market unusualness" indicator.

---

### 1.4 Momentum Decomposition Model

**What it adds**: Separates momentum into interpretable components.

**Components**:
1. **Trend momentum**: Long-term direction (63-day)
2. **Mean reversion pressure**: Distance from moving averages
3. **Relative momentum**: vs sector, vs market
4. **Momentum quality**: Consistency of gains (Sharpe of returns)

**Why decompose**: Raw momentum score hides *why* something is trending. A stock up 20% from steady gains vs up 20% from one spike are very different.

**Visualization**: Stacked bar chart showing momentum breakdown per asset.

---

## Part 2: Data Sources

### 2.1 Reddit Sentiment (via PRAW)

**What it adds**: Retail investor sentiment and attention - orthogonal to news sentiment.

**Subreddits to monitor**:
- r/wallstreetbets (high-risk retail plays)
- r/stocks (mainstream retail)
- r/investing (conservative retail)

**Signals extracted**:
- Mention velocity (sudden attention spikes)
- Sentiment polarity (bullish/bearish language)
- Rocket emoji density (meme stock indicator - seriously, it's predictive)

**Why it matters**: Retail flows move small/mid caps. Crowded trades often reverse. Extreme bullishness is often a sell signal.

**Cost**: Free (Reddit API)

**Visualization**: "Retail Radar" - heatmap of what retail is talking about and how bullish they are.

---

### 2.2 Options Flow Data

**What it adds**: What sophisticated money is betting on.

**Source**: CBOE delayed data (free) or Polygon.io free tier

**Signals**:
- Put/Call ratio (market-wide and per-asset)
- Unusual options activity (large trades)
- Implied volatility skew (fear of downside)
- Term structure (near vs far expiry IV)

**Why it matters**: Options traders are often more sophisticated. Large unusual bets can signal informed trading. IV skew reveals expected distribution asymmetry.

**Visualization**: Options sentiment gauge per asset + market-wide fear/greed from options.

---

### 2.3 VIX Term Structure

**What it adds**: Forward-looking volatility expectations across time horizons.

**Signals**:
- VIX level (current fear)
- VIX futures curve slope (contango = normal, backwardation = panic)
- VIX/VIX3M ratio (short vs medium-term fear)

**Source**: CBOE (free delayed data)

**Why it matters**: Backwardation (near VIX > far VIX) is one of the strongest crisis indicators. Extreme contango suggests complacency.

**Visualization**: VIX term structure curve with historical percentile shading.

---

### 2.4 Insider Transaction Filings

**What it adds**: What company insiders are doing with their own money.

**Source**: SEC EDGAR (free, public data)

**Signals**:
- Cluster buys (multiple insiders buying)
- Insider buy/sell ratio
- Dollar-weighted insider sentiment

**Why it matters**: Insiders sell for many reasons, but they only buy for one. Cluster buying is a strong signal.

**Processing**: Weekly batch job to pull Form 4 filings

**Visualization**: Insider activity indicator per holding + aggregate insider sentiment.

---

### 2.5 Economic Calendar

**What it adds**: Awareness of upcoming market-moving events.

**Events to track**:
- FOMC meetings (Fed decisions)
- CPI/PPI releases (inflation)
- NFP (employment)
- Earnings dates for holdings

**Source**: Finnhub free tier or scraped from investing.com

**Why it matters**: Volatility clusters around events. The system should know when to expect turbulence.

**Visualization**: Calendar overlay on charts marking event dates.

---

## Part 3: Visualization & Dashboard

### 3.1 Design Philosophy

The dashboard should answer these questions at a glance:

1. **What is the market doing?** (Regime, trend, anomalies)
2. **What is the model predicting?** (Forecasts with uncertainty)
3. **Why is it predicting that?** (Feature importance, attention)
4. **What are other signals saying?** (Sentiment, options, insiders)
5. **What did we decide to do?** (Actions, reasoning)
6. **How are we doing?** (Performance, attribution)

### 3.2 Main Chart Component

**Interactive multi-layer chart with**:

```
Layer 1: Price + Volume
├── Candlestick or line chart
├── Volume bars (colored by up/down)
└── Event markers (earnings, FOMC, etc.)

Layer 2: Predictions
├── TFT prediction cone (1/5/21 day forecasts)
├── Confidence intervals (10/50/90%)
└── Historical prediction accuracy overlay

Layer 3: Signals
├── Regime band (color-coded background)
├── Health score line
├── Anomaly markers

Layer 4: Sentiment
├── News sentiment line
├── Reddit sentiment line
├── Options sentiment (put/call ratio)

Layer 5: Decisions
├── Buy/sell markers with reasoning tooltips
├── Position size indicators
└── Stop loss levels
```

**Toggleable layers**: User can show/hide any layer for clarity.

### 3.3 Model Transparency Panel

**Feature Importance View**:
- Bar chart: which features most influenced today's prediction
- Time attention: which historical days the model focused on
- Comparison: "The model weighted volatility heavily today because..."

**Model Agreement Matrix**:
```
           GRU    Transformer    TFT-direction
Regime     ✓ risk_on  ✓ risk_on      -
Direction     -           -       ✓ bullish
Confidence   85%        78%         72%
```

**Decision Audit Trail**:
```
Decision: BUY XLB
├── Health Score: 0.78 (above 0.60 threshold)
├── Regime: risk_on_trend (compatible)
├── TFT Forecast: +2.3% (5-day)
├── Sentiment: Neutral news, Bullish reddit
├── Options: Low put/call ratio
├── Insider: 2 buys this month
└── LLM Check: No red flags identified
```

### 3.4 Sentiment Dashboard

**Three gauges**:
1. **News Sentiment** (GDELT): Institutional/media view
2. **Retail Sentiment** (Reddit): Individual investor view
3. **Smart Money Sentiment** (Options): Derivatives trader view

**Historical context**: Each gauge shows current level vs 1-year range.

**Divergence alerts**: When signals disagree, highlight it. "Retail bullish but options traders hedging heavily."

### 3.5 Risk & Anomaly Panel

**Current Risk State**:
- Portfolio VaR (Value at Risk)
- Maximum drawdown (current vs historical)
- Correlation to SPY
- Beta exposure

**Anomaly Monitor**:
- Market anomaly score (0-100)
- Specific anomalies detected
- Historical anomaly events marked on timeline

### 3.6 Performance Attribution

**What drove returns**:
- By asset (which positions made/lost money)
- By signal (which models were right)
- By regime (performance in each market state)

**Benchmark comparison**:
- vs SPY (market)
- vs 60/40 (traditional allocation)
- vs equal-weight universe

---

## Part 4: Implementation Plan

### Phase 2A: Quick Wins (Week 1)

| Task | Effort | Impact |
|------|--------|--------|
| Switch to Claude Haiku | 30 min | Cost savings |
| Enable RegimeTransformer ensemble | 1 hour | Better regime detection |
| Add ensemble disagreement logic | 2 hours | New uncertainty signal |
| Basic prediction chart component | 4 hours | Foundation for viz |

### Phase 2B: Data Expansion (Week 2)

| Task | Effort | Impact |
|------|--------|--------|
| Fix GDELT ingestion | 2 hours | Real sentiment |
| Add Reddit sentiment | 4 hours | Retail signal |
| Add VIX term structure | 2 hours | Vol expectations |
| Add options put/call ratio | 3 hours | Smart money signal |

### Phase 2C: TFT Forecasting (Week 3)

| Task | Effort | Impact |
|------|--------|--------|
| Implement TFT architecture | 8 hours | Price forecasting |
| Train TFT on historical data | 4 hours | Model ready |
| Add prediction to pipeline | 4 hours | Daily forecasts |
| Prediction cone visualization | 6 hours | See the forecasts |

### Phase 2D: Full Dashboard (Week 4)

| Task | Effort | Impact |
|------|--------|--------|
| Multi-layer chart component | 8 hours | Main viz |
| Model transparency panel | 6 hours | See the reasoning |
| Sentiment dashboard | 4 hours | All signals at glance |
| Decision audit trail | 4 hours | Full transparency |

---

## Part 5: Technical Specifications

### 5.1 TFT Model Specification

```python
TFT Config:
  hidden_size: 64
  attention_heads: 4
  dropout: 0.1
  lstm_layers: 2

  static_features: ['sector', 'asset_class']
  known_future: ['day_of_week', 'month', 'is_earnings_week']
  observed: [
    'close', 'volume', 'return_1d', 'return_5d',
    'vol_21d', 'rsi_14', 'macd', 'news_sentiment',
    'reddit_sentiment', 'put_call_ratio'
  ]

  forecast_horizons: [1, 5, 21]  # days
  quantiles: [0.1, 0.5, 0.9]
```

### 5.2 Data Pipeline Additions

```
Daily Pipeline (10 PM ET):
├── [existing] Ingest prices
├── [existing] Ingest FRED
├── [fix] Ingest GDELT
├── [new] Ingest Reddit sentiment
├── [new] Ingest VIX term structure
├── [new] Ingest options data
├── [existing] Build features
├── [new] Run TFT predictions
├── [modified] Run ensemble regime
├── [existing] Run health model
├── [existing] LLM risk check (Haiku)
├── [existing] Decision engine
├── [existing] Paper trader
├── [existing] LLM weather (Haiku)
└── [existing] Publish artifacts

Weekly Pipeline (Sunday):
├── [new] Ingest insider transactions
└── [new] Update economic calendar
```

### 5.3 Dashboard Data Structure

```json
{
  "charts": {
    "prices": { "ohlcv": [...], "events": [...] },
    "predictions": {
      "horizons": [1, 5, 21],
      "forecasts": { "SPY": { "1d": {...}, "5d": {...} } },
      "attention": { "feature_weights": {...}, "time_weights": {...} }
    }
  },
  "models": {
    "regime": {
      "gru": { "label": "risk_on", "confidence": 0.85 },
      "transformer": { "label": "risk_on", "confidence": 0.78 },
      "ensemble": { "label": "risk_on", "agreement": true }
    },
    "anomaly": { "score": 23, "alerts": [] }
  },
  "sentiment": {
    "news": { "score": 0.12, "percentile": 55 },
    "reddit": { "score": 0.45, "percentile": 78, "trending": ["XLB"] },
    "options": { "put_call": 0.85, "percentile": 40 }
  },
  "decisions": {
    "actions": [...],
    "audit_trail": { "XLB": { "reasons": [...] } }
  }
}
```

---

## Part 6: Cost Analysis

### Monthly Operating Costs (Projected)

| Component | Current | After Phase 2 |
|-----------|---------|---------------|
| AWS Lambda | $6 | $8 (more compute) |
| S3 Storage | $1 | $2 (more data) |
| Claude Haiku LLM | $0 (GPT-4 broken) | $1 |
| Data APIs | $0 | $0 (all free sources) |
| **Total** | **$7** | **$11** |

### Training Costs (One-time, local)

- TFT training: ~2 hours on M-series Mac (free)
- Ensemble training: Already done (just enable)
- Anomaly detector: ~10 minutes (simple model)

---

## Part 7: Success Metrics

### Model Quality
- TFT directional accuracy > 55%
- Regime model agreement > 80% of days
- Anomaly detector catches > 70% of drawdowns with < 20% false positives

### Dashboard Utility
- All predictions visible with reasoning
- Any decision auditable in < 3 clicks
- Full historical data explorable

### Learning Value
- Can explain why any trade was made
- Can see what signals contributed to decisions
- Can compare what model said vs what happened

---

## Appendix: What We're NOT Adding (and Why)

| Candidate | Reason to Skip |
|-----------|----------------|
| Twitter/X sentiment | API now expensive ($100+/mo) |
| High-frequency data | Not relevant for daily trading |
| Fundamental data | Slow-moving, less useful for tactical allocation |
| More LLM calls | One risk check is enough; more adds latency and cost |
| Reinforcement learning | Overkill for current scope; save for Phase 3 |
| Local LLM | User requested to defer; future consideration |

---

## Next Steps

1. **User Review**: Approve this proposal or request modifications
2. **API Key**: Provide Anthropic API key for Claude Haiku
3. **Cursor Handoff**: Cursor implements Phase 2A-2D per this spec
4. **Testing**: Validate each component before moving to next phase

---

*This proposal represents an opinionated, practical path forward. Each addition provides unique signal, maintains low cost, and prioritizes transparency and learning over black-box complexity.*
