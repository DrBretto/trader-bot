# Phase 2 Proposal: Enhanced Models, Data, and Visualization

**Author**: Claude Opus 4.5
**Date**: 2026-01-31
**Status**: Revised after technical review

---

## Executive Summary

This proposal outlines enhancements to transform the investment system from a functional trading bot into a comprehensive learning and decision-transparency platform. The focus is on:

1. **Model ensembling** with disagreement detection and uncertainty quantification
2. **Additional data sources** that provide orthogonal signals (not duplicating existing data)
3. **Rich visualizations** that expose the full decision-making process

**Guiding Principles**:
- Each addition must provide a *new angle*, not duplicate existing signals
- Prefer free/cheap data sources
- Training happens offline; inference is cheap
- Transparency over black-box predictions
- Learning value alongside profit potential
- Pragmatic: skip complex implementations when simpler alternatives work

---

## Part 1: Model Enhancements

### 1.1 Ensemble Regime Classification with Disagreement Detection

**What it adds**: Robustness through model agreement/disagreement as a signal.

**Implementation**:
- Run both RegimeGRU and RegimeTransformer (both already built)
- Compute ensemble prediction (weighted average of probabilities)
- Track disagreement as uncertainty signal

**Key insight**: When models disagree, that's valuable information. Disagreement = uncertainty = reduce position sizes.

```python
Ensemble Logic:
├── Both agree with high confidence → Strong signal, full position sizes
├── Both agree with low confidence → Weak signal, reduce sizes
├── Models disagree → Uncertain regime, defensive posture
└── Disagreement metric = 1 - cosine_similarity(probs_gru, probs_transformer)
```

**Visualization**:
- Regime timeline with confidence shading (dark = agreement, light = disagreement)
- Disagreement gauge (0-100%)

---

### 1.2 Ensemble-Based Prediction Confidence Bands

**What it adds**: Uncertainty quantification for predictions without needing TFT.

**Implementation**:
- Use ensemble standard deviation across models as uncertainty estimate
- Generate prediction bands from historical ensemble error distribution

**Why this over TFT**: TFT is complex to implement and tune. Ensemble uncertainty is simpler, interpretable, and often just as useful for decision-making.

```python
Confidence Bands:
├── Point estimate: ensemble mean prediction
├── 68% band: ± 1 std of ensemble predictions
├── 95% band: ± 2 std of ensemble predictions
└── Calibrate bands using historical prediction errors
```

**Visualization**: Prediction cone showing expected range with shaded confidence intervals.

---

### 1.3 Anomaly Detection via Isolation Forest

**What it adds**: Early warning system for unusual market conditions.

**Why Isolation Forest**:
- Unsupervised: doesn't need labeled "crisis" data
- Fast inference
- Catches "unknown unknowns" - conditions unlike anything in training
- Simple to implement and interpret

**Features to monitor**:
```python
Anomaly Features:
├── Cross-asset correlation (rolling 21-day)
├── Correlation vs historical mean
├── Volume z-score (vs 63-day average)
├── Volatility z-score
├── VIX term structure slope
├── Credit spread change
└── Regime model disagreement
```

**Visualization**:
- Anomaly score gauge (0-100) with color coding (green/yellow/red)
- Historical anomaly timeline with market events marked
- "Current market unusualness" indicator

---

### 1.4 Momentum Decomposition

**What it adds**: Interpretable breakdown of why an asset has momentum.

**The Problem**: A single "momentum score" hides important nuances. Two assets with the same score can have very different risk profiles.

**Components**:

| Component | Calculation | Interpretation |
|-----------|-------------|----------------|
| **Trend Strength** | Slope of 63-day OLS regression on log prices | Steady climb vs sideways chop |
| **Mean Reversion Risk** | Z-score of price vs 21-day and 50-day MA | How stretched; >2 = likely snapback |
| **Relative Momentum** | Return vs sector ETF, vs SPY | True alpha vs rising tide |
| **Momentum Quality** | Rolling 21-day Sharpe of daily returns | Steady gains vs lucky spikes |

**Combined Score**:
```python
momentum_score = (
    0.30 * trend_strength_percentile +
    0.25 * (1 - mean_reversion_risk_percentile) +  # Lower stretch = better
    0.25 * relative_momentum_percentile +
    0.20 * momentum_quality_percentile
)
```

**Example Interpretation**:
```
Asset: XLB
├── Trend Strength: 0.82 (strong upward slope)
├── Mean Reversion Risk: 0.35 (not stretched, room to run)
├── Relative Momentum: 0.71 (outperforming sector)
├── Momentum Quality: 0.68 (consistent gains, not spiky)
└── Combined: 0.72 (high quality momentum, likely to continue)

Asset: MEME_STOCK
├── Trend Strength: 0.45 (choppy, no clear trend)
├── Mean Reversion Risk: 0.92 (way above MAs, stretched)
├── Relative Momentum: 0.88 (outperforming everything)
├── Momentum Quality: 0.15 (one big spike, not steady)
└── Combined: 0.41 (low quality momentum, reversion likely)
```

**Visualization**: Stacked bar chart showing momentum breakdown per asset, color-coded by component.

---

## Part 2: Data Sources

### 2.1 Google Trends (Retail Attention)

**What it adds**: Retail investor attention signal - cleaner than Reddit scraping.

**Why Google Trends over Reddit**:
- No NLP required (just search volume)
- Longer history available
- Easier API (pytrends library)
- Captures broader retail interest
- More stable signal

**Implementation**:
```python
from pytrends.request import TrendReq

def get_retail_attention(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch retail attention from Google Trends.

    NOTE: pytrends is an unofficial scraper - Google can break it anytime.
    Always wrap in try/except and return neutral scores on failure.
    """
    try:
        pytrends = TrendReq()
        results = {}

        for symbol in symbols:
            pytrends.build_payload([f"{symbol} stock"], timeframe='now 7-d')
            data = pytrends.interest_over_time()
            if not data.empty:
                # Normalize to 0-1 scale
                results[symbol] = data[f"{symbol} stock"].iloc[-1] / 100.0
            else:
                results[symbol] = 0.5  # Neutral if no data

        return results

    except Exception as e:
        print(f"Google Trends failed: {e}")
        # Return neutral scores instead of crashing pipeline
        return {symbol: 0.5 for symbol in symbols}
```

**Signals**:
- Attention velocity (sudden spikes = crowded trade warning)
- Attention level vs historical (percentile ranking)
- Attention divergence from price (attention up but price flat = potential move coming)

**Cost**: Free

**Visualization**: "Retail Radar" heatmap showing attention levels across holdings.

---

### 2.2 VIX Term Structure

**What it adds**: Forward-looking volatility expectations across time horizons.

**Why it matters**: VIX term structure is one of the strongest regime indicators.

**Data Points**:
```
Source: CBOE (free delayed data)
├── VIX (30-day implied vol)
├── VIX3M (3-month implied vol)
├── VIX6M (6-month implied vol)
└── VIX futures curve (if available)
```

**Signals**:
| Signal | Calculation | Meaning |
|--------|-------------|---------|
| **Contango** | VIX < VIX3M < VIX6M | Normal, complacent market |
| **Backwardation** | VIX > VIX3M | Panic, fear of near-term |
| **VIX/VIX3M Ratio** | VIX / VIX3M | >1 = stress, <0.9 = calm |
| **Term Slope** | (VIX6M - VIX) / VIX | Steeper = more normal |

**Historical Context**: Backwardation has preceded or accompanied every major selloff.

**Visualization**:
- VIX term structure curve chart
- Current state vs historical percentile
- Contango/backwardation indicator with color coding

---

### 2.3 Options Put/Call Ratio

**What it adds**: Sentiment from sophisticated options traders.

**Source**: CBOE free data (delayed is fine for daily system)

**Signals**:
```python
Put/Call Signals:
├── Equity P/C Ratio (CBOE)
├── Index P/C Ratio (CBOE)
├── Total P/C Ratio
└── 5-day moving average (smoothed signal)

Interpretation:
├── P/C > 1.0: Bearish sentiment (often contrarian bullish)
├── P/C < 0.7: Bullish sentiment (often contrarian bearish)
└── Extreme readings (>1.2 or <0.5) = potential reversal
```

**Why it matters**: Options traders tend to be more sophisticated. Extreme readings often mark sentiment extremes that precede reversals.

**Visualization**: Put/Call gauge with historical percentile shading.

---

### 2.4 Economic Calendar

**What it adds**: Awareness of upcoming market-moving events.

**Implementation**: Simple, lightweight approach.

**Events to Track**:
```python
HIGH_IMPACT_EVENTS = [
    'FOMC',           # Fed rate decisions
    'CPI',            # Inflation
    'NFP',            # Employment
    'GDP',            # Growth
    'PCE',            # Fed's preferred inflation
]

# Per-holding
EARNINGS_CALENDAR = {
    # Fetch from Finnhub free tier or Yahoo Finance
}
```

**Source**: Finnhub free API or investing.com scrape

**Use in System**:
- Flag holdings with earnings within 5 days
- Reduce position sizes before FOMC
- Note expected volatility windows

**Visualization**:
- Calendar overlay on price charts
- Event markers with expected impact level
- Countdown to next major event

---

## Part 3: Visualization & Dashboard

### 3.1 Design Philosophy

The dashboard answers these questions at a glance:

1. **What is the market doing?** → Regime, trend, anomalies
2. **How confident is the model?** → Ensemble agreement, confidence bands
3. **Why does it think that?** → Attention weights, feature importance
4. **What are other signals saying?** → Sentiment gauges, VIX structure
5. **What did we decide?** → Actions with full reasoning
6. **How are we doing?** → Performance attribution

### 3.2 Main Interactive Chart

**Multi-layer chart with toggleable overlays**:

```
Layer 1: Price & Volume (always visible)
├── Candlestick chart
├── Volume bars (green/red by direction)
└── Moving averages (21, 50, 200)

Layer 2: Predictions & Uncertainty
├── Ensemble prediction direction
├── Confidence bands (68%, 95%)
└── Historical prediction markers (✓ right, ✗ wrong)

Layer 3: Regime & Anomaly
├── Background color by regime
├── Regime confidence shading
├── Anomaly markers (unusual conditions)

Layer 4: Momentum Decomposition
├── Trend strength indicator
├── Mean reversion risk zones
├── Momentum quality line

Layer 5: Events & Decisions
├── Economic event markers
├── Earnings dates
├── Buy/sell markers with tooltips
└── Stop loss levels
```

**Interaction**: Click any point to see full model state at that time.

### 3.3 Model Transparency Panel

**Ensemble Agreement View**:
```
┌─────────────────────────────────────────────┐
│ REGIME CLASSIFICATION                        │
├─────────────────────────────────────────────┤
│ GRU Model:         risk_on_trend (85%)      │
│ Transformer:       risk_on_trend (78%)      │
│ ─────────────────────────────────────────── │
│ Ensemble:          risk_on_trend            │
│ Agreement:         ████████████░░ 92%       │
│ Confidence:        ████████░░░░░░ 81%       │
└─────────────────────────────────────────────┘
```

**Attention Weights Panel**:
```
┌─────────────────────────────────────────────┐
│ WHAT THE MODEL FOCUSED ON TODAY             │
├─────────────────────────────────────────────┤
│ spy_vol_21d        ████████████████░ 0.23   │
│ yield_slope        ██████████████░░░ 0.19   │
│ vixy_return_21d    ████████████░░░░░ 0.15   │
│ spy_return_21d     ██████████░░░░░░░ 0.12   │
│ credit_spread      ████████░░░░░░░░░ 0.09   │
│ ...                                          │
└─────────────────────────────────────────────┘
```

**Time Attention** (which historical days mattered):
```
┌─────────────────────────────────────────────┐
│ TEMPORAL ATTENTION (last 21 days)           │
├─────────────────────────────────────────────┤
│ ░░░░▓▓░░░░░░▓▓▓░░░░░██                      │
│ -21d        -10d         -1d    today       │
│                                              │
│ Model focused on: days -5, -4, -3 (recent   │
│ volatility spike) and day -15 (Fed meeting) │
└─────────────────────────────────────────────┘
```

### 3.4 Decision Audit Trail

**Full transparency for every trade**:

```
┌─────────────────────────────────────────────┐
│ DECISION: BUY XLB                           │
├─────────────────────────────────────────────┤
│ ✓ Health Score: 0.78 (threshold: 0.60)      │
│ ✓ Regime: risk_on_trend (compatible)        │
│ ✓ Ensemble Direction: bullish (72% conf)    │
│ ✓ Model Agreement: 89%                      │
│ ─────────────────────────────────────────── │
│ MOMENTUM BREAKDOWN:                          │
│   Trend Strength:      0.82 ████████░░      │
│   Mean Reversion Risk: 0.35 ███░░░░░░░ LOW  │
│   Relative Momentum:   0.71 ███████░░░      │
│   Momentum Quality:    0.68 ██████░░░░      │
│ ─────────────────────────────────────────── │
│ SENTIMENT SIGNALS:                           │
│   Google Trends:   neutral (52nd %ile)      │
│   Put/Call Ratio:  0.82 (bullish)           │
│   VIX Structure:   contango (normal)        │
│ ─────────────────────────────────────────── │
│ LLM RISK CHECK:                              │
│   "No significant risks identified.          │
│    Materials sector benefiting from          │
│    infrastructure spending momentum."        │
│ ─────────────────────────────────────────── │
│ POSITION SIZE: $21,974 (22% of portfolio)   │
│ STOP LOSS: $44.35 (-10% trailing)           │
└─────────────────────────────────────────────┘
```

### 3.5 Sentiment Dashboard

**Three gauges showing different perspectives**:

```
┌─────────────────────────────────────────────┐
│           SENTIMENT SIGNALS                  │
├───────────────┬───────────────┬─────────────┤
│  RETAIL       │  OPTIONS      │  VOLATILITY │
│  (G.Trends)   │  (Put/Call)   │  (VIX)      │
│               │               │             │
│     ┌─┐       │     ┌─┐       │    ┌─┐      │
│   ╱     ╲     │   ╱     ╲     │  ╱     ╲    │
│  │   ●   │    │  │  ●    │    │ │    ●  │   │
│   ╲     ╱     │   ╲     ╱     │  ╲     ╱    │
│     └─┘       │     └─┘       │    └─┘      │
│               │               │             │
│   NEUTRAL     │   BULLISH     │   CALM      │
│   52nd %ile   │   35th %ile   │  28th %ile  │
├───────────────┴───────────────┴─────────────┤
│ DIVERGENCE ALERT: None                       │
└─────────────────────────────────────────────┘
```

**Divergence Detection**: When signals disagree, highlight it prominently.

### 3.6 Anomaly Monitor

```
┌─────────────────────────────────────────────┐
│ MARKET ANOMALY DETECTOR                      │
├─────────────────────────────────────────────┤
│                                              │
│  Anomaly Score: 23/100                       │
│  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  [  NORMAL  ]                                │
│                                              │
│  Contributing Factors:                       │
│  • Cross-asset correlation: normal           │
│  • Volume levels: normal                     │
│  • VIX structure: normal (contango)          │
│  • Model agreement: high (92%)               │
│                                              │
│  Last Anomaly: 2025-12-18 (score: 78)       │
│  "Correlation spike during Fed meeting"     │
│                                              │
└─────────────────────────────────────────────┘
```

### 3.7 Performance Attribution

```
┌─────────────────────────────────────────────┐
│ WHAT'S DRIVING RETURNS                       │
├─────────────────────────────────────────────┤
│ BY POSITION (MTD):                           │
│   SCHD    +$342   ████████████░░░░ +1.6%    │
│   XLB     +$187   ██████░░░░░░░░░░ +0.9%    │
│   VEA     -$54    ██░░░░░░░░░░░░░░ -0.2%    │
│   VLUE    +$23    █░░░░░░░░░░░░░░░ +0.1%    │
│                                              │
│ BY SIGNAL (hit rate this month):             │
│   Regime model:     6/8 correct (75%)        │
│   Health scores:    5/7 correct (71%)        │
│   Momentum decomp:  4/5 correct (80%)        │
│   LLM risk check:   2/2 avoided losses       │
│                                              │
│ BY REGIME (historical):                      │
│   risk_on_trend:    +12.3% (best)            │
│   calm_uptrend:     +8.7%                    │
│   choppy:           +1.2%                    │
│   risk_off_trend:   -2.1%                    │
│   high_vol_panic:   -4.5% (defensive helped) │
│                                              │
└─────────────────────────────────────────────┘
```

---

## Part 4: Implementation Plan

### Phase 2A: Foundation (Week 1)

| Task | Effort | Description |
|------|--------|-------------|
| Switch to Claude Haiku | 30 min | Update API calls to use Anthropic |
| Enable RegimeTransformer | 1 hour | Change config, retrain |
| Ensemble disagreement logic | 2 hours | Compute agreement score, adjust sizing |
| Confidence bands from ensemble | 2 hours | Use prediction std as uncertainty |
| Basic chart component scaffold | 4 hours | React component with layers |

### Phase 2B: Data Expansion (Week 2)

| Task | Effort | Description |
|------|--------|-------------|
| Fix GDELT ingestion | 2 hours | Debug 404 issues |
| Add Google Trends | 3 hours | pytrends integration |
| Add VIX term structure | 2 hours | CBOE data fetch |
| Add put/call ratio | 2 hours | CBOE equity P/C |
| Add economic calendar | 3 hours | Finnhub or scrape |

### Phase 2C: Model Enhancements (Week 3)

| Task | Effort | Description |
|------|--------|-------------|
| Anomaly detection (Isolation Forest) | 4 hours | Train on historical features |
| Momentum decomposition | 4 hours | Implement 4 components |
| Attention weight extraction | 3 hours | Export from transformer |
| Integrate into decision engine | 4 hours | Use new signals in decisions |

### Phase 2D: Dashboard Build (Week 4)

| Task | Effort | Description |
|------|--------|-------------|
| Multi-layer interactive chart | 8 hours | Price + overlays + toggles |
| Model transparency panel | 4 hours | Agreement, attention, features |
| Decision audit trail | 4 hours | Full reasoning display |
| Sentiment gauges | 3 hours | Three-gauge component |
| Anomaly monitor | 2 hours | Score + history |
| Performance attribution | 3 hours | By position, signal, regime |

---

## Part 5: Technical Specifications

### 5.1 Ensemble Configuration

```python
# training/config.py
ENSEMBLE_CONFIG = {
    'models': ['gru', 'transformer'],
    'weights': {'gru': 0.5, 'transformer': 0.5},  # Equal weight
    'disagreement_threshold': 0.3,  # Reduce size if disagreement > 30%
    'low_confidence_threshold': 0.6,  # Reduce size if confidence < 60%
}
```

### 5.2 Momentum Decomposition

```python
# src/utils/momentum_decomposition.py
def decompose_momentum(prices: pd.Series, benchmark: pd.Series) -> dict:
    """
    Decompose momentum into interpretable components.

    Returns:
        dict with keys: trend_strength, mean_reversion_risk,
                       relative_momentum, momentum_quality
    """
    # Trend strength: slope of log-price regression
    log_prices = np.log(prices)
    X = np.arange(len(prices)).reshape(-1, 1)
    slope = LinearRegression().fit(X, log_prices).coef_[0]
    trend_strength = norm.cdf(slope, loc=0, scale=0.001)  # Percentile

    # Mean reversion risk: z-score from moving averages
    ma21 = prices.rolling(21).mean().iloc[-1]
    ma50 = prices.rolling(50).mean().iloc[-1]
    current = prices.iloc[-1]
    z_score = ((current - ma21) / ma21 + (current - ma50) / ma50) / 2
    mean_reversion_risk = norm.cdf(z_score, loc=0, scale=0.03)

    # Relative momentum: return vs benchmark
    asset_return = (prices.iloc[-1] / prices.iloc[-63]) - 1
    bench_return = (benchmark.iloc[-1] / benchmark.iloc[-63]) - 1
    relative = asset_return - bench_return
    relative_momentum = norm.cdf(relative, loc=0, scale=0.1)

    # Momentum quality: Sharpe of daily returns
    daily_returns = prices.pct_change().dropna().tail(21)
    sharpe = daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    momentum_quality = norm.cdf(sharpe, loc=0, scale=0.2)

    return {
        'trend_strength': trend_strength,
        'mean_reversion_risk': mean_reversion_risk,
        'relative_momentum': relative_momentum,
        'momentum_quality': momentum_quality,
        'combined': (
            0.30 * trend_strength +
            0.25 * (1 - mean_reversion_risk) +
            0.25 * relative_momentum +
            0.20 * momentum_quality
        )
    }
```

### 5.3 Anomaly Detection

```python
# src/models/anomaly_detector.py
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    'cross_asset_correlation',
    'correlation_vs_historical',
    'volume_zscore',
    'volatility_zscore',
    'vix_term_slope',
    'credit_spread_change',
    'regime_disagreement',
]

class MarketAnomalyDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

    def fit(self, historical_features: pd.DataFrame):
        self.model.fit(historical_features[ANOMALY_FEATURES])

    def score(self, current_features: pd.Series) -> float:
        """Return anomaly score 0-100 (higher = more anomalous)."""
        raw_score = self.model.decision_function([current_features])[0]
        # Convert to 0-100 scale (more negative = more anomalous)
        return max(0, min(100, 50 - raw_score * 50))
```

### 5.4 Dashboard Data Structure

```json
{
  "timestamp": "2026-01-31T22:00:00Z",

  "regime": {
    "gru": {"label": "risk_on_trend", "confidence": 0.85, "probs": {...}},
    "transformer": {"label": "risk_on_trend", "confidence": 0.78, "probs": {...}},
    "ensemble": {"label": "risk_on_trend", "confidence": 0.81},
    "agreement": 0.92,
    "attention_weights": {
      "features": {"spy_vol_21d": 0.23, "yield_slope": 0.19, ...},
      "temporal": [0.02, 0.03, 0.05, 0.12, 0.18, ...]
    }
  },

  "predictions": {
    "direction": "bullish",
    "confidence": 0.72,
    "bands": {
      "lower_95": -0.02,
      "lower_68": -0.01,
      "point": 0.008,
      "upper_68": 0.018,
      "upper_95": 0.028
    }
  },

  "anomaly": {
    "score": 23,
    "level": "normal",
    "contributing_factors": []
  },

  "sentiment": {
    "google_trends": {"score": 0.52, "percentile": 52, "velocity": "stable"},
    "put_call": {"ratio": 0.82, "percentile": 35, "signal": "bullish"},
    "vix": {"level": 18.5, "term_structure": "contango", "percentile": 28}
  },

  "holdings": [
    {
      "symbol": "XLB",
      "momentum": {
        "trend_strength": 0.82,
        "mean_reversion_risk": 0.35,
        "relative_momentum": 0.71,
        "momentum_quality": 0.68,
        "combined": 0.72
      },
      "decision_audit": {
        "health_score": {"value": 0.78, "passed": true},
        "regime_compatible": true,
        "ensemble_direction": "bullish",
        "model_agreement": 0.89,
        "llm_risk_check": "No significant risks identified.",
        "position_size": 21974,
        "stop_loss": 44.35
      }
    }
  ],

  "calendar": {
    "upcoming": [
      {"date": "2026-02-05", "event": "FOMC", "impact": "high"},
      {"date": "2026-02-12", "event": "CPI", "impact": "high"}
    ],
    "earnings": [
      {"symbol": "XLB", "date": "2026-02-15", "days_until": 15}
    ]
  }
}
```

---

## Part 6: Cost Analysis

### Monthly Operating Costs

| Component | Current | After Phase 2 |
|-----------|---------|---------------|
| AWS Lambda | $6 | $7 |
| S3 Storage | $1 | $2 |
| Claude Haiku | $0 | $1 |
| Data APIs | $0 | $0 (all free) |
| **Total** | **$7** | **$10** |

### Data Source Costs

| Source | Cost | Notes |
|--------|------|-------|
| Google Trends | Free | pytrends library |
| CBOE VIX data | Free | Delayed data sufficient |
| CBOE Put/Call | Free | Delayed data sufficient |
| Finnhub calendar | Free | Free tier covers needs |
| GDELT | Free | Already integrated |

---

## Part 7: What We're NOT Doing (and Why)

| Candidate | Decision | Reason |
|-----------|----------|--------|
| TFT (Temporal Fusion Transformer) | Defer | Complex to implement; ensemble uncertainty achieves similar goal |
| SEC Insider Data | Skip | Too messy to parse reliably |
| Reddit Sentiment | Defer | Google Trends is cleaner, easier |
| Twitter/X | Skip | API too expensive ($100+/mo) |
| Reinforcement Learning | Defer | Overkill for current scope |
| Local LLM | Defer | User requested future consideration |

---

## Appendix: Migration Notes

### Switching to Claude Haiku

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=openai_key)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...]
)

# After (Anthropic)
from anthropic import Anthropic
client = Anthropic(api_key=anthropic_key)
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[...]
)
```

### Enabling Transformer Ensemble

```python
# training/train.py - change model_type
regime_model_gru = train_regime_model(context_df, model_type='gru', ...)
regime_model_transformer = train_regime_model(context_df, model_type='transformer', ...)

# Save both models
save_model(regime_model_gru, 'regime_gru_v{date}.pkl')
save_model(regime_model_transformer, 'regime_transformer_v{date}.pkl')
```

---

## Next Steps

1. **User Approval**: Confirm this revised proposal
2. **API Key**: Provide Anthropic API key for Claude Haiku
3. **Cursor Handoff**: Implement Phase 2A-2D per this spec
4. **Weekly Check-ins**: Review progress, adjust as needed

---

*This proposal prioritizes practical, interpretable enhancements over complex black-box models. Every addition provides transparency into the decision-making process while maintaining low operational costs.*
