# Complete System Specification: Autonomous Daily Investment Decision System
## Version 1.0 - Claude Code Build Target

---

# TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack & Dependencies](#tech-stack--dependencies)
4. [Project File Structure](#project-file-structure)
5. [ETF Universe Definition](#etf-universe-definition)
6. [Data Layer](#data-layer)
7. [Feature Engineering](#feature-engineering)
8. [Baseline Models (Phase 1)](#baseline-models-phase-1)
9. [Decision Engine](#decision-engine)
10. [LLM Integration](#llm-integration)
11. [Training System](#training-system)
12. [AWS Infrastructure](#aws-infrastructure)
13. [Frontend Dashboard](#frontend-dashboard)
14. [Deployment & Automation](#deployment--automation)
15. [Implementation Phases](#implementation-phases)
16. [Testing & Validation](#testing--validation)

---

# EXECUTIVE SUMMARY

## What We're Building

A fully autonomous daily investment decision system that:
- Ingests market data from free APIs
- Generates buy/sell signals using AI models + qualitative risk assessment
- Executes paper trades with full transparency
- Displays performance in an impressive React dashboard
- Runs completely hands-off with monthly model retraining

## Key Constraints

- **Daily resolution only** (no intraday)
- **Paper trading only** (no real money)
- **AWS cost < $20/month**
- **Local training** (MacBook, monthly, automated)
- **Free data sources** (Stooq, FRED, GDELT)
- **Impressive front-end** for portfolio presentation

## System Flow

```
Daily (10pm ET):
  EventBridge → Lambda → [Ingest → Features → Inference → Decisions → LLM Risk/Weather] → S3 Artifacts

Monthly (1st, 2am):
  launchd → train.py → Models uploaded to S3

Anytime:
  User visits dashboard → S3 static site → Reads latest artifacts → Shows impressive charts
```

---

# ARCHITECTURE OVERVIEW

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (FREE)                      │
├─────────────────────────────────────────────────────────────┤
│  Stooq (OHLCV) │ FRED (Rates) │ GDELT (Sentiment) │ Alpha V│
└────────┬────────────────┬────────────────┬─────────────┬────┘
         │                │                │             │
         ▼                ▼                ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│              DAILY PIPELINE (Lambda, 10pm ET)                │
├─────────────────────────────────────────────────────────────┤
│  1. Ingest prices/macro/sentiment                            │
│  2. Validate data quality                                    │
│  3. Build features (technical indicators)                    │
│  4. Run inference (regime + asset health)                    │
│  5. LLM risk check (bounded, 16 calls max)                   │
│  6. Decision engine (buy/sell/hold)                          │
│  7. Execute paper trades                                     │
│  8. LLM weather blurb (1 call)                               │
│  9. Write all artifacts to S3                                │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    S3 ARTIFACTS (Daily)                      │
├─────────────────────────────────────────────────────────────┤
│  • prices.parquet       • features.parquet                   │
│  • context.parquet      • inference.json                     │
│  • llm_risk.json        • decisions.json                     │
│  • portfolio_state.json • trades.jsonl                       │
│  • weather_blurb.json   • run_report.json                    │
│  • latest.json (pointer to current day)                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│          REACT DASHBOARD (S3 Static + CloudFront)            │
├─────────────────────────────────────────────────────────────┤
│  • Hero metrics (return, Sharpe, drawdown)                   │
│  • Equity curve with regime shading + trade markers          │
│  • Drawdown underwater plot                                  │
│  • Monthly returns heatmap                                   │
│  • Weather report + regime gauge                             │
│  • Portfolio table + candidates table                        │
│  • Recent trades with P&L                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         MONTHLY TRAINING (MacBook via launchd)               │
├─────────────────────────────────────────────────────────────┤
│  1. Download historical data from S3                         │
│  2. Train regime model (Transformer/GRU)                     │
│  3. Train asset health model (Autoencoder)                   │
│  4. Run evolutionary meta-search (genetic algorithm)         │
│  5. Backtest new models + templates                          │
│  6. Upload models/templates to S3                            │
│  7. Update models/latest.json                                │
└─────────────────────────────────────────────────────────────┘
```

---

# TECH STACK & DEPENDENCIES

## Backend (Python)

**Core Libraries:**
```python
# requirements.txt for Lambda
boto3==1.34.19
pandas==2.1.4
numpy==1.24.3
pyarrow==14.0.1
requests==2.31.0
openai==1.6.1
python-dateutil==2.8.2
```

**Training Libraries (local only):**
```python
# requirements-training.txt
torch==2.1.2
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3
boto3==1.34.19
tqdm==4.66.1
```

**Python Version:** 3.11

## Frontend (React)

**Framework & Libraries:**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "recharts": "^2.10.0",
    "date-fns": "^2.30.0",
    "tailwindcss": "^3.4.0",
    "axios": "^1.6.0"
  }
}
```

## AWS Services

- **Lambda:** Daily pipeline execution (Python 3.11 runtime)
- **EventBridge:** Cron scheduling (10pm ET weeknights)
- **S3:** Artifact storage + static website hosting
- **CloudFront:** (Optional) CDN for dashboard
- **IAM:** Least-privilege roles
- **CloudWatch:** Logs + cost monitoring
- **SNS:** (Optional) Failure alerts
- **Secrets Manager:** OpenAI API key storage

## Data APIs

- **Stooq:** Free daily OHLCV (no key required)
- **FRED:** Free macroeconomic data (API key required, free tier)
- **GDELT:** Free sentiment/news aggregates (no key required)
- **Alpha Vantage:** Free tier fallback for critical symbols (API key required)
- **OpenAI:** gpt-4o-mini for LLM calls (~$0.15/1M tokens input, ~$0.60/1M output)

---

# PROJECT FILE STRUCTURE

```
investment-system/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-training.txt
│
├── config/
│   ├── universe.csv                    # ETF universe definition
│   ├── aws_config.json                 # AWS region, bucket names
│   ├── data_sources.json               # API endpoints, keys
│   ├── decision_params.json            # Buy/sell thresholds
│   └── regime_compatibility.json       # Asset class × regime multipliers
│
├── lambda/
│   ├── handler.py                      # Main Lambda entry point
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── ingest_prices.py
│   │   ├── ingest_fred.py
│   │   ├── ingest_gdelt.py
│   │   ├── validate_data.py
│   │   ├── build_features.py
│   │   ├── run_inference.py
│   │   ├── llm_risk_check.py
│   │   ├── decision_engine.py
│   │   ├── paper_trader.py
│   │   ├── llm_weather.py
│   │   └── publish_artifacts.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_regime.py
│   │   ├── baseline_health.py
│   │   └── loader.py                   # Load trained models from S3
│   └── utils/
│       ├── __init__.py
│       ├── s3_client.py
│       ├── data_validation.py
│       ├── feature_utils.py
│       └── logging.py
│
├── training/
│   ├── train.py                        # Main training orchestrator
│   ├── train_regime.py                 # Regime model training
│   ├── train_health.py                 # Health model training
│   ├── evolutionary_search.py          # Policy template evolution
│   ├── backtest.py                     # Walk-forward backtesting
│   ├── models/
│   │   ├── regime_transformer.py
│   │   ├── health_autoencoder.py
│   │   └── architectures.py
│   └── utils/
│       ├── data_loader.py
│       ├── metrics.py
│       └── visualization.py
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── App.jsx                     # Main app component
│   │   ├── index.jsx                   # Entry point
│   │   ├── components/
│   │   │   ├── HeroMetrics.jsx
│   │   │   ├── EquityCurveChart.jsx
│   │   │   ├── DrawdownChart.jsx
│   │   │   ├── MonthlyReturnsHeatmap.jsx
│   │   │   ├── WeatherReport.jsx
│   │   │   ├── PortfolioTable.jsx
│   │   │   ├── CandidatesTable.jsx
│   │   │   └── RecentTrades.jsx
│   │   ├── utils/
│   │   │   ├── dataLoader.js
│   │   │   ├── calculations.js
│   │   │   └── formatters.js
│   │   └── styles/
│   │       └── tailwind.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── infrastructure/
│   ├── lambda_deploy.sh                # Package & deploy Lambda
│   ├── s3_setup.sh                     # Create buckets, enable hosting
│   ├── iam_policies.json               # IAM role definitions
│   ├── eventbridge_rule.json           # Cron schedule
│   └── launchd_plist/
│       └── com.investsys.train.plist   # Monthly training schedule
│
└── tests/
    ├── test_data_ingestion.py
    ├── test_features.py
    ├── test_baseline_models.py
    ├── test_decision_engine.py
    └── test_paper_trader.py
```

---

# ETF UNIVERSE DEFINITION

## Phase 1 Starter Universe (60 symbols)

**File:** `config/universe.csv`

```csv
symbol,asset_class,sector,leverage_flag,eligible
SPY,equity,broad,0,1
QQQ,equity,broad,0,1
IWM,equity,broad,0,1
DIA,equity,broad,0,1
RSP,equity,broad,0,1
VTI,equity,broad,0,1
VOO,equity,broad,0,1
IVV,equity,broad,0,1
VT,equity,global,0,1
VEA,equity,international_dev,0,1
VWO,equity,international_em,0,1
EFA,equity,international_dev,0,1
EEM,equity,international_em,0,1
EWJ,equity,country_japan,0,1
EWZ,equity,country_brazil,0,1
FXI,equity,country_china,0,1
INDA,equity,country_india,0,1
VGK,equity,region_europe,0,1
ARKK,equity,theme_innovation,0,1
XLK,equity,sector_tech,0,1
XLF,equity,sector_financials,0,1
XLE,equity,sector_energy,0,1
XLV,equity,sector_healthcare,0,1
XLI,equity,sector_industrials,0,1
XLY,equity,sector_cons_disc,0,1
XLP,equity,sector_cons_staples,0,1
XLU,equity,sector_utilities,0,1
XLB,equity,sector_materials,0,1
XLC,equity,sector_comm,0,1
VNQ,equity,sector_reit,0,1
IYR,equity,sector_reit,0,1
SOXX,equity,industry_semis,0,1
SMH,equity,industry_semis,0,1
XBI,equity,industry_biotech,0,1
IBB,equity,industry_biotech,0,1
KRE,equity,industry_regional_banks,0,1
XRT,equity,industry_retail,0,1
ITA,equity,industry_aerospace_defense,0,1
IYT,equity,industry_transport,0,1
MTUM,equity,factor_momentum,0,1
QUAL,equity,factor_quality,0,1
VLUE,equity,factor_value,0,1
USMV,equity,factor_minvol,0,1
VIG,equity,factor_dividend_growth,0,1
SCHD,equity,factor_dividend,0,1
VUG,equity,style_growth,0,1
VTV,equity,style_value,0,1
TLT,bond,treas_long,0,1
IEF,bond,treas_intermediate,0,1
SHY,bond,treas_short,0,1
TIP,bond,treas_tips,0,1
LQD,bond,credit_investment_grade,0,1
HYG,bond,credit_high_yield,0,1
AGG,bond,aggregate,0,1
BND,bond,aggregate,0,1
MUB,bond,muni,0,1
GLD,commodity,gold,0,1
SLV,commodity,silver,0,1
DBC,commodity,broad_commodities,0,1
USO,commodity,oil,0,1
UNG,commodity,natural_gas,0,1
UUP,fx,usd,0,1
FXE,fx,eur,0,1
VIXY,vol,volatility,0,1
```

## Required Proxy Symbols

These **must be present** for the system to run:
- **SPY** (benchmark)
- **QQQ** (tech benchmark)
- **IWM** (small-cap benchmark)
- **TLT** (long-duration treasuries)
- **IEF** (intermediate treasuries)
- **HYG** (high-yield credit)
- **LQD** (investment-grade credit)
- **GLD** (gold/safe haven)
- **VIXY** (volatility proxy)

---

# DATA LAYER

## 6.1 Stooq (Primary Price Source)

**Endpoint:** `https://stooq.com/q/d/l/?s={symbol}&i=d`

**Response Format:** CSV with columns:
```
Date,Open,High,Low,Close,Volume
2025-01-27,450.23,452.10,448.50,451.80,85342100
```

**Implementation:**
```python
import requests
import pandas as pd
from io import StringIO

def fetch_stooq_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        lookback_days: How many days of history (default 365)
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Keep only recent data
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        df = df[df['date'] >= cutoff]
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"Stooq fetch failed for {symbol}: {e}")
        return pd.DataFrame()
```

**Validation Requirements:**
- Must have data for >= 95% of universe
- Latest date must be within 2 business days
- No missing OHLCV values
- Volume > 0

## 6.2 Alpha Vantage (Fallback)

**Endpoint:** `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}&outputsize=full`

**Usage:** Only for critical symbols if Stooq fails (SPY, QQQ, IWM, TLT, IEF, HYG, LQD, GLD + current holdings)

**Free Tier Limit:** 25 calls/day

**Implementation:**
```python
def fetch_alphavantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    """Fallback price source for critical symbols."""
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            return pd.DataFrame()
            
        ts = data['Time Series (Daily)']
        
        records = []
        for date_str, values in ts.items():
            records.append({
                'date': pd.to_datetime(date_str),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
            
        df = pd.DataFrame(records).sort_values('date')
        return df.tail(365)  # Keep last year
        
    except Exception as e:
        print(f"Alpha Vantage fetch failed for {symbol}: {e}")
        return pd.DataFrame()
```

## 6.3 FRED (Macroeconomic Data)

**Endpoint:** `https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={key}&file_type=json`

**Required Series:**
```python
FRED_SERIES = {
    'DGS2': '2-Year Treasury Yield',
    'DGS10': '10-Year Treasury Yield',
    'VIXCLS': 'VIX Close (backup)',
    'DCOILWTICO': 'WTI Oil Price',
    'DEXUSEU': 'USD/EUR Exchange Rate'
}
```

**Implementation:**
```python
def fetch_fred_series(series_id: str, api_key: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch daily series from FRED.
    
    Returns:
        DataFrame with columns: date, value
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if 'observations' not in data:
            return pd.DataFrame()
            
        records = []
        for obs in data['observations']:
            if obs['value'] != '.':  # FRED uses '.' for missing
                records.append({
                    'date': pd.to_datetime(obs['date']),
                    'value': float(obs['value'])
                })
                
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"FRED fetch failed for {series_id}: {e}")
        return pd.DataFrame()
```

**Validation:**
- Latest value within 5 business days is acceptable
- Forward-fill missing values (max 5 days)

## 6.4 GDELT (Sentiment Aggregates)

**Approach:** Daily bulk files (no real-time needed)

**URL Pattern:** `http://data.gdeltproject.org/gdeltv2/{YYYYMMDD}.gkgcounts.csv.zip`

**Implementation:**
```python
import zipfile
from io import BytesIO

def fetch_gdelt_daily_aggregate(date: pd.Timestamp) -> dict:
    """
    Fetch GDELT daily aggregate features.
    
    Returns:
        dict with keys: doc_count, avg_tone, tone_std, neg_tone_share
    """
    date_str = date.strftime('%Y%m%d')
    url = f"http://data.gdeltproject.org/gdeltv2/{date_str}.gkgcounts.csv.zip"
    
    try:
        response = requests.get(url, timeout=30)
        
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            csv_name = f"{date_str}.gkgcounts.csv"
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep='\t', header=None, 
                                names=['date', 'count', 'type', 'fips', 'name'])
        
        # Aggregate to single daily features
        doc_count = len(df)
        
        # Tone is in a separate file, simplified here:
        # In production, also fetch {YYYYMMDD}.gkg.csv.zip and parse tone
        
        return {
            'gdelt_doc_count': doc_count,
            'gdelt_avg_tone': 0.0,  # Placeholder - implement from gkg.csv
            'gdelt_tone_std': 0.0,
            'gdelt_neg_tone_share': 0.0
        }
        
    except Exception as e:
        print(f"GDELT fetch failed for {date_str}: {e}")
        return {
            'gdelt_doc_count': 0,
            'gdelt_avg_tone': 0.0,
            'gdelt_tone_std': 0.0,
            'gdelt_neg_tone_share': 0.0
        }
```

**MVP Simplification:**
For Phase 1, GDELT can be **optional**. Set all values to 0 and mark as `gdelt_available=False` in context.parquet.

---

# FEATURE ENGINEERING

## 7.1 Per-Asset Features

**Computed for each symbol daily:**

```python
def compute_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features for a single asset.
    
    Input:
        df: DataFrame with columns [date, open, high, low, close, volume]
        
    Output:
        df with additional feature columns
    """
    df = df.sort_values('date').copy()
    
    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_21d'] = df['close'].pct_change(21)
    df['return_63d'] = df['close'].pct_change(63)
    
    # Volatility (annualized)
    df['vol_21d'] = df['return_1d'].rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['return_1d'].rolling(63).std() * np.sqrt(252)
    
    # Drawdown
    df['peak_63d'] = df['close'].rolling(63, min_periods=1).max()
    df['drawdown_63d'] = (df['close'] - df['peak_63d']) / df['peak_63d']
    
    # Trend (log slope over 63 days)
    df['log_price'] = np.log(df['close'])
    df['trend_63d'] = df['log_price'].diff(63) / 63
    
    # Liquidity proxy
    df['volume_ma21'] = df['volume'].rolling(21).median()
    
    return df

def compute_relative_strength(asset_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative strength vs SPY.
    
    Both DataFrames must have aligned dates.
    """
    merged = asset_df.merge(spy_df[['date', 'return_21d', 'return_63d']], 
                            on='date', suffixes=('', '_spy'))
    
    merged['rel_strength_21d'] = merged['return_21d'] - merged['return_21d_spy']
    merged['rel_strength_63d'] = merged['return_63d'] - merged['return_63d_spy']
    
    return merged
```

## 7.2 Global Context Features

**Computed once per day:**

```python
def compute_context_features(spy_df: pd.DataFrame, 
                             tlt_df: pd.DataFrame,
                             hyg_df: pd.DataFrame, 
                             ief_df: pd.DataFrame,
                             vixy_df: pd.DataFrame,
                             fred_data: dict,
                             gdelt_data: dict) -> pd.DataFrame:
    """
    Compute market-wide context features.
    
    Returns:
        Single-row DataFrame with context for the latest date
    """
    latest_date = spy_df['date'].max()
    
    # Index returns (already computed by compute_asset_features)
    spy_latest = spy_df[spy_df['date'] == latest_date].iloc[0]
    
    # Rates
    rate_2y = fred_data.get('DGS2', 0)
    rate_10y = fred_data.get('DGS10', 0)
    yield_slope = rate_10y - rate_2y
    
    # Credit spread proxy (HYG - IEF performance)
    hyg_latest = hyg_df[hyg_df['date'] == latest_date].iloc[0]
    ief_latest = ief_df[ief_df['date'] == latest_date].iloc[0]
    credit_spread_proxy = hyg_latest['return_21d'] - ief_latest['return_21d']
    
    # Risk-off proxy (SPY - TLT trend)
    tlt_latest = tlt_df[tlt_df['date'] == latest_date].iloc[0]
    risk_off_proxy = spy_latest['return_21d'] - tlt_latest['return_21d']
    
    # Vol proxy
    vixy_latest = vixy_df[vixy_df['date'] == latest_date].iloc[0] if len(vixy_df) > 0 else {}
    
    context = {
        'date': latest_date,
        'spy_return_1d': spy_latest['return_1d'],
        'spy_return_21d': spy_latest['return_21d'],
        'spy_vol_21d': spy_latest['vol_21d'],
        'rate_2y': rate_2y,
        'rate_10y': rate_10y,
        'yield_slope': yield_slope,
        'credit_spread_proxy': credit_spread_proxy,
        'risk_off_proxy': risk_off_proxy,
        'vixy_return_21d': vixy_latest.get('return_21d', 0),
        **gdelt_data
    }
    
    return pd.DataFrame([context])
```

## 7.3 Feature Storage Schema

**features.parquet** columns:
```
date            : datetime64[ns]
symbol          : str
return_1d       : float64
return_5d       : float64
return_21d      : float64
return_63d      : float64
vol_21d         : float64
vol_63d         : float64
drawdown_63d    : float64
trend_63d       : float64
volume_ma21     : float64
rel_strength_21d: float64
rel_strength_63d: float64
```

**context.parquet** columns:
```
date                 : datetime64[ns]
spy_return_1d        : float64
spy_return_21d       : float64
spy_vol_21d          : float64
rate_2y              : float64
rate_10y             : float64
yield_slope          : float64
credit_spread_proxy  : float64
risk_off_proxy       : float64
vixy_return_21d      : float64
gdelt_doc_count      : int64
gdelt_avg_tone       : float64
gdelt_tone_std       : float64
gdelt_neg_tone_share : float64
```

---

# BASELINE MODELS (PHASE 1)

## 8.1 Baseline Regime Model

**Output:** One of 5 regime labels + probabilities

**Regimes:**
- `calm_uptrend`
- `risk_on_trend`
- `risk_off_trend`
- `choppy`
- `high_vol_panic`

**Logic:**

```python
def baseline_regime_model(context: pd.Series) -> dict:
    """
    Deterministic baseline regime classification.
    
    Args:
        context: Single row from context.parquet
        
    Returns:
        {
            'regime_label': str,
            'regime_probs': dict[str, float],
            'regime_embedding': list[float]  # dummy for now
        }
    """
    spy_21d_ret = context['spy_return_21d']
    spy_21d_vol = context['spy_vol_21d']
    credit_stress = context['credit_spread_proxy']
    vixy_21d_ret = context['vixy_return_21d']
    
    # Compute percentiles from historical data (stored in S3)
    # For baseline, use hardcoded thresholds:
    vol_p85 = 0.25  # 25% annualized vol
    vol_p50 = 0.18
    vol_p40 = 0.15
    
    # Rules (in order):
    if (vixy_21d_ret > 0.20 or spy_21d_vol > vol_p85) and spy_21d_ret < -0.08:
        label = 'high_vol_panic'
        
    elif spy_21d_ret < -0.05 or credit_stress < -0.03:
        label = 'risk_off_trend'
        
    elif spy_21d_ret > 0.06 and spy_21d_vol < vol_p40:
        label = 'calm_uptrend'
        
    elif abs(spy_21d_ret) < 0.02 and spy_21d_vol > vol_p50:
        label = 'choppy'
        
    else:
        label = 'risk_on_trend'
    
    # Generate dummy probabilities (deterministic baseline)
    probs = {regime: 0.0 for regime in ['calm_uptrend', 'risk_on_trend', 
                                         'risk_off_trend', 'choppy', 'high_vol_panic']}
    probs[label] = 1.0
    
    # Dummy embedding (for compatibility)
    embedding = [0.0] * 8
    
    return {
        'regime_label': label,
        'regime_probs': probs,
        'regime_embedding': embedding
    }
```

## 8.2 Baseline Asset Health Model

**Output:** Health score (0-1), volatility bucket, behavior classification

**Logic:**

```python
def baseline_health_model(features_df: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    """
    Deterministic baseline health scoring.
    
    Args:
        features_df: DataFrame with all symbols' features
        latest_date: Date to score
        
    Returns:
        DataFrame with columns: symbol, health_score, vol_bucket, behavior, latent
    """
    # Filter to latest date
    latest = features_df[features_df['date'] == latest_date].copy()
    
    if len(latest) == 0:
        return pd.DataFrame()
    
    # Compute cross-sectional ranks (0-1)
    latest['mom_63_rank'] = latest['return_63d'].rank(pct=True)
    latest['mom_21_rank'] = latest['return_21d'].rank(pct=True)
    latest['vol_63_rank'] = latest['vol_63d'].rank(pct=True)
    latest['dd_63_rank'] = (-latest['drawdown_63d']).rank(pct=True)  # Less negative = better
    latest['rs_63_rank'] = latest['rel_strength_63d'].rank(pct=True)
    
    # Composite scores
    latest['momentum'] = 0.6 * latest['mom_63_rank'] + 0.4 * latest['mom_21_rank']
    latest['risk'] = 0.6 * latest['vol_63_rank'] + 0.4 * (1 - latest['dd_63_rank'])
    
    # Health formula
    latest['health_score'] = (
        0.45 * latest['momentum'] + 
        0.35 * latest['rs_63_rank'] + 
        0.20 * (1 - latest['risk'])
    ).clip(0, 1)
    
    # Volatility bucket
    latest['vol_bucket'] = pd.cut(
        latest['vol_63_rank'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['low', 'med', 'high']
    )
    
    # Behavior classification
    def classify_behavior(row):
        if row['mom_63_rank'] > 0.66 and row['dd_63_rank'] > 0.5:
            return 'momentum'
        else:
            return 'mixed'
    
    latest['behavior'] = latest.apply(classify_behavior, axis=1)
    
    # Dummy latent vector
    latest['latent'] = [[0.0] * 16] * len(latest)
    
    return latest[['symbol', 'health_score', 'vol_bucket', 'behavior', 'latent']]
```

## 8.3 Model Output Schema

**inference.json structure:**

```json
{
  "date": "2025-01-27",
  "model_versions": {
    "regime": "baseline_v1",
    "health": "baseline_v1"
  },
  "regime": {
    "label": "risk_on_trend",
    "probs": {
      "calm_uptrend": 0.0,
      "risk_on_trend": 1.0,
      "risk_off_trend": 0.0,
      "choppy": 0.0,
      "high_vol_panic": 0.0
    },
    "embedding": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "asset_health": [
    {
      "symbol": "SPY",
      "health_score": 0.78,
      "vol_bucket": "low",
      "behavior": "momentum",
      "latent": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  ]
}
```

---

# DECISION ENGINE

## 9.1 Portfolio Constraints

**Defaults (stored in config/decision_params.json):**

```json
{
  "max_positions": 8,
  "max_position_weight": 0.20,
  "max_sector_weight": 0.35,
  "min_cash_reserve_by_regime": {
    "calm_uptrend": 0.10,
    "risk_on_trend": 0.10,
    "choppy": 0.20,
    "risk_off_trend": 0.40,
    "high_vol_panic": 0.40
  },
  "leveraged_constraints": {
    "max_weight": 0.10,
    "max_hold_days": 10,
    "stop_multiplier": 0.6
  },
  "buy_score_threshold": 0.65,
  "min_health_buy": 0.60,
  "sell_health_threshold": 0.35,
  "sell_health_days": 3,
  "reduce_health_drop": 0.20,
  "trailing_stop_base": 0.10,
  "trailing_stop_leveraged": 0.06,
  "min_order_dollars": 250
}
```

## 9.2 Regime Compatibility Table

**File:** `config/regime_compatibility.json`

```json
{
  "risk_on_trend": {
    "equity": 1.10,
    "bond": 0.95,
    "commodity": 1.00,
    "vol": 0.90,
    "sector_tech": 1.15,
    "sector_energy": 1.10,
    "sector_utilities": 0.90,
    "treas_long": 0.90
  },
  "risk_off_trend": {
    "equity": 0.85,
    "bond": 1.10,
    "commodity": 0.95,
    "vol": 1.05,
    "sector_tech": 0.80,
    "sector_cons_staples": 1.10,
    "sector_utilities": 1.10,
    "treas_long": 1.15
  },
  "high_vol_panic": {
    "equity": 0.50,
    "bond": 1.20,
    "commodity": 1.10,
    "vol": 1.30,
    "treas_long": 1.25,
    "credit_high_yield": 0.60
  },
  "choppy": {
    "equity": 0.95,
    "bond": 1.05,
    "commodity": 1.00,
    "vol": 1.00
  },
  "calm_uptrend": {
    "equity": 1.10,
    "bond": 0.95,
    "commodity": 1.00,
    "vol": 0.85
  }
}
```

## 9.3 Candidate Scoring

```python
def score_candidates(asset_health_df: pd.DataFrame,
                    features_df: pd.DataFrame,
                    universe_df: pd.DataFrame,
                    regime_label: str,
                    regime_compat: dict) -> pd.DataFrame:
    """
    Score all eligible candidates for buying.
    
    Returns:
        DataFrame with: symbol, score, health_score, reason_code
    """
    # Merge asset health with features and universe metadata
    merged = asset_health_df.merge(features_df, on='symbol')
    merged = merged.merge(universe_df, on='symbol')
    
    # Filter to eligible only
    merged = merged[merged['eligible'] == 1]
    
    # Compute base score components (using ranks)
    merged['momentum_score'] = 0.6 * merged['mom_63_rank'] + 0.4 * merged['mom_21_rank']
    merged['risk_penalty'] = 0.6 * merged['vol_63_rank'] + 0.4 * (1 - merged['dd_63_rank'])
    merged['relative_score'] = merged['rs_63_rank']
    
    # Base score
    merged['base_score'] = (
        0.45 * merged['health_score'] +
        0.25 * merged['momentum_score'] +
        0.20 * merged['relative_score'] -
        0.10 * merged['risk_penalty']
    )
    
    # Apply regime compatibility multiplier
    def get_multiplier(row):
        # Check sector first, then asset_class
        sector_key = row['sector']
        asset_key = row['asset_class']
        
        mult = regime_compat.get(regime_label, {}).get(sector_key, 
                regime_compat.get(regime_label, {}).get(asset_key, 1.0))
        return mult
    
    merged['regime_multiplier'] = merged.apply(get_multiplier, axis=1)
    merged['final_score'] = (merged['base_score'] * merged['regime_multiplier']).clip(0, 1)
    
    # Add reason codes
    merged['reason_code'] = 'SCORED'
    
    return merged[['symbol', 'final_score', 'health_score', 'vol_bucket', 'reason_code']]
```

## 9.4 Buy Rules

```python
def filter_buy_candidates(scored_df: pd.DataFrame,
                         current_holdings: list,
                         params: dict,
                         regime_label: str,
                         llm_risk_flags: dict) -> pd.DataFrame:
    """
    Apply buy filters.
    
    Returns:
        DataFrame of buy-eligible candidates
    """
    candidates = scored_df.copy()
    
    # Filter: not already held
    candidates = candidates[~candidates['symbol'].isin(current_holdings)]
    
    # Filter: score threshold
    candidates['pass_score'] = candidates['final_score'] >= params['buy_score_threshold']
    
    # Filter: health threshold
    candidates['pass_health'] = candidates['health_score'] >= params['min_health_buy']
    
    # Filter: vol bucket (high vol allowed only in calm_uptrend with exceptional score)
    def check_vol(row):
        if row['vol_bucket'] == 'high':
            return regime_label == 'calm_uptrend' and row['final_score'] > 0.80
        return True
    
    candidates['pass_vol'] = candidates.apply(check_vol, axis=1)
    
    # Filter: LLM veto
    def check_llm_veto(row):
        symbol = row['symbol']
        if symbol in llm_risk_flags:
            return not llm_risk_flags[symbol].get('structural_risk_veto', False)
        return True
    
    candidates['pass_llm'] = candidates.apply(check_llm_veto, axis=1)
    
    # Filter: panic mode (only bonds/commodities/defensives)
    if regime_label == 'high_vol_panic':
        candidates['pass_regime'] = candidates['asset_class'].isin(['bond', 'commodity'])
    else:
        candidates['pass_regime'] = True
    
    # Apply all filters
    candidates = candidates[
        candidates['pass_score'] &
        candidates['pass_health'] &
        candidates['pass_vol'] &
        candidates['pass_llm'] &
        candidates['pass_regime']
    ]
    
    # Sort by score descending
    candidates = candidates.sort_values('final_score', ascending=False)
    
    return candidates
```

## 9.5 Sell/Reduce Rules

```python
def evaluate_holdings(portfolio_state: dict,
                     asset_health_df: pd.DataFrame,
                     prices_df: pd.DataFrame,
                     params: dict,
                     regime_label: str,
                     llm_risk_flags: dict) -> list:
    """
    Evaluate each holding for SELL or REDUCE signals.
    
    Returns:
        List of dicts: [{'symbol': 'SPY', 'action': 'SELL', 'reason': 'STOP_HIT'}, ...]
    """
    actions = []
    
    for holding in portfolio_state['holdings']:
        symbol = holding['symbol']
        entry_price = holding['entry_price']
        entry_date = holding['entry_date']
        current_price = prices_df[prices_df['symbol'] == symbol]['close'].iloc[-1]
        peak_price = holding['peak_price']
        
        # Get current health
        health_row = asset_health_df[asset_health_df['symbol'] == symbol]
        if len(health_row) == 0:
            continue
        current_health = health_row['health_score'].iloc[0]
        
        # Days held
        days_held = (pd.Timestamp.now() - pd.to_datetime(entry_date)).days
        
        # Trailing stop
        stop_pct = params['trailing_stop_base']
        if holding.get('leverage_flag', 0) == 1:
            stop_pct = params['trailing_stop_leveraged']
        
        trailing_stop_price = peak_price * (1 - stop_pct)
        
        # CHECK SELL TRIGGERS
        
        # 1. Stop hit
        if current_price <= trailing_stop_price:
            actions.append({
                'symbol': symbol,
                'action': 'SELL',
                'reason': 'STOP_HIT',
                'details': f'Price {current_price:.2f} <= stop {trailing_stop_price:.2f}'
            })
            continue
        
        # 2. Health collapse (3 consecutive days)
        # (Requires tracking health history - simplified here)
        if current_health <= params['sell_health_threshold']:
            actions.append({
                'symbol': symbol,
                'action': 'SELL',
                'reason': 'HEALTH_COLLAPSE',
                'details': f'Health {current_health:.2f} <= {params["sell_health_threshold"]}'
            })
            continue
        
        # 3. Panic mode (asset not allowed)
        if regime_label == 'high_vol_panic':
            asset_class = holding.get('asset_class', 'equity')
            if asset_class not in ['bond', 'commodity']:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'REGIME_PANIC',
                    'details': f'Asset class {asset_class} not allowed in panic'
                })
                continue
        
        # 4. LLM structural risk veto
        if symbol in llm_risk_flags:
            if llm_risk_flags[symbol].get('structural_risk_veto', False):
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'LLM_VETO',
                    'details': llm_risk_flags[symbol].get('one_sentence_rationale', '')
                })
                continue
        
        # 5. Leveraged max hold days
        if holding.get('leverage_flag', 0) == 1:
            if days_held >= params['leveraged_constraints']['max_hold_days']:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'LEVERAGE_HOLD_CAP',
                    'details': f'Held {days_held} days >= max {params["leveraged_constraints"]["max_hold_days"]}'
                })
                continue
        
        # CHECK REDUCE TRIGGERS
        
        # 6. Health drop
        entry_health = holding.get('entry_health', current_health)
        peak_health = holding.get('peak_health', entry_health)
        if current_health < peak_health - params['reduce_health_drop']:
            actions.append({
                'symbol': symbol,
                'action': 'REDUCE',
                'reason': 'HEALTH_DROP',
                'details': f'Health dropped {peak_health - current_health:.2f}'
            })
            continue
        
        # 7. Regime shift to choppy/risk-off
        if regime_label in ['choppy', 'risk_off_trend']:
            entry_regime = holding.get('entry_regime', 'risk_on_trend')
            if entry_regime == 'calm_uptrend' or entry_regime == 'risk_on_trend':
                actions.append({
                    'symbol': symbol,
                    'action': 'REDUCE',
                    'reason': 'REGIME_SHIFT',
                    'details': f'Regime changed from {entry_regime} to {regime_label}'
                })
                continue
    
    return actions
```

## 9.6 Position Sizing

```python
def compute_position_size(symbol: str,
                         target_weight: float,
                         portfolio_value: float,
                         current_price: float,
                         vol_bucket: str,
                         regime_label: str,
                         params: dict,
                         llm_confidence_adj: float = 0.0) -> dict:
    """
    Compute target shares and dollars for a position.
    
    Returns:
        {'shares': int, 'dollars': float, 'final_weight': float}
    """
    # Base target
    base_dollars = portfolio_value * target_weight
    
    # Adjust by volatility bucket
    vol_adj = {'low': 1.10, 'med': 1.0, 'high': 0.80}[vol_bucket]
    
    # Adjust by regime
    regime_adj = {
        'calm_uptrend': 1.10,
        'risk_on_trend': 1.10,
        'choppy': 0.90,
        'risk_off_trend': 0.80,
        'high_vol_panic': 0.50
    }[regime_label]
    
    # Adjust by LLM confidence
    llm_adj = 1.0 - llm_confidence_adj  # llm_confidence_adj is 0.0-0.5 reduction
    
    # Final target
    adjusted_dollars = base_dollars * vol_adj * regime_adj * llm_adj
    
    # Shares
    shares = int(adjusted_dollars / current_price)
    
    # Check minimum
    if shares * current_price < params['min_order_dollars']:
        return {'shares': 0, 'dollars': 0, 'final_weight': 0.0}
    
    actual_dollars = shares * current_price
    final_weight = actual_dollars / portfolio_value
    
    return {
        'shares': shares,
        'dollars': actual_dollars,
        'final_weight': final_weight
    }
```

## 9.7 Decision Output Schema

**decisions.json structure:**

```json
{
  "date": "2025-01-27",
  "regime": "risk_on_trend",
  "actions": [
    {
      "action": "BUY",
      "symbol": "XLK",
      "shares": 45,
      "price": 220.50,
      "dollars": 9922.50,
      "weight": 0.099,
      "reason": "SCORE_0.78_HEALTH_0.82",
      "score": 0.78,
      "health": 0.82,
      "vol_bucket": "med"
    },
    {
      "action": "SELL",
      "symbol": "XLE",
      "shares": 30,
      "price": 95.20,
      "dollars": 2856.00,
      "reason": "STOP_HIT",
      "details": "Price 95.20 <= stop 96.50"
    }
  ],
  "pass_filters": {
    "price_coverage": 0.983,
    "context_freshness": 1,
    "degraded_mode": false
  }
}
```

---

# LLM INTEGRATION

## 10.1 LLM Call Budget

**Hard caps:**
- **Risk checks:** 16 calls/day max (holdings + top 5 candidates + 3 shock list)
- **Weather blurb:** 1 call/day
- **Total:** 17 calls/day = ~510 calls/month

**Cost estimate (gpt-4o-mini):**
- Input: ~500 tokens/call = 255k tokens/month = $0.04
- Output: ~150 tokens/call = 76.5k tokens/month = $0.05
- **Total: ~$0.09/month**

## 10.2 LLM Risk Check

**Prompt Template:**

```python
RISK_CHECK_PROMPT = """
You are a financial risk analyst. Analyze this ETF for structural risks.

ETF: {symbol} - {name}
Asset Class: {asset_class}
Sector: {sector}

Recent Performance:
- 5-day return: {return_5d:.2%}
- 21-day return: {return_21d:.2%}
- Current volatility: {vol_21d:.1%}

Context:
- Market regime: {regime}
- GDELT sentiment: {gdelt_tone:.2f}

Task:
1. Identify any red flags: regulatory risk, litigation, fraud allegations, leveraged ETF decay, liquidity concerns
2. Rate severity: 0=none, 1=monitor, 2=caution, 3=critical
3. Should this be vetoed from trading? (yes/no)
4. Confidence adjustment: 0.0-0.5 (how much to reduce position size)
5. One-sentence rationale

Respond ONLY with JSON:
{{
  "risk_flags": ["regulatory", "liquidity"],
  "severity": 2,
  "structural_risk_veto": false,
  "confidence_adjustment": 0.1,
  "one_sentence_rationale": "Elevated regulatory scrutiny but no immediate threat"
}}
"""

def call_llm_risk_check(symbol: str, 
                       metadata: dict, 
                       performance: dict,
                       context: dict,
                       api_key: str) -> dict:
    """
    Call OpenAI API for risk assessment.
    
    Returns:
        Dict matching schema above, or empty dict on failure
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = RISK_CHECK_PROMPT.format(
        symbol=symbol,
        name=metadata.get('name', symbol),
        asset_class=metadata['asset_class'],
        sector=metadata['sector'],
        return_5d=performance['return_5d'],
        return_21d=performance['return_21d'],
        vol_21d=performance['vol_21d'],
        regime=context['regime'],
        gdelt_tone=context.get('gdelt_avg_tone', 0)
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial risk analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate schema
        required_keys = ['risk_flags', 'severity', 'structural_risk_veto', 
                        'confidence_adjustment', 'one_sentence_rationale']
        if not all(k in result for k in required_keys):
            return {}
        
        return result
        
    except Exception as e:
        print(f"LLM risk check failed for {symbol}: {e}")
        return {}
```

**llm_risk.json structure:**

```json
{
  "date": "2025-01-27",
  "status": "ok",
  "calls_made": 12,
  "risks": {
    "XLK": {
      "risk_flags": [],
      "severity": 0,
      "structural_risk_veto": false,
      "confidence_adjustment": 0.0,
      "one_sentence_rationale": "No significant risks detected"
    },
    "ARKK": {
      "risk_flags": ["volatility", "concentration"],
      "severity": 2,
      "structural_risk_veto": false,
      "confidence_adjustment": 0.2,
      "one_sentence_rationale": "High concentration in speculative tech names warrants caution"
    }
  }
}
```

## 10.3 Weather Blurb

**Prompt Template:**

```python
WEATHER_BLURB_PROMPT = """
You are a portfolio manager writing a daily market brief.

Date: {date}
Regime: {regime}

Portfolio State:
- Total value: ${portfolio_value:,.0f}
- Cash: ${cash:,.0f} ({cash_pct:.1%})
- Positions: {num_positions}
- Day return: {day_return:+.2%}

Market Context:
- SPY: {spy_return:.2%} (21d: {spy_21d:.2%})
- VIX proxy: {vixy_return:.2%}
- Treasury 10y: {rate_10y:.2%}
- Credit spread: {credit_spread:.2%}

Today's Actions:
- Buys: {buys_summary}
- Sells: {sells_summary}

Task:
Write a brief, professional daily update (80-140 words) explaining:
1. Current market conditions (regime)
2. Why actions were taken
3. Risk outlook

Then provide 3 bullet takeaways (<15 words each).

Respond with JSON:
{{
  "headline": "12 word headline",
  "blurb": "80-140 word narrative",
  "takeaways": ["bullet 1", "bullet 2", "bullet 3"]
}}
"""

def call_llm_weather_blurb(snapshot: dict, api_key: str) -> dict:
    """
    Generate daily weather report.
    
    Returns:
        {'headline': str, 'blurb': str, 'takeaways': list[str]}
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = WEATHER_BLURB_PROMPT.format(**snapshot)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a portfolio manager. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"LLM weather blurb failed: {e}")
        return {
            "headline": "Daily Update",
            "blurb": "System running normally.",
            "takeaways": ["Portfolio active", "Monitoring conditions", "Risk managed"]
        }
```

**weather_blurb.json structure:**

```json
{
  "date": "2025-01-27",
  "headline": "Risk-On Momentum Continues, Tech Leading",
  "blurb": "Markets maintained positive momentum in a risk-on environment, with the S&P 500 up 1.2% over 21 days. Credit spreads remain tight, and volatility is subdued. We added exposure to technology (XLK) and semiconductors (SOXX) based on strong health scores and regime compatibility. One position (XLE) was stopped out after hitting trailing stop levels. Current cash reserve at 12% aligns with risk-on posture. Monitoring for any regime shift signals.",
  "takeaways": [
    "Tech sector showing strong momentum",
    "Credit conditions supportive",
    "Risk-on regime intact"
  ]
}
```

---

# TRAINING SYSTEM

## 11.1 Training Overview

**Frequency:** Monthly (1st of month, 2am)

**Duration:** 1-2 hours (local MacBook)

**Outputs:**
- `models/regime_v{N}.pkl`
- `models/health_v{N}.pkl`
- `templates/policy_v{N}.json`
- `models/latest.json` (pointer)
- `backtests/v{N}/results.json`

**Trigger:** launchd plist (macOS scheduled task)

## 11.2 Training Pipeline

**File:** `training/train.py`

```python
#!/usr/bin/env python3
"""
Main training orchestrator.

Usage:
    python train.py --config config/aws_config.json
"""

import argparse
import boto3
import pandas as pd
from datetime import datetime
from train_regime import train_regime_model
from train_health import train_health_model
from evolutionary_search import run_evolutionary_search
from backtest import run_backtest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    bucket = config['s3_bucket']
    s3 = boto3.client('s3')
    
    print(f"[{datetime.now()}] Training started")
    
    # 1. Download historical data from S3
    print("Downloading historical data...")
    prices_df = download_parquet_from_s3(s3, bucket, 'data/prices_historical.parquet')
    features_df = download_parquet_from_s3(s3, bucket, 'data/features_historical.parquet')
    context_df = download_parquet_from_s3(s3, bucket, 'data/context_historical.parquet')
    
    # 2. Train regime model
    print("Training regime model...")
    regime_model = train_regime_model(context_df, features_df)
    regime_version = datetime.now().strftime('%Y%m%d')
    regime_path = f'models/regime_v{regime_version}.pkl'
    
    with open(regime_path, 'wb') as f:
        pickle.dump(regime_model, f)
    
    # Upload to S3
    s3.upload_file(regime_path, bucket, regime_path)
    print(f"Regime model uploaded: {regime_path}")
    
    # 3. Train health model
    print("Training health model...")
    health_model = train_health_model(features_df, context_df)
    health_version = regime_version
    health_path = f'models/health_v{health_version}.pkl'
    
    with open(health_path, 'wb') as f:
        pickle.dump(health_model, f)
    
    s3.upload_file(health_path, bucket, health_path)
    print(f"Health model uploaded: {health_path}")
    
    # 4. Run evolutionary search
    print("Running evolutionary policy search...")
    best_templates = run_evolutionary_search(
        prices_df, features_df, context_df,
        regime_model, health_model
    )
    
    template_version = regime_version
    template_path = f'templates/policy_v{template_version}.json'
    
    with open(template_path, 'w') as f:
        json.dump(best_templates, f, indent=2)
    
    s3.upload_file(template_path, bucket, template_path)
    print(f"Policy templates uploaded: {template_path}")
    
    # 5. Backtest new models
    print("Running backtest...")
    backtest_results = run_backtest(
        prices_df, features_df, context_df,
        regime_model, health_model, best_templates[0]
    )
    
    backtest_path = f'backtests/v{regime_version}/results.json'
    os.makedirs(os.path.dirname(backtest_path), exist_ok=True)
    
    with open(backtest_path, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    s3.upload_file(backtest_path, bucket, backtest_path)
    print(f"Backtest results uploaded: {backtest_path}")
    
    # 6. Update latest pointers
    latest_config = {
        'version': regime_version,
        'timestamp': datetime.now().isoformat(),
        'regime_model': regime_path,
        'health_model': health_path,
        'policy_template': template_path,
        'backtest': backtest_path
    }
    
    latest_path = 'models/latest.json'
    with open(latest_path, 'w') as f:
        json.dump(latest_config, f, indent=2)
    
    s3.upload_file(latest_path, bucket, latest_path)
    print(f"Latest config updated: {latest_path}")
    
    print(f"[{datetime.now()}] Training complete!")

if __name__ == '__main__':
    main()
```

## 11.3 Evolutionary Meta-Search

**File:** `training/evolutionary_search.py`

```python
import numpy as np
from typing import List, Dict
from backtest import run_backtest

class PolicyGenome:
    """Represents a policy template as a genome."""
    
    def __init__(self, random_init=True):
        if random_init:
            self.genes = {
                'buy_score_threshold': np.random.uniform(0.55, 0.80),
                'min_health_buy': np.random.uniform(0.50, 0.75),
                'sell_health_threshold': np.random.uniform(0.20, 0.50),
                'trail_stop_nonlev': np.random.uniform(0.06, 0.15),
                'trail_stop_lev': np.random.uniform(0.03, 0.10),
                'max_positions': np.random.randint(4, 13),
                'max_position_weight': np.random.uniform(0.10, 0.30),
                'cash_reserve_calm': np.random.uniform(0.05, 0.20),
                'cash_reserve_risk_on': np.random.uniform(0.05, 0.20),
                'cash_reserve_choppy': np.random.uniform(0.15, 0.35),
                'cash_reserve_risk_off': np.random.uniform(0.30, 0.60),
                'cash_reserve_panic': np.random.uniform(0.30, 0.60),
                'leverage_max_days': np.random.randint(3, 16),
                'score_weight_health': np.random.uniform(0.3, 0.6),
                'score_weight_momentum': np.random.uniform(0.15, 0.35),
                'score_weight_relative': np.random.uniform(0.10, 0.30),
                'score_weight_risk': np.random.uniform(0.05, 0.15)
            }
            
            # Normalize score weights to sum to 1
            score_weights = [
                self.genes['score_weight_health'],
                self.genes['score_weight_momentum'],
                self.genes['score_weight_relative'],
                self.genes['score_weight_risk']
            ]
            total = sum(score_weights)
            self.genes['score_weight_health'] /= total
            self.genes['score_weight_momentum'] /= total
            self.genes['score_weight_relative'] /= total
            self.genes['score_weight_risk'] /= total
    
    def mutate(self, rate=0.1):
        """Mutate genes with given probability."""
        for key, value in self.genes.items():
            if np.random.rand() < rate:
                if isinstance(value, int):
                    # Integer genes
                    if key == 'max_positions':
                        self.genes[key] = np.clip(value + np.random.randint(-2, 3), 4, 12)
                    elif key == 'leverage_max_days':
                        self.genes[key] = np.clip(value + np.random.randint(-2, 3), 3, 15)
                else:
                    # Float genes
                    noise = np.random.normal(0, 0.05)
                    self.genes[key] = np.clip(value + noise, 0, 1)
        
        # Re-normalize score weights
        if any(k.startswith('score_weight') for k in self.genes):
            score_keys = [k for k in self.genes if k.startswith('score_weight')]
            total = sum(self.genes[k] for k in score_keys)
            for k in score_keys:
                self.genes[k] /= total
    
    def crossover(self, other: 'PolicyGenome') -> 'PolicyGenome':
        """Create offspring via crossover."""
        child = PolicyGenome(random_init=False)
        child.genes = {}
        
        for key in self.genes:
            # 50% chance to inherit from each parent
            child.genes[key] = self.genes[key] if np.random.rand() < 0.5 else other.genes[key]
        
        return child

def evaluate_fitness(genome: PolicyGenome, 
                     prices_df, features_df, context_df,
                     regime_model, health_model) -> Dict[str, float]:
    """
    Evaluate fitness via walk-forward backtest.
    
    Returns:
        {'return': float, 'drawdown': float, 'turnover': float, 'stability': float}
    """
    # Convert genome to policy params
    policy_params = genome.genes.copy()
    
    # Run backtest on multiple windows (3-year windows, quarterly steps)
    windows = []
    start_date = prices_df['date'].min()
    end_date = prices_df['date'].max()
    
    current = start_date
    while current + pd.Timedelta(days=3*365) <= end_date:
        window_end = current + pd.Timedelta(days=3*365)
        windows.append((current, window_end))
        current += pd.Timedelta(days=90)  # Quarterly step
    
    results = []
    for window_start, window_end in windows:
        window_prices = prices_df[(prices_df['date'] >= window_start) & 
                                   (prices_df['date'] <= window_end)]
        window_features = features_df[(features_df['date'] >= window_start) & 
                                       (features_df['date'] <= window_end)]
        window_context = context_df[(context_df['date'] >= window_start) & 
                                     (context_df['date'] <= window_end)]
        
        result = run_backtest(window_prices, window_features, window_context,
                             regime_model, health_model, policy_params)
        results.append(result)
    
    # Aggregate metrics
    returns = [r['total_return'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    turnovers = [r['turnover'] for r in results]
    
    fitness = {
        'median_return': np.median(returns),
        'median_drawdown': np.median(drawdowns),
        'median_turnover': np.median(turnovers),
        'stability': -np.std(returns)  # Lower variance = better
    }
    
    return fitness

def run_evolutionary_search(prices_df, features_df, context_df,
                            regime_model, health_model,
                            population_size=50,
                            generations=20) -> List[Dict]:
    """
    Run genetic algorithm to find optimal policy templates.
    
    Returns:
        List of top 5 policy templates (dicts)
    """
    print(f"Starting evolutionary search: {population_size} pop, {generations} gens")
    
    # Initialize population
    population = [PolicyGenome() for _ in range(population_size)]
    
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        # Evaluate fitness
        fitness_scores = []
        for i, genome in enumerate(population):
            if i % 10 == 0:
                print(f"  Evaluating genome {i+1}/{population_size}")
            
            fitness = evaluate_fitness(genome, prices_df, features_df, context_df,
                                      regime_model, health_model)
            
            # Composite score (multi-objective)
            # Maximize return, minimize drawdown, minimize turnover, maximize stability
            composite = (
                fitness['median_return'] * 2.0 +
                -abs(fitness['median_drawdown']) * 1.5 +
                -fitness['median_turnover'] * 0.3 +
                fitness['stability'] * 1.0
            )
            
            fitness_scores.append((genome, fitness, composite))
        
        # Sort by composite fitness
        fitness_scores.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  Best fitness: {fitness_scores[0][2]:.4f}")
        
        # Selection (top 50%)
        survivors = [x[0] for x in fitness_scores[:population_size//2]]
        
        # Crossover + mutation to create next generation
        offspring = []
        while len(offspring) < population_size - len(survivors):
            parent1 = np.random.choice(survivors)
            parent2 = np.random.choice(survivors)
            child = parent1.crossover(parent2)
            child.mutate(rate=0.1)
            offspring.append(child)
        
        population = survivors + offspring
    
    # Final evaluation and selection
    final_fitness = []
    for genome in population:
        fitness = evaluate_fitness(genome, prices_df, features_df, context_df,
                                  regime_model, health_model)
        composite = (
            fitness['median_return'] * 2.0 +
            -abs(fitness['median_drawdown']) * 1.5 +
            -fitness['median_turnover'] * 0.3 +
            fitness['stability'] * 1.0
        )
        final_fitness.append((genome, fitness, composite))
    
    final_fitness.sort(key=lambda x: x[2], reverse=True)
    
    # Promote top 5 templates
    top_templates = []
    for i in range(5):
        genome, fitness, score = final_fitness[i]
        template = {
            'rank': i+1,
            'params': genome.genes,
            'fitness': fitness,
            'composite_score': score
        }
        top_templates.append(template)
    
    return top_templates
```

## 11.4 Backtesting

**File:** `training/backtest.py`

```python
def run_backtest(prices_df, features_df, context_df,
                 regime_model, health_model, policy_params) -> dict:
    """
    Run walk-forward backtest with given models and policy.
    
    Returns:
        {
            'total_return': float,
            'max_drawdown': float,
            'sharpe_ratio': float,
            'turnover': float,
            'win_rate': float,
            'equity_curve': list[dict]
        }
    """
    # Initialize portfolio
    portfolio = {
        'value': 100000,
        'cash': 100000,
        'holdings': [],
        'trades': []
    }
    
    equity_curve = []
    
    # Simulate daily decisions
    dates = sorted(context_df['date'].unique())
    
    for date in dates:
        # Get regime
        context_row = context_df[context_df['date'] == date].iloc[0]
        regime_output = regime_model.predict(context_row)  # Or baseline
        regime_label = regime_output['regime_label']
        
        # Get asset health
        features_today = features_df[features_df['date'] == date]
        health_output = health_model.predict(features_today)  # Or baseline
        
        # Run decision engine (simplified here)
        # ... (use decision engine logic from above)
        
        # Execute trades
        # ... (paper trader logic)
        
        # Record equity
        portfolio_value = portfolio['cash'] + sum(
            h['shares'] * get_price(h['symbol'], date, prices_df)
            for h in portfolio['holdings']
        )
        
        equity_curve.append({
            'date': date,
            'value': portfolio_value
        })
    
    # Calculate metrics
    equity_series = pd.Series([e['value'] for e in equity_curve])
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    max_drawdown = (equity_series / equity_series.cummax() - 1).min()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Turnover (total trades / avg portfolio value)
    total_traded = sum(abs(t['dollars']) for t in portfolio['trades'])
    avg_value = equity_series.mean()
    turnover = total_traded / avg_value if avg_value > 0 else 0
    
    # Win rate
    winning_trades = [t for t in portfolio['trades'] if t.get('pnl', 0) > 0]
    win_rate = len(winning_trades) / len(portfolio['trades']) if portfolio['trades'] else 0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'turnover': turnover,
        'win_rate': win_rate,
        'equity_curve': equity_curve
    }
```

---

# AWS INFRASTRUCTURE

## 12.1 Lambda Function

**Runtime:** Python 3.11  
**Memory:** 3GB  
**Timeout:** 15 minutes  
**Layers:** pandas, numpy, pyarrow, requests (use AWS Data Wrangler layer or custom)

**Handler:** `lambda/handler.py`

```python
import json
import boto3
from datetime import datetime
from steps import (
    ingest_prices, ingest_fred, ingest_gdelt,
    validate_data, build_features, run_inference,
    llm_risk_check, decision_engine, paper_trader,
    llm_weather, publish_artifacts
)

s3 = boto3.client('s3')
secrets = boto3.client('secretsmanager')

def lambda_handler(event, context):
    """
    Main daily pipeline entry point.
    """
    start_time = datetime.now()
    print(f"[{start_time}] Pipeline started")
    
    # Load config
    bucket = event.get('bucket', 'investment-system-data')
    config = load_config_from_s3(bucket)
    
    # Get API keys from Secrets Manager
    openai_key = get_secret('investment-system/openai-key')
    fred_key = get_secret('investment-system/fred-key')
    alphavantage_key = get_secret('investment-system/alphavantage-key')
    
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Step 1: Ingest prices
        print("[1/10] Ingesting prices...")
        prices_df = ingest_prices.run(config['universe'], alphavantage_key)
        
        # Step 2: Ingest FRED
        print("[2/10] Ingesting FRED data...")
        fred_df = ingest_fred.run(fred_key)
        
        # Step 3: Ingest GDELT
        print("[3/10] Ingesting GDELT...")
        gdelt_data = ingest_gdelt.run(run_date)
        
        # Step 4: Validate data
        print("[4/10] Validating data...")
        validation = validate_data.run(prices_df, fred_df, config)
        
        if validation['degraded_mode']:
            print("WARNING: Running in degraded mode")
        
        # Step 5: Build features
        print("[5/10] Building features...")
        features_df, context_df = build_features.run(prices_df, fred_df, gdelt_data)
        
        # Step 6: Run inference
        print("[6/10] Running inference...")
        inference_output = run_inference.run(features_df, context_df, config)
        
        # Step 7: LLM risk check
        print("[7/10] LLM risk check...")
        llm_risks = llm_risk_check.run(
            inference_output, features_df, context_df, openai_key, config
        )
        
        # Step 8: Decision engine
        print("[8/10] Running decision engine...")
        decisions = decision_engine.run(
            inference_output, llm_risks, features_df, config, validation
        )
        
        # Step 9: Paper trader
        print("[9/10] Executing paper trades...")
        portfolio_state, trades = paper_trader.run(decisions, prices_df, bucket)
        
        # Step 10: LLM weather blurb
        print("[10/10] Generating weather blurb...")
        weather = llm_weather.run(
            inference_output, decisions, portfolio_state, context_df, openai_key
        )
        
        # Publish all artifacts to S3
        print("Publishing artifacts to S3...")
        publish_artifacts.run(
            bucket, run_date,
            prices_df, context_df, features_df,
            inference_output, llm_risks, decisions,
            portfolio_state, trades, weather, validation
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"[{end_time}] Pipeline completed in {duration:.1f}s")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'date': run_date,
                'duration_seconds': duration
            })
        }
        
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        
        # Write failure report
        failure_report = {
            'status': 'failed',
            'date': run_date,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        s3.put_object(
            Bucket=bucket,
            Key=f'daily/{run_date}/run_report.json',
            Body=json.dumps(failure_report, indent=2)
        )
        
        # Send alert (optional)
        # sns.publish(TopicArn=..., Message=...)
        
        return {
            'statusCode': 500,
            'body': json.dumps(failure_report)
        }

def load_config_from_s3(bucket):
    """Load config files from S3."""
    # Implementation...
    pass

def get_secret(secret_name):
    """Retrieve secret from Secrets Manager."""
    # Implementation...
    pass
```

## 12.2 IAM Role

**Policy:** `infrastructure/iam_policies.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::investment-system-data",
        "arn:aws:s3:::investment-system-data/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:*:secret:investment-system/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
      "Resource": "arn:aws:sns:us-east-1:*:investment-system-alerts"
    }
  ]
}
```

## 12.3 EventBridge Rule

**Schedule:** Run at 10:00 PM ET on weeknights

**Cron Expression:** `cron(0 2 ? * TUE-SAT *)` (2 AM UTC = 10 PM ET previous day, Tue-Sat = Mon-Fri)

**Target:** Lambda function

**Payload:**
```json
{
  "bucket": "investment-system-data",
  "source": "eventbridge-scheduled"
}
```

## 12.4 S3 Bucket Structure

```
investment-system-data/
├── config/
│   ├── universe.csv
│   ├── decision_params.json
│   └── regime_compatibility.json
│
├── daily/
│   ├── 2025-01-27/
│   │   ├── prices.parquet
│   │   ├── context.parquet
│   │   ├── features.parquet
│   │   ├── inference.json
│   │   ├── llm_risk.json
│   │   ├── decisions.json
│   │   ├── portfolio_state.json
│   │   ├── trades.jsonl
│   │   ├── weather_blurb.json
│   │   └── run_report.json
│   └── latest.json
│
├── data/
│   ├── prices_historical.parquet
│   ├── features_historical.parquet
│   └── context_historical.parquet
│
├── models/
│   ├── regime_v20250201.pkl
│   ├── health_v20250201.pkl
│   └── latest.json
│
├── templates/
│   ├── policy_v20250201.json
│   └── latest.json
│
└── backtests/
    └── v20250201/
        ├── results.json
        └── equity_curve.parquet
```

## 12.5 Cost Breakdown

**Monthly AWS Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| Lambda | 30 invocations × 3GB × 10min | $3.00 |
| S3 Storage | 5GB data + versioning | $0.15 |
| S3 Requests | ~1000 PUTs, ~5000 GETs | $0.01 |
| CloudWatch Logs | 500MB/month | $0.25 |
| Secrets Manager | 3 secrets | $1.20 |
| Data Transfer | Minimal (< 1GB) | $0.10 |
| **AWS Total** | | **$4.71** |

**API Costs:**

| Service | Usage | Cost |
|---------|-------|------|
| OpenAI (gpt-4o-mini) | 510 calls/month | $0.10 |
| FRED | Free (< 1000/day) | $0.00 |
| Stooq | Free | $0.00 |
| GDELT | Free | $0.00 |
| Alpha Vantage | Free tier | $0.00 |
| **API Total** | | **$0.10** |

**Grand Total: ~$5/month** (well under $20 budget)

---

# FRONTEND DASHBOARD

## 13.1 React App Structure

**Entry Point:** `frontend/src/index.jsx`

```jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/tailwind.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

**Main App:** `frontend/src/App.jsx`

```jsx
import React, { useState, useEffect } from 'react';
import HeroMetrics from './components/HeroMetrics';
import EquityCurveChart from './components/EquityCurveChart';
import DrawdownChart from './components/DrawdownChart';
import MonthlyReturnsHeatmap from './components/MonthlyReturnsHeatmap';
import WeatherReport from './components/WeatherReport';
import PortfolioTable from './components/PortfolioTable';
import CandidatesTable from './components/CandidatesTable';
import RecentTrades from './components/RecentTrades';
import { loadLatestData, loadHistoricalData } from './utils/dataLoader';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        
        // Load latest snapshot
        const latest = await loadLatestData();
        
        // Load historical data for charts
        const historical = await loadHistoricalData();
        
        setData({
          ...latest,
          historical
        });
        
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    }
    
    fetchData();
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-2xl">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-red-500 text-xl">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-3xl font-bold">AI Investment System</h1>
        <p className="text-gray-400 text-sm mt-1">
          Paper Trading Dashboard • Last Updated: {data.date}
        </p>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 space-y-8">
        
        {/* Hero Metrics */}
        <HeroMetrics data={data} />

        {/* Weather Report (Right Sidebar on Desktop) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Equity Curve */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4">Performance vs Benchmarks</h2>
              <EquityCurveChart data={data.historical} />
            </div>

            {/* Drawdown */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4">Drawdown Analysis</h2>
              <DrawdownChart data={data.historical} />
            </div>

            {/* Monthly Returns Heatmap */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4">Monthly Returns</h2>
              <MonthlyReturnsHeatmap data={data.historical} />
            </div>
          </div>

          {/* Right Sidebar */}
          <div className="space-y-8">
            <WeatherReport data={data} />
            <RecentTrades trades={data.recent_trades} />
          </div>
        </div>

        {/* Portfolio & Candidates */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <PortfolioTable holdings={data.portfolio.holdings} />
          <CandidatesTable candidates={data.candidates} />
        </div>

      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-4 mt-12">
        <p className="text-gray-500 text-sm text-center">
          Paper Trading Only • Not Financial Advice • Built with React + AWS
        </p>
      </footer>
    </div>
  );
}

export default App;
```

## 13.2 Hero Metrics Component

**File:** `frontend/src/components/HeroMetrics.jsx`

```jsx
import React from 'react';
import { formatPercent, formatCurrency } from '../utils/formatters';

function HeroMetrics({ data }) {
  const metrics = data.metrics;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      
      {/* Total Return */}
      <MetricCard
        label="Total Return"
        value={formatPercent(metrics.total_return)}
        positive={metrics.total_return > 0}
        large
      />

      {/* vs S&P 500 */}
      <MetricCard
        label="vs S&P 500"
        value={formatPercent(metrics.alpha)}
        positive={metrics.alpha > 0}
      />

      {/* Sharpe Ratio */}
      <MetricCard
        label="Sharpe Ratio"
        value={metrics.sharpe.toFixed(2)}
        positive={metrics.sharpe > 1}
      />

      {/* Max Drawdown */}
      <MetricCard
        label="Max Drawdown"
        value={formatPercent(metrics.max_drawdown)}
        positive={false}
        inverse
      />

      {/* Win Rate */}
      <MetricCard
        label="Win Rate"
        value={formatPercent(metrics.win_rate)}
        positive={metrics.win_rate > 0.5}
      />

      {/* Current Positions */}
      <MetricCard
        label="Positions / Cash"
        value={`${data.portfolio.holdings.length} / ${formatPercent(data.portfolio.cash_pct)}`}
      />
      
    </div>
  );
}

function MetricCard({ label, value, positive, large, inverse }) {
  const colorClass = positive
    ? 'text-green-400'
    : inverse
    ? 'text-red-400'
    : 'text-gray-300';

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="text-gray-400 text-sm mb-1">{label}</div>
      <div className={`${large ? 'text-3xl' : 'text-2xl'} font-bold ${colorClass}`}>
        {value}
      </div>
    </div>
  );
}

export default HeroMetrics;
```

## 13.3 Equity Curve Chart

**File:** `frontend/src/components/EquityCurveChart.jsx`

```jsx
import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceArea
} from 'recharts';
import { format } from 'date-fns';

function EquityCurveChart({ data }) {
  // data.equity_curve: [{date, portfolio_value, spy_value, regime}, ...]
  
  const chartData = data.equity_curve.map(d => ({
    date: new Date(d.date).getTime(),
    portfolio: (d.portfolio_value / data.equity_curve[0].portfolio_value - 1) * 100,
    spy: (d.spy_value / data.equity_curve[0].spy_value - 1) * 100,
    regime: d.regime
  }));

  // Compute regime shading zones
  const regimeZones = [];
  let currentZone = null;

  chartData.forEach((d, i) => {
    if (d.regime !== currentZone?.regime) {
      if (currentZone) {
        currentZone.x2 = d.date;
        regimeZones.push(currentZone);
      }
      currentZone = {
        regime: d.regime,
        x1: d.date,
        x2: null
      };
    }
  });
  if (currentZone) {
    currentZone.x2 = chartData[chartData.length - 1].date;
    regimeZones.push(currentZone);
  }

  const regimeColors = {
    calm_uptrend: 'rgba(34, 197, 94, 0.1)',
    risk_on_trend: 'rgba(34, 197, 94, 0.05)',
    choppy: 'rgba(251, 191, 36, 0.1)',
    risk_off_trend: 'rgba(239, 68, 68, 0.1)',
    high_vol_panic: 'rgba(239, 68, 68, 0.2)'
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        
        {/* Regime shading */}
        {regimeZones.map((zone, i) => (
          <ReferenceArea
            key={i}
            x1={zone.x1}
            x2={zone.x2}
            fill={regimeColors[zone.regime] || 'rgba(0,0,0,0)'}
          />
        ))}

        <XAxis
          dataKey="date"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(ts) => format(new Date(ts), 'MMM yyyy')}
          stroke="#9CA3AF"
        />
        <YAxis
          label={{ value: 'Return (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
          stroke="#9CA3AF"
        />
        <Tooltip
          labelFormatter={(ts) => format(new Date(ts), 'MMM dd, yyyy')}
          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
          formatter={(value) => `${value.toFixed(2)}%`}
        />
        <Legend />

        <Line
          type="monotone"
          dataKey="portfolio"
          stroke="#10B981"
          strokeWidth={3}
          name="Portfolio"
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="spy"
          stroke="#6B7280"
          strokeWidth={2}
          strokeDasharray="5 5"
          name="S&P 500"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default EquityCurveChart;
```

## 13.4 Drawdown Chart

**File:** `frontend/src/components/DrawdownChart.jsx`

```jsx
import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from 'recharts';
import { format } from 'date-fns';

function DrawdownChart({ data }) {
  // Compute drawdown
  const chartData = [];
  let peak = data.equity_curve[0].portfolio_value;

  data.equity_curve.forEach(d => {
    if (d.portfolio_value > peak) {
      peak = d.portfolio_value;
    }
    const drawdown = ((d.portfolio_value - peak) / peak) * 100;

    chartData.push({
      date: new Date(d.date).getTime(),
      drawdown
    });
  });

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="date"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(ts) => format(new Date(ts), 'MMM yyyy')}
          stroke="#9CA3AF"
        />
        <YAxis
          label={{ value: 'Drawdown (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
          stroke="#9CA3AF"
        />
        <Tooltip
          labelFormatter={(ts) => format(new Date(ts), 'MMM dd, yyyy')}
          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
          formatter={(value) => `${value.toFixed(2)}%`}
        />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#EF4444"
          fill="#EF4444"
          fillOpacity={0.3}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default DrawdownChart;
```

## 13.5 Monthly Returns Heatmap

**File:** `frontend/src/components/MonthlyReturnsHeatmap.jsx`

```jsx
import React from 'react';
import { format, startOfMonth, endOfMonth } from 'date-fns';

function MonthlyReturnsHeatmap({ data }) {
  // Compute monthly returns
  const monthlyReturns = {};

  data.equity_curve.forEach((d, i) => {
    const monthKey = format(new Date(d.date), 'yyyy-MM');
    if (!monthlyReturns[monthKey]) {
      monthlyReturns[monthKey] = {
        start: d.portfolio_value,
        end: d.portfolio_value
      };
    }
    monthlyReturns[monthKey].end = d.portfolio_value;
  });

  const months = Object.keys(monthlyReturns).sort();
  const years = [...new Set(months.map(m => m.split('-')[0]))];

  const heatmapData = years.map(year => {
    const row = { year };
    for (let m = 1; m <= 12; m++) {
      const monthKey = `${year}-${String(m).padStart(2, '0')}`;
      if (monthlyReturns[monthKey]) {
        const ret = (monthlyReturns[monthKey].end / monthlyReturns[monthKey].start - 1) * 100;
        row[m] = ret;
      } else {
        row[m] = null;
      }
    }
    return row;
  });

  const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function getColor(value) {
    if (value === null) return 'bg-gray-700';
    if (value > 10) return 'bg-green-700';
    if (value > 5) return 'bg-green-600';
    if (value > 0) return 'bg-green-500';
    if (value > -5) return 'bg-red-500';
    if (value > -10) return 'bg-red-600';
    return 'bg-red-700';
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="p-2"></th>
            {monthLabels.map((label, i) => (
              <th key={i} className="p-2 text-gray-400">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {heatmapData.map(row => (
            <tr key={row.year}>
              <td className="p-2 font-semibold text-gray-300">{row.year}</td>
              {monthLabels.map((_, i) => {
                const value = row[i + 1];
                return (
                  <td key={i} className="p-2">
                    <div
                      className={`h-10 rounded flex items-center justify-center ${getColor(value)}`}
                      title={value !== null ? `${value.toFixed(1)}%` : 'N/A'}
                    >
                      {value !== null && <span className="text-white font-semibold">{value.toFixed(1)}%</span>}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default MonthlyReturnsHeatmap;
```

## 13.6 Weather Report

**File:** `frontend/src/components/WeatherReport.jsx`

```jsx
import React from 'react';

function WeatherReport({ data }) {
  const { weather, regime } = data;

  const regimeColors = {
    calm_uptrend: 'bg-green-600',
    risk_on_trend: 'bg-green-500',
    choppy: 'bg-yellow-500',
    risk_off_trend: 'bg-red-500',
    high_vol_panic: 'bg-red-700'
  };

  const regimeLabels = {
    calm_uptrend: 'Calm Uptrend',
    risk_on_trend: 'Risk-On',
    choppy: 'Choppy',
    risk_off_trend: 'Risk-Off',
    high_vol_panic: 'High Volatility'
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h2 className="text-xl font-semibold mb-4">Market Weather</h2>

      {/* Regime Indicator */}
      <div className="mb-6">
        <div className="text-gray-400 text-sm mb-2">Current Regime</div>
        <div className={`${regimeColors[regime]} rounded-lg p-4 text-center`}>
          <span className="text-2xl font-bold text-white">
            {regimeLabels[regime]}
          </span>
        </div>
      </div>

      {/* Weather Blurb */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">{weather.headline}</h3>
        <p className="text-gray-300 text-sm leading-relaxed">
          {weather.blurb}
        </p>
      </div>

      {/* Takeaways */}
      <div>
        <div className="text-gray-400 text-sm mb-2">Key Takeaways</div>
        <ul className="space-y-2">
          {weather.takeaways.map((item, i) => (
            <li key={i} className="flex items-start">
              <span className="text-green-400 mr-2">•</span>
              <span className="text-gray-300 text-sm">{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default WeatherReport;
```

## 13.7 Portfolio Table

**File:** `frontend/src/components/PortfolioTable.jsx`

```jsx
import React from 'react';
import { formatCurrency, formatPercent } from '../utils/formatters';

function PortfolioTable({ holdings }) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h2 className="text-xl font-semibold mb-4">Current Holdings</h2>
      
      {holdings.length === 0 ? (
        <div className="text-gray-400 text-center py-8">No positions</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left p-2 text-gray-400">Symbol</th>
                <th className="text-right p-2 text-gray-400">Shares</th>
                <th className="text-right p-2 text-gray-400">Entry</th>
                <th className="text-right p-2 text-gray-400">Current</th>
                <th className="text-right p-2 text-gray-400">P&L</th>
                <th className="text-right p-2 text-gray-400">Weight</th>
              </tr>
            </thead>
            <tbody>
              {holdings.map(h => {
                const pnl = ((h.current_price - h.entry_price) / h.entry_price) * 100;
                const pnlColor = pnl > 0 ? 'text-green-400' : 'text-red-400';
                
                return (
                  <tr key={h.symbol} className="border-b border-gray-700">
                    <td className="p-2 font-semibold">{h.symbol}</td>
                    <td className="p-2 text-right">{h.shares}</td>
                    <td className="p-2 text-right">${h.entry_price.toFixed(2)}</td>
                    <td className="p-2 text-right">${h.current_price.toFixed(2)}</td>
                    <td className={`p-2 text-right font-semibold ${pnlColor}`}>
                      {formatPercent(pnl / 100)}
                    </td>
                    <td className="p-2 text-right">{formatPercent(h.weight)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default PortfolioTable;
```

## 13.8 Data Loader Utility

**File:** `frontend/src/utils/dataLoader.js`

```javascript
const S3_BASE_URL = 'https://investment-system-data.s3.amazonaws.com';

export async function loadLatestData() {
  // Load latest.json to get current date
  const latestRes = await fetch(`${S3_BASE_URL}/daily/latest.json`);
  const latest = await latestRes.json();
  
  const date = latest.date;
  
  // Load all daily artifacts
  const [
    weather,
    portfolio,
    decisions,
    inference,
    trades
  ] = await Promise.all([
    fetch(`${S3_BASE_URL}/daily/${date}/weather_blurb.json`).then(r => r.json()),
    fetch(`${S3_BASE_URL}/daily/${date}/portfolio_state.json`).then(r => r.json()),
    fetch(`${S3_BASE_URL}/daily/${date}/decisions.json`).then(r => r.json()),
    fetch(`${S3_BASE_URL}/daily/${date}/inference.json`).then(r => r.json()),
    fetch(`${S3_BASE_URL}/daily/${date}/trades.jsonl`).then(r => r.text())
  ]);
  
  // Parse trades
  const tradesArray = trades.split('\n').filter(Boolean).map(line => JSON.parse(line));
  
  return {
    date,
    weather,
    portfolio,
    decisions,
    inference,
    regime: inference.regime.label,
    recent_trades: tradesArray.slice(-10),
    candidates: decisions.actions.filter(a => a.action === 'BUY'),
    metrics: {
      // Computed from portfolio state...
      total_return: 0.15, // Placeholder
      alpha: 0.03,
      sharpe: 1.2,
      max_drawdown: -0.08,
      win_rate: 0.65
    }
  };
}

export async function loadHistoricalData() {
  // Load equity curve from historical files
  // This requires aggregating daily portfolio_state.json files
  // For MVP, we can create a separate equity_curve.json artifact
  
  const equityCurveRes = await fetch(`${S3_BASE_URL}/data/equity_curve.json`);
  const equityCurve = await equityCurveRes.json();
  
  return {
    equity_curve: equityCurve
  };
}
```

## 13.9 S3 Static Hosting

**Setup Script:** `infrastructure/s3_setup.sh`

```bash
#!/bin/bash

BUCKET="investment-system-data"
REGION="us-east-1"

# Create bucket if not exists
aws s3 mb s3://${BUCKET} --region ${REGION} || true

# Enable static website hosting on /daily/latest/ path
aws s3 website s3://${BUCKET} --index-document index.html

# Upload React build
cd frontend
npm run build
aws s3 sync build/ s3://${BUCKET}/dashboard/ --delete

# Set bucket policy for public read on /dashboard/*
aws s3api put-bucket-policy --bucket ${BUCKET} --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::'${BUCKET}'/dashboard/*"
  }]
}'

echo "Dashboard deployed to: http://${BUCKET}.s3-website-${REGION}.amazonaws.com/dashboard/"
```

---

# DEPLOYMENT & AUTOMATION

## 14.1 Lambda Deployment

**Script:** `infrastructure/lambda_deploy.sh`

```bash
#!/bin/bash

FUNCTION_NAME="investment-system-daily-pipeline"
REGION="us-east-1"
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/investment-system-lambda-role"

# Package Lambda
cd lambda
zip -r ../lambda_package.zip . -x "*.pyc" -x "__pycache__/*"
cd ..

# Create or update function
aws lambda create-function \
  --function-name ${FUNCTION_NAME} \
  --runtime python3.11 \
  --role ${ROLE_ARN} \
  --handler handler.lambda_handler \
  --zip-file fileb://lambda_package.zip \
  --timeout 900 \
  --memory-size 3072 \
  --region ${REGION} \
  || \
aws lambda update-function-code \
  --function-name ${FUNCTION_NAME} \
  --zip-file fileb://lambda_package.zip \
  --region ${REGION}

echo "Lambda deployed: ${FUNCTION_NAME}"
```

## 14.2 launchd Monthly Training

**File:** `infrastructure/launchd_plist/com.investsys.train.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.investsys.train</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/YOUR_USERNAME/investment-system/training/train.py</string>
        <string>--config</string>
        <string>/Users/YOUR_USERNAME/investment-system/config/aws_config.json</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Day</key>
        <integer>1</integer>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/investment-system/logs/train.log</string>
    
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/investment-system/logs/train_error.log</string>
    
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

**Installation:**

```bash
# Copy plist to LaunchAgents
cp infrastructure/launchd_plist/com.investsys.train.plist ~/Library/LaunchAgents/

# Load the job
launchctl load ~/Library/LaunchAgents/com.investsys.train.plist

# Verify
launchctl list | grep investsys
```

---

# IMPLEMENTATION PHASES

## Phase 1: Core Pipeline (Baseline Models)

**Goal:** Get daily pipeline running with baseline heuristics (no ML yet)

**Tasks:**
1. Set up S3 bucket structure
2. Create `config/universe.csv` and other config files
3. Implement data ingestion (Stooq, FRED, GDELT)
4. Implement data validation
5. Implement feature engineering
6. Implement baseline regime model (deterministic rules)
7. Implement baseline asset health model (rank-based scoring)
8. Implement decision engine (buy/sell/sizing)
9. Implement paper trader (track portfolio state)
10. Implement LLM integration (risk + weather)
11. Create Lambda function
12. Deploy Lambda + EventBridge schedule
13. Test end-to-end

**Deliverable:** Daily pipeline running and producing artifacts in S3

**Duration:** 1-2 weeks

## Phase 2: Machine Learning Models

**Goal:** Replace baseline models with trained Transformer/GRU/Autoencoder

**Tasks:**
1. Implement regime model architecture (Transformer or GRU)
2. Implement asset health model architecture (Autoencoder)
3. Create training data loaders
4. Implement training loops
5. Implement model evaluation
6. Create model versioning system
7. Integrate trained models into Lambda inference step
8. Test inference with real models

**Deliverable:** Trained models replace baselines, decisions improve

**Duration:** 2-3 weeks

## Phase 3: Frontend Dashboard

**Goal:** Build impressive React dashboard

**Tasks:**
1. Set up React project with Vite
2. Implement data loader utilities
3. Build HeroMetrics component
4. Build EquityCurveChart with regime shading
5. Build DrawdownChart
6. Build MonthlyReturnsHeatmap
7. Build WeatherReport component
8. Build PortfolioTable
9. Build CandidatesTable
10. Build RecentTrades component
11. Style with Tailwind CSS
12. Deploy to S3 static hosting
13. Test responsiveness

**Deliverable:** Live dashboard showing portfolio performance

**Duration:** 1 week

## Phase 4: Evolutionary Meta-Search

**Goal:** Auto-discover optimal policy templates

**Tasks:**
1. Implement PolicyGenome class
2. Implement crossover and mutation
3. Implement fitness evaluation (walk-forward)
4. Implement evolutionary loop
5. Integrate template promotion
6. Test on historical data

**Deliverable:** Monthly template optimization working

**Duration:** 1 week

## Phase 5: Automation & Polish

**Goal:** Hands-off operation

**Tasks:**
1. Set up launchd for monthly training
2. Add cost monitoring
3. Add error alerting (SNS)
4. Write README and documentation
5. Add unit tests
6. Create deployment scripts
7. Final end-to-end testing

**Deliverable:** Fully autonomous system

**Duration:** 1 week

---

# TESTING & VALIDATION

## 16.1 Unit Tests

**Test Coverage:**
- Data ingestion (mock API responses)
- Feature calculations (known inputs → expected outputs)
- Baseline models (deterministic outputs)
- Decision engine logic (edge cases)
- Position sizing math
- Paper trader accounting

**Example:** `tests/test_features.py`

```python
import pytest
import pandas as pd
from lambda.utils.feature_utils import compute_asset_features

def test_returns_calculation():
    data = {
        'date': pd.date_range('2025-01-01', periods=100),
        'close': [100 + i for i in range(100)]  # Linear growth
    }
    df = pd.DataFrame(data)
    
    result = compute_asset_features(df)
    
    # Check 21-day return
    assert result.iloc[-1]['return_21d'] == pytest.approx(0.21, rel=0.01)
```

## 16.2 Integration Tests

**Test Scenarios:**
- Full pipeline run with sample data
- Degraded mode triggering
- LLM fallback handling
- Artifact writing to S3

## 16.3 Backtesting Validation

**Baseline Benchmarks:**
1. Buy & hold SPY
2. Equal-weight sector rotation
3. Simple momentum (no ML)

**Acceptance Criteria:**
- Sharpe ratio > 0.8
- Max drawdown < 20%
- Outperform SPY in risk-adjusted terms

---

# APPENDIX

## A. Configuration Files

### aws_config.json

```json
{
  "s3_bucket": "investment-system-data",
  "region": "us-east-1",
  "lambda_function": "investment-system-daily-pipeline",
  "secrets": {
    "openai_key": "investment-system/openai-key",
    "fred_key": "investment-system/fred-key",
    "alphavantage_key": "investment-system/alphavantage-key"
  }
}
```

### data_sources.json

```json
{
  "stooq": {
    "base_url": "https://stooq.com/q/d/l/",
    "timeout": 10
  },
  "fred": {
    "base_url": "https://api.stlouisfed.org/fred/series/observations",
    "series": ["DGS2", "DGS10", "VIXCLS", "DCOILWTICO"]
  },
  "gdelt": {
    "base_url": "http://data.gdeltproject.org/gdeltv2/"
  },
  "alphavantage": {
    "base_url": "https://www.alphavantage.co/query",
    "rate_limit": 25
  }
}
```

## B. Glossary

- **Regime:** Market state classification (calm, risk-on, risk-off, choppy, panic)
- **Health Score:** 0-1 metric indicating asset's risk-adjusted attractiveness
- **Trailing Stop:** Dynamic stop-loss that rises with price but never falls
- **Position Weight:** % of portfolio allocated to a single holding
- **Turnover:** Total trading volume relative to portfolio size
- **Sharpe Ratio:** Risk-adjusted return metric (higher = better)
- **Drawdown:** % decline from peak portfolio value
- **Walk-Forward Testing:** Backtesting method that simulates real-time decisions

---

# END OF SPECIFICATION

**This document is complete and ready for Claude Code to execute.**

**Next Steps:**
1. Review this spec for any final tweaks
2. Feed to Claude Code
3. Monitor build progress through phases
4. Test each phase before proceeding
5. Deploy and go live!

**Estimated Total Build Time:** 6-8 weeks for full system

**Good luck! 🚀**
