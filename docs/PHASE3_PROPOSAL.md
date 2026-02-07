PHASE 3 EXPANSION PLAN – ENSEMBLE MARKET INTELLIGENCE SYSTEM

Overview

Expand the system from a 2-model ensemble into a multi-axis market intelligence engine, powered by four new orthogonal predictive “experts,” all running on daily cadence: 1. Macro & Credit Regime Score 2. Vol-of-Vol / Options Uncertainty Score 3. Cross-Asset Fragility Score 4. Entropy / Distribution-Shift Score

These integrate with the existing GRU (risk-on trend) and Transformer (panic regime) to form a 6-expert regime engine, with full visualization on the dashboard.

Claude will implement all components in backend ingestion, signal computation, storage, decision fusion, and frontend visualization.

⸻

SECTION 1 — NEW DAILY DATA INGESTION

Claude must expand the daily ETL to ingest and store the following new series:

1.1 Macro & Credit Inputs
• 10-year Treasury yield
• 3-month Treasury yield
• HY credit spread proxy: HYG price minus IEF or LQD (or actual HY spread if available)
• Optional but recommended: ISM PMI, Unemployment Rate, CPI Surprise Index (monthly updates; store when available)

1.2 Volatility Complex (Options-based)
• VIX Index
• VVIX Index
• CBOE SKEW Index
• VIX term structure: front month vs 3-month (fetch VIX9D + VIX for short-term)

1.3 Cross-Asset Panel (for fragility calculation)

Daily closing prices for:
• SPY
• QQQ
• IWM
• TLT
• HYG
• GLD
• EFA (international equities)
• EEM (emerging markets)

1.4 Distribution / Return Series
• Daily returns already exist → Claude uses them.

All new features must be stored in the daily parquet alongside existing features.

⸻

SECTION 2 — NEW SIGNAL COMPUTATION MODULES

Claude will create a new directory:

signals/
macro_credit.py
vol_uncertainty.py
fragility.py
entropy_shift.py

Each module must compute a single scalar output per day, stored in the database.

⸻

2.1 Macro & Credit Regime Score

Output:
macro_credit_score ∈ [-1.0, +1.0]

Inputs:
• Yield curve slope = (10Y – 3M)
• HY spread proxy = normalized (HYG – IEF) or LQD spread
• Macro indicators (PMI, unemployment, CPI surprise) when available

Rules:
• Strong inversion + widening HY spread → push score toward -1
• Healthy slope + narrowing HY spread → push score toward +1
• PMI < 50 → negative pressure
• PMI > 52 → positive pressure

This module provides the slow-moving backdrop for all regime decisions.

⸻

2.2 Vol-of-Vol / Options Uncertainty Score

Outputs:
• vol_uncertainty_score ∈ [0,1]
• vol_regime_label ∈ {calm, unstable_calm, panic}

Inputs:
• VIX
• VVIX
• SKEW
• VIX term structure

Rules:
• VVIX percentile > 80 while VIX < 60 → unstable_calm
• VVIX > 80 + VIX > 80 + term structure inverted → panic
• Otherwise → calm

vol_uncertainty_score is a normalized blend of VVIX, SKEW, and term structure.

This module detects panic before panic.

⸻

2.3 Cross-Asset Fragility Score

Output:
• fragility_score ∈ [0,1]

Inputs:
• Daily returns of SPY, QQQ, IWM, TLT, HYG, GLD, EFA, EEM

Procedure: 1. Compute a 60-day rolling correlation matrix 2. Compute PCA and measure variance explained by the first 1–3 components 3. Combine:
• average correlation
• top-factor absorption
into a normalized fragility score.

High fragility means:
• Markets are tightly coupled
• Single shocks propagate quickly
• Trend / vol signals require caution

This provides a systemic structural risk dimension.

⸻

2.4 Entropy / Distribution-Shift Score

Outputs:
• entropy_score (continuous)
• entropy_shift_flag (boolean)

Method: 1. Compute rolling return histogram (60–120 days) 2. Compute Shannon entropy 3. Normalize entropy by historical z-score 4. Mark entropy_shift_flag = True if z-score > 1.5 for 3 consecutive days

High entropy =
• The return distribution is changing
• Patterns may be breaking
• Reduce trust in model outputs

This is the Infotropy sensor.

⸻

SECTION 3 — ENSEMBLE DECISION FUSION UPGRADE

The final regime decision function becomes:

final_regime = decide_regime_v3(
trend_risk_on_prob,
panic_prob,
macro_credit_score,
vol_uncertainty_score,
vol_regime_label,
fragility_score,
entropy_score,
entropy_shift_flag
)

Claude must upgrade the decision engine to:

3.1 Apply Multi-Axis Logic
• High panic_prob OR vol_regime_label == “panic” → hard risk-off
• High trend_risk_on_prob + macro positive + vol calm → full risk-on
• High trend + macro negative OR vol unstable → partial risk-on
• Fragility_score > threshold → cap allocation regardless of signal
• Entropy_shift_flag → reduce confidence in all predictions

3.2 Produce Final Outputs

Claude must compute and store:
• regime_label
• regime_confidence
• position_size_modifier
• risk_throttle_factor

This feeds into the portfolio engine.

⸻

SECTION 4 — DASHBOARD EXPANSION

Claude must add 5 new panels to the dashboard.

4.1 Regime Strip + Expert Scores

A horizontal bar showing the final regime each day.
Below it, multi-line mini-charts for:
• trend_risk_on_prob
• panic_prob
• macro_credit_score
• vol_uncertainty_score
• fragility_score
• entropy_score

4.2 Macro & Credit Panel
• Line chart of macro_credit_score
• Overlays for yield curve slope & HY spread

4.3 Volatility Complex Panel
• VIX & VVIX lines
• Shaded background representing vol_regime_label
• vol_uncertainty_score as a top line

4.4 Fragility Panel
• Line of fragility_score
• Threshold markers
• Optional rolling correlation heatmap shown on hover

4.5 Entropy / Infotropy Panel
• entropy_score as line
• entropy_shift_flag as event markers
• Overlay with SPY for context

⸻

SECTION 5 — STORAGE & SCHEMA CHANGES

Claude must update the backend DB schemas:

New columns:

macro_credit_score
vol_uncertainty_score
vol_regime_label
fragility_score
entropy_score
entropy_shift_flag
position_size_modifier
risk_throttle_factor

All must be version-controlled and documented.

⸻

SECTION 6 — BACKTEST & REPORT EXPANSION

Claude must extend the backtester to evaluate:
• Profit impact of fragility gating
• Profit impact of macro backdrop
• Regime accuracy vs SPY forward returns
• Entropy warnings vs drawdowns

And output:
• Sharpe changes
• Drawdown changes
• Hit rate changes
• Regime confusion matrix

⸻

SECTION 7 — DEPLOYMENT PLAN

Claude will execute in this order:

Step 1: Add new data sources to ETL

Step 2: Add new signal modules

Step 3: Add fields to database + daily storage

Step 4: Expand regime decision engine

Step 5: Update portfolio allocator with throttles

Step 6: Add dashboard panels + backend endpoints

Step 7: Run full-system backtest

Step 8: Deploy updates
