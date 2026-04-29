# Performance Audit — Since Hybrid Path

Date: 2026-04-28

## Context

Operator concern: bot performance has felt soft "the last month or two since the hybrid path"; market has shifted; want to know if it's an algorithm problem or a random bad-luck spell.

Hybrid timeline (from `git log`):
- 2026-03-21 — `feat: hybrid ranking system with shadow deployment` (30fb365). Hybrid blend in shadow.
- 2026-04-06 — `Update decision params with latest optimizer results` (c4e8831). Decision-params bundle refresh; new buy basket entered Apr 6–8.

## Data sources

- S3 `s3://investment-system-data/dashboard/dashboard.json` (snapshot 2026-04-28 13:45 ET, 199-day equity curve, 75 closed round-trips, 150 fills).
- S3 `s3://investment-system-data/dashboard/timeseries.json` (187 daily rows of regime/expert/throttle history).
- S3 `s3://investment-system-data/daily/latest.json` (today's snapshot pointer).

## Headline numbers

| Period | Days | Bot return | SPY return | Bot Sharpe | Bot MDD |
|---|---:|---:|---:|---:|---:|
| All-time (since 2025-08-04) | 199 | **+6.25%** | +3.58% | 1.57 | -1.47% |
| Pre-hybrid (Aug → Mar 20) | 161 | **+2.34%** | **-4.67%** | 0.90 | -1.36% |
| Hybrid live (since Apr 6) | 20 | +4.05% | **+9.12%** | 4.99 | -1.47% |
| Last 30 cal days | 26 | +3.96% | **+13.03%** | 4.22 | -1.47% |
| Last 14 cal days | 13 | +0.56% | +3.19% | 1.27 | -1.47% |

Interpretation: **the bot beat SPY by ~7 points when SPY was down (pre-hybrid window), and is now lagging SPY by ~9 points in the rally (last 30 days).** All-time it's still ahead +6.25% vs +3.58%.

Today is the all-time-low drawdown print (-1.47%). The portfolio peaked at $107,833 on 2026-04-17; it has drifted down ~1.5% on declining-volume sideways action while SPY ground higher. Psychologically that's the "worried" feeling.

## Verdict — algo problem, not random bad luck

The lag is **systematic and traceable to one specific subsystem**: the Phase 6 fragility / vol-uncertainty gates are pinned high and dragging the position-size modifier to ~half-size even though the primary regime model is correctly calling `risk_on_trend` with 0.92 confidence.

### Throttle audit (timeseries.json)

| Period | Regime mix | position_size_modifier (avg) | risk_throttle_factor (avg) | fragility_score (avg) | vol_uncertainty (avg) |
|---|---|---:|---:|---:|---:|
| Pre-hybrid baseline | 87% choppy, 7% panic | **0.86** | 0.07 | 0.57 | 0.10 |
| Hybrid live (Apr 6+) | 80% risk_on_trend | **0.49** | 0.20 | **0.98** | 0.68 |
| Last 14 days | 100% risk_on_trend | **0.51** | 0.20 | **0.98** | 0.62 |

Key reads:
- **fragility_score is pinned at 0.98** for all 15 days of the hybrid-live window. Inputs: `avg_correlation = 0.60`, `pc1_explained = 0.68` — i.e. the cross-asset panel really is strongly co-moving (typical in a single-factor melt-up rally). The fragility model treats this as "everything moving together = mass-de-risk vulnerability" and throttles.
- **vol_uncertainty = 0.68** because VIX percentile is elevated despite low realized vol; the gate sees this as "calm but vol-priced uncertainty" and throttles.
- These two signals combine to halve the position-size modifier even when the regime is correctly `risk_on_trend`.

### Cash drag

Current `cash_pct = 35.1%`, `gross_exposure = 64.9%`. With SPY +9% over the rally, ~35% cash alone is a ~3.15-point drag on benchmark-relative return. Combined with the half-size position modifier on the invested portion, effective exposure to the rally is roughly 30–50% of full-deployment.

### Stock-picking is fine

The bot bought the right names on Apr 6–8 — ARKK, VUG, FXI, XBI, XLF, XRT, XLK, SLV — a clean risk-on basket. These positions are all still open (no closes since Mar 21; the only Mar 23–25 sells were forced by a panic_prob spike of 0.997 and exited commodities/utilities). Unrealized P&L on those 8 names is +4% in 3 weeks, which is fine in absolute terms — just under-sized.

Pre-hybrid trade-level stats (75 round-trips):
- Win rate 57.3%, avg win +3.16%, avg loss -3.60%, expectancy +0.52%/trade, avg hold 8.5 days.
- Slight magnitude asymmetry against operator (loss > win), but win-rate edge keeps expectancy positive.

### Worst-day check

Bot's biggest single-day TWR drops are -1.24% (Feb 6), -1.20% (Feb 13), -1.08% (Apr 23) — all tiny and on days SPY was also down. Daily volatility is structurally lower than SPY. The throttling is doing its job on the downside; the cost is upside capture.

## Diagnosis

**Not bad luck.** The algorithm is doing what the Phase 6 fragility/vol-uncertainty gates were designed to do, but the calibration is too conservative for the current regime. Specifically:

1. **fragility_score saturates at high cross-asset correlation.** A 0.60 average correlation in a single-factor rally is normal-bullish, not fragile. The current gate treats correlation-rises as exclusively a panic-precursor signal.
2. **vol_uncertainty couples to VIX percentile** rather than to realized-vs-implied spread. VIX percentile of 0.64 in a calm uptrend is benign; current gate treats it like uncertainty.
3. **The gates don't relax when the primary regime model is high-confidence `risk_on_trend`.** Currently fragility-throttle and regime-confidence-up don't talk to each other; they multiply.

Reasonable fixes (in escalating scope):

- **Calibration only (lowest risk):** raise the fragility/vol-uncertainty thresholds so 0.98 fragility + 0.68 vol-uncertainty in confirmed `risk_on_trend` doesn't drop position_size below ~0.75. This is a parameter sweep against the existing walk-forward framework.
- **Gate logic:** make fragility/vol-uncertainty *conditional on regime*. Apply full throttle in `choppy` / `high_vol_panic`; apply soft throttle (or none) in `risk_on_trend` with `regime_confidence > 0.8`.
- **Cash deployment rule:** at 35% cash with regime risk-on-trend held >5 days, allow scaling-in of existing winners or selecting from a smaller candidate set rather than letting cash sit.

None of these are emergency fixes. The bot is not bleeding; it's just under-deploying.

## Recommended next actions

1. **Don't touch live params today.** Confirm the diagnosis with operator first.
2. **Run the existing walk-forward framework** (`scripts/run_hybrid_walk_forward.py`) with a fragility-threshold sweep to see what calibration would have captured more of recent rallies without losing the pre-hybrid drawdown protection.
3. **Decide between calibration-only vs gate-logic redesign** before any code change. Both have plan docs that should be authored if the operator greenlights.
4. **Continue monitoring**: if SPY rolls over and fragility was actually correct, today's underperformance reverses fast and the gates pay back.

## Numbers to sit with

- The bot was ALSO ~9 points behind SPY in the late-2025 leg of the rally? No — the bot was AHEAD of SPY by 7 points pre-hybrid because SPY was negative. The relative position only flipped in the last ~30 days when SPY started running.
- Sharpe 1.57 all-time, 4.99 in hybrid-live window, MDD -1.47%. By any conventional risk-adjusted measure this is excellent. The drag is purely beta — the bot is running too low to capture the rally.

## Follow-ups

- [ ] Operator decision: calibration sweep, gate-logic redesign, or hold and watch?
- [ ] If sweeping: dedicated plan doc with explicit search space + walk-forward gate criteria before any param change.
- [ ] Close the trader-bot/frontend uncommitted drift separately (operator confirmed: "ignore for now, fix later").
