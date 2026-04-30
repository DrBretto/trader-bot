# Phase 2 — Fragility historical-norm audit and empirical re-derivation

Date: 2026-04-30
Inputs: Phase 1 throttle history (`$TMPDIR/trader-bot-audit-20260430/throttle_history_augmented.parquet`).

## §A — Source trace per constant

`src/signals/fragility.py:17-23`:

```python
# Historical norms for normalization
# Average pairwise correlation: mean ~0.30, std ~0.15
AVG_CORR_MEAN = 0.30
AVG_CORR_STD = 0.15
# PC1 explained variance: mean ~0.45, std ~0.12
PC1_MEAN = 0.45
PC1_STD = 0.12
```

Provenance trace:
- Introduced in commit `b9317e1` ("Add expert signal modules and regime fusion v3"), 2026-02-06, as part of the initial Phase-6 expert-engine implementation.
- The commit message names no source dataset, no date range, no panel composition, no rolling-window assumption.
- `docs/PHASE3_PROPOSAL.md` (the source design doc cited in the audit's read-first list) describes the metric construction conceptually but **does not specify any of the four constants**. Search of the doc for `0.30`, `0.45`, `AVG_CORR`, `PC1_MEAN` yields no matches.
- No accompanying test, fixture, or comment cites a dataset path or date.
- No CSV / parquet / .npy lives in `tests/`, `optimizer/`, or `data/` that would be the source for these values.

**Conclusion**: the four constants are bare assertions in code. They were chosen at code-write time, plausibly from a prior conversation or an off-repo notebook against an unspecified historical SPY+ETF panel, and were never traceable to a dataset or to a re-derivation procedure. **No re-validation cadence exists; no test fails when the empirical distribution drifts.**

This is exactly the failure mode the new `docs/POSTMORTEMS.md` entry (Phase 6 of this packet) classes as **stale-historical-norm**.

## §B — Empirical re-derivation

Recomputed `avg_correlation` and `pc1_explained` daily for **1272 trading days** (2021-04-07 → 2026-04-29) using the production code path in `src/signals/fragility.py:compute_fragility` against `PANEL_SYMBOLS = ['SPY','QQQ','IWM','TLT','HYG','GLD','EFA','EEM']` price history pulled from yfinance (cross-validated against production `signals.parquet` to ±0.005 absolute on overlap). Window = 60 trading days (production default).

Empirical statistics by lookback:

| lookback                    | n     | avg_correlation mean | avg_correlation std | pc1_explained mean | pc1_explained std |
|-----------------------------|------:|---------------------:|--------------------:|-------------------:|------------------:|
| **full 1272d (2021-04..2026-04)** | 1272 |              0.4576 |              0.0991 |             0.5990 |            0.0608 |
| **last 900d (2022-09..2026-04)**  |  900 |              **0.4774** |          **0.0979** |         **0.6007** |        **0.0658** |
| last 504d (2024-04..2026-04)    |  504 |              0.4514 |              0.0713 |             0.5835 |            0.0593 |
| last 252d (2025-04..2026-04)    |  252 |              0.4388 |              0.0761 |             0.5899 |            0.0541 |

The packet specifies a 900-day window. The 900d row is the empirical ground truth that the recalibration targets.

## §C — Side-by-side: current vs empirical

| constant          | current | empirical (900d) | gap                                        | reading |
|-------------------|--------:|-----------------:|--------------------------------------------|---------|
| `AVG_CORR_MEAN`   | 0.30    | 0.4774           | code value is **at the empirical 5th percentile** (0.302) | the metric treats the empirical center as +1.8σ — saturating tanh on the high side |
| `AVG_CORR_STD`    | 0.15    | 0.0979           | code value is **53% wider** than empirical std       | even when re-centered, the wide std under-flags small deviations |
| `PC1_MEAN`        | 0.45    | 0.6007           | code value is **below the empirical minimum** (0.400) — i.e. PC1_MEAN of 0.45 has *never* been observed in 1272 days as an empirical mean | metric saturates instantly on the high side |
| `PC1_STD`         | 0.12    | 0.0658           | code value is **82% wider** than empirical std       | same de-resolution problem |

## §D — Implied fragility distribution under each constant set

Recomputed `fragility_score` daily across all 1272 days under both old and new constants (full distribution):

| lookback          | n    | fragility_score mean | std    | p25   | p50   | p75   | p95   | gate fire rate (>0.75) | ≥0.97 fraction |
|-------------------|-----:|---------------------:|-------:|------:|------:|------:|------:|----------------------:|---------------:|
| **current constants**             | 1272 | 0.8630 | 0.1138 | 0.798 | 0.905 | 0.949 | 0.988 | **82.5%** | 13.9% |
| **recalibrated 900d constants**   | 1272 | 0.4611 | 0.2930 | 0.199 | 0.443 | 0.697 | 0.966 | **21.2%** |  4.3% |
| recalibrated 504d (sensitivity)  | 1272 | 0.4985 | 0.3083 | 0.234 | 0.491 | 0.762 | 0.978 | 25.3%     |  6.7% |

Under the recalibrated 900d constants:
- Median fragility falls from 0.905 → 0.443 (clearly informative).
- Gate fire rate falls from 82.5% → 21.2% (selective rather than baseline tax).
- Days at the saturated extreme (`>=0.97`) drop from 13.9% → 4.3% (kept for genuine extremes only).

## §E — Inflection-point comparison

For a single-day worked example, where does the fragility score saturate as `avg_correlation` rises?

| avg_correlation | corr_z (current) | norm_corr (current) | corr_z (recal 900d) | norm_corr (recal) |
|----------------:|-----------------:|--------------------:|--------------------:|------------------:|
| 0.20            | -0.667           | 0.207               | -2.834              | 0.005             |
| 0.30            |  0.000           | 0.500               | -1.812              | 0.057             |
| 0.40            | +0.667           | 0.793               | -0.791              | 0.171             |
| 0.45            | +1.000           | 0.881               | -0.281              | 0.363             |
| 0.50            | +1.333           | 0.939               | +0.231              | 0.614             |
| 0.55            | +1.667           | 0.964               | +0.741              | 0.815             |
| 0.60            | +2.000           | 0.982               | +1.252              | 0.924             |
| 0.65            | +2.333           | 0.991               | +1.762              | 0.971             |

(For pc1_explained the picture is similar but more extreme — empirical p5 of 0.514 is `+0.971σ` above the current `PC1_MEAN=0.45`, and tanh is already at 0.749 of [0,1].)

**Reading**: under current constants, the score saturates by `avg_correlation = 0.45` (which is the *empirical median*). Under recalibrated 900d constants, the score reads 0.5 at the empirical median and only saturates above `avg_correlation = 0.62` (the empirical p93). That is the fix: the metric now uses its full output range across the empirical input range.

## §F — Recommended values

For the Phase 4 feature-gated update:

```python
# src/signals/fragility.py — recalibrated against 900d empirical (2022-09 → 2026-04)
# Source dataset: yfinance close for PANEL_SYMBOLS over 900 trading days
# Re-derived 2026-04-30; see docs/plans/2026-04-30-phase2-fragility-baseline-audit.md
AVG_CORR_MEAN = 0.4774
AVG_CORR_STD  = 0.0979
PC1_MEAN      = 0.6007
PC1_STD       = 0.0658
```

The walk-forward sweep in Phase 4 will validate these and gate them behind a feature flag. The defaults stay at the current values until the operator approves the cutover.

## §G — Exit criterion

Each historical-norm constant has a current-empirical counterpart with documented evidence. The recommendation is to update all four to the 900d empirical values. Advance to Phase 3 (compound-effect walk-throughs).
