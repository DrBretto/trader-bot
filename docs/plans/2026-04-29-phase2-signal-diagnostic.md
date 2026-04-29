# Phase 2 — Comprehensive 26-field signal diagnostic

Date: 2026-04-29
Source data: `/tmp/claude-501/trader-bot-audit/timeseries.json` (188 rows, 2025-08-04 → 2026-04-29).

Per packet §Required method step 3: for every column, compute distinct values, monotonicity (up/down/flat day diffs), max consecutive-flat run, dominant-value share, and correlation with `position_size_modifier` and `risk_throttle_factor`. Flag every dead/stuck/ratchet/step signal and trace to its source.

**Headline discovery — F-11 (new):** the first 125 rows (2025-08-04 → 2026-01-30) are **structurally different** from the remaining 63 rows. *Twelve* signals share the exact same first-change date — `2026-01-31`. Pre-pivot is either backfill or pre-deploy placeholder; post-pivot is real. This invalidates any backtest or "all-time outperformance" claim that treats the timeseries as homogeneous. See §F-11 below.

## §A — 26-field health table

Card legend: `distinct` = unique rounded values; `dom_val@%` = dominant value and its share of all rows; `maxrun` = longest run of consecutive identical values; `u/d/f` = up/down/flat day-to-day diffs.

| # | Column | Type | distinct | dom_val | dom_pct | maxrun | u/d/f | Pass/Flag | Trace |
|---|--------|------|---------:|--------:|--------:|-------:|------:|-----------|-------|
| 1  | `date`                       | str  | 188 | n/a       | n/a      | n/a | n/a       | OK         | direct |
| 2  | `final_regime_label`         | cat  | 4   | choppy    | 77.1%    | n/a | n/a       | OK         | `regime_fusion.decide_regime_v3` |
| 3  | `regime_confidence`          | num  | 61  | 1.0000    | 68.1%    | 125 | 29/33/125 | **STUCK pre-pivot** | inference output, default 1.0 |
| 4  | `trend_risk_on_prob`         | num  | 58  | 1.0000    | 67.6%    | 125 | 29/31/127 | **STUCK pre-pivot** | inference output |
| 5  | `panic_prob`                 | num  | 61  | 0.0000    | 68.1%    | 125 | 27/35/125 | **STUCK pre-pivot** | inference output |
| 6  | `macro_credit_score`         | num  | 64  | -0.5431   | 66.5%    | 125 | 33/30/124 | **STUCK pre-pivot** | `macro_credit.compute_macro_credit` |
| 7  | `yield_slope_10y_3m`         | num  | 12  | 0.0000    | **89.4%** | 168 | 10/8/169 | **DEAD until 2026-04-02** — `DGS3MO` not present in `fred_df` | `compute_macro_credit:slope_z_score` |
| 8  | `hy_spread_proxy`            | num  | 63  | 0.0000    | 66.5%    | 125 | 34/28/125 | STUCK pre-pivot only | macro_credit |
| 9  | `vol_uncertainty_score`      | num  | 21  | 0.1000    | **89.4%** | 168 | 8/12/167 | **DEAD until 2026-04-02 — F-1** | `vol_uncertainty.compute_vol_uncertainty` |
| 10 | `vol_regime_label`           | cat  | **1** | calm    | **100%** | 188 | n/a       | **DEAD all-time — only `calm` ever observed**; `unstable_calm` and `panic` paths unreachable | `vol_uncertainty.py:101-107` (regime label) |
| 11 | `vix_percentile`             | num  | 21  | 0.1000    | **89.4%** | 168 | 8/12/167 | DEAD until 2026-04-02 (corr 1.0 with #9 — F-3) | `vol_uncertainty.py:71-74` |
| 12 | `vvix_percentile`            | num  | **1** | 0.5000  | **100%** | 188 | 0/0/187 | **DEAD all-time — F-3** | `vol_uncertainty.py:82` (neutral fallback) |
| 13 | `skew_value`                 | num  | **1** | 0.0000  | **100%** | 188 | 0/0/187 | **DEAD all-time — F-3** | `vol_uncertainty.py:117` (None fallback) |
| 14 | `fragility_score`            | num  | 63  | 0.5000    | 66.5%    | 125 | 39/23/125 | **STUCK pre-pivot at neutral 0.5; SATURATED post-pivot — F-2** | `fragility.compute_fragility` |
| 15 | `avg_correlation`            | num  | 63  | 0.0000    | 66.5%    | 125 | 39/23/125 | STUCK pre-pivot (0.0 = neutral fallback) | `fragility.py:_neutral_result` |
| 16 | `pc1_explained`              | num  | 63  | 0.0000    | 66.5%    | 125 | 40/22/125 | STUCK pre-pivot | `fragility.py:_neutral_result` |
| 17 | `entropy_score`              | num  | 46  | 0.5000    | 66.5%    | 125 | 30/17/140 | STUCK pre-pivot | `entropy_shift.compute_entropy_shift` |
| 18 | `entropy_z_score`            | num  | 63  | 0.0000    | 66.5%    | 125 | 30/32/125 | STUCK pre-pivot | entropy_shift |
| 19 | `entropy_shift_flag`         | bool | **1** | False   | **100%** | 188 | 0/0/187 | **DEAD all-time — entropy gate has never fired** | F-12 below |
| 20 | `entropy_consecutive_days`   | num  | **1** | 0       | **100%** | 188 | 0/0/187 | DEAD all-time | entropy_shift |
| 21 | `entropy_above_threshold`    | bool | **1** | False   | **100%** | 188 | 0/0/187 | DEAD all-time | entropy_shift |
| 22 | `position_size_modifier`     | num  | 34  | 1.0000    | 66.5%    | 125 | 23/26/138 | STUCK pre-pivot | `regime_fusion.decide_regime_v3` |
| 23 | `risk_throttle_factor`       | num  | **3** | 0.0000  | 67.0%    | 125 | 7/6/174   | **COARSE — only 3 distinct values across 188 days**; binary-ish gate behavior | `regime_fusion.py:226` (clamped sum) |
| 24 | `override_reason`            | cat  | 3   | macro_downgrade | 73.4% | n/a | n/a    | OK structurally; macro_downgrade dominance is calibration F-13 | `regime_fusion.decide_regime_v3` |
| 25 | `spy_close`                  | num  | 186 | 677.58    | 1.1%     | 2   | 97/89/1   | **MISLABELED — column written from `ctx.get('spy_return_1d', 0)`** | `publish_artifacts.py:434` |
| 26 | `portfolio_value`            | num  | 59  | 100000    | **67.0%** | 126 | 33/26/128 | **NOT TRADED before 2026-01-29** — F-11 confirms | bot was in cash for the first 125 days |

### Flagged categories

- **DEAD all-time (5)**: `vol_regime_label`, `vvix_percentile`, `skew_value`, `entropy_shift_flag`, `entropy_consecutive_days`, `entropy_above_threshold`. Already covered by F-3 / F-9 (vol complex) and §F-12 below (entropy gate dead).
- **DEAD until 2026-04-02 (3)**: `yield_slope_10y_3m`, `vol_uncertainty_score`, `vix_percentile`. Same pivot date as F-1; explained by FRED data-plane fix. Yield slope is a new sub-finding (`DGS3MO` missing for 168 days — see §F-13 below).
- **STUCK pre-pivot (12)**: every signal except `date`, `final_regime_label`, `override_reason`, and the always-dead set. Pivot at 2026-01-31. See F-11.
- **MISLABELED (1)**: `spy_close` is sourced from `spy_return_1d`. See §F-14 below.
- **COARSE (1)**: `risk_throttle_factor` has only 3 distinct values across 188 days. Almost binary in practice. See §F-15 below.

### Correlations with throttle outputs (post-pivot only, n=63)

To ground "which signals actually drive sizing right now," compute Pearson correlation of each signal against `position_size_modifier` and `risk_throttle_factor` over the live (post-2026-01-31) subset:

| signal                  | corr(pos_mod) | corr(risk_throttle) | reading |
|-------------------------|--------------:|--------------------:|---------|
| `regime_confidence`     | +0.543        | +0.334              | high regime confidence raises sizing — desired |
| `panic_prob`            | -0.627        | **+0.847**          | panic_prob is the dominant driver of risk_throttle (panic_override) |
| `macro_credit_score`    | +0.351        | -0.103              | weak |
| `vol_uncertainty_score` | +0.351        | -0.261              | weak; counterintuitive sign on throttle (because the override paths are unreachable — F-3) |
| `fragility_score`       | +0.125        | +0.119              | **near-zero correlation despite gating** — fragility is so saturated post-pivot that its variation is too small to drive correlations even though it caps `pos_mod` at 0.60 |
| `avg_correlation`       | +0.207        | +0.025              | inputs to fragility — nearly orthogonal to throttle in current regime |
| `pc1_explained`         | +0.208        | +0.017              | same |
| `entropy_score`         | -0.116        | +0.320              | barely used |

**Reading**: in the current production regime, the only signals doing significant work are `regime_confidence` and `panic_prob`. Fragility caps without modulating; vol_uncertainty doesn't reach its hard-override paths; entropy is dormant. The "5-expert" Phase 6 design is, in production, *one* expert (panic from ensemble) plus a soft cap (fragility).

---

## §B — New findings discovered during diagnostic

### F-11 — Timeseries is structurally inhomogeneous: pre-2026-01-31 rows are not produced by the same pipeline as post

**Evidence**: the first day on which *any* of these signals first deviates from its initial value is 2026-01-31, identical across all twelve:
`regime_confidence`, `trend_risk_on_prob`, `panic_prob`, `macro_credit_score`, `fragility_score`, `avg_correlation`, `pc1_explained`, `entropy_score`, `entropy_z_score`, `position_size_modifier`. Plus `portfolio_value` first deviates from $100,000 on 2026-02-02 (after the first trade on 2026-01-29). For 125 consecutive days every numeric signal except `macro_credit_score` and `hy_spread_proxy` is held at a neutral fallback value (1.0 for confidences, 0.0 for panic/correlations, 0.5 for fragility/entropy/vol). Macro_credit is held at exactly `-0.5430889522` and `0.0` (a single distinct slope) for 125 days.

**Diagnosis**: either (a) the timeseries dataset was *backfilled* from the live pipeline starting 2026-01-31 with synthetic placeholders for prior dates; or (b) the live signal pipeline did not actually run before 2026-01-31, and the bot's daily pipeline was using fallback values for every signal. Both interpretations have the same downstream consequence: **no calibration / backtest based on this timeseries can treat 2025-08-04 → 2026-01-30 as ground truth**.

**Bot was in cash from 2025-08-04 to 2026-01-29.** The first 5 trades were all on 2026-01-29 (VLUE, XLB, VEA, SCHD). Portfolio value held at exactly $100,000 for 125 consecutive days. The prior audit's "Bot return: +2.34% pre-hybrid" and "+7 points outperformance" credits the bot with skill it did not exercise — for 125 of 161 pre-hybrid days the bot was 100% in cash while SPY drew down. **Sitting in cash is not alpha.**

**Recommended fix summary**:
1. Add a `pipeline_status` column to `timeseries.json` rows that records `{not_yet_deployed, backfill_placeholder, live_signal}` so future readers cannot accidentally treat backfill rows as real.
2. Add a `_select_active_segment` extension in `dashboard_metrics.py:171-223` that detects long flat runs in `regime_confidence` (or any pivot signal) and excludes them from the canonical equity curve, the way it already handles cutover discontinuities.
3. Update Phase-5 walk-forward harness to refuse to run on date ranges containing a backfill segment.

**Risk class**: LIVE-IMPACT for any backtest, OBSERVABILITY for the dashboard's all-time figures.

### F-12 — Entropy gate has never fired in production

`entropy_shift_flag = False` and `entropy_consecutive_days = 0` and `entropy_above_threshold = False` for all 188 days. Either (a) the SPY-return entropy of the last several years has been below the threshold uniformly; (b) the threshold is mis-set; (c) the gate logic doesn't reach this code path. The entropy expert is one of the four "Phase 6 experts." Two of the other three (vol-complex, fragility) are degraded; if entropy is also dead, then in production the 6-expert engine is **two effective experts** (regime ensemble + macro). 

**Recommended fix summary**: Phase 5 calibration sweep should compute the live distribution of `entropy_z_score` over 900-day historical SPY returns and pick a threshold that fires ~5-15% of historical days, not zero. Without that, entropy is just consuming compute.

### F-13 — Macro `yield_slope_10y_3m` is dead until 2026-04-02 because `DGS3MO` is missing in `fred_df`

`yield_slope_10y_3m = 0.00` for 168/188 days (89.4%), then steps to `0.6700` on 2026-04-02 — same pivot date as F-1. `compute_macro_credit` returns `slope = 0.0` when `rate_3m == 0` (line 7-12 of `macro_credit.py`'s slope handling). Same root cause class as F-1: a FRED series was not being ingested or wasn't surviving validation, then the data-plane fix on 2026-04-02 brought it online.

`src/steps/ingest_fred.py:11` does declare `DGS3MO` in `FRED_SERIES`, so the series is requested. But at run time it appears the series was either (a) returned empty by FRED, (b) lost in a feature-engineering step, (c) filtered out by `validate_data` in degraded mode. Phase 1 finding for the same data-plane bucket as F-1.

**Recommended fix summary**: in `ingest_fred.run`, treat empty or stale-by-N-days returns for any series in `FRED_SERIES` as a hard `degraded_reason='fred_<series>_unavailable'` warning rather than silently passing an empty frame. Phase-2 health card automation will catch the next instance.

### F-14 — `spy_close` column is sourced from `spy_return_1d` (mis-labeling)

`src/steps/publish_artifacts.py:434`:

```python
'spy_close': float(ctx.get('spy_return_1d', 0)) if hasattr(ctx, 'get') else 0.0,
```

The column name says `spy_close`. The source is `spy_return_1d`. The data shows large numerical values (mean 599.82, max 696.16) — i.e. SPY closing prices, not returns. So *despite* the broken assignment, the actual values that land in production look like real SPY closes. This means either (a) `ctx.get('spy_return_1d', 0)` is returning the close instead of the return (semantic drift in `build_features`), or (b) some other code path is overwriting the column.

**Recommended fix summary**: rename the field to `spy_close` or `spy_return_1d` consistently and assert the column matches the schema name in tests. This is a small bug but it's exactly the kind of latent inconsistency that breaks future calibration code.

### F-15 — `risk_throttle_factor` is effectively binary (3 distinct values across 188 days)

The factor is a sum of increments that are clipped at 1.0 (`regime_fusion.py:251-255`). Increments observed: `panic_risk_throttle = 1.0`, `unstable_risk_throttle = 0.7`, `fragility_throttle_increment = 0.2`, `entropy_throttle_increment = 0.15`. With entropy dead (F-12) and unstable_calm unreachable (F-3), the factor in production can only be `0.0`, `0.2`, or `1.0` — exactly the 3 distinct values observed. The hardware is theoretically capable of a finer signal; the upstream pipeline is starving it.

**Risk class**: OBSERVABILITY/LIVE-IMPACT — the throttle is a primary sizing input but in production it has no resolution.

---

## §C — Cross-references to Phase 1

| Phase 1 ID | Phase 2 confirmation in health table |
|------------|--------------------------------------|
| F-1  | column 9 (`vol_uncertainty_score`): 89.4% at 0.10, 168-day run |
| F-2  | column 14 (`fragility_score`): post-pivot saturation visible in last-30-day range 0.0935; tanh saturation reproducer in §F-2 |
| F-3  | columns 12, 13: `vvix_percentile=0.5` and `skew_value=0.0` for 100% of rows |
| F-4  | not visible in timeseries.json (per-holding); live in `dashboard.json` only |
| F-5  | derived from F-4 |
| F-6  | columns 14, 17, 22 etc — many neutral-fallback values held for 125 days with no `degraded_reason` recorded in timeseries |
| F-7  | derived from regime_fusion code; data shows fragility_score corr w/ pos_mod is +0.125 (capping not modulating) |
| F-8  | upstream cause of F-1; not a runtime symptom in current data (vol_uncertainty has been live since Apr 2) |
| F-9  | latent — needs SKEW data to actually arrive |
| F-10 | `dashboard.json metrics`: total_value=$106,248 vs broker=$96,170 |

End Phase 2.
