# Phase 1 — Throttle distribution diagnostic (1272-day empirical)

Date: 2026-04-30
Packet: `docs/plans/2026-04-30-fragility-and-throttle-calibration-audit-packet.md`
Computation source: `$TMPDIR/trader-bot-audit-20260430/throttle_history_augmented.parquet` (1272 trading days, 2021-04-07 → 2026-04-29) recomputed via the production code paths in `src/signals/fragility.py`, `src/signals/entropy_shift.py`, `src/signals/vol_uncertainty.py`. Panel inputs: yfinance close prices for `PANEL_SYMBOLS = ['SPY','QQQ','IWM','TLT','HYG','GLD','EFA','EEM']` plus `^VIX`. Sample scripts saved alongside the parquet.

Cross-validation: production `signals.parquet` rows for 2026-04-07 → 2026-04-29 match recomputed values within ±0.005 on `fragility_score`, ±0.005 on `avg_correlation`, ±0.003 on `pc1_explained` — the small delta is yfinance vs Stooq prices and is well below the resolution this audit is making claims at.

## §A — Health card per throttle (legend)

`distinct` = unique values rounded to 4 dp, `n=1272`. `pX` = X-th percentile. `extremum%` = fraction of days at observed min or max. `gate_fire_rate` = fraction of days the production gate fires given the production threshold. `verdict ∈ {informative | saturated | dead | misnormalized | window-quiet}`.

| throttle / input            | n    | distinct | mean   | std    | p5     | p25   | p50   | p75   | p95   | extremum% | gate fire rate         | verdict                |
|-----------------------------|-----:|---------:|-------:|-------:|-------:|------:|------:|------:|------:|----------:|-----------------------:|------------------------|
| `avg_correlation`           | 1272 | 1054     | 0.4576 | 0.0991 | 0.302  | 0.375 | 0.473 | 0.520 | 0.637 |  0.1/0.1  | n/a (input)            | **misnormalized** (constant `AVG_CORR_MEAN=0.30` is at empirical P5; `AVG_CORR_STD=0.15` is 51% wider than empirical std 0.099) |
| `pc1_explained`             | 1272 |  951     | 0.5990 | 0.0608 | 0.514  | 0.556 | 0.594 | 0.638 | 0.712 |  0.1/0.1  | n/a (input)            | **misnormalized** (constant `PC1_MEAN=0.45` is below empirical min 0.40; `PC1_STD=0.12` is 97% wider than empirical std 0.061) |
| `pc2_explained`             | 1272 |  686     | 0.1633 | 0.0252 | 0.127  | 0.144 | 0.162 | 0.178 | 0.210 |  0.1/0.1  | not consumed by gate    | informative (unused diagnostic) |
| `fragility_score`           | 1272 | 1004     | 0.8630 | 0.1138 | 0.639  | 0.798 | 0.905 | 0.949 | 0.988 |  0.1/0.1  | **82.5%** (>0.75)       | **saturated** (median 0.905; P75 0.949; 50.8% of days ≥ 0.90; **last 42d median 0.965, last 11d median 0.975** — operator's "stuck ≥ 0.97" symptom is empirically real for the recent window. Cause is misnormalization of the two inputs above.) |
| `entropy_score`             | 1272 |  795     | 0.8559 | 0.0687 | 0.730  | 0.830 | 0.866 | 0.902 | 0.943 |  0.1/0.1  | n/a (input)            | informative |
| `entropy_z_score`           | 1272 | 1250     | -0.144 | 1.314  | -2.266 | -1.228| 0.005 | 0.788 | 1.759 |  0.1/0.1  | 28.6% above |z|>1.5      | informative |
| `entropy_shift_flag`        | 1272 |    2     |  n/a   | n/a    |  -     |  -    |  -    |  -    |  -    |  -        | **22.9%** (consec≥3)    | **window-quiet** — fired 27 distinct streaks across the 1272-day history (last 5y), but within the 186-day production-data window (2025-08-04..2026-04-29) it fired only 1 day. Prior audit's F-12 "entropy gate has never fired in production" was a window artifact, not a permanently-broken signal. |
| `vol_uncertainty_score`     | 1272 | 1182     | 0.4846 | 0.2418 | 0.102  | 0.303 | 0.471 | 0.660 | 0.899 |  0.1/0.1  | n/a (consumed by override paths) | informative on its own; **paths it gates are unreachable** |
| `vol_regime_label`          | 1272 |    1     | n/a    | n/a    | -      | -     | -     | -     | -     |  -        | **0.0%** ("panic" / "unstable_calm" never reached) | **dead** — always "calm" because `vvix_history` is `None` in production (vvix_pctile defaults to 0.5), so the panic path `vix_pctile > 0.80 AND vvix_pctile > 0.80` and the unstable_calm path `vvix_pctile > 0.80 AND vix_pctile < 0.60` are both unreachable by construction. **Same root cause as F-3 from prior audit; calibration baselines for this gate are downstream of fixing VVIX ingest.** |
| `vix_value`                 | 1272 |  866     | 19.226 | 5.236  | 12.92  | 15.59 | 17.93 | 21.77 | 30.06 |  0.1/0.1  | n/a (input)            | informative |
| `regime_confidence` (post-pivot only, n=64) | 64 |  37 | 0.700 | 0.339 | 0.057 | 0.350 | 0.870 | 1.000 | 1.000 | n/a | n/a (consumed by F-7 relax path, currently off) | informative |
| `position_size_modifier` (post-pivot only, n=64) | 64 | 30 | 0.474 | 0.151 | 0.250 | 0.300 | 0.488 | 0.522 | 0.600 | **at clip-min 0.25**: 14.1%; **at fragility cap 0.60**: 4.7% | n/a (output) | **clamp-bound** — 33% of post-pivot days at 0.30 (regime_adj for `choppy`/`risk_off` × 1.0 fragility_cap), 14% at the 0.25 clip-min floor — 47% at or near the bottom of the modifier's allowed range. The production output is structurally pegged near floor. |
| `effective_exposure_multiplier` (post-pivot, n=64) | 64 | 30 | 0.444 | 0.155 | 0.225 | 0.270 | 0.448 | 0.475 | 0.540 | similar | n/a | clamp-bound (mirrors above) |

(Throttle-output rows are post-pivot-only because the production timeseries pre-pivot had structurally inhomogeneous defaults — see prior audit's F-11 retraction.)

## §B — Sensitivity to a 1σ shift in the input

For each gate consumer, what is the position-size delta of moving the input by 1 empirical-σ?

| input → gate              | empirical σ (1272d) | corresponding fragility delta under current constants | corresponding fragility delta under recalibrated constants | gate_action | sensitivity verdict |
|---------------------------|--------------------:|-----------------------------------------------------:|-----------------------------------------------------------:|-------------|---------------------|
| avg_correlation 1σ shift  | 0.099               | tanh-saturated above 0.55 → ≈0 above; ≈0.30 around 0.30 (high resolution only in tails of empirical distribution) | full ±1σ resolution centered on empirical median           | binary cap  | **misnormalized** — current constants put the empirical center on the saturated side of tanh |
| pc1_explained 1σ shift    | 0.061               | similar — empirical median 0.594 is +1.2σ above PC1_MEAN=0.45 → already saturated | ±1σ resolution                                            | binary cap  | **misnormalized** |
| fragility_score 1σ shift  | 0.114               | ≥0.97 (saturated); above 0.75 threshold either way     | informative                                                | binary cap  | **saturated** under current constants |
| entropy_z_score 1σ shift  | 1.314               | input directly drives gate                              | n/a                                                        | shift flag  | informative |
| ensemble_multiplier proxy | post-pivot std ≈ 0.10 | applied directly to position_size_mod                  | n/a                                                        | multiplicative | informative; **see §D — applied twice in the chain** |

## §C — Operator hypothesis confirmation

| hypothesis | confirmed? | evidence |
|------------|-----------|----------|
| **H-1 — Fragility historical norms are stale** | **CONFIRMED** | `AVG_CORR_MEAN=0.30` is at the 5th percentile of empirical 900d avg_correlation (true mean 0.477). `PC1_MEAN=0.45` is below the empirical minimum (0.40 in 900d). The constants were chosen against a dataset whose date and provenance is **not documented anywhere in the codebase or repo**. Phase 2 source-traces this. |
| **H-2 — Saturation = gate permanently on** | **CONFIRMED for the current window**, **partially confirmed historically** | Last 42 production days: `fragility_score` median 0.965; gate threshold 0.75; gate fires every day. Across the full 1272-day window: gate fires on 82.5% of days. Yes, in the current window the cap is a baseline tax. Across history it's better than that but still over-firing. |
| **H-3 — Compounding chain is over-throttling** | **CONFIRMED** | Phase 3 walk-throughs detail this. Snapshot: post-pivot `position_size_modifier` is at 0.30 on 33% of days and at 0.25 (clip-min floor) on 14% of days. 47% of days output is at or near the floor of [0.25, 1.0]. **Additional finding: `ensemble_multiplier` is applied twice in the chain** (`regime_fusion` line 262 × `decision_engine.compute_position_size` line 440). At ensemble_multiplier=0.85, this turns a ~0.51 modifier into a ~0.43 effective adjustment on every trade — see §D below. |
| **H-4 — 2026-04-29 sweep was scoped too narrowly** | **CONFIRMED** | The prior sweep varied `fragility_threshold` and `fragility_position_cap` only — it did **not** vary `AVG_CORR_MEAN/STD` or `PC1_MEAN/STD`. With the input metric saturated, no setting on the gate parameters can recover resolution. |
| **H-5 — Other throttles may have analogous staleness** | **PARTIALLY CONFIRMED** | `vol_regime_label` is 100% "calm" — but the cause is missing VVIX inputs (F-3 from prior audit), not stale thresholds. `vix_thresholds` themselves (`p20=13, p50=17, p80=25, p95=30`) are reasonable against empirical VIX distribution (median 17.93). `entropy_z_threshold=1.5` produces a 22.9% above-threshold rate over 1272d — reasonable. **Net: the staleness story is concentrated in fragility's normalization constants; the other throttles are limited by upstream-data issues (VVIX/SKEW), not by stale thresholds.** |

## §D — Findings discovered during execution

### F-2030-A — `ensemble_multiplier` is applied twice in the position-size chain

Code locations:
- `src/signals/regime_fusion.py:262` — `position_size_mod *= ensemble_multiplier` (multiplies into the value stored as `position_size_modifier` in the fusion result).
- `src/steps/decision_engine.py:440` — `ensemble_adj = ensemble_multiplier`; `:445` — `expert_adj = position_size_modifier`; `:452-454` — `adjusted_dollars = (... * ensemble_adj * expert_adj * throttle_adj)`.

The comment at `decision_engine.py:442-444` documents the intent ("`position_size_modifier` already includes `ensemble_multiplier` via regime_fusion, but we keep `ensemble_adj` here for backward compat when expert_signals is None") but the legacy fallback is unconditional — `ensemble_multiplier` is always multiplied in regardless of whether `expert_signals` was populated.

Empirical impact: with `ensemble_multiplier ≈ 0.85` (recent post-pivot median) and `position_size_modifier ≈ 0.51` (which already includes the 0.85), the chain produces `0.85 × 0.51 = 0.4335` instead of the intended `0.51`. That is a ~15% over-throttle on every trade in v3 mode.

Risk class: SIZING — silent over-throttling of every position sized by the v3 fusion path.

**Per packet rule "do not silently fix during the audit"**, this is logged here for operator review and not fixed in this packet's branches. The prior audit (2026-04-29) F-1 through F-10 did not surface it; either the comment-as-intent confused review, or the issue post-dates Phase-6 design.

**Recommended follow-up**: a dedicated branch fixing the multiplication chain so `ensemble_multiplier` is applied exactly once. Test that locks the property: `compute_position_size(...)` × `decide_regime_v3(...)` results should not square `ensemble_multiplier` when both are populated.

### F-2030-B — The fragility binary cap design vs. the score's continuous nature

`fragility_position_cap=0.60` is applied *binary*: if `fragility_score > 0.75` then `position_size_mod = min(position_size_mod, 0.60)`, else no action. The score itself is continuous on `[0, 1]` and (under recalibrated constants) carries empirical resolution. The current design discards resolution by using only the threshold-crossing event.

This is design, not a bug. Logged as a **design observation** for operator consideration: a continuous variant (e.g., `cap = 1.0 - 0.5 * fragility_score`) would let the metric express "more fragile = more cap" without an arbitrary 0.75 threshold. Out of scope for this packet (which is scoped to recalibrating constants); flagging because the calibration recommendation in Phase 4 will only be partial without operator decision on this axis.

## §E — Throttle list with verdict & follow-up

| throttle                  | verdict          | Phase 2 / Phase 4 follow-up                                                               |
|---------------------------|------------------|-------------------------------------------------------------------------------------------|
| `fragility_score`         | **saturated**    | Phase 2 source-traces constants. Phase 4 ships recalibrated constants behind feature gate. |
| `avg_correlation` (input) | **misnormalized** | Phase 2 traces constant; Phase 4 updates it.                                              |
| `pc1_explained` (input)   | **misnormalized** | Same.                                                                                    |
| `vol_uncertainty_score`   | informative on its own | No calibration change. The dependent `vol_regime_label` deadness is upstream (F-3, missing VVIX); out of this packet's scope. |
| `vol_regime_label`        | dead             | Out of scope (upstream data, not calibration). Carry to a separate branch.                 |
| `entropy_score` / `entropy_z_score` | informative | No calibration change.                                                                    |
| `entropy_shift_flag`      | window-quiet (not dead) | Prior audit's F-12 reframed: not "never fires", but "rare and window-dependent". 22.9% historical fire rate is acceptable. |
| `regime_confidence`       | informative      | No calibration change.                                                                    |
| `position_size_modifier`  | clamp-bound      | Driven by upstream throttles + clip floor 0.25. Phase 3 walk-throughs detail.              |
| `ensemble_multiplier`     | informative      | **F-2030-A: applied twice in chain — separate fix follow-up.**                            |

## §F — Exit criterion

At least one throttle is identifiable as `saturated` or `misnormalized`: **YES**. `fragility_score` is saturated; `avg_correlation` and `pc1_explained` are misnormalized. The operator hypothesis is confirmed at the input layer. Advance to Phase 2 (source-trace the constants and re-derive empirically).
