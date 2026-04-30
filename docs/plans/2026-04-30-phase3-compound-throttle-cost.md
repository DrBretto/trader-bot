# Phase 3 — Compound throttle cost: three representative-day walk-throughs

Date: 2026-04-30

## Method

For each representative day, walk the position-sizing chain step-by-step using the production code logic in `src/signals/regime_fusion.decide_regime_v3` and `src/steps/decision_engine.compute_position_size`. Compare four configurations:

- **CURRENT** — production constants (`AVG_CORR_MEAN=0.30`, `AVG_CORR_STD=0.15`, `PC1_MEAN=0.45`, `PC1_STD=0.12`); threshold 0.75; cap 0.60; F-7 relax off.
- **RECAL** — recalibrated 900d constants (`0.4774 / 0.0979 / 0.6007 / 0.0658`); threshold 0.75; cap 0.60; F-7 relax off.
- **RECAL + threshold 0.85** — recalibrated constants and a raised gate threshold.
- **RECAL + F-7 relax** — recalibrated constants with F-7 relax-in-risk-on enabled at `fragility_relax_confidence=0.80`.

Inputs are pulled from `dashboard/timeseries.json` for each date. `ensemble_multiplier` is back-solved from the production `position_size_modifier` (`prod_pos_mod / 0.60` for non-panic days, since regime_fusion ends with a multiply by `ensemble_multiplier`). The `decision_engine` chain assumes `base_dollars = $20k = 20% of $100k portfolio`, `vol_adj=1.0` (med), `llm_adj=1.0`, and the production `regime_adj` for the day's regime label.

The `decision_engine` chain in production multiplies by both `ensemble_adj` (i.e. `ensemble_multiplier` directly) **and** `expert_adj` (i.e. `position_size_modifier`, which already includes `ensemble_multiplier` from `regime_fusion`). The "with F-2030-A fixed" line shows what the chain would be if that double-application were removed (covered as a follow-up finding in Phase 1 §D).

## §A — Day 1: 2026-04-23 (rally / risk-on)

Production data: `final_regime_label='risk_on_trend'`, `regime_confidence=0.91`, `panic_prob=0.074`, `avg_correlation=0.587`, `pc1_explained=0.678`, `fragility_score=0.978`, `position_size_modifier=0.509`. Back-solved `ensemble_multiplier ≈ 0.848`.

| variant                  | fragility | gate fired? | position_size_mod (after fusion) | exposure $ (chain) | exposure $ (if F-2030-A fixed) |
|--------------------------|----------:|:-----------:|---------------------------------:|-------------------:|-------------------------------:|
| CURRENT                  |    0.9785 | yes         | 0.509                            |          $8,542.96 |                     $10,074.24 |
| RECAL                    |    0.9089 | yes         | 0.509                            |          $8,542.96 |                     $10,074.24 |
| RECAL + threshold 0.85   |    0.9089 | yes         | 0.509                            |          $8,542.96 |                     $10,074.24 |
| RECAL + F-7 relax        |    0.9089 | **no — relaxed** | 0.848                       |         **$15,820.29** |                 **$18,656.00** |

**Reading**: the rally day has `avg_correlation=0.587` and `pc1_explained=0.678` — **at the empirical 87th percentile** (avg_corr) and **94th percentile** (pc1) over the 900-day window. It is genuinely a high-coupling day. Both the current metric (0.978) and the recalibrated metric (0.909) read it as fragile and fire the gate. **Recalibration alone does not change rally-day behavior** because the rally has correlations near historical extremes — even the empirically-correct metric reads them as fragile.

The behavior change comes from one of three knobs:
1. **F-7 relax** turns off the gate when the ensemble is high-confidence (0.91 ≥ 0.80) in a risk-on regime — position grows from $8.5k → $15.8k (1.85×).
2. **F-2030-A fix** (separate bug) gives $10.1k vs $8.5k regardless of fragility config (+18%).
3. **Combined (RECAL + F-7 + F-2030-A fix)** gives $18.7k — 2.18× current production sizing.

The Phase-2 calibration is necessary but not sufficient for rally relief; it has to be paired with either the F-7 relax or the F-2030-A fix to produce visibly different rally-day behavior. The Phase 4 sweep tests these combinations.

## §B — Day 2: 2026-03-25 (peak panic — `panic_override` fires)

Production data: `final_regime_label='high_vol_panic'`, `panic_prob=0.992`, `avg_correlation=0.522`, `pc1_explained=0.621`, `fragility_score=0.948`, `position_size_modifier=0.250`. Override reason: `panic_override`. Assumed `ensemble_multiplier=0.85`.

| variant      | fragility | panic_override? | position_size_mod | exposure $ |
|--------------|----------:|:---------------:|------------------:|-----------:|
| CURRENT      |    0.9478 | yes             | 0.250 (clip-min)  |  $1,062.50 |
| RECAL        |    0.6794 | yes             | 0.250 (clip-min)  |  $1,062.50 |

**Reading**: the panic gate fires at `panic_prob=0.992 > 0.70`, hard-overrides the regime to `high_vol_panic`, and forces `position_size_mod=0.25` (clip-min) and `risk_throttle=1.0`. **Fragility is not consulted on this path** — the recalibration is irrelevant for true panics, which are caught by the panic-probability override.

This is the protective property the recalibration must preserve. **Confirmed**: panic-day exposure is $1,063 either way. Safety is unchanged.

The recal fragility on this day (0.679) is below the 0.75 threshold — meaning, on this day, the *fragility* gate would NOT have fired even without the panic override. That is consistent with the empirical reading: avg_correlation 0.522 and pc1 0.621 are above the empirical median but not at the extremum. The actual panic signal here is `panic_prob=0.992`, not the cross-asset correlation. The `fragility` metric correctly treats this as "moderately elevated, not extreme".

## §C — Day 3: 2026-04-08 (choppy / mixed, just past March panic)

Production data: `final_regime_label='choppy'`, `regime_confidence=0.98`, `panic_prob=0.081`, `avg_correlation=0.563`, `pc1_explained=0.660`, `fragility_score=0.971`, `position_size_modifier=0.440`. Back-solved `ensemble_multiplier ≈ 0.733`.

| variant                  | fragility | gate fired? | position_size_mod (after fusion) | exposure $ | exposure $ (F-2030-A fixed) |
|--------------------------|----------:|:-----------:|---------------------------------:|-----------:|----------------------------:|
| CURRENT                  |    0.9709 | yes         | 0.440                            |  $5,222.45 |                   $7,124.76 |
| RECAL                    |    0.8558 | yes         | 0.440                            |  $5,222.45 |                   $7,124.76 |
| RECAL + threshold 0.85   |    0.8558 | yes         | 0.440                            |  $5,222.45 |                   $7,124.76 |

**Reading**: the choppy day has `avg_correlation=0.563`, almost as high as the rally day. Both current and recalibrated metrics fire the gate. The regime label being `choppy` (low confidence in risk-on) means F-7 relax is not eligible, so even in the F-7 variant the gate would still fire. **Choppy days are throttled identically under all configurations**; recalibration does not weaken caution in mixed regimes.

The F-2030-A fix would lift exposure from $5.2k → $7.1k (+36%) on this kind of day regardless of fragility config.

## §D — Compound-cost summary

| representative day | regime              | fragility CURRENT → RECAL | exposure CURRENT | exposure recommended config        | delta                  |
|--------------------|---------------------|---------------------------|-----------------:|-----------------------------------:|-----------------------:|
| 2026-04-23 rally   | risk_on_trend       | 0.978 → 0.909             |        $8,542.96 | RECAL + F-7 relax: $15,820.29       | **+85% participation** |
| 2026-03-25 panic   | high_vol_panic      | 0.948 → 0.679             |        $1,062.50 | unchanged: $1,062.50                | **+0% (safety preserved)** |
| 2026-04-08 choppy  | choppy              | 0.971 → 0.856             |        $5,222.45 | unchanged: $5,222.45                | **+0% (caution preserved)** |

## §E — Findings to surface in the RETURN doc

1. **Recalibration alone is necessary but not sufficient.** The metric becomes informative again, but the current high-correlation regime keeps the gate firing on rally days. To get rally relief, recalibration must be paired with the F-7 relax knob (already wired, currently off).
2. **Panic-day protection is preserved** because `panic_override` fires on `panic_prob > 0.70`, an upstream condition that recalibration does not touch.
3. **Choppy-day caution is preserved** because the gate still fires under recalibration when correlations are mid-elevated and ensemble confidence in risk-on is not high.
4. **F-2030-A** (`ensemble_multiplier` applied twice) is a separate, multiplicative bug independent of fragility calibration. It costs ~15-25% on every v3 trade regardless of regime. Logged in Phase 1 §D for operator review; deliberately not silently fixed in this audit.

## §F — Exit criterion

Three representative-day walk-throughs are produced. The operator can read them and form an opinion: the recalibration is mathematically correct, preserves panic-day safety, preserves choppy-day caution, and — by itself — does not produce the rally relief the lag suggests is needed. The Phase 4 sweep tests whether the recalibration combined with F-7 relax (or with the threshold raised) crosses the gate metrics in §Acceptance test of the packet.
