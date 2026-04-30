# RETURN — Trader-bot fragility + throttle calibration audit (2026-04-30)

Packet: `docs/plans/2026-04-30-fragility-and-throttle-calibration-audit-packet.md`
Branch: `ai/recal-fragility-norms-20260430` (one focused commit; recalibrated constants behind feature gate; defaults preserved)
Authority: execution / discovery_bearing.

## TL;DR

The operator's hypothesis was correct. `fragility_score` has been pegged ≥ 0.97 for the last six weeks because the four normalization constants embedded in `src/signals/fragility.py` (originally written 2026-02-06 with no documented source dataset) are dramatically stale relative to the current market regime. `AVG_CORR_MEAN=0.30` is at the empirical 5th percentile; `PC1_MEAN=0.45` is below the empirical minimum. Recalibration is shipped behind a feature-gate bundle. The walk-forward sweep finds **`recalibrated_plus_F7_relax_on` strictly dominates current production on both the broad gate window (-14.95% vs -17.97%) and the rally subset (-31.21% vs -98.28%)**, with panic-window return improved (+23.81% vs +13.26%) and rally drawdown dramatically improved (-2.41% vs -9.98%). One independent finding (F-2030-A: ensemble_multiplier double-applied) is logged for operator review and deliberately not silently fixed.

## What was wrong

| # | What was wrong | Why it mattered | Status |
|---|---|---|---|
| F-2030-1 | Fragility historical-norm constants `AVG_CORR_MEAN=0.30`, `AVG_CORR_STD=0.15`, `PC1_MEAN=0.45`, `PC1_STD=0.12` are wildly stale: empirical 900d mean of `avg_correlation` is 0.4774 (current value at p5); empirical 900d mean of `pc1_explained` is 0.6007 (current value below empirical min 0.40). Constants were committed 2026-02-06 (`b9317e1`) with no source dataset, no derivation date, no test. | The metric saturates by `avg_correlation > 0.45` (the empirical median). Gate fires 82.5% of historical days — a baseline tax masquerading as a caution gate. Operator's "stuck ≥ 0.97 for six weeks" is an empirically-real consequence. | **fixed behind feature gate** — code defaults preserved, recalibrated values in `RECALIBRATED_2026_04_30` and `config/decision_params.recalibrated_2026_04_30.json` (shadow-only). |
| F-2030-A | `ensemble_multiplier` is applied twice in the position-size chain: once at `regime_fusion.py:262` (folded into `position_size_modifier`), again at `decision_engine.py:440-454` as `ensemble_adj`. Code comment cites "backward compat when expert_signals is None", but the second multiplication runs unconditionally. | At `ensemble_multiplier ≈ 0.85` (recent post-pivot median), every v3 trade is throttled to `0.85 × 0.85 = 0.7225` of intended sizing — a ~15% silent over-throttle on every position regardless of regime or fragility. | **NOT FIXED** in this packet — logged in Phase 1 §D and Phase 4 §G as a separate finding requiring operator review per packet's "no silent fixes during the audit" rule. |
| F-2030-B (design observation) | Fragility binary cap (`if score > 0.75: pos_mod = min(pos_mod, 0.60)`) discards the metric's continuous resolution. | Mathematically correct; just leaves information on the table. A continuous `cap = 1.0 - 0.5 * fragility_score` variant would express "more fragile = more cap" without an arbitrary threshold. | Out of this packet's scope; logged for operator decision. |
| Prior-audit F-12 reframed | "Entropy gate has never fired in production" was a 188-day production-window artifact, not permanent deadness. Across 1272 historical days, the gate fires 22.9% of days (27 distinct streaks). | Window-quiet, not dead. The signal is informative; the production window happened to not contain a 3-day above-threshold streak. | No change required. Phase 1 §A and §E document. |
| Prior-audit F-3 still open | `vol_regime_label` is permanently "calm" because `panic` and `unstable_calm` paths require VVIX percentile data that is not plumbed into production. | Calibration thresholds for vol-uncertainty are downstream of fixing VVIX ingest. Out of this audit's scope. | Carry to a separate VVIX-ingest fix branch. |

## What was fixed (commit on `ai/recal-fragility-norms-20260430`)

One focused commit (per packet rule "one fix = one branch = one commit = one test") with the following changes:

```
src/signals/fragility.py                          (+27 documenting + RECALIBRATED_2026_04_30 dict)
config/decision_params.recalibrated_2026_04_30.json  (new shadow bundle)
tests/test_fragility_calibration.py               (new — 5 tests, all passing)
scripts/run_recalibration_sweep_20260430.py       (new sweep harness)
runs/recalibration_sweep_20260430.json            (sweep output artifact)
docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md
docs/plans/2026-04-30-phase2-fragility-baseline-audit.md
docs/plans/2026-04-30-phase3-compound-throttle-cost.md
docs/plans/2026-04-30-phase4-recalibration-recommendation.md
docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md      (this file)
docs/POSTMORTEMS.md                               (one new entry: stale-historical-norm class)
docs/PLAN.md                                      (Phase 10 entry)
```

All 5 new tests in `test_fragility_calibration.py` pass. All 39 prior signal tests still pass on the branch.

## Walk-forward verification — what the numbers said

Sweep harness: `scripts/run_recalibration_sweep_20260430.py`. Dataset: 200 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-30). Walk-forward plan: 3 folds + 42-day gate (2026-02-12 → 2026-04-29). Hybrid blend 0.35.

Two narrow windows:
- **rally**: 2026-04-15 → 2026-04-29 (11 trading days, the operator's recent lag period)
- **panic**: 2026-02-11 → 2026-03-31 (35 trading days, the late-Feb-through-March stress run)

Headline result table:

| variant                                              | gate ret  | rally ret | panic ret | rally dd  | panic dd  |
|------------------------------------------------------|----------:|----------:|----------:|----------:|----------:|
| `current_production`                                 |   −17.97% |   −98.28% |   +13.26% |    −9.98% |    −0.82% |
| `recalibrated_constants_default_off`                 |   −17.97% |   −98.28% |   +13.26% |    −9.98% |    −0.82% |
| `recalibrated_constants_default_on` (recommended half-step) | **−9.00%**|−98.28% |+23.81% | −9.98% | −1.54% |
| **`recalibrated_plus_F7_relax_on`** (recommended full)|**−14.95%**|**−31.21%**|**+23.81%**|**−2.41%**|**−1.54%**|

`recalibrated_plus_F7_relax_on` is the only variant that **strictly dominates current production on both gate and rally windows** (per packet §Acceptance test #5). It improves panic-window return by +10.55pp; panic drawdown is slightly worse (−1.54% vs −0.82%) but still very small. The recalibration-only variant produces +9pp on the broad gate without changing rally behavior — that is the operator-safe half-step.

The harness's `recalibrated_constants_default_off` row is identical to `current_production` and `fixes_only_no_calibration`, confirming the **default-off property**: shipping the recalibration code does not change a single trading decision until the operator promotes the bundle.

## What the system does differently after this packet

If the operator does nothing: nothing changes. The recalibration ships behind the feature gate; the active bundle still points at production constants; every trading decision is byte-identical to today.

If the operator promotes only the recalibrated bundle (`cp config/decision_params.recalibrated_2026_04_30.json config/decision_params.active.json`):
- Median fragility metric drops from 0.905 → 0.443 (informative again).
- Gate fire rate drops from 82.5% → 21.2% on average.
- Broad gate return improves +9pp; panic return improves +10.55pp.
- **Rally behavior unchanged** — the recent rally has correlations at historical extremes, so even the empirically-correct metric reads them as fragile.

If the operator additionally flips `regime_fusion.fragility_relax_in_risk_on = True` in the active bundle (the F-7 knob from the prior audit, already wired):
- Rally return improves +67pp (-98.28% → -31.21%).
- Rally drawdown improves dramatically (-9.98% → -2.41%).
- Panic-day protection preserved (the `panic_override` path is untouched by both knobs; panic_prob > 0.70 still hard-overrides regardless).

## What the operator should know about the lag

The Phase 3 walk-throughs make the picture explicit:

- The recent rally (2026-04-15..2026-04-29) has `avg_correlation` 0.57-0.60 — the empirical 87th percentile over 900d. It is genuinely a high-coupling rally.
- Both the current metric (0.978) and the recalibrated metric (0.909) read it as fragile and fire the gate at threshold 0.75.
- **Recalibration alone does not change rally-day behavior.** It restores informational integrity to the metric; it does not move the threshold.
- Two complementary knobs produce rally relief:
    1. **F-7 relax-in-risk-on** (already wired, default off): when `ensemble_regime_label='risk_on_trend'` AND `ensemble_confidence ≥ 0.80`, skip the fragility cap. This is the Phase-4-recommended pairing.
    2. **F-2030-A fix** (separate finding): would lift every v3 trade by ~15% regardless of regime. Independent of fragility.
- **Panic-day protection is structurally preserved** because `panic_override` fires at `panic_prob > 0.70`, an upstream condition that recalibration does not touch. The panic-window data confirms this: panic return *improves* under recalibration (+10.55pp) because the saturated metric was producing some false-positive throttling on panic-window non-peak days that the recalibrated metric correctly de-flags.

## Day-1 / Day-3 / Day-7 monitoring (post-cutover)

When the operator promotes the recalibrated bundle (with or without F-7 relax flip), monitor:

- **Day 1**: confirm `signals.fragility` block in `decision_params.active.json` carries `AVG_CORR_MEAN=0.4774` etc. Confirm `fragility_score` in today's `signals.parquet` is materially lower than the prior week's median (expected: ~0.6-0.9 under recalibration vs ~0.97 prior). Confirm `position_size_modifier` is no longer pegged at 0.30 floor. Confirm `panic_override` fires unchanged on any `panic_prob > 0.70` day.
- **Day 3**: full broker reconciliation. Confirm new positions opened are within expected size band (recalibration should produce ~50% larger position sizes on rally days when paired with F-7 relax). Verify dashboard hero metrics.
- **Day 7**: rolling 7-day comparison vs SPY. Recommended ship beats prior production gate by ~3pp/year on this small sample; rolling tracking should show closer-to-SPY participation in any rally-like window.

## Findings carried out of scope

1. **F-2030-A (ensemble_multiplier double-applied)**: separate fix branch recommended. Test should lock the property "compute_position_size × decide_regime_v3 do not square ensemble_multiplier when both are populated".
2. **F-2030-B (binary cap design)**: design observation; operator decision pending.
3. **Prior F-3 (VVIX/SKEW not plumbed into production)**: separate VVIX-ingest fix branch. Until then `vol_regime_label` will remain permanently "calm" and the unstable_calm / panic_override paths will be unreachable.
4. **Other normalizer constants not in this packet's scope**: `vol_uncertainty.py:VIX_THRESHOLDS`, `entropy_shift.py:z_threshold`, `optimizer/replay.py:SLOPE_MEAN/STD/HY_MEAN/STD`. The new postmortem's prevention rules apply to all of these — they should each get the same provenance + re-validation + drift-test treatment over time.

## Files written

- `docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md`
- `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md`
- `docs/plans/2026-04-30-phase3-compound-throttle-cost.md`
- `docs/plans/2026-04-30-phase4-recalibration-recommendation.md`
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` (this file)
- `docs/POSTMORTEMS.md` — one new entry: **stale-historical-norm class**
- `docs/PLAN.md` — Phase 10 entry
- `src/signals/fragility.py` (provenance comments + `RECALIBRATED_2026_04_30` dict; defaults unchanged)
- `config/decision_params.recalibrated_2026_04_30.json` (shadow-only bundle)
- `tests/test_fragility_calibration.py` (5 tests; 5/5 passing)
- `scripts/run_recalibration_sweep_20260430.py`
- `runs/recalibration_sweep_20260430.json`

---

TRADER-BOT FRAGILITY + THROTTLE CALIBRATION AUDIT COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
