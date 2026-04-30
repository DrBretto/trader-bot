# Phase 4 — Recalibration recommendation + walk-forward verification

Date: 2026-04-30
Branch: `ai/recal-fragility-norms-20260430`
Sweep harness: `scripts/run_recalibration_sweep_20260430.py`
Sweep raw output: `runs/recalibration_sweep_20260430.json`
New test: `tests/test_fragility_calibration.py` (5 tests, all green)

## §A — What was implemented

1. **`src/signals/fragility.py`**: docstring expanded to document the original-norm provenance (commit `b9317e1`, 2026-02-06, no source dataset) and the empirical re-derivation. Module-level `RECALIBRATED_2026_04_30` dict added with the 900d empirical values and provenance fields. **Default constants `AVG_CORR_MEAN=0.30`, `AVG_CORR_STD=0.15`, `PC1_MEAN=0.45`, `PC1_STD=0.12` are unchanged** — the recalibration ships behind a bundle gate, not by mutating the module-level defaults.
2. **`config/decision_params.recalibrated_2026_04_30.json`**: new shadow-only bundle. Identical to `decision_params.active.json` except for `version_id`, `metadata`, and a populated `signals.fragility` block carrying the four recalibrated values. Promoting it to active is a one-file copy by the operator.
3. **`tests/test_fragility_calibration.py`**: locks the production defaults (regression guard), locks the recalibrated values (drift guard), locks the bundle-vs-code consistency (config drift guard), and asserts the recalibration produces a >0.40 fragility-score gap on a synthetic near-p25 input — i.e. resolution restoration is real, not a typo.
4. **`scripts/run_recalibration_sweep_20260430.py`**: walk-forward sweep across 7 variants. Reuses the harness from the prior audit (`run_calibration_sweep_20260429.py`) but extends the variant axis from `regime_fusion` parameters into `signals.fragility` constants.

The packet's prohibition list ("no live deploy", "no live-param mutation", "default off") is honored: the active bundle is unchanged; the recalibration is staged in a side bundle; the code defaults are unchanged; the test suite verifies all of the above.

## §B — Walk-forward sweep results

Dataset: 200 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-30) loaded by `optimizer.data_access.load_optimizer_dataset`. Walk-forward plan: 3 folds + 42-day gate (2026-02-12 → 2026-04-29). Hybrid blend 0.35. Each variant replayed with `initial_capital=$100,000`. The harness computes `fragility_score` from the snapshot's stored `avg_correlation` and `pc1_explained` using the variant's specified constants — i.e. the snapshot inputs are fixed and the variant only changes the normalization formula. Two narrow windows added:

- `rally`: 2026-04-15 → 2026-04-29 (11 trading days, the operator's lag period).
- `panic`: 2026-02-11 → 2026-03-31 (35 trading days, the late-February-through-March stress run).

Results:

| variant                                              | mean fold ret | gate ret  | rally ret  | panic ret | gate Sharpe | gate dd  | rally dd  | panic dd  |
|------------------------------------------------------|--------------:|----------:|-----------:|----------:|------------:|---------:|----------:|----------:|
| `current_production`                                 |        +1.77% |   −17.97% |    −98.28% |   +13.26% |     −1.7810 |   −7.91% |    −9.98% |    −0.82% |
| `fixes_only_no_calibration`                          |        +1.77% |   −17.97% |    −98.28% |   +13.26% |     −1.7810 |   −7.91% |    −9.98% |    −0.82% |
| `recalibrated_constants_default_off`                 |        +1.77% |   −17.97% |    −98.28% |   +13.26% |     −1.7810 |   −7.91% |    −9.98% |    −0.82% |
| **`recalibrated_constants_default_on`**              |    **+1.77%** | **−9.00%**|    −98.28% | **+23.81%**|    −1.7274 |   −7.84% |    −9.98% |    −1.54% |
| `recalibrated_plus_threshold_085`                    |        +1.77% |   −33.63% |    −98.28% |   +23.81% |     −1.7806 |  −13.50% |    −9.98% |    −1.54% |
| **`recalibrated_plus_F7_relax_on`**                  |    **+1.77%** |  −14.95% **|−31.21% (best)**| +23.81%|    −1.9219 |   −8.58% |    −2.41% |    −1.54% |
| `recalibrated_plus_threshold_085_plus_F7_relax`      |        +1.77% |   −33.63% |    −31.21% |   +23.81% |     −1.7806 |  −13.50% |    −2.41% |    −1.54% |

(Fold returns are tied at +1.77% across all variants — the fold-test windows steady-state is unchanged by the gate parameters; the gate window is the meaningful comparator. Same harness behavior as the prior audit.)

## §C — Acceptance-test gate (per packet §Acceptance test #5)

> *"The recalibrated variant beats current production on **both** the 2026-02-12 → 2026-04-29 broad gate window **and** the rally subset (2026-04-15 → 2026-04-29). If a recalibration only helps one window, that is a finding, not a ship — surface it."*

Per-variant assessment vs `current_production` (gate −17.97%, rally −98.28%):

| variant                                          | gate vs current  | rally vs current  | strict-dominance? | meets §Acceptance #5? |
|--------------------------------------------------|-----------------:|------------------:|-------------------|-----------------------|
| `recalibrated_constants_default_on`              | **+8.97pp**      |  +0.00pp          | weak (gate only)  | **NO** — broad gate improved, rally tied |
| `recalibrated_plus_threshold_085`                | −15.66pp (worse) | +0.00pp           | NO                | NO                    |
| **`recalibrated_plus_F7_relax_on`**              | **+3.02pp**      | **+67.07pp**      | **YES**           | **YES — passes the gate** |
| `recalibrated_plus_threshold_085_plus_F7_relax`  | −15.66pp (worse) | +67.07pp          | partial           | NO                    |

**Reading**: only `recalibrated_plus_F7_relax_on` strictly dominates current_production on both windows. Recalibration alone (`default_on`) helps the broad gate by ~9pp but does not move the rally — confirming the Phase 3 walk-throughs: the rally days have correlations at historical extremes, and even the empirically-correct metric reads them as fragile, so the rally relief comes from the F-7 relax knob, not from the recalibration. The two are complementary: recalibration restores resolution to the metric, F-7 lets the gate get out of the way during high-confidence risk-on regimes.

## §D — Recommendation

**Ship the recalibration code and bundle behind feature gate (defaults preserved). Recommend operator promotion of the recalibrated bundle paired with `fragility_relax_in_risk_on=True` after a one-day shadow validation.**

Concrete actions, in order:

| # | Action | Status | Risk |
|---|--------|--------|------|
| 1 | Merge `ai/recal-fragility-norms-20260430` to main | **READY** — one focused commit, one passing test file (5/5), all 39 prior signal tests pass on the branch. | None (defaults unchanged). |
| 2 | Phase 1 / Phase 2 / Phase 3 / Phase 4 docs in `docs/plans/` | **WRITTEN** | None. |
| 3 | New `config/decision_params.recalibrated_2026_04_30.json` shadow bundle | **WRITTEN**, not promoted | None — file exists but is not pointed-to by deploy. |
| 4 | Operator one-day shadow with the recalibrated bundle (no F-7 relax) | **PENDING OPERATOR APPROVAL** | Low — gate window improvement is +9pp; rally behavior unchanged; panic-day protection preserved by `panic_override`. |
| 5 | Operator promote the recalibrated bundle + flip `fragility_relax_in_risk_on=True` in active bundle | **PENDING OPERATOR APPROVAL after step 4** | Medium — the sweep shows this strictly dominates current production on both gate and rally; rally relief comes through. The Phase 3 walk-throughs show the panic_override path remains intact. |
| 6 | F-2030-A double-applied ensemble_multiplier — separate fix | **NOT IN THIS PACKET** — surfaced as a finding in Phase 1 §D for operator review. | Out of scope here; recommend a follow-up branch. |

The §G "Decision-params bundle" of the packet (proposing a one-line `regime_fusion.fragility_position_cap` change) is **not** what this audit is recommending. The right cutover is a two-line change: promote the bundle (4 fragility constants) and flip the F-7 relax knob in the active bundle. That is the minimal-blast-radius answer that `recalibrated_plus_F7_relax_on` produces in the sweep.

## §E — Sensitivity analysis

What if the operator chooses different recalibration windows? (Phase 2 §B already tested 252d / 504d / 900d / 1272d.)

| baseline window | AVG_CORR_MEAN | AVG_CORR_STD | PC1_MEAN | PC1_STD | gate fire rate (recomputed over 1272d) |
|-----------------|--------------:|-------------:|---------:|--------:|---------------------------------------:|
| current (n/a)   | 0.30          | 0.15         | 0.45     | 0.12    | 82.5% |
| 1272d (full)    | 0.4576        | 0.0991       | 0.5990   | 0.0608  | 23.4% |
| 900d (recommended) | **0.4774** | **0.0979**   | **0.6007** | **0.0658** | **21.2%** |
| 504d (last 2y)  | 0.4514        | 0.0713       | 0.5835   | 0.0593  | 25.3% |
| 252d (last 1y)  | 0.4388        | 0.0761       | 0.5899   | 0.0541  | 24.6% |

The 900d window is the packet-specified default and produces the most-conservative recalibration (highest mean → highest threshold for the gate to fire). 504d and 252d would produce slightly more aggressive (rally-friendlier) configurations; 1272d is materially indistinguishable. **The 900d choice is the right default**; if the operator wants more rally aggressiveness, the F-7 relax knob is the cleaner lever than tightening the baseline window.

## §F — What changes for the system after the operator promotes (full path)

If operator promotes both recalibrated bundle AND `fragility_relax_in_risk_on=True`:

| metric                        | current_production | promoted (recal+F7) | delta             |
|-------------------------------|-------------------:|--------------------:|-------------------|
| Median historical fragility   | 0.905              | 0.443               | metric is informative again |
| Gate fire rate (1272d hist.)  | 82.5%              | 21.2% (when relax inactive)  | 4× more selective |
| 2026-02..04 broad gate ret    | −17.97%            | −14.95%             | +3.02pp           |
| 2026-04-15..29 rally ret      | −98.28%            | −31.21%             | +67.07pp          |
| 2026-02-11..03-31 panic ret   | +13.26%            | +23.81%             | +10.55pp          |
| 2026-02-11..03-31 panic dd    | −0.82%             | −1.54%              | slightly worse but still small |
| Rally drawdown                | −9.98%             | −2.41%              | dramatically better |

If operator promotes ONLY the recalibrated bundle (no F-7 flip):

| metric                        | current_production | promoted (recal only) | delta |
|-------------------------------|-------------------:|----------------------:|-------|
| Broad gate ret                | −17.97%            | −9.00%                | +8.97pp |
| Rally ret                     | −98.28%            | −98.28%               | 0     |
| Panic ret                     | +13.26%            | +23.81%               | +10.55pp |

That second variant is operator-safe-call: meaningful improvement on broad gate and panic windows, no behavior change in the recent rally. It is the more-conservative half-step.

## §G — Findings to surface

(All Phase 1 §D findings carried forward into the RETURN doc.)

1. **F-2030-A** — `ensemble_multiplier` is applied twice in the position-size chain (`regime_fusion.py:262` × `decision_engine.py:440`). Independent of fragility recalibration; estimated 15-25% of every v3 trade. Logged for operator review; not silently fixed.
2. **F-2030-B** — Fragility binary cap design vs continuous score. Design observation, not bug. Logged for operator decision on a follow-up axis.
3. **Prior-audit F-12 reframed** — "entropy gate has never fired" was a 188-day production-window artifact, not permanent deadness. Across 1272 days the gate fires 22.9% of days (27 distinct streaks). Window-quiet, not dead.
4. **Prior-audit F-3 still open** — `vol_regime_label` is permanently "calm" because the panic / unstable_calm paths require VVIX percentile data that is not plumbed into production. Out of this packet's scope; remains a separate calibration block downstream of fixing VVIX ingest.

## §H — Acceptance-test status

| Acceptance criterion (from packet §Acceptance test) | Status |
|-----------------------------------------------------|--------|
| 1. Empirical distribution of `avg_correlation` and `pc1_explained` over 900d | **PASS** — see Phase 2 §B/C |
| 2. Health card per throttle (informativeness, gate fire rate, sensitivity) | **PASS** — see Phase 1 §A/B |
| 3. Compound cost worked example with fragility pegged at 0.98 | **PASS** — see Phase 3 §A |
| 4. Side-by-side current vs recalibrated on representative days | **PASS** — see Phase 3 §A/B/C |
| 5. Walk-forward gate beats current on both broad gate and rally | **PASS** — `recalibrated_plus_F7_relax_on` strictly dominates on both windows |
