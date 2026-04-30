# RETURN — Trader-bot ensemble multiplier double-application investigation (2026-04-30)

Packet: `docs/plans/2026-04-30-ensemble-multiplier-double-application-fix-packet.md`
Branch: `ai/fix-ensemble-double-apply-20260430` (one focused commit; gate behind feature flag; defaults preserved)
Authority: execution / bounded.

## TL;DR

**The audit's F-2030-A finding was a false positive.** `ensemble_multiplier` is **not** applied twice in the production position-sizing chain. The v3 production callers at `decision_engine.py:798` and `:828` pass `ensemble_multiplier=1.0` to `compute_position_size` whenever `expert_signals is not None`, neutralizing the second multiplication inside the function. The v2 callers leave `position_size_modifier` at its 1.0 default, neutralizing the first multiplication. Either way, the multiplier is applied exactly once.

The function is mathematically *susceptible* to double-application if called naively (with both arguments non-neutral), but no production call site invokes that pattern. The `ensemble_multiplier_already_applied` gate ships behind a default-off feature flag as defense-in-depth — useful for catching a future regression in the line-798 defense, but with no measurable effect under any current caller. The 2x2 walk-forward confirms this empirically: `current_production`, `double_fix_on_only`, `recal_on_only`, and `both_on` produce byte-identical trading results within their respective recalibration columns.

Per packet rule "if the bug does not reproduce — i.e., if some code path the audit missed already deduplicates — that is the finding. Surface it; do not silently ship a no-op fix" — the gate is shipped as documented defense-in-depth, not as a fix. Recommendation: do not promote.

## What was wrong

| # | Claimed | Actual | Evidence |
|---|---------|--------|----------|
| F-2030-A | `ensemble_multiplier` is applied twice in the position-size chain — once at `regime_fusion.py:262` (folded into `position_size_modifier`) and again at `decision_engine.py:440-454` (as `ensemble_adj`). Estimated 15-25% over-throttle on every v3 trade. | The function `compute_position_size` *would* double-apply if called with both arguments non-neutral, but no production caller invokes that pattern. The v3 callers at `decision_engine.py:798` and `:828` pass `ensemble_multiplier=1.0` whenever `expert_signals is not None`. The v2 callers pass `position_size_modifier=1.0` (the default). Either way, single application. | Code at `decision_engine.py:798`: `ensemble_multiplier if expert_signals is None else 1.0`. The 2x2 walk-forward shows `current_production` and `double_fix_on_only` byte-identical across the 42-day broad gate, the 11-day rally, and the 35-day panic windows. |
| Comment-as-misleading-evidence | The comment at `decision_engine.py:442-444` ("we keep ensemble_adj here for backward compat when expert_signals is None") suggested an intentional double-application. | The comment was correct *given the caller defense at line 798*, but read in isolation it implies a bug. The audit chased the comment-and-math without checking the call sites. | The original audit (Phase 1 §D of `2026-04-30-phase1-throttle-distribution-diagnostic.md`) cited the function definition and the upstream `regime_fusion.py:262` but not `decision_engine.py:798`. |

## What was fixed (commit on `ai/fix-ensemble-double-apply-20260430`)

One focused commit:

```
src/steps/decision_engine.py             (+19 / -3 — gate flag + rewritten comment)
tests/test_decision_engine.py            (+93 — 6 new tests in TestEnsembleMultiplierCallerPatterns)
scripts/run_ensemble_double_fix_sweep_20260430.py  (new — 2x2 sweep harness)
runs/ensemble_double_fix_sweep_20260430.json       (sweep output artifact)
docs/plans/2026-04-30-ensemble-multiplier-double-application-fix-packet.md
docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md
docs/plans/2026-04-30-ensemble-double-fix-RETURN.md  (this file)
docs/POSTMORTEMS.md                      (one new entry: known-bug-documented-as-feature class)
docs/PLAN.md                             (Phase 11 entry)
```

The fix:
- New gate `ensemble_multiplier_already_applied` (read via `decision_engine_overrides.position_size.ensemble_multiplier_already_applied`, default `False`). When `True`, forces `ensemble_adj=1.0` regardless of the passed `ensemble_multiplier`. Defense-in-depth: protects against a future regression that drops the v3 caller defense at line 798.
- Comment at `decision_engine.py:440-470` rewritten to document the caller defense pattern explicitly. A future reader does not need to chase call sites to understand the contract.
- Tests in `TestEnsembleMultiplierCallerPatterns` (6 tests):
    1. `test_v3_caller_pattern_single_application`: locks the v3 production pattern (caller passes 1.0, expects single application via `position_size_modifier`).
    2. `test_v2_caller_pattern_single_application`: locks the v2 / pre-hybrid pattern (caller passes ensemble, `position_size_modifier=1.0` default).
    3. `test_unsafe_caller_pattern_double_applies_without_gate`: documents the function's susceptibility under naive call patterns.
    4. `test_gate_on_neutralizes_unsafe_caller`: gate fix forces correct math even with naive callers.
    5. `test_gate_on_with_v3_caller_is_no_op`: critical — proves the gate is safe to ship default-on under the actual production path.
    6. `test_gate_default_off_explicit_false_no_op`: explicit-False matches no-override default.

All 16 decision-engine tests pass (10 prior + 6 new). All 39 prior signal tests pass. No regressions.

**The packet's required `config/decision_params.recalibrated_2026_04_30.json` extension was not produced** because the gate is a no-op under correct callers — promoting it via config does not change any trading decision. The recalibration packet's recommendation stands as the only operator-action item; this packet adds defense-in-depth code but does not require a config promotion.

## Walk-forward 2x2

Dataset: 200 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-30). 3 folds + 42-day gate (2026-02-12 → 2026-04-29). Hybrid blend 0.35.

| variant                       | gate ret  | rally ret  | panic ret | rally dd  | panic dd |
|-------------------------------|----------:|-----------:|----------:|----------:|---------:|
| `current_production`          |  −17.97%  |   −98.28%  |  +13.26%  |    −9.98% |   −0.82% |
| `double_fix_on_only`          |  −17.97%  |   −98.28%  |  +13.26%  |    −9.98% |   −0.82% |
| `recal_on_only`               |   −9.00%  |   −98.28%  |  +23.81%  |    −9.98% |   −1.54% |
| `both_on`                     |   −9.00%  |   −98.28%  |  +23.81%  |    −9.98% |   −1.54% |
| `both_on_plus_F7_relax`       |  −14.95%  |   −31.21%  |  +23.81%  |    −2.41% |   −1.54% |

**Reading**: the double-fix gate has zero measurable effect (rows 1 vs 2, rows 3 vs 4 are byte-identical). The recalibration is the only material change in the matrix (rows 1-2 vs rows 3-4); F-7 relax is the only further material change (rows 3-4 vs row 5). The double-fix gate is a no-op under all current callers.

## What the system does differently after this packet

If the operator does nothing: nothing changes. The gate ships default-off; promoting it to default-on would also change nothing (because the v3 caller already passes 1.0).

If the operator promotes the gate via config (`decision_engine.position_size.ensemble_multiplier_already_applied = true` in any active bundle): nothing changes. The gate is harmless documentation for the existing correct behavior.

The test suite now actively prevents a regression in the line-798 defense. If a future refactor drops `ensemble_multiplier if expert_signals is None else 1.0` and reverts to passing `ensemble_multiplier` directly, the v3-caller-pattern test will fail and force the developer to re-add the defense (or to enable the gate to compensate).

## What the operator should know

1. **The 2026-04-30 fragility audit's F-2030-A finding was wrong.** The prior audit's RETURN doc and Phase 4 doc both list F-2030-A as a "separate finding requiring operator review, not silently fixed." That review, performed in this packet, found no bug. The recalibration audit's primary findings (stale historical norms, recommended bundle promotion ± F-7 relax) are unaffected — they stand on their own evidence.
2. **The defense-in-depth gate ships as code, not as policy.** The new gate flag exists to catch a future regression that breaks the line-798 defense. Today it is a no-op under all callers. There is no operator action.
3. **The unchanged recommendation from the 2026-04-30 fragility audit RETURN doc**: promote `decision_params.recalibrated_2026_04_30.json` (recalibration-only) for +9pp on the broad gate window with no rally or panic regression. For full rally relief, additionally flip `regime_fusion.fragility_relax_in_risk_on=True` (the F-7 knob from the 2026-04-29 audit). Both are gated and operator-controlled.
4. **The bigger lesson goes into POSTMORTEMS.** "Backward compat" in a code comment is a flag not a closure; function-in-isolation analysis is incomplete. The new postmortem entry codifies these for future audits.

## Day-1 / Day-3 / Day-7 monitoring

There is nothing for the operator to monitor specifically about this packet's fix. The gate is default-off and a no-op under correct callers — there is no behavior change to verify. If the operator promotes the recalibrated bundle (the recommendation from the prior packet), the day-1/3/7 monitoring listed in `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` applies unchanged.

## Findings carried out of scope

- **Other compute_position_size call sites**: the audit looked only at `decision_engine.py:798` and `:828`. The watchlist sub-call at `:569` is downstream of `:828` and inherits the same `ensemble_multiplier=1.0` defense. No additional susceptibilities were found.
- **Other multiplicative chains in the codebase**: a brief grep for similar "kept for backward compat" comment patterns found none in the trading hot path. The pattern that produced this false-positive is localized to `compute_position_size`.

## Files written

- `docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md` (2x2 sweep + 3 representative-day walk-throughs)
- `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` (this file)
- `docs/POSTMORTEMS.md` — one new entry: **known-bug-documented-as-feature class**
- `docs/PLAN.md` — Phase 11 entry
- `src/steps/decision_engine.py` (rewritten comment at lines 440-470, new gate flag — defense-in-depth, no-op under correct callers)
- `tests/test_decision_engine.py` (6 new tests in `TestEnsembleMultiplierCallerPatterns`)
- `scripts/run_ensemble_double_fix_sweep_20260430.py`
- `runs/ensemble_double_fix_sweep_20260430.json`

---

TRADER-BOT ENSEMBLE DOUBLE-APPLICATION FIX COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
