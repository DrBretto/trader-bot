# Walk-forward 2x2 — Ensemble multiplier double-application investigation

Date: 2026-04-30
Branch: `ai/fix-ensemble-double-apply-20260430`
Sweep harness: `scripts/run_ensemble_double_fix_sweep_20260430.py`
Sweep raw output: `runs/ensemble_double_fix_sweep_20260430.json`

## TL;DR

The 2x2 walk-forward shows all four cells produce **byte-identical** trading results. The "double-application" bug surfaced by the 2026-04-30 fragility audit (F-2030-A) does not actually fire in production — `decision_engine.py:798` and `:828` already pass `ensemble_multiplier=1.0` to `compute_position_size` whenever `expert_signals is not None`, which is the v3 production path. The function-internal multiplication is mathematically susceptible to double-application when called with both `ensemble_multiplier` and `position_size_modifier` non-neutral, but no production call site does this. **The packet's premise was wrong; per packet rule "if the bug does not reproduce, that is the finding."**

The gate (`ensemble_multiplier_already_applied`) ships behind a default-off feature flag as defense-in-depth, with new tests locking both production caller patterns plus the gate's behavior. The 2×2 walk-forward is the empirical confirmation that the gate is a no-op under the current call sites.

## §A — Reproducer (Phase 2 of method): the bug at the code level

Original audit claim: `regime_fusion.py:262` folds `ensemble_multiplier` into `position_size_mod`, then `decision_engine.py:440-454` multiplies by `ensemble_adj=ensemble_multiplier` AND `expert_adj=position_size_modifier` — both multiplications fire.

Reading the actual call sites:

```python
# src/steps/decision_engine.py:798 (main buy path, inside run())
position = compute_position_size(
    symbol,
    portfolio_value,
    current_price,
    candidate['vol_bucket'],
    regime_label,
    params,
    llm_conf_adj,
    ensemble_multiplier if expert_signals is None else 1.0,   # <-- v3 defense
    position_size_modifier,
    risk_throttle_factor,
    ...
)
```

```python
# src/steps/decision_engine.py:828 (watchlist for dashboard)
watchlist = _build_watchlist(
    scored, current_holdings, features_df, portfolio_value,
    regime_label, params,
    ensemble_multiplier if expert_signals is None else 1.0,   # <-- same v3 defense
    position_size_modifier, risk_throttle_factor,
    ...
)
```

Both v3 call sites pass `ensemble_multiplier=1.0` to the function whenever `expert_signals is not None`. Inside `compute_position_size`, that means `ensemble_adj = 1.0` and the second multiplication is a no-op. Net: `ensemble_multiplier` is applied exactly once — through `position_size_modifier`, where regime_fusion folded it in.

The v2 path (`expert_signals is None`) passes `ensemble_multiplier` directly and `position_size_modifier=1.0` (the default). Same answer: applied exactly once, through `ensemble_adj`.

## §B — 2x2 walk-forward results

Dataset: 200 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-30). Walk-forward plan: 3 folds + 42-day gate (2026-02-12 → 2026-04-29). Hybrid blend 0.35.

Two narrow windows:
- `rally`: 2026-04-15 → 2026-04-29 (11 trading days)
- `panic`: 2026-02-11 → 2026-03-31 (35 trading days)

Variants:
- `current_production` — recal off, fix off (control; expected to match prior recalibration sweep's `current_production` exactly)
- `double_fix_on_only` — recal off, fix on (this packet's fix in isolation)
- `recal_on_only` — recal on, fix off (matches recalibration audit's recommended ship)
- `both_on` — recal on, fix on (combined ship)
- `both_on_plus_F7_relax` — bonus: also flip the F-7 relax knob

| variant                       | mean fold  | gate ret  | rally ret  | panic ret | gate Sharpe | gate dd  | rally dd  | panic dd |
|-------------------------------|-----------:|----------:|-----------:|----------:|------------:|---------:|----------:|---------:|
| `current_production`          |    +1.77%  |  −17.97%  |   −98.28%  |  +13.26%  |   −1.7810   |  −7.91%  |    −9.98% |   −0.82% |
| **`double_fix_on_only`**      |    +1.77%  |  −17.97%  |   −98.28%  |  +13.26%  |   −1.7810   |  −7.91%  |    −9.98% |   −0.82% |
| `recal_on_only`               |    +1.77%  |   −9.00%  |   −98.28%  |  +23.81%  |   −1.7274   |  −7.84%  |    −9.98% |   −1.54% |
| **`both_on`**                 |    +1.77%  |   −9.00%  |   −98.28%  |  +23.81%  |   −1.7274   |  −7.84%  |    −9.98% |   −1.54% |
| `both_on_plus_F7_relax`       |    +1.77%  |  −14.95%  |   −31.21%  |  +23.81%  |   −1.9219   |  −8.58%  |    −2.41% |   −1.54% |

**Reading**: `current_production` and `double_fix_on_only` are byte-identical across every metric; `recal_on_only` and `both_on` are byte-identical across every metric. The double-fix gate has zero measurable effect because the v3 caller defense already produces correct math. (The `both_on_plus_F7_relax` row is included as the audit's recommended-full-promotion path for context.)

The 2x2 contract from §Acceptance test #3:

| (recal off / fix off) | (recal off / fix on) |
|----------------------:|---------------------:|
| broad gate −17.97%    | broad gate −17.97%   |
| rally −98.28%         | rally −98.28%        |
| panic +13.26%         | panic +13.26%        |

| (recal on / fix off) | (recal on / fix on) |
|---------------------:|--------------------:|
| broad gate −9.00%    | broad gate −9.00%   |
| rally −98.28%        | rally −98.28%       |
| panic +23.81%        | panic +23.81%       |

The two columns of each row are identical → fix-on is a no-op vs fix-off. The two rows differ only because of the recalibration → recalibration is the only material change in the matrix.

## §C — Per-day position-size walk-throughs (3 representative days)

Reusing the three days from the prior audit's Phase 3 (`2026-04-23` rally, `2026-03-25` panic, `2026-04-08` choppy), we walk the chain under the four cells of the 2×2.

For each day, the production caller pattern is `ensemble_multiplier=1.0, position_size_modifier=X` where X already includes the ensemble multiplier from regime_fusion. The chain at `compute_position_size` is then:

`adjusted_dollars = base_dollars × vol_adj × regime_adj × llm_adj × ensemble_adj × expert_adj × throttle_adj`

with `ensemble_adj=1.0` (because the caller passed 1.0) regardless of the gate flag.

### 2026-04-23 — rally / risk-on

Production inputs: `regime_label='risk_on_trend'`, `ensemble_multiplier=0.848`, fragility cap fired, `position_size_modifier=0.509` (= 1.0 × 0.60 × 0.848).

| variant                      | ens_mult passed by caller | pos_mod_modifier | ens_adj | expert_adj | adjusted_dollars (chain) |
|------------------------------|--------------------------:|-----------------:|--------:|-----------:|-------------------------:|
| (recal off, fix off)         |                       1.0 |            0.509 |     1.0 |      0.509 |                $11,198 ≈ |
| (recal off, fix on)          |                       1.0 |            0.509 |     1.0 |      0.509 |                $11,198 ≈ |
| (recal on,  fix off)         |                       1.0 |            0.509 |     1.0 |      0.509 |                $11,198 ≈ |
| (recal on,  fix on)          |                       1.0 |            0.509 |     1.0 |      0.509 |                $11,198 ≈ |

(The recalibration changes `fragility_score` from 0.978 → 0.909 but the gate still fires at threshold 0.75, so `fragility_position_cap=0.60` is still applied. The cap floor is what produces 0.509 either way.)

Chain: $20k × vol(1.0) × regime(1.10) × llm(1.0) × ens(1.0) × expert(0.509) × throttle(0.90) = $10,078. Identical across all four cells.

### 2026-03-25 — peak panic (panic_override fires)

Production inputs: `panic_prob=0.992`, `final_regime_label='high_vol_panic'`, `position_size_modifier=0.25` (clip-min), `risk_throttle_factor=1.0`.

| variant                | ens_mult passed | pos_mod | ens_adj | expert_adj | adjusted_dollars |
|------------------------|----------------:|--------:|--------:|-----------:|-----------------:|
| all four cells         |             1.0 |    0.25 |     1.0 |       0.25 |          $1,250  |

The 0.25 floor on `position_size_modifier` is set by `panic_override` and does not depend on ensemble or fragility. The gate flag and the recalibration are both irrelevant on this day. Panic protection is preserved.

### 2026-04-08 — choppy

Production inputs: `regime_label='choppy'`, `ensemble_multiplier=0.733`, `position_size_modifier=0.440` (= 1.0 × 0.60 × 0.733).

| variant                      | ens_mult passed | pos_mod | ens_adj | expert_adj | adjusted_dollars |
|------------------------------|----------------:|--------:|--------:|-----------:|-----------------:|
| all four cells               |             1.0 |   0.440 |     1.0 |      0.440 |          $7,128  |

Chain: $20k × 1.0 × 0.90 × 1.0 × 1.0 × 0.440 × 0.90 = $7,128. Identical across all four cells.

## §D — Acceptance test status

Per packet §Acceptance test #1-4:

1. **What was wrong, in plain English?** Nothing was wrong — the prior audit's F-2030-A finding was a false positive. The function `compute_position_size` is mathematically susceptible to double-applying `ensemble_multiplier` if called with both arguments non-neutral, but neither the v3 production caller (`decision_engine.py:798`, `:828`) nor the v2 caller invokes this pattern. The v3 caller passes `ensemble_multiplier=1.0` whenever `expert_signals is not None`; the v2 caller leaves `position_size_modifier` at its 1.0 default. Either way, the multiplier is applied exactly once.
2. **What changed in the code?** A defense-in-depth gate (`ensemble_multiplier_already_applied`, default False) was added to `src/steps/decision_engine.py:440-470` so that a future regression in caller code (e.g., dropping the `if expert_signals is None else 1.0` defense at line 798) cannot reintroduce the double-multiply. The gate is a no-op under all current callers; promoting it via config is also a no-op. The original misleading comment ("we keep ensemble_adj here for backward compat when expert_signals is None") was replaced with one that documents the actual caller defense pattern.
3. **What does the walk-forward say?** All four cells of the 2×2 produce byte-identical trading results. The gate has no measurable effect because the production code already applies the multiplier correctly. Recalibration alone changes the broad gate window by +9pp, the rally subset by 0pp, and the panic subset by +10.55pp (matches the prior audit's findings). The combined `both_on` cell matches `recal_on_only`.
4. **Are the prior 2026-04-29 + 2026-04-30 fixes preserved?** All 39 prior signal tests still pass on this branch. All 16 decision-engine tests (10 prior + 6 new) pass. The recalibration branch (`ai/recal-fragility-norms-20260430`) is on a separate branch; its 5 fragility-calibration tests are not in this branch's test-run scope but pass when checked out.

## §E — Recommendation

Per packet §Required method step 6:

> "The recommendation is the variant that strictly dominates on broad gate AND rally AND panic. If `both_on` strictly dominates, it is the recommended promotion. If the result is mixed (e.g., `double_fix_on_only` improves rally but worsens panic), surface the tradeoff explicitly; do not weaken the gate to declare a winner."

`double_fix_on_only` is byte-identical to `current_production` on every window. There is no tradeoff to surface — there is no effect. **Recommendation: ship the gate code as defense-in-depth (default off); do not promote it. Operator action is unchanged from the prior audit's recommendation: promote `decision_params.recalibrated_2026_04_30.json` (recalibration-only or with F-7 relax) for the actual measurable improvement; the double-fix gate stays off because turning it on changes nothing under correct callers.**

The promotion of the gate flag would only matter if a future caller breaks the v3 defense. The new tests at `tests/test_decision_engine.py::TestEnsembleMultiplierCallerPatterns` lock both production caller patterns explicitly so a regression there is caught at the unit level.

## §F — Findings

1. **Audit-error finding (false-positive class)**: F-2030-A was a function-in-isolation analysis. The prior audit (Phase 1 §D of `2026-04-30-phase1-throttle-distribution-diagnostic.md`) cited the function definition (`compute_position_size:440-454`) and the regime_fusion application (`regime_fusion.py:262`) but did not check the call sites of `compute_position_size`. Both production callers defend by passing `ensemble_multiplier=1.0` in v3 mode. Reading the function definition without checking the call sites produced a believable-but-wrong claim. **The fix is: every claim about a chain's behavior must read every call site, not just the function definition.** Captured in the new postmortem entry.
2. **Comment-as-misleading-evidence**: the original comment at `decision_engine.py:442-444` ("we keep ensemble_adj here for backward compat when expert_signals is None") was correct *given the caller defense*, but read in isolation it strongly implies the function is a bug. The reframed comment now documents the caller pattern explicitly so a future reader does not need to chase the call sites to understand the contract.
