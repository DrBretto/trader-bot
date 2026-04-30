# PKT-TRADER-BOT-ENSEMBLE-DOUBLE-APPLICATION-FIX-20260430

Packet Owner:
- Claude (fresh executor session, instantiated in `/Users/drbretto/Desktop/Projects/trader-bot/`)

Date:
- 2026-04-30

Authority surface:
- `CLAUDE.md` (project workflow, plan-doc discipline)
- `docs/PLAN.md` (current roadmap; Phase 9 + Phase 10 closeouts)
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` (the audit that surfaced this finding as F-2030-A)

Authority level:
- execution

Discovery mode:
- bounded (the finding is named and code-located; the executor verifies the fix and measures interaction with the recalibration shipped earlier today, but is not expected to surface new findings beyond the ensemble-double-application surface)

## What this packet exists to fix

F-2030-A from the 2026-04-30 fragility audit: `ensemble_multiplier` is applied twice in the position-sizing chain.

- **Application 1**: `src/signals/regime_fusion.py:262` — `position_size_mod *= ensemble_multiplier`. The `position_size_modifier` returned by `decide_regime_v3` already has the ensemble multiplier folded in.
- **Application 2**: `src/steps/decision_engine.py:440` and `:452-454` — `ensemble_adj = ensemble_multiplier`, then `adjusted_dollars = base_dollars * vol_adj * regime_adj * llm_adj * ensemble_adj * expert_adj * throttle_adj`. The expert_adj **is** `position_size_modifier` (line 445), which already contains `ensemble_multiplier`.

Net effect: every v3 trade that passes expert signals into `compute_position_size` has `ensemble_multiplier` multiplied in twice. With a typical post-hybrid `ensemble_multiplier ≈ 0.85`, that is a **~15% silent over-throttle** on every position. The comment at `decision_engine.py:442-444` is self-aware ("position_size_modifier already includes ensemble_multiplier via regime_fusion, but we keep ensemble_adj here for backward compat when expert_signals is None") — meaning the bug was *known and documented* but never resolved. The "backward compat when expert_signals is None" framing is the smoking gun: in the v3 production path, expert_signals is not None and the double-application fires.

## Operator hypothesis (the question this packet exists to answer)

Removing the double-application is mechanically straightforward, but the blast radius is every v3 trade. With the fragility recalibration also sitting default-off behind a feature gate (shipped 2026-04-30 on `ai/recal-fragility-norms-20260430`), there are now two independent sizing changes pending. The audit shipped them measured separately; this packet measures them together.

The walk-forward must answer:

1. Does removing the double-application produce a meaningful sizing change in isolation?
2. Does the recalibration + double-application-removed combination produce a *better* result than either change alone, or is the recalibration sweep result already capturing most of the win?
3. Is there any window (panic, rally, broad gate) where removing the double-application *worsens* outcomes — i.e., where the silent 15% throttle was load-bearing protection rather than a bug?

If H-3 is "no" everywhere, the fix is a free win and ships behind a default-off gate alongside the recalibration. If "yes" anywhere, that is a finding requiring operator review — surface it; do not weaken the fix to ship.

## Scope

Fix the double-application of `ensemble_multiplier` in `src/steps/decision_engine.py`, behind a feature gate defaulted off. Run the same four-variant walk-forward harness used in the 2026-04-30 audit, comparing all combinations of {recalibration off/on} × {double-application fix off/on}. Recommend a default-on configuration for both ship paths. **Do not cut over to live until the operator approves.**

## Stop condition

The fix is merged behind a feature gate, the four-variant walk-forward result is documented, and `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` ends with the literal final line:

```
TRADER-BOT ENSEMBLE DOUBLE-APPLICATION FIX COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
```

## Acceptance test

The operator opens the RETURN doc and within 15 minutes can answer:

1. **What was wrong, in plain English?** The ensemble multiplier was multiplied twice into every v3 position size; this is a ~15% silent throttle on every trade.
2. **What changed in the code?** One file (`src/steps/decision_engine.py`), a small region around line 440–454, gated behind a config flag (defaulted off) so the operator can promote independently of the recalibration.
3. **What does the walk-forward say?** A 2×2 table — {recal off, recal on} × {double-fix off, double-fix on} — with broad-gate, rally-subset, and panic-subset returns + drawdowns for each cell. The cell to ship is the one that strictly dominates (or, if no cell strictly dominates, the most defensible compromise with the tradeoff named).
4. **Are the prior 2026-04-29 + 2026-04-30 fixes preserved?** Tests still pass; no regressions in dashboard reconciliation, signal computation, or fragility recalibration.

## Reference outputs

- **Negative reference (do NOT produce)**: a fix that silently flips the default to "on" because the walk-forward looked good. The default ships **off**, like the recalibration. Operator promotes both via config.
- **Negative reference (do NOT produce)**: a "backward compat" defense of the double-application as intentional. The comment in the code is wrong; the v3 production path passes `expert_signals` and thus hits the double-multiply. If the executor finds the v2 / pre-hybrid path *did* depend on the double-multiply, the fix preserves v2 behavior with an explicit branch — but the v3 path is unambiguously a bug.
- **Negative reference (do NOT produce)**: a bundled commit that mixes this fix with the recalibration. They ship as separate gates, separate commits, separate tests.
- **Positive reference shape — fix branch**: branch `ai/fix-ensemble-double-apply-20260430`; one focused commit; one new unit test in `tests/test_decision_engine.py` that locks the single-application math and asserts equivalence-to-baseline when the gate is off.
- **Positive reference shape — RETURN table**: 2×2 walk-forward table with three columns per cell (broad gate / rally / panic), each showing return % and max drawdown %.

## Write surface

- `src/steps/decision_engine.py` (the fix, behind a feature gate)
- `tests/test_decision_engine.py` (new test or extend existing)
- `config/decision_params.recalibrated_2026_04_30.json` (extend the existing shadow bundle to also enable the double-fix when promoted, OR create a sibling bundle — executor's call based on what cleanly composes)
- `docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md` (the 2×2 walk-forward result table + per-day position-size deltas on three representative days from the prior audit's Phase 3)
- `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` (final stop-condition file)
- `docs/POSTMORTEMS.md` (one new entry: known-bug-documented-as-feature class — the comment at `decision_engine.py:442-444` documented the bug as backward-compat behavior; the lesson is that "backward compat" is a smoking-gun phrase requiring follow-up review, not a closure)
- `docs/PLAN.md` (Phase 11 entry summarizing the fix)

**Specifically PROHIBITED**:
- Any live deploy or live-param mutation.
- Any change to `frontend/src/` (operator WIP).
- Any change to `src/signals/regime_fusion.py:262` — the application there is correct (it is *the* canonical place ensemble_multiplier is folded into `position_size_mod`). The fix is in `decision_engine.py`, which is the duplicate.
- Removing or weakening the comment at `decision_engine.py:442-444` *silently*. If the comment is updated, the new comment must describe the *current* (post-fix) behavior accurately.
- Any rollback of the 2026-04-29 or 2026-04-30 fixes.

## Read-first list

- `CLAUDE.md`
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` — the audit that flagged F-2030-A. Understand which constants and behaviors it shipped (default-off recalibration on `ai/recal-fragility-norms-20260430`).
- `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` — the prior audit's full picture; F-7 (fragility relax knob) is the only other live default-off knob.
- `src/steps/decision_engine.py` — read the full `compute_position_size` function (line 380–481).
- `src/signals/regime_fusion.py` — confirm line 262 and the `position_size_modifier` return value at line 295 carry `ensemble_multiplier`. The fix should not assume; it should verify.
- The existing `ai/recal-fragility-norms-20260430` branch — read its diff. The fix here should compose cleanly when both branches are merged.
- Walk-forward harness (whichever script the 2026-04-30 audit used; likely a sibling of `scripts/run_calibration_sweep_20260429.py`).
- 900-day data cache from prior runs; refresh from `s3://investment-system-data/daily/` if older than 24h.

## Flow position

This packet is the **bug-fix follow-on** to the 2026-04-30 calibration audit. The recalibration ships default-off; this fix ships default-off; both can be promoted independently or together via config.

## Required method

### 1. Read the Read-first list

Specifically internalize the recalibration branch's structure so the fix composes cleanly.

### 2. Confirm the bug at code level

- Read `regime_fusion.py:260–295` and trace `position_size_mod` through the return.
- Read `decision_engine.py:380–481` and trace `ensemble_adj` and `expert_adj` into `adjusted_dollars`.
- Confirm with a small reproducer: build a fake `compute_position_size` call with `ensemble_multiplier=0.85` and `position_size_modifier=0.85` (representing 0.85 × 0.85 = 0.7225 from regime_fusion's perspective), and show that the resulting `adjusted_dollars` is multiplied by `0.85 × 0.7225 = 0.614` instead of `0.7225`.
- If the bug does not reproduce — i.e., if some code path the audit missed already deduplicates — that is the finding. Surface it; do not silently ship a no-op fix.

### 3. Implement the fix

The fix is gated. Two acceptable shapes — executor picks based on cleaner composition with the recalibration branch:

- **Shape A** (recommended if it composes cleanly): a config flag `ensemble_multiplier_already_applied` (default `False` to preserve current behavior), read via `decision_engine_overrides`. When `True`, set `ensemble_adj = 1.0` instead of `ensemble_multiplier`. The flag's name documents *why* it is set, not just *what* it does.
- **Shape B**: a new parameter to `compute_position_size` — `ensemble_already_applied: bool = False` — passed by the v3 caller (`expert_signals is not None`) and not passed by the v2 caller. Same effect, slightly more explicit.

Shape A is preferred because it can be promoted via the same `decision_params` bundle that promotes the recalibration. Shape B is acceptable if the executor finds the call sites cleaner with an explicit argument. Either way, the v2 (`expert_signals is None`) code path keeps current behavior unchanged.

### 4. Lock the fix with a test

`tests/test_decision_engine.py` (new test or extend existing):

- Test 1: gate off (default) — `compute_position_size` produces byte-identical output to the current code on a fixed set of inputs. Locks the no-op-by-default contract.
- Test 2: gate on — with `ensemble_multiplier=0.85` and `position_size_modifier=0.7225` (0.85 already folded in), `adjusted_dollars` matches the single-application math (`0.7225` factor, not `0.614`).
- Test 3: v2 path (`expert_signals is None`) — gate has no effect; v2 behavior preserved.

### 5. Walk-forward 2×2

Reuse the harness from the 2026-04-30 audit (whichever sweep script computed the recalibration result). Variants:

- `current_production` — recal off, double-fix off (control)
- `recal_on_only` — recal on, double-fix off (matches the audit's recommended ship)
- `double_fix_on_only` — recal off, double-fix on (this packet's fix in isolation)
- `both_on` — recal on, double-fix on (combined ship)

For each: broad-gate window (2026-02-12 → 2026-04-29), rally subset (2026-04-15 → 2026-04-29), panic subset (late-March 2026). Report return % and max drawdown %.

Output: `docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md`. Include the same three representative-day position-size walk-throughs from the prior audit's Phase 3, extended to show the four variants side-by-side.

### 6. Recommendation

The recommendation is the variant that strictly dominates on broad gate AND rally AND panic. If `both_on` strictly dominates, it is the recommended promotion. If the result is mixed (e.g., `double_fix_on_only` improves rally but worsens panic), surface the tradeoff explicitly; do not weaken the gate to declare a winner.

### 7. Postmortem

`docs/POSTMORTEMS.md` — one new entry: **known-bug-documented-as-feature class**. The comment at `decision_engine.py:442-444` documented the double-application as intentional backward-compat behavior, and that documentation is what kept the bug alive through the prior audit's Phase 1 sweep. Prevention: any code comment containing "backward compat" or "kept for legacy" is a follow-up flag, not a closure; comments justifying behavior on legacy grounds get reviewed for "is the legacy path still live?" each audit cycle.

### 8. Update `docs/PLAN.md`

Phase 11 entry summarizing the fix and its walk-forward result.

### 9. Stop

Write `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` ending with the literal final line in §Stop condition.

## Constraints

- **Read-only on `frontend/src/`** — operator WIP.
- **Read-only on prior audit docs** — preserve as historical record.
- **No live deploy or live-param mutation.**
- **Default off.** The fix ships gated; operator promotes via config.
- **One fix = one branch = one commit = one test.** Do not bundle with the recalibration branch; the recalibration is already on its own branch.
- **No silent fixes during the audit.** If a related bug surfaces (e.g., a similar double-application pattern elsewhere in the chain), log it as an addendum finding; do not absorb it into this fix.
- **Read code, not memory.** `file:line` for every claim about behavior.
- **AWS cost rule** (per `CLAUDE.md`): no S3 versioning; no new persistent infra.

## Required return

1. Branch `ai/fix-ensemble-double-apply-20260430` with one focused commit, one passing test, one paragraph of rationale in the commit message.
2. `docs/plans/2026-04-30-ensemble-double-fix-walk-forward.md` — 2×2 walk-forward table; three representative-day walk-throughs; recommendation row.
3. `config/decision_params.recalibrated_2026_04_30.json` extended (or sibling bundle) with the `ensemble_multiplier_already_applied: true` flag set so operator can promote in one config swap.
4. `docs/POSTMORTEMS.md` entry: known-bug-documented-as-feature class.
5. `docs/PLAN.md` Phase 11 entry.
6. `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md` — synthesis: what was wrong, what was fixed, what the 2×2 walk-forward shows, what the operator promotes via config, ending with the literal final line in §Stop condition.

## Hard rules during execution

- Read code before claiming behavior.
- One fix = one branch = one commit = one test.
- Default off. Operator promotes.
- "Backward compat" in a comment is a flag, not a closure.
- If the walk-forward shows the fix worsens any window, surface it — do not weaken the result to ship.
