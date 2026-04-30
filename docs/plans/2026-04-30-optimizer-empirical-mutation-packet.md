# PKT-TRADER-BOT-OPTIMIZER-EMPIRICAL-MUTATION-20260430

Packet Owner:
- Claude (fresh executor session, instantiated in `/Users/drbretto/Desktop/Projects/trader-bot/`)

Date:
- 2026-04-30

Authority surface:
- `CLAUDE.md` (project workflow, plan-doc discipline)
- `docs/PLAN.md` (current roadmap)
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` (the audit whose existence is the meta-finding this packet addresses)
- `optimizer/contracts/live_params_contract.md` (existing optimizer contract surface)

Authority level:
- execution

Discovery mode:
- bounded (the four-part scope is fixed; executor verifies and implements but does not expand the surface; new findings get logged for operator review, not silently absorbed)

## What this packet exists to fix

The weekly optimizer (launchd Sunday 03:30, `optimizer/cli.py run`) covers fragility / macro / vol normalization constants in scope but cannot find empirical-distribution-based improvements. The 2026-04-30 fragility audit discovered that `AVG_CORR_MEAN=0.30` was at the empirical 5th percentile of 900-day data (true mean ≈ 0.4774) and `PC1_MEAN=0.45` was below the empirical minimum — and the optimizer had been running weekly for months without surfacing this. **The audit was needed because this loop wasn't doing its job.**

Three distinct issues compound:

1. **Bounds are scale-of-value, not scale-of-input.** `optimizer/discovery/param_inventory.json:898–899` shows `fragility.AVG_CORR_MEAN` with `min: -5.0, max: 5.0` (`source: inferred_from_usage`, `needs_bounds_confirmation: true`). The actual statistic is a correlation coefficient with 900-day empirical range ≈ 0.23–0.69. A search space that is two orders of magnitude wider than the meaningful range cannot converge on the right value via random mutation in finite generations. Same pattern affects `AVG_CORR_STD`, `PC1_MEAN`, `PC1_STD`, and likely other normalization constants flagged with `needs_bounds_confirmation: true`.

2. **No empirical-statistic seeding.** The genetic operator mutates randomly within bounds. For a normalization constant whose "right" value is the empirical mean of a live data stream, the GA has no mechanism to *compute* that value and inject it as a candidate. It can only stumble toward it.

3. **Persistent rejection is silent.** Recent runs have rejected 100% of challengers. Without operator-visible escalation, "the optimizer rejected everything again" looks identical to "the optimizer is healthy and nothing needed changing." The failure mode is invisible until a separate audit surfaces it.

## Operator hypothesis (bounded)

The four-part fix below restores the optimizer's ability to find calibration-class drift on its own. The verification gate is sharp: the first weekly run after the fix must propose `AVG_CORR_MEAN ≈ 0.477 ± 0.02` as a candidate (the empirically re-derived value the 2026-04-30 audit landed on). If it does not, the empirical-mutation operator is broken — investigate before promoting.

## Scope

Four-part fix to `optimizer/`, gated behind a feature flag, defaulted off until the first run after the fix validates the verification gate. Specifically out of scope: changing the active fragility constants (already shipped via `decision_params.recalibrated_2026_04_30.json`); changing the GA's selection or crossover operators; expanding empirical mutation to non-normalization constants.

### Part 1 — `parameter_class` field on the inventory

Extend the schema of `optimizer/discovery/param_inventory.json` entries with an optional `parameter_class` field. Two values defined for this packet:

- `parameter_class: "normalization_constant"` — the value is meant to represent the empirical mean / std / quantile of an input distribution. Bounds for these derive from the input distribution (e.g., empirical p1 / p99 of `avg_correlation` over the configured backtest window), not from value-scale heuristics. Mutation respects those bounds.
- `parameter_class: "decision_threshold"` (default for everything else) — current behavior preserved; existing bounds and mutation logic apply.

Mark the fragility / macro / vol normalization constants as `normalization_constant`. Specifically, the executor enumerates these by reading every `where_defined` source file and tagging any constant that is used as a normalization mean / std / percentile-baseline. The set is bounded — likely under 20 constants total — and the executor lists each one with code citation in the Phase 1 doc before edits.

### Part 2 — Empirical re-derivation candidate operator

In `evolution/genetic.py` (or `optimizer/champion_challenger.py` — executor picks based on which is the cleaner integration point; document the choice), add a new candidate-generation operator: **empirical re-derivation candidate**.

Once per generation, for each parameter tagged `parameter_class: normalization_constant`:

1. Compute the live-data statistic the constant represents. For `AVG_CORR_MEAN`: the mean of `avg_correlation` over the optimizer's configured backtest window. For `PC1_STD`: the std of `pc1_explained` over the same window. The mapping from constant name to "what statistic does this represent" lives in the inventory entry's `notes` field or a new `empirical_statistic` field — executor's call.
2. Inject that value as one candidate genome for the generation. The candidate is opt-in per-constant via the `parameter_class` tag; constants without the tag are unaffected.
3. The candidate competes with random-mutation candidates under the existing fitness + guardrail pipeline. No special path; if the empirical candidate doesn't pass guardrails, it loses like any other.

This is *seeding*, not *replacement*. The GA still mutates and crosses; the empirical candidate is one extra genome per generation, not a substitute for evolution.

### Part 3 — Calibration-only delta detector + per-class guardrail set

Calibration-only changes (parameters tagged `normalization_constant`, with no other parameter changed) shouldn't need to clear `min_round_trips_total=20` (`optimizer/guardrails.py:63`) because they don't change *whether* the bot trades, only *how much*. Add a "calibration-only delta" detector in `optimizer/guardrails.py`:

1. If a challenger genome differs from the champion *only* in `normalization_constant`-tagged parameters, classify it as a calibration-only delta.
2. For calibration-only deltas, swap the standard guardrail config for a separate `calibration_guardrail_config` — likely keeping `max_drawdown_cap`, `cost_ratio_cap`, `min_fold_ann_return` but dropping `min_round_trips_total` and `min_gate_round_trips` (since the recalibration changes position size, not trade frequency). Final list is the executor's call based on which guardrails are *behavior-conditioning* vs. *outcome-conditioning*; document the rationale.
3. Calibration-class guardrails are stricter on outcome-quality (no widening of `max_drawdown_cap`) and looser only on trade-count gates that don't apply.

### Part 4 — Persistent-rejection alert

When `optimizer/cli.py run` completes a weekly cycle and rejects all challengers, increment a counter persisted in `optimizer/persistence.py`'s state. If three consecutive weekly runs reject all challengers, fire an SNS notification (or whichever notification channel the project already uses for ops alerts — executor reads `infrastructure/` and `automation/` to find the canonical path before adding a new one).

Notification body must include:
- Run timestamps for the three rejected cycles.
- Top rejected challenger from the most recent cycle and its guardrail-failure breakdown.
- Suggested operator action: "if calibration-class drift is suspected, re-run with `--empirical-mutation-debug` to see whether the empirical-re-derivation candidates are being proposed and rejected, or whether they are not being proposed at all."

The 3-in-a-row threshold is the trigger; the counter resets when a cycle promotes anything.

## Stop condition

Four-part fix is implemented behind a feature flag (defaulted off), tests pass, the next weekly run (or a manual dry-run trigger) is verified to propose an empirical-re-derivation candidate for `AVG_CORR_MEAN` within the §Verification gate tolerance, and `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` ends with the literal final line:

```
TRADER-BOT OPTIMIZER EMPIRICAL-MUTATION COMPLETE — AWAITING OPERATOR FIRST-RUN VERIFICATION
```

## Verification gate (load-bearing)

The first weekly run (or dry-run via `optimizer/cli.py run --empirical-mutation-debug`, the executor adds the flag) after the fix must propose `fragility.AVG_CORR_MEAN ≈ 0.477 ± 0.02` as a candidate genome. If it does not:

- The empirical-mutation operator is broken. **Investigate before promoting.** Do not flip the feature flag on.
- Likely failure modes: the live-data statistic is computed against the wrong window (mismatch with the audit's 900-day reference), the constant-to-statistic mapping is wrong (inventory entry's `empirical_statistic` field misnames the statistic), or the candidate is being generated but rejected by a bounds check before it reaches the fitness evaluator.

The `±0.02` tolerance accommodates: data refreshes since the audit, minor windowing differences, the executor's choice of statistic estimator (mean vs. winsorized mean vs. median).

## Acceptance test

The operator opens the RETURN doc and within 20 minutes can answer:

1. **What changed in the optimizer?** A four-line summary: bounds source switched for normalization constants, empirical-re-derivation operator added, calibration-only guardrail path added, persistent-rejection alert wired.
2. **What's behind the feature flag?** All four pieces gate together via one flag (`OPTIMIZER_EMPIRICAL_MUTATION_ENABLED` or equivalent), defaulted off. The verification gate must pass before the operator flips it on.
3. **Did the verification gate pass?** Either yes (with the proposed `AVG_CORR_MEAN` value cited and the dry-run output attached) or no (with the diagnosis of why the empirical-mutation operator didn't propose the expected candidate).
4. **What stays the same?** The active fragility constants in `decision_params.recalibrated_2026_04_30.json` are unchanged; the GA's selection / crossover operators are unchanged; non-normalization constants are unaffected.
5. **What does the persistent-rejection alert look like in practice?** A sample notification body, generated from a synthetic 3-run-rejection scenario, included in the RETURN doc.

## Reference outputs

- **Negative reference (do NOT produce)**: a fix that auto-overrides the constants to the empirical statistic. The empirical value is a *candidate*, not a *replacement*. The GA's fitness + guardrail pipeline decides whether it wins.
- **Negative reference (do NOT produce)**: a fix that broadens guardrails for *all* challengers because calibration-only deltas need looser rules. The looser path applies *only* to calibration-only deltas; mixed deltas stay on the strict path.
- **Negative reference (do NOT produce)**: a SNS notification that fires every week. The 3-in-a-row threshold is load-bearing — it filters healthy "no change needed" cycles from genuine "the optimizer is rejecting everything" failures.
- **Negative reference (do NOT produce)**: a bundled commit. Each of the four parts ships as its own commit (one branch is fine if the parts compose, but the commit history must let the operator revert any of the four independently).
- **Positive reference shape — inventory entry update**: every parameter tagged `normalization_constant` carries (a) the new tag, (b) updated `allowed_range.min` / `max` derived from empirical p1 / p99 with `source: empirical_distribution_<window>`, (c) an `empirical_statistic` field naming the statistic (`mean(avg_correlation)`, `std(pc1_explained)`, etc.), and (d) the `notes` field documenting which audit or analysis informed the bounds.
- **Positive reference shape — RETURN doc**: a four-section doc, one per scope item, plus a "Verification gate result" section with the dry-run output literal-quoted.

## Write surface

- `optimizer/discovery/param_inventory.json` (extend schema; tag normalization constants; update bounds for tagged entries)
- `optimizer/champion_challenger.py` and/or `evolution/genetic.py` (new empirical-re-derivation operator; executor picks integration point)
- `optimizer/guardrails.py` (calibration-only delta detector + alternative guardrail config)
- `optimizer/cli.py` (dry-run flag if needed for verification gate; persistent-rejection counter wiring)
- `optimizer/persistence.py` (rejection-counter state)
- `infrastructure/` or `automation/` (SNS or equivalent notification path — executor reads existing surfaces before adding new ones)
- `tests/test_optimizer_*.py` (new tests: empirical-mutation operator proposes the expected candidate; calibration-only detector classifies correctly; persistent-rejection alert fires on the 3rd run not the 1st or 2nd)
- `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md` (Phase 1 output: the list of normalization constants with their empirically-derived bounds and statistic mappings)
- `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` (final stop-condition file)
- `docs/POSTMORTEMS.md` (one new entry: stale-bounds-block-empirical-convergence class — the failure mode where a search space wider than the meaningful range prevents convergence on the empirical statistic)
- `docs/PLAN.md` (Phase 12 entry summarizing the optimizer fix)

**Specifically PROHIBITED**:
- Any change to active fragility constants in `src/signals/fragility.py` or in `decision_params.recalibrated_2026_04_30.json` — they are already shipped on the recalibration branch.
- Any change to the GA's selection or crossover operators (out of scope).
- Any change to `frontend/src/` (operator WIP).
- Any rollback of the 2026-04-29, 2026-04-30 audit, or ensemble-double-fix work.
- Any new persistent infra. SNS notification piggybacks on whatever ops-alert path already exists.

## Read-first list

- `CLAUDE.md`
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` — the meta-finding this packet addresses.
- `optimizer/cli.py` — the weekly entry point.
- `optimizer/champion_challenger.py` — the candidate-generation pipeline.
- `optimizer/guardrails.py` (lines 50–92 specifically) — the guardrail check stack the calibration-only path will branch from.
- `optimizer/discovery/param_inventory.json` (lines 890–1030 specifically) — the fragility-constant entries and the surrounding schema. Confirm the operator's claim about `min: -5.0, max: 5.0` bounds at lines 898–899.
- `evolution/genetic.py` — the mutation / crossover surface.
- `optimizer/persistence.py` — where rejection-counter state lives.
- `optimizer/contracts/live_params_contract.md` — the existing contract surface for what the optimizer is allowed to touch.
- `infrastructure/` and `automation/` — find the canonical notification path before adding a new one.
- launchd plist for the Sunday 03:30 cron — confirm the cadence assumption.

## Flow position

This packet is the **optimizer-level fix** that prevents future stale-historical-norm incidents from requiring an operator-driven audit. The 2026-04-30 fragility audit was the symptom; the optimizer's blindness to empirical-distribution drift is the disease. The fix ships gated; the verification gate determines whether the operator promotes it. After promotion, the weekly optimizer is expected to find calibration drift on its own and surface it through champion-challenger promotion or the persistent-rejection alert.

## Required method

### 1. Read the Read-first list

Specifically confirm the `min: -5.0, max: 5.0` bounds at `param_inventory.json:898–899` and the `min_round_trips_total` guardrail at `guardrails.py:63`. The packet's claims rest on these.

### 2. Phase 1 — Bounds derivation + inventory tagging

Walk the inventory, identify every parameter that is functionally a normalization constant (`AVG_CORR_MEAN`, `AVG_CORR_STD`, `PC1_MEAN`, `PC1_STD` from fragility; analogous constants in macro / vol expert signals if present). For each:

- Read the `where_defined` source file. Confirm the constant is used as a normalization mean / std / percentile-baseline.
- Compute the empirical p1 / p99 of the underlying input statistic over the optimizer's configured backtest window (likely the same 900-day cache the prior audits used).
- Tag with `parameter_class: normalization_constant` and update `allowed_range.min` / `max` accordingly. Add the `empirical_statistic` field naming the statistic.

Output: `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md` — the list with code citations and bounds.

**Exit criterion**: every normalization constant has a tag, derived bounds, and a named statistic. The list is exhaustive within fragility / macro / vol expert signals; constants outside that scope are noted but untagged.

### 3. Phase 2 — Empirical re-derivation operator

Implement the operator per Part 2 of §Scope. The integration point is the executor's call between `optimizer/champion_challenger.py` and `evolution/genetic.py`; document the choice with one paragraph in the commit message.

Test: on a fixed input dataset, the operator proposes a candidate genome with `fragility.AVG_CORR_MEAN` set to the empirical mean of `avg_correlation` over the test window. Lock the test data so future drift is detectable.

### 4. Phase 3 — Calibration-only guardrail path

Implement Part 3 of §Scope. Detector classifies challengers as calibration-only or mixed; calibration-only path uses an alternative guardrail config; mixed path uses the standard config unchanged.

Test: a challenger differing only in `AVG_CORR_MEAN` is classified calibration-only and clears guardrails with a trade-count below `min_round_trips_total=20`. A challenger differing in any non-normalization-constant parameter is classified mixed and stays on the strict path.

### 5. Phase 4 — Persistent-rejection alert

Implement Part 4 of §Scope. Counter persists across runs; resets on any promotion; fires on 3rd consecutive rejection.

Test: synthetic three-run scenario triggers the notification on run 3, not on runs 1 or 2; a promotion in run 2 prevents the run-3 alert.

### 6. Verification gate

Run a dry-run cycle (`optimizer/cli.py run --empirical-mutation-debug` or whatever flag the executor adds). Confirm the dry-run proposes `fragility.AVG_CORR_MEAN ≈ 0.477 ± 0.02` as a candidate genome. Quote the dry-run output in the RETURN doc.

If the gate fails: do not flip the feature flag on; document the diagnosis and stop.

### 7. Postmortem

`docs/POSTMORTEMS.md` — one new entry: **stale-bounds-block-empirical-convergence class**. Document the failure mode (search space orders of magnitude wider than meaningful range; random mutation cannot converge in finite generations on the empirical statistic) and the prevention (any normalization constant in the optimizer inventory must have bounds derived from its input distribution, not from value-scale heuristics, and any inventory entry with `needs_bounds_confirmation: true` is a follow-up flag rather than a closure).

### 8. Update `docs/PLAN.md`

Phase 12 entry summarizing the optimizer fix and the verification-gate outcome.

### 9. Stop

Write `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` ending with the literal final line in §Stop condition.

## Constraints

- **Default off.** Feature flag gates all four parts. Operator promotes after verifying the gate.
- **Read-only on `frontend/src/`** — operator WIP.
- **Read-only on prior audit docs and the active fragility constants** — preserve as historical record / live shipped state.
- **No live deploy or live-param mutation.**
- **No bundled commits.** Each of the four scope items ships as its own commit (one branch is acceptable).
- **No silent fixes.** If a related bug surfaces (e.g., another inventory entry with mis-scaled bounds outside the named scope), log it as an addendum finding; do not fix in this packet.
- **No new persistent infra.** SNS / notification path piggybacks on existing ops-alert surface.
- **Read code, not memory.** `file:line` for every claim about behavior.
- **AWS cost: $0.** Optimizer runs locally on the MacBook via launchd; this fix adds no cloud compute.

## Required return

1. Branch `ai/optimizer-empirical-mutation-20260430` with four commits (one per scope item), each with one focused unit test.
2. Updated `optimizer/discovery/param_inventory.json` with `parameter_class` tags and empirical bounds for every normalization constant in scope.
3. New empirical-re-derivation operator in `optimizer/champion_challenger.py` or `evolution/genetic.py` (with the choice documented).
4. New calibration-only guardrail path in `optimizer/guardrails.py`.
5. Persistent-rejection alert wired through `optimizer/cli.py` + `optimizer/persistence.py` + the existing notification surface.
6. `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md` — the bounds-derivation table.
7. `docs/POSTMORTEMS.md` entry: stale-bounds-block-empirical-convergence class.
8. `docs/PLAN.md` Phase 12 entry.
9. `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` — synthesis: what changed, what stays the same, verification-gate result, sample notification body, ending with the literal final line in §Stop condition.

## Hard rules during execution

- Read code before claiming behavior. `file:line` for every claim.
- One scope item = one commit = one test.
- Default off. Operator promotes after verification gate.
- Verification gate is load-bearing: `AVG_CORR_MEAN ≈ 0.477 ± 0.02`. If the dry-run misses it, the operator is broken — investigate, do not flip the flag.
- "Backward compat" or "needs_bounds_confirmation: true" or "inferred_from_usage" in any inventory entry is a flag, not a closure.
- No new persistent infra. AWS cost stays at $0.
