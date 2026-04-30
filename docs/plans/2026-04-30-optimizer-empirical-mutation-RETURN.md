# RETURN — Optimizer empirical-mutation fix (2026-04-30)

Packet: `docs/plans/2026-04-30-optimizer-empirical-mutation-packet.md`
Branch: `ai/optimizer-empirical-mutation-20260430` (four focused commits, one per scope item)
Authority: execution / bounded.

## TL;DR

The weekly optimizer was running for months without finding empirically-correct calibration. Four-part fix shipped behind a feature flag (`OptimizerConfig.enable_empirical_mutation`, defaulted off): inventory bounds tightened to empirical p1/p99 with `parameter_class` tagging, empirical-re-derivation candidate operator added, calibration-only guardrail path added, persistent-rejection alert wired through the existing `[TraderBot]` SNS topic. Verification gate dry-run **PASSED**: proposed `fragility.AVG_CORR_MEAN = 0.4847` (target 0.477 ± 0.02).

The active fragility constants in `decision_params.recalibrated_2026_04_30.json` are unchanged. The GA's selection / crossover operators are unchanged. Non-normalization constants are unaffected. AWS cost stays at $0.

## What changed

| # | Scope item | Files | Commit | Test count |
|---|------------|-------|--------|-----------:|
| 1 | Inventory bounds + `parameter_class` tagging | `optimizer/discovery/param_inventory.json` | `1715fc1` | n/a |
| 2 | Empirical re-derivation candidate operator | `optimizer/param_space.py`, `optimizer/champion_challenger.py`, `optimizer/cli.py`, `optimizer/config.py` | `472c21c` | 17 |
| 3 | Calibration-only guardrail path | `optimizer/guardrails.py`, `optimizer/champion_challenger.py` | `672e999` | 3 (+ 17 from Phase 2 share `is_calibration_only_diff`) |
| 4 | Persistent-rejection alert | `optimizer/persistence.py`, `optimizer/champion_challenger.py`, `optimizer/alerts.py` | `a1c6e39` | 8 |

All 31 new + existing optimizer tests pass. No regressions in 39 signal tests or 10 decision-engine tests.

## What's behind the feature flag

A single config flag — `OptimizerConfig.enable_empirical_mutation` (yaml key `enable_empirical_mutation`) — gates the empirical-re-derivation candidate operator. Defaulted False. When True, the operator runs once per cycle and injects one extra candidate per generation; when False, the optimizer runs identically to its pre-2026-04-30 behavior except for the corrected inventory bounds.

The dry-run flag `optimizer/cli.py run --empirical-mutation-debug` always computes the candidate (independent of the runtime flag), so the operator can verify the verification gate before flipping the runtime flag on.

The other three pieces ship always:
- **Inventory bounds + tagging** — pure metadata correction. The legacy `inferred_from_usage, needs_bounds_confirmation: true` markers were a self-aware TODO that had been ignored for months. New bounds are documented at `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md`. Random mutation now samples within empirical-realistic bounds, which is an improvement regardless of the empirical-mutation operator.
- **Calibration-only guardrail path** — only fires when the challenger's diff is *exclusively* in `parameter_class: "normalization_constant"` parameters. With no challengers of that class proposed (i.e., `enable_empirical_mutation: false`), the path is unreachable.
- **Persistent-rejection alert** — ships always. Fires only when the optimizer rejects 3+ consecutive challengers, which under default-off behavior happens iff the GA is failing for non-calibration reasons. The alert is itself observability, not a behavior change.

The packet's "default off" framing is honored: in the no-promote default state, the optimizer's behavior is materially the same (with corrected bounds and alerting). Only flipping the flag on changes the candidate-injection behavior.

## Verification gate result

Run command:
```
AWS_PROFILE=personal .venv/bin/python -m optimizer.cli run --config config/optimizer.yaml --empirical-mutation-debug
```

Output (excerpted):
```json
{
  "mode": "empirical_mutation_debug",
  "dataset_size": 177,
  "date_range": "2025-08-04 to 2026-04-29",
  "feature_flag": {
    "enable_empirical_mutation": false,
    "note": "Flag is config-controlled. This dry-run computes the empirical candidate regardless of the flag, so the operator can verify the verification gate before flipping the flag on."
  },
  "normalization_constants_with_empirical_statistic": [
    {
      "name": "fragility.AVG_CORR_MEAN",
      "current_value": 0.3881,
      "empirical_statistic": {"fn": "mean", "field": "avg_correlation", "exclude_zero": true},
      "proposed_value": 0.4846527238905304,
      "within_bounds": true,
      "allowed_range": {"min": 0.3881, "max": 0.5999}
    },
    {
      "name": "fragility.PC1_MEAN",
      "current_value": 0.5206,
      "proposed_value": 0.5929214216430337,
      "within_bounds": true,
      "allowed_range": {"min": 0.5206, "max": 0.687}
    },
    {
      "name": "macro_credit.SLOPE_MEAN",
      "current_value": 0.6655,
      "proposed_value": 0.6043749999999998,
      "within_bounds": true,
      "allowed_range": {"min": 0.55, "max": 0.6655}
    }
    // ... 5 more constants with empirical_statistic ...
    // ... 12 vol_uncertainty bins with empirical_statistic: null ...
  ],
  "verification_gate": {
    "target_param": "fragility.AVG_CORR_MEAN",
    "target_value": 0.477,
    "tolerance": 0.02,
    "proposed_value": 0.4846527238905304,
    "within_tolerance": true
  }
}
```

**Verification gate PASSED.** `proposed_value = 0.4847`, target = 0.477 ± 0.02 → within tolerance.

A small note on `current_value: 0.3881` for AVG_CORR_MEAN (not the production `0.30`): the new bounds [0.3881, 0.5999] clamp the legacy production value at the lower bound when loaded into the genome representation. That clamp is on the optimizer's internal model only — the active production bundle (`config/decision_params.active.json`) still has `signals.fragility = {}` (using the code default `0.30` from `src/signals/fragility.py:19`). The optimizer's champion is therefore evaluated at the clamped 0.3881; the empirical candidate proposes 0.4847; a promotion would write 0.4847 into the active bundle. Either path leads to a recalibrated production state; the recalibration audit's existing `decision_params.recalibrated_2026_04_30.json` shadow bundle remains the cleaner manual-promotion path.

## Sample persistent-rejection alert body

Generated via `tests/test_optimizer_rejection_streak.py::TestAlertBody::test_subject_carries_traderbot_prefix` and reproduced here in cleartext:

```
Subject: [TraderBot] Optimizer rejected 3 consecutive cycles (threshold 3)

OPTIMIZER PERSISTENT-REJECTION ALERT

The weekly optimizer has rejected all challengers for 3 consecutive cycles
(alert threshold = 3).

Recent rejected cycles:
    - run-3 (2026-04-30T03:42:00Z) decision=not_promoted
    - run-2 (2026-04-23T03:32:00Z) decision=not_promoted
    - run-1 (2026-04-16T03:35:00Z) decision=not_promoted

Most recent cycle:
  run_id: run-3
  champion_version_before: hybrid-ranking-035-v1
  challenger_version: opt-20260430T...
  reason_codes: ['runtime_error']
  Challenger wf_objective: 0.123
  Challenger guardrails passed: False
  Calibration-only path used: False
  Failed guardrail checks:
    - min_round_trips_total: value=0.0 threshold=20.0

Suggested operator action:
  If calibration-class drift is suspected (i.e., normalization
  constants are stale relative to current empirical distribution),
  re-run with --empirical-mutation-debug to inspect what the
  empirical-re-derivation candidate is proposing:

    .venv/bin/python -m optimizer.cli run --empirical-mutation-debug

  Output shows the proposed values for every normalization
  constant and whether the verification gate
  (AVG_CORR_MEAN ~ 0.477 +/- 0.02) is satisfied.

  If the debug output looks right but the GA never proposes the
  empirical candidate, check that
  optimizer.yaml -> enable_empirical_mutation: true.

  If the debug output is wildly off, check the
  parameter_inventory.json `empirical_statistic` field per
  constant and the underlying signal_row data quality.
```

The subject prefix `[TraderBot]` matches the daily-pipeline alerts the operator already filters on, so existing email rules route it without configuration.

## What stays the same

- `decision_params.active.json` and `decision_params.recalibrated_2026_04_30.json` — both unchanged. The recalibration audit's recommended ship path is unaltered by this packet.
- `evolution/genetic.py` — the GA's selection, crossover, and mutation operators are unchanged. The empirical-re-derivation operator lives in `optimizer/param_space.py` (the helper) and `optimizer/champion_challenger.py` (the integration point).
- Non-normalization constants — `entropy_shift.z_threshold`, `vol_uncertainty.regime_thresholds.*`, the various `*.window_days` / `*.MIN_DAYS` / `*.MIN_SYMBOLS` integers — all stay tagged `decision_threshold` with their existing bounds and mutation logic.
- Existing 233 non-normalization inventory entries — unchanged in semantics; just default-tagged with `parameter_class: "decision_threshold"`.
- Active SNS topic — the existing `investment-system-alerts` topic is reused. No new topic, no new subscription, no new IAM, no new cost line.

## Operator runbook

After merging this branch and the verification gate has been reviewed:

1. **Inspect the dry-run output** (already done as part of this RETURN; re-run if the dataset has refreshed):
   ```
   AWS_PROFILE=personal .venv/bin/python -m optimizer.cli run --config config/optimizer.yaml --empirical-mutation-debug
   ```
   Confirm `verification_gate.within_tolerance: true`.

2. **Flip the runtime flag** in `config/optimizer.yaml`:
   ```yaml
   enable_empirical_mutation: true
   ```
   This is a one-line edit. The next weekly cycle (Sunday 03:30) will then inject the empirical candidate.

3. **Monitor the first run after promotion** at `runs/optimizer/<run_id>/run.log`. Look for the line:
   ```
   Empirical-mutation candidate proposed for 8 normalization constants: fragility.AVG_CORR_MEAN=0.4847, ...
   ```
   If the line is absent, the runtime flag is off; if it's present but the candidate isn't promoted, check `runs/optimizer/<run_id>/promotion_decision.json` for `reason_codes`.

4. **Watch for a calibration-only promotion in the lineage**: the next promoted version in `runs/optimizer/active_params_lineage.json` should differ from the champion only in `parameter_class: normalization_constant` parameters. If it does, the path worked end-to-end.

5. **If the persistent-rejection alert fires later**, follow the runbook in the alert body (which references the dry-run command in step 1).

## Findings carried out of scope

- **Discovery is not re-runnable today.** `optimizer/discovery/param_inventory.json` was generated once and treated as authoritative metadata. Adding a quarterly re-discovery + diff workflow is logged for follow-up; this packet only updates the entries in scope (fragility / macro / vol).
- **Raw VIX/VVIX/SKEW values are not in `signal_row`.** The 12 vol_uncertainty percentile-bin constants get tighter bounds but no empirical-mutation candidate. To upgrade them, plumb raw values through `src/steps/publish_artifacts.py:_build_timeseries_row` into `signal_row`. Out of this packet's scope.
- **Other normalization-class candidates outside fragility / macro / vol.** Possibly worth tagging in the future: `entropy_shift.z_threshold` (could be derived from the empirical entropy_z_score distribution). Logged for next packet.
- **The `current_value` clamp at genome load.** The new tight bounds clamp the legacy production value (e.g., `AVG_CORR_MEAN=0.30` → `0.3881` at the genome boundary). The optimizer's internal model of the champion is therefore slightly different from the production constant. This is conservative-correct (sets a higher bar for promotion) but logged as a documentation surface.

## Files written

- `optimizer/discovery/param_inventory.json` (Phase 1 — bounds + tagging)
- `optimizer/param_space.py` (Phase 2 — `parse_empirical_statistic`, `compute_empirical_statistic`, `empirical_genome`, `is_calibration_only_diff`)
- `optimizer/config.py` (Phase 2 + Phase 4 — `enable_empirical_mutation`, `rejection_streak_alert_threshold`)
- `optimizer/champion_challenger.py` (Phase 2 + Phase 3 + Phase 4 wiring)
- `optimizer/cli.py` (Phase 2 — `--empirical-mutation-debug` flag)
- `optimizer/guardrails.py` (Phase 3 — `calibration_only` parameter, `CALIBRATION_GUARDRAIL_DROPS`)
- `optimizer/persistence.py` (Phase 4 — `update_rejection_streak`)
- `optimizer/alerts.py` (Phase 4 — `build_persistent_rejection_alert_body`, `send_optimizer_alert`)
- `tests/test_optimizer_empirical_mutation.py` (17 tests for Phase 2)
- `tests/test_optimizer_guardrails.py` (3 new tests for Phase 3)
- `tests/test_optimizer_rejection_streak.py` (8 tests for Phase 4)
- `docs/plans/2026-04-30-optimizer-inventory-bounds-derivation.md` (Phase 1 derivation table)
- `docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md` (this file)
- `docs/POSTMORTEMS.md` — one new entry: **stale-bounds-block-empirical-convergence class**
- `docs/PLAN.md` — Phase 12 entry

---

TRADER-BOT OPTIMIZER EMPIRICAL-MUTATION COMPLETE — AWAITING OPERATOR FIRST-RUN VERIFICATION
