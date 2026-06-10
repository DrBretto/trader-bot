# PKT-TB-004 — Control-Stack Attribution (committee)

## Context

Every position size routes through a multiplicative control stack (vol adjust, regime
adjust, LLM confidence, ensemble disagreement, expert modifier, risk throttle) plus
sell triggers, steered by a regime label whose published confidence was 0.0426 on the
2026-06-10 night run. No layer has ever been individually attributed. This packet
produces a per-mechanism P&L attribution by counterfactual replay, plus E2 verdicts on
three named candidates (exposure-targeting trim, regime confidence calibration, VIXY
hold policy). Packet: `committee/packets/PKT-TB-004-CONTROL-STACK-ATTRIBUTION-V1-20260609.md`.

## Substrate established at orientation (2026-06-09)

- **Two replay harnesses.**
  - `optimizer/replay.py::run_replay_for_dates` — full history available
    **2025-08-04 → present (~235 trading days)**, starts from $100k cash, seeded
    slippage RNG (`random_seed`), transaction-cost model from
    `src/utils/transaction_costs.py` (per-sector half-spread bps + uniform ±2bps
    slippage), regime fusion recomputed from signal rows, ensemble overrides
    supported. **`llm_risks` hardcoded `{}`** — the LLM layer is structurally OFF in
    this harness today.
  - `src/utils/three_line_replay/replay_engine.py::run_variant` — canon-line
    continuation seeded from the real 2026-03-11 book, holdout segment only,
    consumes stored `daily/<d>/llm_risk.json`, supports champion strategy overlays
    (`extend_relax_choppy(0.50)` + `topup_psm(1.1)`), **no transaction-cost model**
    in `_execute_intents`.
- **Holdout boundary**: `holdout_start: 2026-03-11`
  (`config/optimizer.committee_20260606.json`). The probe precedent
  (`runs/sizing_lead_20260606/sizing_probe.py`) loads the full dataset with
  `cfg.holdout_start=""` and splits dates itself.
- **LLM artifacts** (`llm_risk.json`, `decisions.json`) exist from **2026-01-29**
  onward; before that the LLM step did not run.
- **`regime_confidence` is `1.0 − ensemble_disagreement`** (cosine-based) under
  no-override fusion (`src/signals/regime_fusion.py:89`), NOT a 5-class probability —
  0.0426 means the GRU and Transformer probability vectors were near-orthogonal.
  (Under hard overrides it becomes `max(panic_prob, vol_uncertainty_score)`.)
- **Existing toggle surface** (no code needed): `position_size.vol_adj.*`,
  `position_size.regime_adj.*`, `position_size.throttle_scale`,
  `ensemble_overrides.multiplier.{clip_min,clip_max}`, sell-trigger params
  (`trailing_stop_*`, `sell_health_threshold`, `reduce_health_drop`,
  `leveraged_constraints.max_hold_days`), `min_cash_reserve_by_regime`,
  `buy_score_threshold_by_regime`. **Code flags needed**: neutralize ensemble
  multiplier inside `decide_regime_v3` (folds into psm at step 6), force expert
  modifier to 1.0, wire stored `llm_risk.json` into the optimizer harness.
- Active params: `beat-champion-participation-2026-06-06-v1` (ranking_blend 0.35,
  max_position_weight 0.30, regime-conditional buy thresholds).
- Branch base: `60b128c` on `ai/dead-broker-purge` (PKT-TB-001, 30 commits unmerged
  to main) — branched `ai/control-stack-attribution` from the current working line.

## Plan

- [ ] Declare panel; dispatch Causal Attribution Methodologist to prescribe the
      ablation battery (cells, seeds-per-cell, brackets, interaction rules,
      holdout-look budget, pre-registration).
- [ ] Implement replay flags (src/ + tests/) for the toggles that need code.
- [ ] Run the battery; write ATTRIBUTION_MATRIX.md + per-cell JSON manifests.
- [ ] LLM Veto Auditor: score every historical veto/downsize vs subsequent outcomes.
- [ ] Model Calibration Specialist: confidence history diagnosis + calibration
      candidates (+ downstream counterfactual replay if supported).
- [ ] Risk Architect: exposure-targeting trim + VIXY policy candidates behind flags,
      run to E2.
- [ ] Analyst attribution reads; Skeptic validity attack (what ablations CANNOT say).
- [ ] CANDIDATE_VERDICTS.md (cost-model-sensitivity flags), cut list, E2-only
      implementations.
- [ ] COMMITTEE_REPORT.md (PKT-TB-002 field set + required final line); commits.

## Execution Log

- 2026-06-09: Orientation complete (packet, PACKET_EXECUTION, COMMITTEE_FORMAT,
  EVIDENCE_PROTOCOL, ADVERSARIAL_REINTERPRETATION_METHOD, EVIDENCE_LADDER,
  decision_engine.py, regime_fusion.py, both replay harnesses, data_access,
  active params, S3 inventory). Branch + run dir created.

## Follow-ups

- (none yet)
