# PKT-TRADER-BOT-FRAGILITY-AND-THROTTLE-CALIBRATION-AUDIT-20260430

Packet Owner:
- Claude (fresh executor session, instantiated in `/Users/drbretto/Desktop/Projects/trader-bot/`)

Date:
- 2026-04-30

Authority surface:
- `CLAUDE.md` (project workflow, plan-doc discipline)
- `docs/PLAN.md` (current roadmap; Phase 6 expert-engine spec; Phase 9 prior-audit closeout)
- `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` (prior audit's RETURN; F-1 through F-10 + retracted F-11; explicitly flags F-2 fragility tanh-saturation as still-open)
- `docs/plans/2026-04-29-phase5-calibration-recommendation.md` (the sweep that left F-7 knob off by default)
- `docs/PHASE3_PROPOSAL.md` (origin of fragility / vol_uncertainty signals and historical-norm constants)

Authority level:
- execution

Discovery mode:
- discovery_bearing (executor must surface follow-on calibration defects beyond the named hypotheses; the operator's hypothesis is a starting frame, not a conclusion)

## Operator hypothesis (the question this packet exists to answer)

The prior audit (2026-04-29) found ten findings, fixed four, and left F-2 (fragility tanh-saturation) and F-7 (regime-conditional relax) as a configurable knob defaulted **off** because the calibration sweep showed no setting strictly dominated current production. Six weeks later, `fragility_score` is still pegged ≥ 0.97 every day. The operator's working hypothesis is:

1. The algorithm core (signal generation, regime fusion, decision engine, ranking) is fundamentally sound.
2. The **calibration baselines** the throttles are normalized against (`AVG_CORR_MEAN=0.30`, `AVG_CORR_STD=0.15`, `PC1_MEAN=0.45`, `PC1_STD=0.12` in `src/signals/fragility.py`) are stale relative to the current market regime. A metric saturated near 1.0 daily is not measuring fragility — it is measuring "we are in any market that exists in 2026."
3. The compounding effect across the multiplicative throttle chain (`fragility_position_cap` → `entropy_size_multiplier` → `ensemble_multiplier` → final clamps) is making the bot structurally too timid.
4. The sweep that left F-7 off may have been the right call **for the F-2 metric as written** but wrong for the **system goal**. Choosing between "panic-window protection" and "rally-window participation" is a false dichotomy if the underlying metric is informationally dead.

The audit either confirms this with code-level + data-level evidence, or refutes it with the same.

## Scope

Audit the full position-sizing throttle pipeline end-to-end against 900-day historical signal data, characterize every multiplicative input for stuck-on / saturated / informationally-dead behavior, and produce a calibration recommendation that makes each throttle fire on **deviations from current empirical distribution** rather than on stale historical norms. Recalibrate constants where indicated. Verify the recalibrated system on walk-forward backtest. **Do not cut over to live until the operator approves.**

This packet is **not** a re-do of the 2026-04-29 audit. F-1, F-3, F-4, F-5, F-6, F-8, F-9 are already fixed and should be assumed-correct unless the executor finds a regression. The scope here is specifically: **the fragility metric's calibration baseline + every multiplicative throttle's empirical distribution + the compounding effect on position size.**

## Stop condition

A new calibration recommendation exists, every multiplicative throttle has a documented empirical-distribution health card, the recalibrated constants are merged behind a feature gate (defaulted off until operator green-light), the walk-forward verification shows the recalibrated system meets the gate metrics in §Acceptance test, and `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` ends with the literal final line:

```
TRADER-BOT FRAGILITY + THROTTLE CALIBRATION AUDIT COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
```

## Acceptance test

The operator opens the RETURN doc and within 30 minutes can answer:

1. **Is the fragility metric actually informative right now?** Specifically: what is the empirical distribution of `avg_correlation` and `pc1_explained` over the last 900 days, and at what thresholds do they actually discriminate panic from non-panic? Does the prior calibration baseline (`AVG_CORR_MEAN=0.30`) reflect the current regime, or is it from a pre-2024 dataset?
2. **Which throttles in the chain are actually firing on signal vs. firing on stale normalization?** Each of `fragility_score`, `entropy_shift_flag`, `ensemble_multiplier`, `vol_uncertainty_score`, `regime_confidence` gets a one-paragraph empirical health card: distinct values across 900 days, fraction of days at extremum, fraction of days the throttle's gate fires, sensitivity of position size to a 1σ shift in the input.
3. **What is the compound cost in position size of "always-pegged fragility"?** A worked example showing: with fragility pegged at 0.98 and `fragility_position_cap=0.X`, what fraction of the un-throttled position survives? Is that justified by panic-window protection in the empirical record, or is it dead weight?
4. **What does the recalibrated system do differently?** A side-by-side: same input day, current production sizing vs. recalibrated sizing. The recalibrated system should produce visibly different (typically larger) positions on rally days while preserving the protective behavior on the 2026-02-11 → 2026-03-31 panic window.
5. Walk-forward gate: the recalibrated variant beats current production on **both** the 2026-02-12 → 2026-04-29 broad gate window **and** the rally subset (2026-04-15 → 2026-04-29). If a recalibration only helps one window, that is a finding, not a ship — surface it.

## Reference outputs

- **Negative reference (do NOT produce)**: a sweep that re-runs the same eight regime_fusion variants from 2026-04-29 and concludes "no setting dominates, ship nothing." That was the failure mode of the prior sweep — it varied `fragility_position_cap` and `fragility_threshold` without questioning whether the fragility *input* itself was meaningful. **Question the metric, not just the gate parameters.**
- **Negative reference (do NOT produce)**: a recommendation that simply hardcodes `fragility_relax_in_risk_on=True` and ships. The 2026-04-29 sweep already showed that's not a free win. The right answer is more likely a recalibrated `AVG_CORR_MEAN` / `PC1_MEAN` rooted in the actual empirical distribution, with `AVG_CORR_STD` / `PC1_STD` adjusted to match. The opt-in knob may then be retired or kept as a separate axis.
- **Negative reference (do NOT produce)**: a finding that "the algorithm is correct, just over-conservative" without code+data evidence. That framing was the original failure mode (`docs/plans/2026-04-28-performance-audit-since-hybrid.md`); the same trap applies here. If a throttle's behavior looks defensible, double-check by: (a) reading the constant's source provenance, (b) plotting the empirical distribution, (c) computing the cost in position size.
- **Positive reference shape — throttle health card**: `throttle_name | input_signal | constant_dependencies | empirical_distribution_900d | fraction_at_extremum | gate_fire_rate | sensitivity_to_1σ_shift | verdict (informative | saturated | dead | misnormalized)`
- **Positive reference shape — calibration recommendation row**: `constant_name | current_value | proposed_value | source_evidence (data path + computation) | impact_on_gate_fire_rate | impact_on_position_size_distribution | impact_on_walk_forward_gate`

## Write surface

- `docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md` (Phase 1 output: 900-day empirical distributions of every throttle input)
- `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md` (Phase 2 output: source-trace and empirical re-derivation of `AVG_CORR_MEAN`/`PC1_MEAN` and friends)
- `docs/plans/2026-04-30-phase3-compound-throttle-cost.md` (Phase 3 output: worked compound-effect analysis on representative days)
- `docs/plans/2026-04-30-phase4-recalibration-recommendation.md` (Phase 4 output: proposed constants + walk-forward verification)
- `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` (final stop-condition file)
- `src/signals/fragility.py` (constants update behind feature gate)
- `src/signals/regime_fusion.py` (only if Phase 4 finds the gate parameters need adjustment)
- `config/` (new decision-params bundle if recalibration is non-trivial)
- `tests/test_fragility_calibration.py` (new test locking the empirical-baseline derivation)
- `docs/POSTMORTEMS.md` (one new entry: stale-historical-norm class)
- `docs/PLAN.md` (Phase 10 entry summarizing the calibration audit)

**Specifically PROHIBITED**:
- Any live deploy or live-param mutation.
- Any change to `frontend/src/` (operator has separate WIP).
- Any retroactive edit of the prior audit's RETURN doc or its phase docs (preserve as historical record).
- Bundled commits — one fix = one branch = one commit = one test, per `CLAUDE.md`.
- Removing or weakening any of the 2026-04-29 fixes (F-1/F-3/F-4/F-5/F-6/F-8/F-9). If a recalibration appears to require it, that is a finding requiring operator review, not a silent rollback.

## Read-first list

- `CLAUDE.md`
- `docs/PLAN.md` (Phase 9 closeout)
- `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` — **read this first.** It documents the 10 findings and the four fixes that already shipped. The current packet builds on, does not redo, that work.
- `docs/plans/2026-04-29-phase5-calibration-recommendation.md` — the sweep that decided F-7 stays off by default. Understand its assumptions before challenging its conclusion.
- `docs/plans/2026-04-29-phase2-signal-diagnostic.md` — the prior signal health table (re-run on current data; flag any signal whose distribution has shifted since 2026-04-29).
- `docs/plans/2026-04-28-performance-audit-since-hybrid.md` — the original framing-trap audit. **Read this to internalize the failure mode this packet must not repeat.**
- `docs/PHASE3_PROPOSAL.md` — the origin of the fragility metric and its historical-norm constants. Trace `AVG_CORR_MEAN=0.30` to its source dataset and date.
- `src/signals/fragility.py` — full file.
- `src/signals/regime_fusion.py` — focus on §"Caution Gate: Fragility" (around line 200) and the multiplicative chain through `position_size_mod`.
- `src/signals/compute_signals.py` — how the inputs are assembled.
- `src/steps/decision_engine.py` — focus on `position_size_modifier` and `ensemble_multiplier` consumption (around line 388–480).
- 900-day historical data: `s3://investment-system-data/daily/` (snapshot to `$TMPDIR/trader-bot-audit-20260430/` if not already cached). Refresh with `AWS_PROFILE=personal aws s3 cp ... --recursive`.

## Flow position

This packet is the **calibration follow-on** to the 2026-04-29 algorithm-fix cycle. It assumes the prior audit's fixes are in place and asks the next question down: with the bugs out, are the **constants** the corrected algorithm runs on still the right ones? The final phase is the operator-gated live cutover after Phase 4 verification passes the §Acceptance test gate.

## Required method

The executor chooses how to deliberate, in what order, and what intermediate artifacts are useful, within the following minimal procedural floor:

### 1. Read the Read-first list

Especially the prior audit's RETURN. Internalize what was already fixed; do not re-fix.

### 2. Phase 1 — Throttle distribution diagnostic

For every multiplicative throttle and every input that feeds one, compute the 900-day empirical distribution. Specifically:

- `fragility_score` itself, plus its inputs `avg_correlation` and `pc1_explained` (compute from raw panel-symbol prices, not from the cached `signals.parquet` — the question is whether the *current* metric is informative on the *current* market, so recompute fresh with the production code path).
- `entropy_score` and `entropy_shift_flag`.
- `ensemble_multiplier` and `ensemble_disagreement`.
- `vol_uncertainty_score` (post-F-1 fix; should be ≠ 0.10 floor on most days).
- `regime_confidence` and the discrete `final_regime_label`.
- `position_size_modifier` (the output) and `effective_exposure_multiplier`.

For each, the health card columns from the §Reference outputs positive shape. Output: `docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md`.

**Exit criterion**: at least one throttle is identifiable as `saturated` or `misnormalized` *or* the diagnostic shows every throttle is `informative` and the operator hypothesis is refuted. Either result advances Phase 2.

### 3. Phase 2 — Fragility baseline audit

For each historical-norm constant in `src/signals/fragility.py` (`AVG_CORR_MEAN`, `AVG_CORR_STD`, `PC1_MEAN`, `PC1_STD`):

- Trace its source. Where did `0.30` come from? `docs/PHASE3_PROPOSAL.md` is the likely origin; verify and date the dataset.
- Empirically re-derive each constant from the 900-day production-equivalent dataset (same panel symbols, same window, same return computation).
- Compute the implied *current* `fragility_score` distribution under both old and re-derived constants.
- Identify the inflection point: at what `avg_correlation` value does the current metric saturate? At what value would a re-calibrated metric saturate? What is the gap?

Output: `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md`.

**Exit criterion**: each historical-norm constant has a *current-empirical* counterpart, and the executor has a defensible answer to "should the constants be updated, and to what?"

### 4. Phase 3 — Compound throttle cost

Pick three representative days from the 900-day dataset:
- a confirmed risk-on day with rally-window context (e.g., 2026-04-23)
- a confirmed panic day (e.g., 2026-02-11 or any day in the late-March panic)
- a choppy / mixed day

For each, walk the position-sizing chain step by step:
- Input: raw signal values for that day.
- After fragility gate: `position_size_mod` value.
- After entropy gate.
- After ensemble multiplier.
- After final clamps.
- Final `effective_exposure_multiplier`.

Compute the same chain under the Phase 2 recalibrated constants. Report the delta. Quantify: is the delta defensible (the panic day still gets protection) or does it expose the system (the recalibration removes appropriate caution)?

Output: `docs/plans/2026-04-30-phase3-compound-throttle-cost.md`.

**Exit criterion**: the operator can read three example-day walk-throughs and form an opinion about whether the recalibration is sane.

### 5. Phase 4 — Recalibration recommendation

Codify the recommended constant updates. For each:
- One focused branch (`ai/recal-<constant>`).
- One commit updating the constant behind a feature gate (defaulted **off** — `regime_fusion_overrides` or equivalent).
- One unit test that locks the empirical-baseline derivation (so a future drift in the dataset is detectable).

Run walk-forward verification with the same harness as 2026-04-29's Phase 5 (`scripts/run_calibration_sweep_20260429.py` or its sibling), comparing:
- `current_production` (no-op control)
- `fixes_only_no_calibration` (post-2026-04-29 baseline; should match current)
- `recalibrated_constants_default_off` (recalibration shipped behind gate; same behavior as control)
- `recalibrated_constants_default_on` (recalibration active)
- (optional) `recalibrated_plus_F7_relax_on` (the prior knob also flipped)

Each variant gets:
- Broad-gate window return + drawdown.
- Rally-subset window return + drawdown.
- Panic-subset window return + drawdown.
- Day-by-day position-size delta vs. `current_production`.

Recommendation is the variant that meets the §Acceptance test gate. If no variant meets the gate, **that is the finding** — surface it in the RETURN doc; do not weaken the gate to ship something.

Output: `docs/plans/2026-04-30-phase4-recalibration-recommendation.md`.

### 6. Postmortem capture

One new entry in `docs/POSTMORTEMS.md`: **stale-historical-norm class** — the failure mode of normalization constants embedded in code at write-time and never re-validated against current data. Include the *why* (informally: regimes shift; constants chosen against a pre-2024 dataset will no longer span the current input range; the metric saturates and goes informationally dead) and the *prevention* (every normalization constant should carry: source dataset path, source dataset date, re-validation cadence, test that fails when the empirical distribution drifts > X% from the constant).

### 7. Update `docs/PLAN.md`

Add a Phase 10 entry summarizing the calibration audit, the constants updated, and the operator-gated cutover status.

### 8. Stop

Write `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` ending with the literal final line in §Stop condition.

## Constraints

- **Read-only on `frontend/src/`** — operator has WIP.
- **Read-only on the prior audit docs** (`2026-04-29-*`) — preserve as historical record.
- **No live deploy or live-param mutation.** Operator-gated.
- **No bundled commits.** One constant update = one branch = one commit = one test.
- **No silent fixes** during the audit. If a non-calibration bug surfaces, log it as a finding; do not fix without operator review.
- **Read code + data, not memory.** Every claim about a constant's behavior must cite `file:line` for the code and a data path + computation for the empirical claim.
- **Question the metric, not just the gate.** This is the central discipline of this packet. If the only finding is "tweak `fragility_threshold` from 0.75 to 0.80", that is the trap — the threshold is a knob on a metric the audit must first re-validate.
- **Default off.** Every recalibration ships behind a gate, defaulted off. Operator flips it on after reviewing Phase 3 walk-throughs and Phase 4 sweep.
- **AWS cost rule** (per `CLAUDE.md`): no S3 versioning; no new persistent infra. Calibration runs reuse existing harness.

## Required return

Under the write surface above:

1. `docs/plans/2026-04-30-phase1-throttle-distribution-diagnostic.md` — full health-card table for every throttle input across 900 days; flagged-throttle follow-up list.
2. `docs/plans/2026-04-30-phase2-fragility-baseline-audit.md` — source-trace per constant; empirically re-derived counterparts; recommended values with evidence.
3. `docs/plans/2026-04-30-phase3-compound-throttle-cost.md` — three representative-day walk-throughs comparing current vs. recalibrated chain output.
4. Phase 4 fix-branches merged to `main` (or a single verification branch), each with one passing test, one focused commit, one paragraph of rationale in the commit message.
5. `docs/plans/2026-04-30-phase4-recalibration-recommendation.md` — sweep table; recommended constants; sensitivity analysis; gate-metric pass/fail per variant.
6. New decision-params bundle file in `config/` if the recommendation is non-trivial.
7. New `docs/POSTMORTEMS.md` entry: stale-historical-norm class.
8. `docs/PLAN.md` Phase 10 entry.
9. `docs/plans/2026-04-30-fragility-throttle-audit-RETURN.md` — synthesis: what was found, what was recalibrated, what the system does differently, what to monitor, ending with the literal final line in §Stop condition.

---

## Hypotheses to confirm or disconfirm (starting points, not exhaustive)

These are the operator's working hypotheses. The Phase 1–3 diagnostic is expected to confirm, refine, or disconfirm each.

### H-1 — Fragility historical norms are stale

`AVG_CORR_MEAN=0.30` and `PC1_MEAN=0.45` were chosen against a dataset whose date and provenance is unverified in current code. The current market regime (2024–2026) likely has structurally higher mean cross-asset correlation; if true, the metric saturates almost daily and the gate fires unconditionally. Confirm by re-deriving from the 900-day production-equivalent panel.

### H-2 — Saturation is functionally equivalent to the gate being permanently on

If `fragility_score ≥ 0.97` for 6 weeks running and `fragility_threshold = 0.75`, then the gate is firing every day. The position-size cap (`fragility_position_cap`) is then a constant multiplier on every position, not a conditional one. That is not a "caution gate" — it is a baseline tax. If confirmed, either the metric needs recalibration (H-1) or the gate needs redesign (separate question — log as a finding, do not fix in this packet).

### H-3 — The compounding chain is over-throttling

`position_size_mod` passes through fragility cap → entropy multiplier → ensemble multiplier → final clamps. Each step is multiplicative. If two or more are stuck near their throttling extremum simultaneously, the compound effect is multiplicatively worse than any single throttle implies. Phase 3 walk-throughs are the test.

### H-4 — The 2026-04-29 sweep was scoped too narrowly

That sweep varied `fragility_threshold`, `fragility_position_cap`, and the `F-7` relax-in-risk-on knob. It did **not** vary `AVG_CORR_MEAN`, `AVG_CORR_STD`, `PC1_MEAN`, `PC1_STD` — the constants that determine where the metric saturates in the first place. If those are the real lever, the sweep's "no setting dominates" conclusion is correct *for the wrong axis*.

### H-5 — Other throttles may have analogous staleness

`vol_uncertainty_score` was just fixed (F-1, 2026-04-29) — but its threshold logic uses VIX percentiles whose binning may be similarly date-locked. `entropy_shift_flag` uses cutoffs that may share the same provenance. Phase 1 diagnostic catches these by health-carding every throttle uniformly.

### H-N+ — Additional findings surfaced during execution

Add findings here. Required: each one gets a code location (`file:line`), a data-path-and-computation reproducer, a fix summary, a risk class. Do not silently fix. Do not absorb into another finding.

---

## Hard rules during execution

- Read code + data before claiming behavior. `file:line` for code; data path + computation for empirical claims.
- Each fix = one branch = one commit = one test.
- One postmortem entry (stale-historical-norm class) is non-negotiable.
- Operator gates live cutover. Do not deploy without explicit approval.
- If a finding feels small and rationalize-able, that is the trap. Surface it harder.
- Every recalibration ships behind a feature gate, defaulted off.
- Question the metric, not just the gate. That is the discipline this packet exists to enforce.
