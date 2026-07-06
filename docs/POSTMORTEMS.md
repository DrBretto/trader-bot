# Post-Mortems

This file documents significant struggles encountered during development to identify patterns and improve workflows.

## Purpose

Post-mortems are documented here when:
- **3+ retry attempts** on the same operation
- **>15 minutes** resolution time
- **Workaround required** for tooling/dependency/environment issues

## Review Cadence

Review this file **monthly** to identify patterns. When patterns emerge, codify them into `@CLAUDE.md` or `@.cursorrules` as preventive rules.

## Entry Format

Each entry follows this structure:

```markdown
## [Date] - [Category] - [Title]

**Task**: [What were you trying to accomplish?]
**Struggle**: [What went wrong? What errors did you encounter?]
**Resolution**: [How did you resolve it? What was the fix/workaround?]
**Time Lost**: [How much time was spent resolving this?]
**Prevention**: [How can this be prevented in the future? What should be added to CLAUDE.md or .cursorrules?]
```

## Categories

- `DEPENDENCY` - Dependency/package management issues
- `TOOLING` - Tool configuration or availability issues
- `CONFIG` - Configuration/environment issues
- `AWS` - AWS-specific issues
- `BUILD` - Build/compilation issues
- `TEST` - Testing framework or test execution issues
- `DEPLOY` - Deployment issues
- `UNDERSTANDING` - Code understanding or pattern recognition issues

## Example Entry

**DELETE AFTER FIRST REAL ENTRY**

## 2026-07-06 — AWS — Canon line reverted to a contaminated value at publish; date-only gate was blind

**Task**: CU-04 — stop the morning/midday publish path from reverting the displayed canon line.
**Struggle**: The publish gate (`_verify_ledger_or_hold`) checked only that the rendered
terminal matched the ledger it read (parity-against-self) and that the terminal DATE did
not regress. When the source ledger read returned a contaminated same-date value
(2026-07-02 flipped 114271.38 → 121147.52), parity passed (rendered == contaminated
source) and the date was unchanged, so the gate published the reverted line with no block
and no alarm. The watchdog only caught it the next day, only for canon `value`.
**Resolution**: Added a publish-time value-revert guard (`guard_publish_not_reverted`)
that checks the rendered canon `value` + SPY `benchmark` against the corrected clean_v2
ledger (hard-pinned, independent of the parity source) and refuses a known-contaminated
value or a same-date divergence → HOLD + SNS alarm. Extended the watchdog to canon + SPY +
challenger, and added an in-cycle midday value-revert detection to close the Monday /
same-cycle timing gaps. Proven by the non-destructive `publish-revert-diag` reality-test.
**Retry Count**: 1 (root-caused directly).
**Prevention**: A publish gate must validate VALUES against an INDEPENDENT corrected
reference, not parity against the same (possibly contaminated) source it reads; and cover
every displayed line, not just canon.

## 2026-06-07 — DEPLOY — Blank dashboard from stale CloudFront index.html after rebuild

**Task**: Dashboard at https://trader-bot.infotrope.io stopped rendering (blank white page) after a model-promotion redeploy.
**Struggle**: Browser console showed `Failed to load module script: Expected a JavaScript-or-Wasm module script but the server responded with a MIME type of "text/html"`. `#root` was empty. Confusingly, `curl` of the site returned a *healthy* `index.html` (referencing `index-X6mCuWAC.js`, which existed in S3 and served as `text/javascript`), while the browser received a *different, stale* `index.html` referencing `index-CD8sRK5V.js` — a hash from a previous build that no longer existed in the bucket. S3 origin was fully self-consistent; the inconsistency was at the CloudFront edge.
**Resolution**: Ran `aws cloudfront create-invalidation --distribution-id E10EHVNQ0CELM2 --paths "/*" --profile personal`. After it completed (~30s), `#root` rendered (219K chars), no console errors. Root cause: Vite content-hashes asset filenames and the deploy `s3 sync … --delete` removes the prior build's bundle, but `index.html` is served under CloudFront's default `Managed-CachingOptimized` behavior (24h edge TTL). The edge kept serving old HTML pointing at a now-deleted JS hash → SPA fallback returns `index.html` (text/html) for the missing `.js` → strict-MIME refusal → blank page. Intermittent/per-edge-node, so it looked fine from some machines.
**Retry Count**: 1 (diagnosed via `frontend/diag-runtime.mjs` browser harness + direct S3/CloudFront inspection).
**Prevention**: Frontend redeploys MUST run a CloudFront invalidation after the S3 sync. Added as mandatory step 3 + a caution block in `docs/DEPLOY.md`. Any tooling/agent that triggers a frontend rebuild (incl. model-promotion pipelines that regenerate dashboard assets) must include the invalidation. Verify post-deploy with the diag harness: `#root` non-empty, no MIME console errors.

## 2025-01-23 - DEPENDENCY - Pandas Lambda size

**Task**: Deploy Python Lambda function with pandas dependency
**Struggle**: Standard pandas package too large for Lambda deployment package size limits. Deployment failed with "Package size exceeds 50MB" error.
**Resolution**: Used pandas layer from AWS or switched to lighter alternative (pandas-lite, pyarrow). Created separate Lambda layer for pandas.
**Time Lost**: 45 minutes
**Prevention**: Check package sizes before Lambda deployment. Document large dependencies that require layers. Add to CLAUDE.md: "For Lambda deployments, check package sizes and use layers for dependencies >10MB."

---

## New Entries

<!-- Add new entries below, newest first -->

## 2026-04-29 — UNDERSTANDING — First-pass-audit-framing-trap (rationalize-the-anomaly)

**Task**: Diagnose why the bot lagged SPY by 9 points in the last 30 days after beating SPY by 7 points the prior 7 months. Prior audit (`docs/plans/2026-04-28-performance-audit-since-hybrid.md`) framed the lag as "the algorithm is doing what the Phase-6 gates were designed to do, just over-conservative for the current regime."

**Struggle**: That framing accepted four concrete bugs as design intent:
- vol_uncertainty was 0.10 (the lowest-bin VIX-percentile floor) for 168 of 187 production days — silently disabled, not "calibrated tight."
- fragility was framed as a "one-way ratchet"; the actual code defect was tanh saturation at high cross-asset correlation, which collapses the score to ≈1.0 above avg_corr 0.55 — different cause, same symptom, requires a different fix.
- VVIX/SKEW were never wired (always at the neutral 0.5 / 0.0 fallback). The vol "complex" was a single-input VIX percentile in disguise.
- 125 of the first 161 "live history" days were synthetic backfill. The bot was 100% in cash for 6 months. The "+7 points pre-hybrid outperformance" credited the bot with skill it did not exercise.

The deeper failure mode: when an audit's first pass produces a soothing narrative, the narrative crowds out evidence. "Working as designed" disposes of the question without inspecting the code or the data distribution.

**Resolution**: A second pass anchored to file:line code citations and reproducer Python snippets against snapshotted data. Each "working as designed" claim was tested:
- Reproduce the exact `vol_uncertainty=0.10` floor by passing `vix=0` to `_percentile_score(0, VIX_THRESHOLDS)` → returned 0.1 — code-confirmed.
- Compute fragility's day-to-day diff distribution → 29 ups / 18 downs / 139 flat — disproved the ratchet hypothesis but uncovered tanh saturation by examining `(np.tanh(corr_z)+1)/2` at production avg_correlation values.
- Check `vvix_percentile` distinct values → exactly 1 (the 0.5 neutral fallback) → fully confirmed.
- Find first-deviation date for every numeric signal in `timeseries.json` → all twelve hit 2026-01-31 → discovered F-11 (the dataset is structurally inhomogeneous).

**Retry Count**: 2 audits (the first one rationalized; the second one diagnosed).

**Prevention**:

1. **Every behavioral claim in an audit cites file:line.** Memory of how the system "should" work is not evidence. (Codified in this packet's "Hard rules during execution".)
2. **Every "working as designed" hypothesis must be falsified before being accepted.** Find the line of code that should produce the claimed behavior, write a small reproducer that exercises it, run the reproducer against the snapshotted data. If the code says one thing and the data says another, the data wins.
3. **Disconfirm the friendly hypothesis first.** When the first read is "this is fine," that is the hypothesis to attack hardest. The audit framing trap is a confirmation-bias trap. The packet rule "If a finding feels small and rationalize-able, that is the trap. Surface it harder." formalizes this.
4. **Anchor on data shape, not data values.** Distinct-value counts, longest-flat-runs, and dominant-value share each surfaced findings that scalar means hid (`vvix_percentile.mean=0.5` is the same scalar value whether the signal is healthy or dead — but `distinct=1` is unmistakable).
5. **Backfill / placeholder data must be tagged structurally** (the F-6 `signal_status` columns added in Phase 4 are the hardware for this). A neutral fallback that is byte-identical to a real value is invisible until something blows up.

---

## 2026-04-29 — UNDERSTANDING — One-way-ratchet class (saturated normalizers, slow-window smoothing)

**Task**: Diagnose `fragility_score` behavior — symptoms suggested a "ratchet" (only goes up, never resets); the fix to that hypothesis would have been "add a downward decay term."

**Struggle**: The hypothesis was wrong even though the symptom was right. Reading `src/signals/fragility.py:101-108`:

```python
corr_z = (avg_correlation - AVG_CORR_MEAN) / AVG_CORR_STD     # mean=0.30, std=0.15
norm_corr = (np.tanh(corr_z) + 1) / 2
```

`tanh` saturates: at `avg_correlation = 0.55`, `norm_corr ≈ 0.964`; at `avg_correlation = 0.65`, `norm_corr ≈ 0.991`. Above 0.55, the function loses essentially all resolution — the score collapses to ≈1.0 regardless of whether the underlying correlation is rising for fearful (pre-shock) reasons or euphoric (single-factor rally) reasons.

The 60-day rolling window compounds this: today's correlation matrix replaces only ~1.7% of the window each day, so daily change in `avg_correlation` is small. Combined with tanh saturation, the score moves slowly *and* saturates — it looks like a ratchet, but the deeper class is "saturating normalizer over slow-changing inputs." Adding decay would not help — the decay would be just as compressed as the upward motion, both hidden in the saturation tails.

**Resolution**: Reframed the finding (Phase 1 F-2). The fix is two changes:

1. Replace tanh with a piecewise / percentile-based mapping calibrated against a long-horizon historical distribution of `avg_correlation`. This restores resolution above 0.55.
2. Optionally shorten the rolling window or expose a short-vs-long divergence as the actual signal.

**Retry Count**: hypothesis #1 (ratchet) → falsified by data, replaced with hypothesis #2 (saturation) → confirmed by code reading.

**Prevention**:

1. **When a normalizer maps unbounded input to bounded output, check where the input lives in the input distribution before trusting the output.** A saturated normalizer is a dead-zone signal even if all the mathematical machinery is correct. tanh at z=2 is 0.964, at z=3 is 0.995 — that's a 3x increase in the input that produces a 3% increase in the output. If the input variable lives in that range, the signal is pinned.
2. **For any rolling statistic, log the day-over-day change distribution.** A signal whose daily change has a tiny IQR is by definition slow-moving regardless of what the statistic claims to measure.
3. **The whole class of "I have a normalized score in [0,1]" signals deserves saturation audits.** This applies to `vol_uncertainty_score`, `entropy_score`, and any future expert that maps an unbounded measurement into a probability-like output. Phase 5 calibration should re-fit normalizer parameters from long-run distributions, not use designer-time defaults.

---

## 2026-04-29 — CONFIG — Silent-fallback-signals class (degraded inputs invisible at output)

**Task**: Diagnose F-1 (`vol_uncertainty=0.10` for 168 days) and F-3 (VVIX/SKEW dead all-time).

**Struggle**: Each signal block in `src/signals/compute_signals.py` follows a pattern:

```python
try:
    result['vol_uncertainty'] = compute_vol_uncertainty(...)
except Exception as e:
    result['vol_uncertainty'] = {
        'vol_uncertainty_score': 0.5,            # neutral fallback
        ...,
        'degraded_reason': str(e),
    }
```

This is not in itself wrong. Two specific problems compound it:

1. **The fallback values are also valid scores.** A `vol_uncertainty_score=0.5` produced by a clean computation looks identical to one produced by an exception. The `degraded_reason` field is attached to the dict but never propagated to the rolling timeseries (`publish_artifacts._build_timeseries_row`). 168 days of `0.10` (an even more deceptive value — the *floor* of the percentile bin) sat in production unnoticed.
2. **Per-input degradation is not exposed at all.** When VVIX returns empty from Stooq, `compute_vol_uncertainty` proceeds with `vvix=None` and the composite weighting collapses to `score = vix_pctile`. The output dict has no field that records "VVIX was missing." So the same `0.55` score could be a healthy VIX+VVIX+SKEW composite or a single-input VIX percentile; you cannot tell from the artifact.

**Resolution**: Phase 4 fix `f3e490a` added `<signal>_status` columns to every timeseries row: `'ok' | 'partial:<inputs>' | 'degraded:<reason>'`. Phase 4 fix `b591ccd` raised the vol_uncertainty block on `vix_value <= 0` instead of silently passing it through, and added an `inputs_degraded` list that tracks per-input availability (vvix_missing, skew_missing, vix_history_short).

**Retry Count**: undetected for ~8 months (production); diagnosed in one pass during Phase 1.

**Prevention**:

1. **Every signal that has a fallback path must propagate that fact to the artifact.** A degraded signal is a different signal from a healthy one even if the score is numerically the same. Treat that as a hard architectural rule.
2. **Per-input degradation is a separate concern from per-block degradation.** `compute_vol_uncertainty` failing entirely is rare; missing a single input (VVIX, SKEW, vix_history) is *common* and currently silent. The new `inputs_degraded` list is the architectural fix.
3. **Phase-2 health-card automation:** any rolling timeseries column whose `distinct_values == 1` for more than N days should fire a health alert. F-3 would have been caught in week 1 with this check.
4. **Fallback values should be visibly suspicious.** A fallback `0.10` for a percentile signal is worse than `0.5` because it looks like a real low-percentile reading. When choosing fallbacks, prefer values that make the failure obvious in a chart (e.g. `NaN` rendered as a dropped point, or a sentinel like `-1.0` outside the normal range). Do this at signal-design time, not after the fact.
5. **The same class manifested again in F-13** (`yield_slope_10y_3m` was 0.0 for 168 days because `DGS3MO` was missing in `fred_df`). This is a system-wide pattern, not a one-off bug. Any signal that depends on a FRED series, a Stooq fetch, or a model artifact must surface a `degraded_reason` when the dependency fails — and the timeseries row must carry that flag forward.

---

## 2026-04-30 — UNDERSTANDING — Stale-historical-norm class (normalizer constants embedded once, never re-validated)

**Task**: Diagnose why `fragility_score` has been pegged ≥ 0.97 for six straight weeks (operator's complaint that motivated the 2026-04-30 audit packet).

**Struggle**: The fragility metric normalizes two inputs (`avg_correlation`, `pc1_explained`) via z-scores against four hardcoded constants:

```python
# src/signals/fragility.py:17-23 (as-written, 2026-02-06)
AVG_CORR_MEAN = 0.30
AVG_CORR_STD = 0.15
PC1_MEAN = 0.45
PC1_STD = 0.12
```

The comment above them said "Average pairwise correlation: mean ~0.30, std ~0.15" / "PC1 explained variance: mean ~0.45, std ~0.12" — bare assertions, no source dataset, no derivation date, no panel composition, no rolling window assumption. Empirical re-derivation from 1272 trading days (2021-04 → 2026-04) using the production code path against the production panel found:

| constant | code value | empirical 900d | gap |
|----------|-----------:|---------------:|-----|
| `AVG_CORR_MEAN` | 0.30 | 0.4774 | code value is at the empirical 5th percentile |
| `AVG_CORR_STD`  | 0.15 | 0.0979 | code value is 53% wider than empirical |
| `PC1_MEAN`      | 0.45 | 0.6007 | code value is below the empirical minimum (0.40) |
| `PC1_STD`       | 0.12 | 0.0658 | code value is 82% wider than empirical |

The metric had been treating the empirical median as ≈ +1.8σ above the mean — squarely on the saturated side of `tanh`. Result: gate fires on 82.5% of historical days; 50.8% of days score ≥ 0.90; the operator's reported "stuck at 0.97 for six weeks" was directly produced by this. The metric was not measuring fragility — it was measuring "we are in any market that exists in 2025-2026."

**Why it is a class, not a one-off**:

- `vol_uncertainty.py` carries similarly hardcoded `VIX_THRESHOLDS = {'p20': 13, 'p50': 17, 'p80': 25, 'p95': 30}` etc. Empirically over 1272d the VIX median is 17.93 — these are still close, but the same pattern. **No test fails when the empirical distribution drifts**.
- `entropy_shift.py` has `z_threshold=1.5` hardcoded as a default; the consecutive-days flag has not fired in the 188-day production window. (Cross-check shows it does fire ~22.9% of the 1272-day historical window — i.e. it was window-quiet, not dead — but no automated mechanism would have alerted if it had been mis-set.)
- `macro_credit.py` has `slope_mean`/`slope_std`/`hy_mean`/`hy_std` defaults in `optimizer/replay.py` (`SLOPE_MEAN=1.5`, `SLOPE_STD=1.0`, `HY_SPREAD_MEAN=0.0`, `HY_SPREAD_STD=0.02`). Same pattern: hardcoded, undocumented source, no re-validation cadence.

**Resolution**: Phase 4 of the 2026-04-30 audit shipped:

1. `RECALIBRATED_2026_04_30` constants added to `src/signals/fragility.py` with provenance fields (`source_dataset`, `derivation_date`, `re_validation_cadence`).
2. `config/decision_params.recalibrated_2026_04_30.json` ships the recalibrated values behind a bundle gate. Code defaults preserved; cutover is a one-file copy by the operator.
3. `tests/test_fragility_calibration.py` locks both the original defaults (regression guard) and the recalibrated values (drift guard), and asserts a >0.40 fragility-score gap on synthetic near-p25 input.

**Retry Count**: discovered after one prior audit (2026-04-29) had reframed F-2 as "tanh-saturation, calibration knob default off because no setting strictly dominated." The prior audit was correct *at the gate-parameter axis* and wrong *at the input-normalization axis* — a textbook case of First-pass-audit-framing-trap (cited above).

**Prevention**:

1. **Every normalization constant must carry provenance.** Inline at the constant: source dataset path, source dataset date, panel composition, rolling-window assumption. If you cannot write that comment, you do not have a constant; you have a guess.
2. **Every normalization constant must carry a re-validation cadence and a test that fails when the empirical distribution drifts.** The new `tests/test_fragility_calibration.py` is the template. Per-quarter rerun of the empirical re-derivation, fail the test if the constant has drifted > 15% from the empirical mean, and require an audit doc to update it. Drift between `compute_X.py` and the empirical world is a statistic-quality bug, not a magical constant change.
3. **A normalizer's saturation is NOT the same as the underlying signal's saturation.** When debugging a metric that is pinned, *first* check where the input variable lives in the input distribution. Phase 1 of this audit found this in fifteen minutes; the prior audit missed it for an axis-of-investigation reason. The diagnostic order should be: input distribution → normalizer parameters → output distribution → gate parameters. Working that order in reverse produces no-setting-dominates findings, because the loss has already happened upstream.
4. **Bundle-vs-code drift is its own class.** The recalibrated values ship in `config/decision_params.recalibrated_2026_04_30.json`. The Python module's `RECALIBRATED_2026_04_30` constants must agree. The new test verifies this — the failure mode where someone updates the bundle but not the constant (or vice versa) is silent without it.
5. **The 2026-02-06 commit that introduced these constants did not include a `tests/` change locking them to a derivation procedure.** That is the root cause. Future expert-engine work must include such a test as a precondition for merge — the constant in the file is the assertion, the test is the source of truth.

---

## 2026-04-30 — UNDERSTANDING — Known-bug-documented-as-feature class (and false-positive findings from function-in-isolation analysis)

**Task**: Implement the F-2030-A fix from the 2026-04-30 fragility audit. The audit claimed `ensemble_multiplier` was applied twice in the position-size chain — once at `regime_fusion.py:262` (folded into `position_size_modifier`) and again at `decision_engine.py:440-454` (as `ensemble_adj`) — producing a silent ~15% over-throttle on every v3 trade.

**Struggle**: The bug did not reproduce in production. Reading the actual call sites:

```python
# src/steps/decision_engine.py:798
position = compute_position_size(
    ...,
    ensemble_multiplier if expert_signals is None else 1.0,  # v3 defense
    position_size_modifier,
    ...
)
```

Whenever `expert_signals is not None` (the v3 production path), the caller passes `ensemble_multiplier=1.0` — neutralizing the second multiplication inside `compute_position_size`. The v2 caller (`expert_signals is None`) leaves `position_size_modifier=1.0` (the default), neutralizing the first multiplication. Either way, the multiplier is applied exactly once. The 2x2 walk-forward confirmed this empirically: `current_production` and `double_fix_on_only` produce byte-identical fills across the 42-day broad gate, the 11-day rally, and the 35-day panic windows. Same gate Sharpe, same drawdowns, same trip count.

The audit's mistake was reading the function definition (`compute_position_size:440-454`) and the upstream multiplication (`regime_fusion.py:262`) in isolation, without enumerating the call sites of `compute_position_size`. A function that *would* double-apply if called naively does not double-apply if no caller invokes the naive pattern. The audit produced a believable claim from incomplete inspection.

The original comment at `decision_engine.py:442-444` was a contributing factor:

```python
# Expert signal adjustments (position_size_modifier already includes
# ensemble_multiplier via regime_fusion, but we keep ensemble_adj here
# for backward compat when expert_signals is None)
```

Read in isolation, the comment strongly implies the function is buggy ("we keep ensemble_adj here for backward compat" sounds like documenting a known wart, not a contract). It does not mention the caller defense — the reader must chase the call sites to find the `if expert_signals is None else 1.0` guard. The audit chased the wrong thread.

**Resolution**:

1. The fix-branch ships the `ensemble_multiplier_already_applied` gate (default off) as defense-in-depth — if a future refactor drops the v3 defense at line 798, the gate forces correct math regardless of caller pattern.
2. The misleading comment was rewritten to explicitly document the caller defense pattern: "the v3 production callers defend against this by passing `ensemble_multiplier=1.0` whenever `expert_signals is not None`." A future reader does not need to chase the call sites.
3. New tests in `tests/test_decision_engine.py::TestEnsembleMultiplierCallerPatterns` lock both the v3 caller pattern (passes 1.0, expects single application) AND the v2 caller pattern (passes ensemble, expects single application) AND the unsafe pattern (both non-neutral, double-applies without the gate, single-applies with the gate). A regression that breaks the line-798 defense fails the v3 test at the unit level.
4. The walk-forward 2x2 was run anyway as the empirical confirmation: all four cells byte-identical confirms the no-op nature.

**Retry Count**: 1 packet to investigate the audit's claim; the fix is shipped as defense-in-depth, not as a remedy.

**Prevention** (multiple lessons from one investigation):

1. **"Backward compat" / "kept for legacy" / "preserved for X" in a comment is a flag, not a closure.** When you see one of these phrases, the next audit cycle question is *"is the legacy path still live?"* — not *"this is fine, move on."* The comment that kept this bug-shaped pattern alive said exactly the kind of thing that should trigger reinvestigation. Codify in CLAUDE.md or a comment-review checklist: any comment justifying current behavior on legacy / backward-compat grounds is a follow-up flag with a 6-month review timer.
2. **Function-in-isolation analysis is incomplete.** A claim about a function's runtime behavior must enumerate its call sites. Tools: `grep -n "function_name(" .` is a 5-second check. The audit produced a 2,800-line writeup that cited the function definition and its upstream caller (regime_fusion.py:262) but not the downstream callers of compute_position_size (line 798, line 828). The fix to the audit method: every "X is called with Y" claim must include `file:line` references for at least one call site, not just the function body.
3. **A susceptibility is not a bug.** A function that *would* misbehave when called naively is not the same as a function that *does* misbehave. The original comment was correct given the caller defense; the audit treated the susceptibility as a confirmed bug because the susceptibility was easy to demonstrate in a function-only reproducer (passing both arguments non-neutral). The reproducer was right; the inference from reproducer to production behavior was wrong.
4. **The defense-in-depth gate ships anyway.** Even though the audit's claim was wrong, the resulting gate is harmless under correct callers AND useful as future-regression protection AND comes with tests that lock the caller pattern. The cost of shipping is low; the cost of a silent regression in the line-798 defense is high. Default-off so promoting the gate is a separate operator decision; the comment now documents both the caller pattern and the gate's purpose. **An audit producing a defense-in-depth fix from a false-positive finding is still useful work** — provided the RETURN doc surfaces the false-positive clearly, not silently ships the no-op.
5. **First-pass-audit-framing-trap (entry above) repeats here in a new shape.** The 2026-04-29 audit's framing trap was "the algorithm is doing what the gates were designed to do — over-conservative." The 2026-04-30 audit's framing trap was "the comment says backward-compat, the math says double-application — therefore bug." Both rationalized one observation against another without falsifying. Disconfirming the friendly hypothesis (audit's bug-claim) before committing to a fix would have surfaced the caller defense in 5 minutes of grep work.

---

## 2026-04-30 — UNDERSTANDING — Stale-bounds-block-empirical-convergence class

**Task**: Investigate why the weekly optimizer (`launchd` Sundays 3:30 AM) ran for months without ever proposing the empirically-correct fragility constants. The 2026-04-30 fragility audit re-derived `AVG_CORR_MEAN ≈ 0.477` from 900 days of data; the optimizer's `param_inventory.json` had been allowing AVG_CORR_MEAN to be mutated freely with `min: -5.0, max: 5.0` for that entire period and never got close.

**Struggle**: The genetic algorithm's mutation operator samples randomly within the parameter's `allowed_range`. For `AVG_CORR_MEAN`, that range was 10 units wide. The empirical correct band is roughly 0.1 unit wide (p1=0.388 to p99=0.600). Random uniform sampling over a 10-unit range converges on a 0.1-unit band only by accident:

```
P(uniform sample lands in [0.388, 0.600]) ≈ 0.212 / 10.0 ≈ 2.1%
```

With population_size=40 and mutation_rate=0.15, the optimizer evaluates ≈ 6 mutations per parameter per generation. Over 30 generations × N weekly runs, expected hits in the empirical band are linear in attempts, but the *quality* of those hits doesn't improve faster than random because the wf_objective signal in a 200-day backtest is noisy at the calibration-only delta scale. Result: the GA never accumulates improvement; recent weekly runs all rejected with `min_round_trips_total = 0` because random samples in the wide band produce nonsensical normalization values that drive `position_size_modifier` to zero.

The deeper class is **stale bounds blocking empirical convergence**. The `allowed_range` field had `source: inferred_from_usage, needs_bounds_confirmation: true` on every fragility / macro / vol normalization constant — a self-aware marker that the bounds had not been verified against the data they were meant to constrain. That marker was a flag, not a closure. Months passed. The optimizer logged `needs_bounds_confirmation: true` in every run's `param_space_snapshot.json` and nobody noticed.

A second concurrent failure: the guardrail set `min_round_trips_total >= 20` is correct for behaviorally-conditioned changes (e.g. tightening a ranking blend or threshold) but wrong for normalization-constant deltas. A calibration-only delta changes *how much* the bot trades, not *whether* it trades — yet a candidate could fail `min_round_trips_total = 0` with `min_fold_ann_return = -0.999` even when the underlying calibration was directionally correct. The guardrails couldn't distinguish "calibration broke trade frequency" from "calibration is wrong direction." Every recent rejection was on these gates, and the rejection signal was indistinguishable from "everything is fine, no change needed" without a separate alert.

Three issues compounding:
1. Bounds wider than the meaningful range → random mutation cannot converge.
2. No mechanism to propose the empirical statistic as a candidate — only random mutation.
3. Persistent rejection looks identical to "no change needed" without an alert.

**Resolution**: 2026-04-30 four-part fix (`docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md`):

1. **Inventory tagging + tight bounds.** 8 fragility / macro normalization constants tagged `parameter_class: "normalization_constant"` with `empirical_statistic` mappings; bounds re-derived from p1 / p99 of the underlying input distribution. 12 vol_uncertainty percentile bins tagged with `empirical_statistic: null` and bounds derived from historical knowledge (raw VIX/VVIX/SKEW not in the optimizer's signal_row).
2. **Empirical re-derivation candidate operator.** Once per cycle, compute the live-data statistic and inject as a candidate genome. Defaulted off behind `OptimizerConfig.enable_empirical_mutation`.
3. **Calibration-only guardrail path.** When every diff between champion and challenger is in `parameter_class: normalization_constant`, drop `min_round_trips_total`, `min_gate_round_trips`, `min_gate_win_rate`. Outcome-quality gates stay strict.
4. **Persistent-rejection alert.** Counter in `runs/optimizer/rejection_streak.json`; SNS alert via the project's existing `[TraderBot]` topic on the 3rd consecutive rejected cycle. Reset on promotion.

Verification gate: dry-run via `optimizer.cli run --empirical-mutation-debug` proposed `fragility.AVG_CORR_MEAN = 0.4847` (target 0.477 ± 0.02). PASSED.

**Retry Count**: months of weekly runs accumulated rejections silently before the 2026-04-30 fragility audit forced manual recalibration; this packet is the optimizer-level fix to prevent a future cycle of the same class.

**Prevention**:

1. **`needs_bounds_confirmation: true` on an inventory entry is a flag, not a closure.** Any inventory entry with that marker is a follow-up TODO with a 90-day SLA. Codify in `optimizer/discovery/`'s discovery procedure: confirmation must include (a) the source dataset, (b) the empirical p1/p99 of the input, (c) bounds derived from those values. The legacy `inferred_from_usage` source tag is acceptable as a *first* output of discovery but must be replaced by `empirical_distribution_<window>` within one quarter of the parameter being marked `safe_to_optimize: true`.
2. **Bounds for a normalization constant must be derived from its input's distribution, not from the constant's own value scale.** A z-score mean that lives in [0, 1] does not need `min: -5.0, max: 5.0` "just in case." Bounds wider than the input's empirical p1/p99 + 50% are auto-suspicious.
3. **Random mutation is not a substitute for empirical seeding.** When a parameter's "right" value is the empirical mean of a live data stream, the GA needs an explicit candidate-generation operator that *computes* the statistic, not just samples around the current value. The 2026-04-30 fix's empirical-re-derivation candidate is the template.
4. **Guardrails must be class-conditioned, not universal.** `min_round_trips_total` is correct for behavior-conditioning deltas but wrong for normalization-constant deltas. Adding `parameter_class` tagging to the inventory and routing guardrails on it is the architectural fix; the calibration-only path in `optimizer/guardrails.py` is the implementation.
5. **Persistent rejection must be observable.** A weekly run that rejects all challengers looks identical to a healthy "no improvement found, system is fine" cycle without an alert. The rejection-streak counter + SNS notification on threshold crossing is the minimum observability — operators see "the optimizer has been rejecting everything for N weeks" before the underlying drift becomes a separate audit.
6. **Discovery is not a one-time activity.** The `optimizer/discovery/param_inventory.json` was generated once and treated as authoritative metadata. Re-running discovery against the live system is a non-feature in the current toolchain. Add to the operator runbook: re-run discovery quarterly, diff the output against the committed inventory, surface mismatches as findings. (Out of this packet's scope; logged for follow-up.)

## 2026-06-06 - UNDERSTANDING - Beat-champion: measure the displayed object, not a stripped proxy

**Task**: Make the displayed champion measurably better, out-of-sample, on the real system.
**Struggle**: An earlier pass measured performance on the stripped `optimizer.replay` base config
(no overlays) and concluded "+0.98% / can't beat SPY" — a worse-than-incumbent number reported as a
finding. The displayed line's holdout return is ~+9.2% because of the hand overlays (`relax_choppy`,
`topup`); the base alone is ~+0.98%. Also conflated whole-period (+12.5%) with the 2026-03-11+ replay
slice, and computed SPY over the war-dip slice (+14% snap-back) — internally inconsistent and panic-inducing.
**Resolution**: Always measure through `three_line_replay` WITH overlays (the displayed object). Tune on
the pre-holdout in-sample window, validate on 2026-03-11+, enforce a never-worse floor (a sub-incumbent
number is a failed run). Winner: participation tune (max_position_weight 0.30, topup 1.1, stressed-protected
thresholds) → whole period +12.5% → +15.0%, robust on both windows. Full record: `docs/BEAT_CHAMPION_20260606.md`.
**Retry Count**: ~3 framings before the consistent whole-period comparison landed.
**Prevention**: Pin the performance object (displayed three-line replay, overlays on) and the window
(whole-period vs slice) in any future tuning. Never report a number below the live champion as a result.

## 2026-06-06 - UNDERSTANDING - Regime model trained on ~11 rows; rule-labels cap supervised models

**Task**: Fix the regime GRU (9% accuracy, below 20% random).
**Struggle**: Training silently fell back to `daily/` (~11 rows) while 11 years of context
(`training/data/historical_combined.parquet`, 2,803 rows) sat unused. Also, labels come from a fixed
baseline rule, so supervised regime models can only approximate that rule.
**Resolution**: Retrained on the full corpus → OOS accuracy 9%→73%/67%. Switched forward only (history
untouched; chart marker on the switch date). Flagged unsupervised HMM as the real next step (escapes the
rule-label ceiling). Weights set 0.5/0.5 now that both models are healthy (the 0.2/0.8 down-weight was a
crutch for the broken GRU).
**Retry Count**: 1.
**Prevention**: Assert training row-count > N before saving a regime model; alert if it falls back to `daily/`.

## 2026-06-26 - AWS - Equity line frozen: Lambda's old boto3 silently dropped every nightly append

**Task**: "The bot didn't do anything again last night / won't run more than one day in a row." Investigate why the displayed line stops advancing.
**Struggle**: The pipeline was actually firing fine every night (EventBridge → Lambda all green, artifacts published, trades executed). The real failure was buried in the publish step: the equity-ledger append writes its content-addressed leaf with `put_object(..., IfNoneMatch="*")` (S3 conditional write). The Lambda image pins `boto3==1.34.19`, which predates S3 conditional writes, so the call raised `ParamValidationError: Unknown parameter in input: "IfNoneMatch"`. `publish_artifacts.py` swallows any append error as non-fatal ("equity ledger append skipped — line holds"), so the frontier never advanced — frozen at 2026-06-24 while runs kept "succeeding." From the dashboard it looked like the bot did nothing.
**Resolution**: Made the write-once put version-independent in `src/canon/equity_ledger.py` and `src/utils/corrections.py`: on a `ParamValidationError` for `IfNoneMatch`, fall back to a HEAD-check (idempotent no-op if the leaf exists) + a plain put. The leaf key is a content hash, so write-once is already guaranteed by the key — `IfNoneMatch` was only belt-and-suspenders. Added old-SDK regression tests (`FakeS3OldSdk`) asserting the line advances across consecutive days without the parameter. Rebuilt + redeployed the container; manual night invoke confirmed the frontier advanced 2026-06-24 → 2026-06-26 and the log now reads "equity leaf written WITHOUT IfNoneMatch" instead of "append skipped — line holds".
**Retry Count**: 1 (root-caused from CloudWatch on first pass).
**Prevention**: Never let a swallowed-exception path hide a frontier that isn't advancing — the append should at minimum alert when it no-ops on a new run_date. Keep the SDK-version-independent fallback. If `boto3` is ever bumped past ~1.35, true conditional writes resume automatically with no code change. Note: 2026-06-25 has no leaf (skipped before the fix); the append-only frontier invariant means it can't be backfilled mid-chain — a one-point cosmetic gap, not a functional break.

## 2026-06-26 - AWS - Challenger line was laptop-bound + the staleness watchdog was dead code

**Task**: "I need all three lines (canon, SPY, challenger) to run autonomously for weeks without intervention, and to STOP being told it's running when it isn't."
**Struggle**: Two structural autonomy/trust gaps behind the recurring "works one day then stops":
  1. The canon + SPY lines run in the night Lambda (autonomous), but the **challenger (dotted blue) line was produced by a local macOS launchd job** (`~/Library/LaunchAgents/com.traderbot.shadow.plist`, `shadow_nightly.py`) at 23:30 local — and that job wasn't even loaded. A laptop cron can never be "autonomous for weeks." `shadow_nightly.py` was never baked into the Lambda image (not in `bake_runtime_subset.CODE_FILES`).
  2. `src/brain/monitors.py` had a well-built `stale_publish_handler` / `check_stale_publish` (designed for exactly the laptop-asleep failure) but it was **dead code** — nothing called it and no EventBridge schedule pointed at it. So a frozen line was never alerted. The equity-append failure was also swallowed as "non-fatal, line holds" with no alert (see the boto3 postmortem above).
**Resolution**:
  - Moved the challenger into the cloud: added `shadow_nightly.py` to the bake, fixed its Lambda-hostile bits (file logging → `/tmp` via `SHADOW_LOG_DIR`; `s3_client()` uses the IAM role instead of the hardcoded `personal` profile when `AWS_LAMBDA_FUNCTION_NAME` is set), added an isolated `shadow-publish` handler phase (appends NO equity leaf — cannot affect the canon line), and a new EventBridge rule `investment-system-shadow-trigger` @ 03:30 UTC Tue–Sat. Verified: a cloud invoke advanced `dashboard/shadow_timeseries.json` `as_of` + `shadow_A` to the current day with zero laptop involvement.
  - Anti-betrayal watchdog: extended `monitors.py` with `run_daily_health_check` (checks all THREE lines + chassis liveness against the latest trading day) wired to a `healthcheck` handler phase + EventBridge rule `investment-system-healthcheck-trigger` @ 04:00 UTC Tue–Sat. It ALWAYS emails a ✓/✗ status (SNS email to drbretto82@gmail.com is confirmed) so a silent freeze is caught the same morning, not on Friday.
  - De-swallowed the equity-append failure: it now fires an SNS CRITICAL alert in addition to holding the line.
  - Backfilled the bug-dropped 2026-06-25 canon leaf and rewound a provisional 06-26 (frontier now correctly 06-25; tonight settles 06-26).
**Retry Count**: 2 deploy/invoke cycles for the cloud shadow port (1: `personal` profile not in Lambda; 2: success).
**Prevention**: No producer of a displayed line should live on the laptop. The daily health email is now the single source of "is it actually running" — if it stops arriving or says ✗, that's the signal, no chart-watching required. The laptop `com.traderbot.shadow.plist` is now redundant and should be removed (a loaded copy would double-write shadow_timeseries.json and flap).
