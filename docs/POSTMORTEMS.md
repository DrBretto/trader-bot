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
