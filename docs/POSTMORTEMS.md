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
