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
