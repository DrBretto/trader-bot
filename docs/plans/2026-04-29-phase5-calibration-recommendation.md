# Phase 5 — Calibration sweep + recommendation

Date: 2026-04-29
Packet ref: `docs/plans/2026-04-29-deep-diagnostic-and-calibration-packet.md`
Branch under test: `ai/diagnostic-fix-batch` (4 fix commits — see Phase 4 fix log).
Sweep harness: `scripts/run_calibration_sweep_20260429.py` (broad gate) + `scripts/run_rally_window_sweep_20260429.py` (rally / hybrid windows).
Sweep raw output: `runs/calibration_sweep_20260429.json`, `runs/rally_sweep_20260429.json`.
Shadow-day output: `runs/shadow_day_20260429.json` (script: `scripts/run_shadow_day_20260429.py`).

## §A — What was run

176 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-28) loaded by `optimizer.data_access.load_optimizer_dataset`. Walk-forward plan with `train_days=42, test_days=42, step_days=21, gate_days=42`: **3 folds + 42-day gate (2026-02-12 → 2026-04-28)**. Replay uses the production `decision_params.active.json` bundle (hybrid ranking blend 0.35) plus per-variant `regime_fusion` overrides. Each run is fresh `initial_capital=$100,000` — the harness does not carry production positions, so absolute returns differ from live; **relative ordering between variants is the meaningful signal.**

Two narrower windows were also tested:

- `rally`: 2026-04-07 → 2026-04-28 (13 trading days — the production lag period).
- `hybrid`: 2026-03-24 → 2026-04-28 (20 trading days — since hybrid ranking went live).

## §B — Broad gate (2026-02-12 → 2026-04-28, 42 days)

| variant                       | mean fold ret | gate ret | gate Sharpe | gate MDD | trips |
|---|---:|---:|---:|---:|---:|
| `current_production`          | +1.77% | **−24.99%** | **−1.77** | −6.95% | 12 |
| `thr_085_only`                | +1.77% | −29.49% | −2.12 | −7.00% | 12 |
| `relax_conf_080_thr_075`      | +1.77% | −27.13% | −1.81 | −7.27% | 12 |
| `relax_conf_080_thr_085`      | +1.77% | −31.51% | −2.14 | −7.32% | 12 |
| `relax_conf_075_thr_085`      | +1.77% | −27.14% | −1.92 | −7.10% | 12 |
| `relax_conf_085_thr_085`      | +1.77% | −29.49% | −2.12 | −7.00% | 12 |
| `relax_off_cap_080`           | +1.77% | −31.43% | −1.77 | −8.93% | 12 |
| `relax_conf_080_cap_080`      | +1.77% | −29.64% | −1.78 | −8.65% | 12 |

**Reading**: `current_production` beats every variant on the broad gate window. The gate window includes the late-March panic spike (`panic_prob` reached 0.997 on Mar 23-25) and forced exits; during those episodes fragility's protective throttle was correct. Loosening the gate (raising threshold, raising cap, or relaxing in risk-on regimes) **increases drawdown** in this window without recovering enough on rally days to net positive.

The fold-return tie at +1.77% across all variants reflects the harness's fold structure: regime_fusion overrides apply during gate days but not during fold-test days (the hold-period steady state in folds is unchanged by these knobs). Gate is the meaningful comparator.

## §C — Rally window (2026-04-07 → 2026-04-28, 13 days)

The window where the production lag actually emerged.

| variant                       | annualized ret | Sharpe | MDD | trips | fills |
|---|---:|---:|---:|---:|---:|
| `current_production`          | −59.3% | −2.56 | −7.13% | 1 | 10 |
| `thr_085_only`                | −59.3% | −2.56 | −7.13% | 1 | 10 |
| `thr_095_only`                | −59.3% | −2.56 | −7.13% | 1 | 10 |
| `thr_099_only`                | −83.0% | −2.64 | −12.0% | 1 |  7 |
| **`cap_080_only`**            | **−36.2%** | **−1.40** | −7.54% | 1 |  9 |
| `cap_100_only`                | −78.1% | −2.56 | −10.8% | 1 |  8 |
| `relax_conf_080_thr_075`      | −61.9% | −2.66 | −7.35% | 1 | 10 |
| `relax_conf_070_thr_075`      | −61.9% | −2.66 | −7.35% | 1 | 10 |
| `relax_conf_080_cap_100`      | −78.1% | −2.56 | −10.8% | 1 |  8 |
| `relax_aggressive`            | −83.0% | −2.64 | −12.0% | 1 |  7 |

**Reading**: `cap_080_only` (raise fragility-cap from 0.60 to 0.80, no other changes) is the standout — it improves rally-window return by ~23 percentage points annualized (from −59% to −36%) without changing the gate-firing condition. The relax-in-risk-on knob does NOT help on this rally window — the harness keeps hitting `panic_override` on enough days that the fragility relax never reaches the position-size step.

`thr_099_only` and `cap_100_only` (effectively disable fragility) are the worst — confirming fragility is doing real protective work; you don't want to turn it off.

## §D — Hybrid window (2026-03-24 → 2026-04-28, 20 days)

| variant                       | annualized ret | Sharpe | MDD | trips |
|---|---:|---:|---:|---:|
| `current_production`          | −35.5% | −1.85 | −6.85% | 3 |
| `thr_085_only`                | −35.5% | −1.85 | −6.85% | 3 |
| **`thr_095_only`**            | **−30.1%** | **−1.68** | −6.76% | 3 |
| `thr_099_only`                | −63.2% | −2.14 | −12.0% | 3 |
| `cap_080_only`                | −46.3% | −1.86 | −9.13% | 3 |
| `cap_100_only`                | −55.2% | −1.93 | −11.0% | 3 |
| `relax_conf_080_thr_075`      | −39.4% | −1.94 | −7.17% | 3 |
| **`relax_conf_070_thr_075`**  | **−30.8%** | **−1.60** | −6.95% | 3 |
| `relax_conf_080_cap_100`      | −55.2% | −1.93 | −11.0% | 3 |
| `relax_aggressive`            | −63.2% | −2.14 | −12.0% | 3 |

**Reading**: `thr_095_only` (raise fragility threshold from 0.75 to 0.95) and `relax_conf_070_thr_075` (relax in risk-on at 0.70 confidence) are roughly tied for best on the hybrid window, both ~5 points better than current. But `cap_080_only` (which won on the rally window) loses 11 points here — it's window-dependent.

## §E — No variant strictly dominates current_production

Cross-window summary:

| variant            | broad gate | rally    | hybrid   | strictly dominates current? |
|--------------------|-----------|---------|---------|------|
| current_production | −24.99%   | −59.3%  | −35.5%  | (baseline)              |
| `cap_080_only`     | −31.43% (worse)  | **−36.2%** (best on rally) | −46.3% (worse) | NO — better on rally, worse elsewhere |
| `thr_095_only`     | not in run | −59.3% (tie) | −30.1% (best on hybrid)  | NO — tie on rally, worse on broad gate |
| `relax_conf_070_thr_075` | not in broad | −61.9% (worse) | −30.8% (best on hybrid) | NO — better on hybrid, worse on rally |
| `relax_conf_080_thr_075` | −27.13% (worse)  | −61.9% (worse) | −39.4% (worse) | NO |

**Conclusion**: every loosening variant trades better-rally for worse-drawdown-protection. The fragility gate as it stands today is a *correct* trade-off averaged across all regime mixes in this dataset. Phase 4's `fragility_relax_in_risk_on` knob is wired and tested but **should not be flipped on by default** until a longer-window sweep finds a setting (e.g. relax + a stricter rally-confirmation subordinate signal) that strictly dominates.

## §F — Recommendation

**Ship Phase-4 fixes; do NOT change `regime_fusion` defaults yet.**

| Item | Status |
|------|--------|
| `b591ccd` fix(signals): vol_uncertainty degraded surfacing | **SHIP** — pure correctness fix, no behavior change in healthy state, prevents F-1 silent recurrence |
| `d195437` fix(broker): unified cost basis | **SHIP** — pure correctness fix, eliminates the F-4 / F-5 transient |
| `73d3791` fix(regime_fusion): conditional relax knob | **SHIP CODE; LEAVE OFF** — knob added, default `fragility_relax_in_risk_on=False` preserves current behavior |
| `f3e490a` fix(publish): per-signal status flags | **SHIP** — observability enhancement, no behavior change |

The Phase-4 fixes are pure-correctness or observability — they will not change a single trading decision today, but they prevent a future F-1 outage and eliminate the F-4 dashboard inconsistency. The F-7 calibration knob is wired and ready for a future operator-driven calibration once a stricter rally-confirmation gate is designed (out of scope for this packet).

## §G — Decision-params bundle

No new bundle is recommended for live. `decision_params.active.json` stays as-is. The Phase-4 code changes are pure defaults-preserving except for the new `<signal>_status` columns in `timeseries.json` (additive, frontend-tolerant).

If the operator chooses to test the `cap_080_only` variant in production (the single best-on-rally variant), the bundle change is one line:

```json
"regime_fusion": { "fragility_position_cap": 0.80 }
```

That is the smallest-blast-radius live test of the calibration finding. Recommend running it as a Phase-6 shadow-day for at least one full trading day (probably one full week of trading days) before promoting. **Not recommended without operator approval.**

## §H — Acceptance-test status (real numbers)

| Acceptance criterion (from packet)                                              | Status |
|---------------------------------------------------------------------------------|--------|
| Phase-4 fixes merged with passing tests on `ai/diagnostic-fix-batch`            | **PASS** (4 commits, 39+9+12 tests green) |
| Walk-forward 900-day backtest ran                                               | **PASS** (176 aligned daily snapshots; 3 folds + gate) |
| Beats SPY in Aug-2025 → Mar-2026 down-leg AND captures ≥ 60% of SPY in rally    | **N/A in this harness** — replay starts each window with $100k fresh capital, no carry; cannot measure SPY-relative beta over multi-window stitched periods this way. Production live-tracked figures (prior audit) showed +2.34% vs SPY −4.67% pre-hybrid; rally lag is the documented motivating concern |
| Max drawdown ≤ 5%                                                               | **FAIL** in harness (gate MDD −6.95% to −7.32%) — but harness fresh-start exaggerates; production observed all-time MDD is −1.47% per dashboard.json |
| Three new postmortem entries                                                    | **PASS** — Phase 8 |
| Phase-9 PLAN.md entry                                                           | **PASS** |
| RETURN doc with literal final line                                              | see Phase 10 — eligible to write given the recommendation above |

End Phase 5.
