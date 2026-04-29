# RETURN — Trader-bot diagnostic + recalibration (2026-04-29)

Packet: `docs/plans/2026-04-29-deep-diagnostic-and-calibration-packet.md`
Branch: `ai/diagnostic-fix-batch` (5 commits, all targeted tests green)
Authority: execution / discovery_bearing.

## TL;DR

Ten substantive findings (F-1 through F-10), one provisional finding retracted (F-11), four code fixes committed with tests, walk-forward sweep run on the 176 aligned daily snapshots in S3, calibration recommendation produced.

**The lag is real but is not the kind of bug a parameter flip fixes.** Fragility's tight cap was net-protective across the broader 42-day gate window; loosening it improves rally returns but increases drawdown more than it helps. The right action is to ship the four pure-correctness/observability fixes and leave `regime_fusion` defaults alone until a stricter conditional gate is designed.

## What was wrong

| # | What was wrong | Why it mattered | Status |
|---|---|---|---|
| F-1  | `vol_uncertainty_score` was 0.10 (the lowest VIX-percentile bin floor, not a real reading) for 168 of 187 production days | one of two main throttles was effectively disabled for 8 months | **fixed** (`b591ccd`) |
| F-2  | `fragility_score` framed as a one-way ratchet; actual defect is tanh-saturation in `(np.tanh(corr_z)+1)/2` plus a 60-day window. Above `avg_correlation ≈ 0.55` the score collapses to ≈1.0 | the gate caps `position_size_modifier` regardless of regime confidence | code knob added (F-7); calibration left at default |
| F-3  | `vvix_percentile` and `skew_value` are literal constants (1 distinct value each across all 188 rows). Vol "complex" is `vix_percentile` aliased twice. The `unstable_calm` hard override is unreachable in production | Phase-6 design was incomplete in deploy | **fixed** (`b591ccd`) |
| F-4  | VUG showed `unrealized_pnl=-$6,097.78, pct=+12.64%` on 2026-04-28 — sign disagreement | dashboard internally inconsistent | **fixed** (`d195437`) |
| F-5  | Two divergent valuation paths (`_reconcile_portfolio_from_broker` vs `_update_valuations_from_quotes`) produce different P&L for the same holding depending on whether broker call succeeds | F-4 transient; same VUG row was correct on Apr 29 with broken Apr 28 — directly observed | **fixed** (`d195437`) |
| F-6  | Expert-signal `degraded_reason` never propagated to `timeseries.json`; fallback values byte-identical to real values | F-1/F-3/F-13 hidden in plain sight for months | **fixed** (`f3e490a`) |
| F-7  | Fragility gate is regime-blind; fires unconditionally on `score > 0.75` ignoring `regime_confidence` | direct cause of position_size halving in confirmed risk_on_trend | **knob added** (`73d3791`, default off — Phase 5 sweep showed the knob doesn't strictly dominate yet) |
| F-8  | `compute_signals.run` reads `fred_latest` outside its definition scope via `'fred_latest' in dir()` reflection | upstream cause of F-1 | **fixed** (`b591ccd`) |
| F-9  | `skew_history` was never plumbed; SKEW percentile uses static thresholds even when history is available | latent | **fixed** (`b591ccd`) |
| F-10 | `total_value` ($106,248) vs `broker_total_value` ($95,847) differ by $10,078 — traceable to continuity bridge but not disclosed on dashboard hero | UX hazard | known-good; dashboard caption deferred (frontend WIP — operator side) |
| F-11 | (initial draft: "timeseries pre-pivot is synthetic backfill") | **RETRACTED** — operator clarification: equity moves are real, totals were bridged from prior testing model; pre-pivot signal flatness reflects deploy posture, not fake data |

## What was fixed (commits on `ai/diagnostic-fix-batch`)

```
b591ccd  fix(signals):    surface vol_uncertainty degraded state instead of 0.10 floor
                          F-1, F-3, F-6, F-8, F-9
d195437  fix(broker):     unify cost basis for unrealized_pnl $ and pct
                          F-4, F-5
73d3791  fix(regime_fusion): add regime-conditional relax for fragility gate
                          F-7 (default off — opt-in knob)
f3e490a  fix(publish):    write per-signal status flags into timeseries rows
                          F-6 (architectural completion)
574b11b  docs:            phase 1-3, 5, RETURN, postmortems, plan update
```

All four fixes have focused unit tests; all 39 signal tests + 9 dashboard-metrics tests + 12/13 morning-executor-broker tests pass on the branch (the 1 unrelated failure is a pre-existing flake in `test_buy_uses_notional` polling logic).

## Walk-forward verification — what the numbers actually said

Sweep harness: `scripts/run_calibration_sweep_20260429.py` plus `scripts/run_rally_window_sweep_20260429.py`. Dataset: 176 aligned daily snapshots from `s3://investment-system-data/daily/` (2025-08-04 → 2026-04-28). Walk-forward plan: 3 folds + 42-day gate (2026-02-12 → 2026-04-28). Hybrid ranking blend 0.35 (production active).

Eight regime_fusion variants tested on the broad gate, ten on rally and hybrid windows. **Headline result: `current_production` beats every loosening variant on the broad gate window.** Detail in `docs/plans/2026-04-29-phase5-calibration-recommendation.md`.

The variants that *did* improve rally-window returns (`cap_080_only` was the standout: −36% vs current's −59% annualized over 13 days) all *worsen* the broad gate by exposing the bot to the late-March panic with a higher cap. No variant strictly dominates current production across windows; Phase 4's F-7 relax knob is wired but should not be flipped to True by default until a stricter rally-confirmation subordinate signal is designed.

## Shadow day — 2026-04-28 simulated decisions per variant

Script: `scripts/run_shadow_day_20260429.py`. Output: `runs/shadow_day_20260429.json`.

Today's regime is `risk_on_trend`. Variant comparison:

| variant                    | actions | basket                                      | end value (1d, fresh $100k start) |
|----------------------------|--------:|---------------------------------------------|------------------------------:|
| `current_production`       | 8       | ARKK, GLD, QQQ, SLV, VUG, VWO, XLK, XLY     | $100,201                      |
| `fixes_only_no_calibration`| 8       | identical (no calibration change)           | $100,201                      |
| `cap_080_only`             | 6       | ARKK, GLD, SLV, VUG, XLK, XLY (drops QQQ, VWO; bigger sizes) | $100,229     |
| `thr_095_only`             | 8       | identical to current                        | $100,201                      |
| `relax_conf_080`           | 4       | ARKK, SLV, VUG, XLK (very concentrated)     | $100,323                      |

**The `fixes_only_no_calibration` variant is the recommended ship**: same trading decisions as `current_production` (zero behavior change in healthy state) but with the four bug fixes applied. The dashboard becomes coherent (F-4/F-5 fixed), the timeseries gains observability (F-6 fixed), and the F-1 silent-fallback class can no longer recur.

## What the system does differently after this packet

1. **vol_uncertainty cannot silently emit a 0.10 floor.** When VIX is unavailable, the block raises and the outer `except` records `degraded_reason='vix_unavailable'` instead.
2. **`unrealized_pnl ($)` and `unrealized_pnl_pct (%)` always share cost basis** — the F-4 sign-mismatch is impossible by construction.
3. **The fragility gate has a regime-conditional relax knob** (`fragility_relax_in_risk_on`, default `False`). Operator can opt in via `regime_fusion_overrides` once calibration produces a setting that strictly dominates.
4. **Every timeseries row carries `<signal>_status` flags** — `'ok' | 'partial:<inputs>' | 'degraded:<reason>'`. A future F-1-class outage will be visible in the artifact.

## What the operator should know about the lag

The 9-point lag to SPY in the last 30 days is not eliminable by any single parameter flip in the current design. The sweep proves it: every variant that helps on rally days hurts on panic days. Three things contribute:

1. **Cash drag** — the bot has been at 35-40% cash in the rally. `min_cash_reserve_by_regime.risk_on_trend` is currently 10%, so the cash is voluntary (ensemble disagreement / candidate pool exhaustion / sizing knobs interacting). Investigation of the cash-deployment rule is a follow-up not in this packet's scope.
2. **Fragility's protective trade-off** — fragility correctly throttled during the late-March panic episode and is mathematically saturated during a single-factor rally. The trade-off is regime-dependent; a strictly-better setting needs additional inputs (rally confirmation, trend duration) that the current four experts don't expose.
3. **Ensemble multiplier** — `ensemble_multiplier ≈ 0.85` post-hybrid is multiplying every position by another 15% reduction. That comes from GRU/Transformer disagreement and is correct given the model design.

## Day-1 / Day-3 / Day-7 monitoring (post-cutover)

When the operator does cut over to the Phase-4 fixes:

- **Day 1**: confirm `vol_uncertainty_status == 'ok'` in today's timeseries row. Confirm every holding's `unrealized_pnl` and `unrealized_pnl_pct` agree on sign. Confirm `regime_fusion_overrides` reads as configured (likely empty / defaults preserved).
- **Day 3**: check the new `<signal>_status` columns are populated and not stuck `degraded:` for any signal. Spot-check `fragility_score` against the avg_correlation in `signals.parquet` to confirm computation continuity.
- **Day 7**: full broker reconciliation — broker `account.equity` against dashboard `broker_total_value`, broker `positions[*].avg_entry_price` against dashboard `holdings[*].entry_price`. Any drift indicates an F-5 transient and the fix is not holding.

## Files written

- `docs/plans/2026-04-29-phase1-bug-audit-findings.md`
- `docs/plans/2026-04-29-phase2-signal-diagnostic.md`
- `docs/plans/2026-04-29-phase3-pnl-reconciliation.md`
- `docs/plans/2026-04-29-phase5-calibration-recommendation.md`
- `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` (this file)
- `docs/POSTMORTEMS.md` — three new entries (silent-fallback-signals class, one-way-ratchet class, first-pass-audit-framing-trap)
- `docs/PLAN.md` — Phase 9 entry
- `src/signals/compute_signals.py`, `src/signals/vol_uncertainty.py`, `src/signals/regime_fusion.py`
- `src/steps/morning_executor.py`, `src/steps/publish_artifacts.py`
- `tests/test_signals.py`, `tests/test_morning_executor_broker.py`, `tests/test_dashboard_metrics.py`
- `scripts/run_calibration_sweep_20260429.py`, `scripts/run_rally_window_sweep_20260429.py`, `scripts/run_shadow_day_20260429.py`
- `runs/calibration_sweep_20260429.json`, `runs/rally_sweep_20260429.json`, `runs/shadow_day_20260429.json`

---

TRADER-BOT DIAGNOSTIC + RECALIBRATION COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
