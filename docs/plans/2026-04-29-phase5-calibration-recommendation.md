# Phase 5 — Calibration sweep + recommendation

Date: 2026-04-29
Packet ref: `docs/plans/2026-04-29-deep-diagnostic-and-calibration-packet.md`
Branch under test: `ai/diagnostic-fix-batch` (4 commits — see Phase 4 fix log).

## §A — Hard blocker: walk-forward gate cannot run on the existing dataset

The packet's acceptance test specifies a "walk-forward verification on the 900-day dataset." Two stacked constraints make this impossible right now:

1. **S3 has only ~200 aligned daily snapshots** (`s3://investment-system-data/daily/2025-08-04/` through `…/2026-04-29/`). The packet's "900-day dataset" presumes a backfilled history that does not exist as `daily/{date}/{features,inference,signals,prices}.parquet` artifacts.

2. **Of those 200, the first 125 (2025-08-04 → 2026-01-30) are F-11 backfill** — every pivot signal in `timeseries.json` first deviates from a neutral fallback on 2026-01-31. Pre-pivot rows are not produced by the same pipeline as post-pivot rows. Treating them as ground truth is exactly the kind of silent-fallback failure F-1/F-3/F-6 already documented at the per-signal level — F-11 is the dataset-level instance.

3. **The remaining 63 post-pivot days** are too short for the existing harness's default folds (`42/42/21/42` = train/test/step/gate ≈ 168 days minimum). With a shrunken `10/10/5/10` config, the harness loads 176 snapshots (since `load_optimizer_dataset` doesn't filter by F-11) and produces 30 folds, but those folds are still ≥ 79% backfilled. **Any calibration recommendation produced by sweeping over backfilled signals is not meaningful** — the variables being optimized over (fragility, vol_uncertainty, etc.) had no input variation during 125 of those days.

**Surface, do not weaken the gate.** Per packet rule:
> "No skipping verification. Phase 5 walk-forward gate must pass before Phase 6 begins. If it doesn't pass, that is the finding — surface it; do not weaken the gate."

The walk-forward gate cannot pass right now because the prerequisite dataset does not exist. This is the finding.

## §B — Required dataset rebuild before calibration is meaningful

Two paths to a usable calibration dataset, in increasing cost:

1. **Backfill replay (preferred, ~1 day of compute):** for each historical date `D` in the desired backtest window, re-run `compute_signals.run` against the FRED + price + GDELT data available *as of D*, and write the resulting `signals.parquet` to `daily/D/`. This requires a "frozen as-of" version of the daily ingest steps. The training data already contains `historical_combined.parquet` and `historical_context.parquet` covering more than 900 days — those are the right inputs. Output: a new `daily/{date}/signals.parquet` per date that reflects what the *current* signal stack would have produced on that date.

2. **Synthetic re-deploy (full backfill of all artifacts):** rebuild every `daily/{date}/` artifact set from scratch using historical inputs. Heavier; would also retroactively change inference outputs.

Neither is in scope for this packet. The Phase-4 code fixes are correct and tested independently of this dataset issue. Once the rebuild lands, the calibration sweep below is the right structure.

## §C — Calibration sweep design (executable once §B is complete)

**Variable space**:

| Knob                                       | Source     | Sweep range                      | Justification |
|---|---|---|---|
| `regime_fusion_overrides.fragility_threshold` | regime_fusion | {0.75, 0.80, 0.85, 0.90, 0.95} | F-2: tanh saturation makes 0.75 fire 100% in elevated-corr regimes |
| `regime_fusion_overrides.fragility_position_cap` | regime_fusion | {0.60, 0.70, 0.80, 1.00 (off)} | current 0.60 is the active cap |
| `regime_fusion_overrides.fragility_relax_in_risk_on` | new (F-7) | {False, True} | Phase-4 fix |
| `regime_fusion_overrides.fragility_relax_confidence` | new (F-7) | {0.70, 0.75, 0.80, 0.85} | only relevant when relax=True |
| `signal_overrides.fragility.window_days` | fragility | {20, 30, 60} | F-2: 60d may be too smooth |
| `signal_overrides.fragility.AVG_CORR_MEAN` | fragility | re-fit from long-run distribution | F-2: tanh saturation parameter |
| `regime_fusion_overrides.entropy_size_multiplier` | regime_fusion | {0.70, 0.85, 1.00} | F-12: gate has never fired |
| `decision_params.min_cash_reserve_by_regime` | decision_engine | per-regime grid | cash drag at 35% in `risk_on_trend` |

**Variants to test:**

| Label              | Description |
|---|---|
| `current`          | exact production active params, no fixes — establishes the lag baseline |
| `fixes_only`       | Phase-4 code fixes merged; all calibration knobs at their existing defaults — establishes whether the bug fixes alone narrow the lag |
| `fixes_relax_off`  | fixes merged, fragility_relax disabled (default) — control |
| `fixes_relax_on_conservative` | fixes merged, fragility_relax_in_risk_on=True, confidence threshold 0.85 |
| `fixes_relax_on_moderate`     | fixes merged, fragility_relax_in_risk_on=True, confidence threshold 0.80 |
| `fixes_relax_on_aggressive`   | fixes merged, fragility_relax_in_risk_on=True, confidence threshold 0.70 |
| `fixes_window_short`          | fixes merged, fragility window 20d instead of 60d |
| `fixes_full_recalibration`    | fixes merged, best of relax + window + threshold sweep — the recommendation candidate |

**Gate criteria (from §Acceptance test):**

1. Beats SPY in the Aug-2025 → Mar-2026 down-leg (after F-11 backfill, this is a real comparison).
2. Captures ≥ 60% of SPY in the Apr-2026 → today up-leg.
3. Max drawdown ≤ 5% across the span.
4. Sharpe ≥ baseline incumbent.
5. No fold's annualized return < −10%.

A variant that fails any of these is **not** the recommendation — even if it improves the headline. The packet is explicit that this gate must hold.

**Sensitivity**: each recommended knob's value should be tested at ±10% of the recommended setting. If the metric falls below the gate at the ±10% boundary, the recommendation is on a knife edge and should be reported as such — not concealed.

## §D — Recommendation that I CAN make now (without walk-forward)

Even without the calibration sweep, the Phase-1/2 findings make a first-cut recommendation defensible *as a hypothesis to test once §B is done*:

| Knob                              | Current     | Hypothesis    | Source |
|---|---|---|---|
| `fragility_relax_in_risk_on`      | False       | **True**      | F-7: the production fragility gate is unconditional and produces 0.49 average pos_mod in confirmed risk_on_trend |
| `fragility_relax_confidence`      | n/a         | **0.80**      | matches the regime_confidence cluster (mean 0.87 post-pivot, F-7) |
| `fragility_threshold`             | 0.75        | 0.85          | F-2: post-pivot mean fragility is 0.63; raising threshold leaves the gate active during real saturation episodes (≥ 0.85 in panic) but lets normal trend regimes through |
| `fragility.window_days`           | 60          | 30            | F-2: 60d window cannot meaningfully respond to regime change inside a quarter |
| `entropy_threshold`               | unknown     | recalibrate to ~10th-pctile of historical SPY entropy z-scores | F-12: gate has never fired; if it never fires it's not a gate, it's compute |
| `decision_params.min_cash_reserve_by_regime.risk_on_trend` | 0.10 | 0.05 | observed cash_pct in production = 35-40% even at min 10% — needs investigation but cap could be lower |

Only the F-7 knob (`fragility_relax_in_risk_on`) is shipped as code in Phase 4. The rest are config-bundle changes that need a calibration sweep to size, and the calibration sweep needs §B to be meaningful.

## §E — Decision-params bundle: what would change

If the F-7 hypothesis above is confirmed, the new `config/decision_params.<datestamp>.json` bundle would diff from the current `decision_params.active.json` only in `regime_fusion_overrides`:

```jsonc
"regime_fusion_overrides": {
  "fragility_relax_in_risk_on": true,
  "fragility_relax_confidence": 0.80,
  "fragility_relax_regimes": ["risk_on_trend", "calm_uptrend"],
  "fragility_threshold": 0.85
}
```

Single, reversible diff. **Not promoted to active** by this packet — that's the operator's call after §B + the §C sweep run.

## §F — Acceptance-test status (what passes today vs not)

| Acceptance criterion                                                        | Status                                  |
|-----------------------------------------------------------------------------|-----------------------------------------|
| Phase-4 fixes merged with passing tests on `ai/diagnostic-fix-batch`        | **PASS** (4 commits, all tests green)   |
| Walk-forward 900-day backtest                                               | **BLOCKED** (F-11; §B prerequisite)     |
| Beats SPY in Aug-2025 → Mar-2026 down-leg with ≥ 60% upside in rally        | **CANNOT BE EVALUATED YET**             |
| Max drawdown ≤ 5%                                                            | observed all-time MDD currently -1.77% in production (favorable; doesn't violate gate) |
| Three new postmortem entries                                                | pending — Phase 8                        |
| Phase-9 PLAN.md entry                                                        | pending                                  |
| RETURN doc with literal final line                                          | **NOT YET** — see RETURN doc rationale  |

End Phase 5.
