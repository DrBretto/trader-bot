# Evidence comparison — M4A-in (deviation: a-priori genome, instrument-failed EA) vs apriori (deviation: a-priori genome, instrument-failed EA)

**CHALLENGER-IN ARM (M4-A)** (deviation: a-priori genome, instrument-failed EA) — event_damp_strength 0.5 (mid-range, value-blind), trust M4:+1; the adapter's M4 damp pathway with M4-A exceedance scores.

Holdout boundary: 2026-03-11. Delta = M4A-in (deviation: a-priori genome, instrument-failed EA) − apriori (deviation: a-priori genome, instrument-failed EA).
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | M4A-in (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +8.52% | +8.46% | +0.06% |
| cagr | +36.62% | +36.33% | +0.29% |
| sharpe | +3.5721 | +3.5482 | +0.0238 |
| max_drawdown | +2.75% | +2.75% | +0.00% |
| win_rate | +56.06% | +56.06% | +0.00% |
| realized_round_trips | 163 | 164 | -1 |
| cumulative_transaction_costs | +348.0723 | +348.1598 | -0.0876 |
| avg_gross_exposure | +34.73% | +34.68% | +0.06% |

Paired daily diff (cost-adjusted): n=66, mean=+0.084 bp/day, sd=+0.24 bp/day, t=+2.913, HAC t=+1.842, 95% CI [-0.0000, +0.0000], 90% CI [+0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.09 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | M4A-in (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +3.52% | +3.46% | +0.06% |
| cagr | +21.88% | +21.49% | +0.39% |
| sharpe | +2.4650 | +2.4267 | +0.0383 |
| max_drawdown | +1.92% | +1.92% | +0.00% |
| win_rate | +59.09% | +59.09% | +0.00% |
| realized_round_trips | 115 | 116 | -1 |
| cumulative_transaction_costs | +114.7692 | +114.8568 | -0.0876 |
| avg_gross_exposure | +29.78% | +29.69% | +0.08% |

Paired daily diff (cost-adjusted): n=44, mean=+0.127 bp/day, sd=+0.28 bp/day, t=+3.002, HAC t=+2.178, 95% CI [+0.0000, +0.0000], 90% CI [+0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.12 bp/day**

## Manifests (summaries)

- M4A-in (deviation: a-priori genome, instrument-failed EA) (genome c9b8ee96c859): `{"actions_per_decision_date": 3.687, "arm": "orb1", "cost_bps_of_traded": 3.267, "cost_drag_bps_of_start_nav": 34.573, "final_value_cost_adjusted": 109347.3541586113, "final_value_raw": 109695.73, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 247, "sha256_daily_series": "586951dd3e849554bb634c07ec76543b8c553312032158aacef16f91bd29bdbc", "sha256_timeline": "67e5ba0f2e059bc7d4c3cce382d7fbadf93f1a3b95b974cfe99c13a3ddd444ab", "start_value": 100766.47, "total_cost_dollars": 348.3758, "total_traded_dollars": 1066436.85, "window": "live"}`
- apriori (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.271, "cost_drag_bps_of_start_nav": 34.581, "final_value_cost_adjusted": 109286.5366010158, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "6f86c4e397fe63feb668460f1493d89ec51ba0cbddd3ed4e424a8cf429f3c481", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 348.4634, "total_traded_dollars": 1065404.41, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.

NOTE: INTERPRETATION RAIL: the damp shrinks tilt width on high p_exceed names. Because the underlying a-priori M1 tilt has a NEGATIVE point estimate on this window, any width-reduction mechanically scores positive here; this read cannot separate 'M4 skill' from 'less of a losing tilt'. Pre-registered multiplicity line applies: a single positive-looking line is NOT narratable as a discovery (TOURNAMENT §4.6.5).
