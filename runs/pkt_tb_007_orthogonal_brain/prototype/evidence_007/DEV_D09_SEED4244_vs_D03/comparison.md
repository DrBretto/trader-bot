# Evidence comparison — apriori@seed4244 (deviation: a-priori genome, instrument-failed EA) vs apriori@4242 (deviation: a-priori genome, instrument-failed EA)

**COST-SEED SENSITIVITY** (deviation: a-priori genome, instrument-failed EA) — same config, cost seed 4244. Raw series + timeline asserted byte-identical to D03; only the cost overlay moves.

Holdout boundary: 2026-03-11. Delta = apriori@seed4244 (deviation: a-priori genome, instrument-failed EA) − apriori@4242 (deviation: a-priori genome, instrument-failed EA).
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | apriori@seed4244 (deviation: a-priori genome, instrument-failed EA) | apriori@4242 (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +8.47% | +8.46% | +0.01% |
| cagr | +36.39% | +36.33% | +0.06% |
| sharpe | +3.5557 | +3.5482 | +0.0074 |
| max_drawdown | +2.74% | +2.75% | -0.01% |
| win_rate | +56.06% | +56.06% | +0.00% |
| realized_round_trips | 164 | 164 | 0 |
| cumulative_transaction_costs | +335.3683 | +348.1598 | -12.7916 |
| avg_gross_exposure | +34.68% | +34.68% | +0.00% |

Paired daily diff (cost-adjusted): n=66, mean=+0.018 bp/day, sd=+0.18 bp/day, t=+0.775, HAC t=+0.988, 95% CI [-0.0000, +0.0000], 90% CI [-0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.04 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | apriori@seed4244 (deviation: a-priori genome, instrument-failed EA) | apriori@4242 (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +3.45% | +3.46% | -0.00% |
| cagr | +21.46% | +21.49% | -0.03% |
| sharpe | +2.4245 | +2.4267 | -0.0022 |
| max_drawdown | +1.92% | +1.92% | -0.00% |
| win_rate | +59.09% | +59.09% | +0.00% |
| realized_round_trips | 116 | 116 | 0 |
| cumulative_transaction_costs | +118.6200 | +114.8568 | +3.7633 |
| avg_gross_exposure | +29.69% | +29.69% | +0.00% |

Paired daily diff (cost-adjusted): n=44, mean=-0.009 bp/day, sd=+0.08 bp/day, t=-0.760, HAC t=-1.215, 95% CI [-0.0000, +0.0000], 90% CI [-0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.01 bp/day**

## Manifests (summaries)

- apriori@seed4244 (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.151, "cost_drag_bps_of_start_nav": 33.313, "final_value_cost_adjusted": 109299.31523174058, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "8e7ee198ba42ca5406f88367e4b48ee609462f4321310ca42b9f0e4bfb69ac6d", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 335.6848, "total_traded_dollars": 1065404.41, "window": "live"}`
- apriori@4242 (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.271, "cost_drag_bps_of_start_nav": 34.581, "final_value_cost_adjusted": 109286.5366010158, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "6f86c4e397fe63feb668460f1493d89ec51ba0cbddd3ed4e424a8cf429f3c481", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 348.4634, "total_traded_dollars": 1065404.41, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.
