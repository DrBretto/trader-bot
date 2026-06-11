# Evidence comparison — M2-in (deviation: a-priori genome, instrument-failed EA) vs apriori (deviation: a-priori genome, instrument-failed EA)

**CHALLENGER-IN ARM (M2)** (deviation: a-priori genome, instrument-failed EA) — trust {M1:+1, M2:+1}; marginal contribution of adding M2 (status OPEN, FM6) to the a-priori brain.

Holdout boundary: 2026-03-11. Delta = M2-in (deviation: a-priori genome, instrument-failed EA) − apriori (deviation: a-priori genome, instrument-failed EA).
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | M2-in (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +8.42% | +8.46% | -0.04% |
| cagr | +36.15% | +36.33% | -0.18% |
| sharpe | +3.5529 | +3.5482 | +0.0047 |
| max_drawdown | +2.68% | +2.75% | -0.07% |
| win_rate | +56.06% | +56.06% | +0.00% |
| realized_round_trips | 169 | 164 | 5 |
| cumulative_transaction_costs | +362.1657 | +348.1598 | +14.0059 |
| avg_gross_exposure | +34.35% | +34.68% | -0.33% |

Paired daily diff (cost-adjusted): n=66, mean=-0.054 bp/day, sd=+3.54 bp/day, t=-0.123, HAC t=-0.139, 95% CI [-0.0001, +0.0001], 90% CI [-0.0001, +0.0001], **per-arm MDE(|t|=2) = +0.77 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | M2-in (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +3.40% | +3.46% | -0.05% |
| cagr | +21.12% | +21.49% | -0.37% |
| sharpe | +2.4208 | +2.4267 | -0.0059 |
| max_drawdown | +1.86% | +1.92% | -0.06% |
| win_rate | +59.09% | +59.09% | +0.00% |
| realized_round_trips | 115 | 116 | -1 |
| cumulative_transaction_costs | +126.9667 | +114.8568 | +12.1099 |
| avg_gross_exposure | +29.19% | +29.69% | -0.50% |

Paired daily diff (cost-adjusted): n=44, mean=-0.124 bp/day, sd=+4.30 bp/day, t=-0.191, HAC t=-0.219, 95% CI [-0.0001, +0.0001], 90% CI [-0.0001, +0.0001], **per-arm MDE(|t|=2) = +1.13 bp/day**

## Manifests (summaries)

- M2-in (deviation: a-priori genome, instrument-failed EA) (genome fd0ff602ac7b): `{"actions_per_decision_date": 3.97, "arm": "orb1", "cost_bps_of_traded": 3.371, "cost_drag_bps_of_start_nav": 36.006, "final_value_cost_adjusted": 109240.96491736776, "final_value_raw": 109603.76, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 266, "sha256_daily_series": "6d6f79293504ec0785b4aa71a31b40d61a170df7e09513da01aad0535345fd1f", "sha256_timeline": "0f65659438943f76ab99170021e57ed9c66af8eee2c2224857eead4e6e1bd33b", "start_value": 100759.19, "total_cost_dollars": 362.7951, "total_traded_dollars": 1076308.92, "window": "live"}`
- apriori (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.271, "cost_drag_bps_of_start_nav": 34.581, "final_value_cost_adjusted": 109286.5366010158, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "6f86c4e397fe63feb668460f1493d89ec51ba0cbddd3ed4e424a8cf429f3c481", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 348.4634, "total_traded_dollars": 1065404.41, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.
