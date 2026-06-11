# Evidence comparison — ORB-1@apriori (deviation: a-priori genome, instrument-failed EA) vs incumbent

**DEVIATION PAIR** (deviation: a-priori genome, instrument-failed EA) — chair adjudication 3; value-blind a-priori genome (tilt_gain 0.5, trust {M1:+1}, disp_gain 1.0, dead_zone 0.05, caps at B0 mids). NEVER replaces the registered verdict.

Holdout boundary: 2026-03-11. Delta = ORB-1@apriori (deviation: a-priori genome, instrument-failed EA) − incumbent.
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | ORB-1@apriori (deviation: a-priori genome, instrument-failed EA) | incumbent | delta |
|---|---|---|---|
| total_return | +8.46% | +9.86% | -1.40% |
| cagr | +36.33% | +43.18% | -6.85% |
| sharpe | +3.5482 | +3.6962 | -0.1480 |
| max_drawdown | +2.75% | +2.79% | -0.04% |
| win_rate | +56.06% | +56.06% | +0.00% |
| realized_round_trips | 164 | 73 | 91 |
| cumulative_transaction_costs | +348.1598 | +356.0993 | -7.9395 |
| avg_gross_exposure | +34.68% | +39.07% | -4.39% |

Paired daily diff (cost-adjusted): n=66, mean=-1.983 bp/day, sd=+12.81 bp/day, t=-1.258, HAC t=-0.961, 95% CI [-0.0006, +0.0002], 90% CI [-0.0005, +0.0001], **per-arm MDE(|t|=2) = +4.13 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | ORB-1@apriori (deviation: a-priori genome, instrument-failed EA) | incumbent | delta |
|---|---|---|---|
| total_return | +3.46% | +4.87% | -1.41% |
| cagr | +21.49% | +31.27% | -9.78% |
| sharpe | +2.4267 | +2.8511 | -0.4244 |
| max_drawdown | +1.92% | +1.96% | -0.04% |
| win_rate | +59.09% | +59.09% | +0.00% |
| realized_round_trips | 116 | 42 | 74 |
| cumulative_transaction_costs | +114.8568 | +120.4040 | -5.5472 |
| avg_gross_exposure | +29.69% | +36.06% | -6.37% |

Paired daily diff (cost-adjusted): n=44, mean=-3.128 bp/day, sd=+15.55 bp/day, t=-1.335, HAC t=-1.088, 95% CI [-0.0009, +0.0003], 90% CI [-0.0008, +0.0002], **per-arm MDE(|t|=2) = +5.75 bp/day**

## Manifests (summaries)

- ORB-1@apriori (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.271, "cost_drag_bps_of_start_nav": 34.581, "final_value_cost_adjusted": 109286.5366010158, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "6f86c4e397fe63feb668460f1493d89ec51ba0cbddd3ed4e424a8cf429f3c481", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 348.4634, "total_traded_dollars": 1065404.41, "window": "live"}`
- incumbent (genome None): `{"actions_per_decision_date": 1.806, "arm": "incumbent", "cost_bps_of_traded": 3.34, "cost_drag_bps_of_start_nav": 35.34, "final_value_cost_adjusted": 110696.48068195056, "final_value_raw": 111052.58, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 121, "sha256_daily_series": "71dc554a269355f2fec83cf80b376a66bb65b818b64602a77378513cde091a08", "sha256_timeline": "4f06fcb1974f81a4df52e75388a2524f263e4e9fefe9bd5e754b4c3ec2b35874", "start_value": 100764.57, "total_cost_dollars": 356.0993, "total_traded_dollars": 1066290.77, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.

NOTE: D02 ≡ D01 bit-for-bit, so this comparison IS the D03-vs-D02 M1 utility-attribution read (roster = [M1]). Seed 4242, cost-adjusted, identical dates.
