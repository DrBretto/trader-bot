# Evidence comparison — ORB-1@B0 vs incumbent

**REGISTERED VERDICT PAIR** (FREEZE_ORB1 §1: B0 ships by the anchor remediation rule; hash equality asserted by run_battery_007).

Holdout boundary: 2026-03-11. Delta = ORB-1@B0 − incumbent.
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | ORB-1@B0 | incumbent | delta |
|---|---|---|---|
| total_return | +9.86% | +9.86% | +0.00% |
| cagr | +43.18% | +43.18% | +0.00% |
| sharpe | +3.6962 | +3.6962 | +0.0000 |
| max_drawdown | +2.79% | +2.79% | +0.00% |
| win_rate | +56.06% | +56.06% | +0.00% |
| realized_round_trips | 73 | 73 | 0 |
| cumulative_transaction_costs | +356.0993 | +356.0993 | +0.0000 |
| avg_gross_exposure | +39.07% | +39.07% | +0.00% |

Paired daily diff (cost-adjusted): n=66, mean=+0.000 bp/day, sd=+0.00 bp/day, t=n/a, HAC t=n/a, 95% CI [+0.0000, +0.0000], 90% CI [+0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.00 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | ORB-1@B0 | incumbent | delta |
|---|---|---|---|
| total_return | +4.87% | +4.87% | +0.00% |
| cagr | +31.27% | +31.27% | +0.00% |
| sharpe | +2.8511 | +2.8511 | +0.0000 |
| max_drawdown | +1.96% | +1.96% | +0.00% |
| win_rate | +59.09% | +59.09% | +0.00% |
| realized_round_trips | 42 | 42 | 0 |
| cumulative_transaction_costs | +120.4040 | +120.4040 | +0.0000 |
| avg_gross_exposure | +36.06% | +36.06% | +0.00% |

Paired daily diff (cost-adjusted): n=44, mean=+0.000 bp/day, sd=+0.00 bp/day, t=n/a, HAC t=n/a, 95% CI [+0.0000, +0.0000], 90% CI [+0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.00 bp/day**

## Manifests (summaries)

- ORB-1@B0 (genome 5d4c411764e4): `{"actions_per_decision_date": 1.806, "arm": "orb1", "cost_bps_of_traded": 3.34, "cost_drag_bps_of_start_nav": 35.34, "final_value_cost_adjusted": 110696.48068195056, "final_value_raw": 111052.58, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 121, "sha256_daily_series": "71dc554a269355f2fec83cf80b376a66bb65b818b64602a77378513cde091a08", "sha256_timeline": "4f06fcb1974f81a4df52e75388a2524f263e4e9fefe9bd5e754b4c3ec2b35874", "start_value": 100764.57, "total_cost_dollars": 356.0993, "total_traded_dollars": 1066290.77, "window": "live"}`
- incumbent (genome None): `{"actions_per_decision_date": 1.806, "arm": "incumbent", "cost_bps_of_traded": 3.34, "cost_drag_bps_of_start_nav": 35.34, "final_value_cost_adjusted": 110696.48068195056, "final_value_raw": 111052.58, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 121, "sha256_daily_series": "71dc554a269355f2fec83cf80b376a66bb65b818b64602a77378513cde091a08", "sha256_timeline": "4f06fcb1974f81a4df52e75388a2524f263e4e9fefe9bd5e754b4c3ec2b35874", "start_value": 100764.57, "total_cost_dollars": 356.0993, "total_traded_dollars": 1066290.77, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.

NOTE: sha256(timeline.json), sha256(daily_series.csv) (raw AND cost-adjusted columns), sha256(cost_overlay.json) all EQUAL (runs_battery_007/battery_checks_007.json). Paired deltas are identically zero; every cell below is degenerate by construction.
