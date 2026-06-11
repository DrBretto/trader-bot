# Evidence comparison — R-LLM width (deviation: a-priori genome, instrument-failed EA) vs apriori (deviation: a-priori genome, instrument-failed EA)

**R-LLM FALSIFIER ARM** (TOURNAMENT §4.4.9) (deviation: a-priori genome, instrument-failed EA) — llm_disag 252d-z → T_t × clip(1+0.25·tanh z, 0.75, 1.25) on top of the a-priori genome. KILL pre-registered at E1 paired t < +2.0 vs the unconditioned arm; expected effect O(0.1–0.5) bp/day; expected outcome: not promoted.

Holdout boundary: 2026-03-11. Delta = R-LLM width (deviation: a-priori genome, instrument-failed EA) − apriori (deviation: a-priori genome, instrument-failed EA).
Paired daily-difference stats on identical dates are the required form;
endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | R-LLM width (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +8.46% | +8.46% | +0.00% |
| cagr | +36.34% | +36.33% | +0.01% |
| sharpe | +3.5635 | +3.5482 | +0.0152 |
| max_drawdown | +2.69% | +2.75% | -0.06% |
| win_rate | +57.58% | +56.06% | +1.52% |
| realized_round_trips | 157 | 164 | -7 |
| cumulative_transaction_costs | +370.6826 | +348.1598 | +22.5228 |
| avg_gross_exposure | +34.65% | +34.68% | -0.03% |

Paired daily diff (cost-adjusted): n=66, mean=+0.001 bp/day, sd=+1.48 bp/day, t=+0.006, HAC t=+0.010, 95% CI [-0.0000, +0.0000], 90% CI [-0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.21 bp/day**

## Holdout only (≥ 2026-03-11)

| metric | R-LLM width (deviation: a-priori genome, instrument-failed EA) | apriori (deviation: a-priori genome, instrument-failed EA) | delta |
|---|---|---|---|
| total_return | +3.48% | +3.46% | +0.02% |
| cagr | +21.64% | +21.49% | +0.14% |
| sharpe | +2.4596 | +2.4267 | +0.0329 |
| max_drawdown | +1.87% | +1.92% | -0.05% |
| win_rate | +61.36% | +59.09% | +2.27% |
| realized_round_trips | 110 | 116 | -6 |
| cumulative_transaction_costs | +128.0817 | +114.8568 | +13.2250 |
| avg_gross_exposure | +29.64% | +29.69% | -0.06% |

Paired daily diff (cost-adjusted): n=44, mean=+0.045 bp/day, sd=+1.78 bp/day, t=+0.167, HAC t=+0.297, 95% CI [-0.0000, +0.0000], 90% CI [-0.0000, +0.0000], **per-arm MDE(|t|=2) = +0.30 bp/day**

## Manifests (summaries)

- R-LLM width (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.552, "arm": "orb1", "cost_bps_of_traded": 3.503, "cost_drag_bps_of_start_nav": 36.804, "final_value_cost_adjusted": 109291.98397803213, "final_value_raw": 109662.86, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 238, "sha256_daily_series": "3051063c0d434384099ab7192c4a6a56c029afa86f1be815a93011d21a8c7b2d", "sha256_timeline": "a595c31927937f77eddca66015e4380deba686e7751a9d8deb8e0aaf4ab1e5f3", "start_value": 100769.82, "total_cost_dollars": 370.876, "total_traded_dollars": 1058631.82, "window": "live"}`
- apriori (deviation: a-priori genome, instrument-failed EA) (genome 952a2f5a565e): `{"actions_per_decision_date": 3.672, "arm": "orb1", "cost_bps_of_traded": 3.271, "cost_drag_bps_of_start_nav": 34.581, "final_value_cost_adjusted": 109286.5366010158, "final_value_raw": 109635.0, "first_date": "2026-02-02", "last_date": "2026-06-10", "n_decision_dates": 67, "n_executed_actions": 246, "sha256_daily_series": "6f86c4e397fe63feb668460f1493d89ec51ba0cbddd3ed4e424a8cf429f3c481", "sha256_timeline": "38d3c6096eafe35af654b539c1836bf32ae9b252e836a890c9c33ab1c25bc9a0", "start_value": 100766.47, "total_cost_dollars": 348.4634, "total_traded_dollars": 1065404.41, "window": "live"}`

Forced metric choices carried from TB-006 (stats.run_metrics docstring): round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live record was repaired at points; only paired same-harness comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole (7 td, inside the holdout) are excluded identically in both arms.
