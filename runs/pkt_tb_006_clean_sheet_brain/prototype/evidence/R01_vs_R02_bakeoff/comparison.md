# Evidence comparison — SYN-1 vs incumbent

Holdout boundary: 2026-03-11. Delta = SYN-1 − incumbent.
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 | incumbent | delta |
|---|---|---|---|
| total_return | +1.23% | +8.46% | -7.23% |
| cagr | +4.93% | +37.67% | -32.74% |
| sharpe | +1.6541 | +3.2709 | -1.6168 |
| max_drawdown | +1.32% | +2.78% | -1.46% |
| win_rate | +51.56% | +54.69% | -3.12% |
| realized_round_trips | 4 | 74 | -70 |
| cumulative_transaction_costs | +0.0000 | +353.2172 | -353.2172 |
| avg_gross_exposure | +25.72% | +38.13% | -12.41% |

Paired daily diff: n=64, mean=-10.96 bp/day, sd=+60.29 bp/day, t=-1.45, HAC t=-1.32, 95% CI [-0.0027, +0.0005], MDE(|t|=2)=+0.0017

## Holdout only

| metric | SYN-1 | incumbent | delta |
|---|---|---|---|
| total_return | +1.30% | +4.98% | -3.68% |
| cagr | +7.67% | +32.07% | -24.39% |
| sharpe | +2.4980 | +2.8869 | -0.3889 |
| max_drawdown | +0.59% | +1.95% | -1.36% |
| win_rate | +54.55% | +59.09% | -4.55% |
| realized_round_trips | 0 | 42 | -42 |
| cumulative_transaction_costs | +0.0000 | +132.1051 | -132.1051 |
| avg_gross_exposure | +25.71% | +36.50% | -10.78% |

Paired daily diff: n=44, mean=-8.28 bp/day, sd=+59.50 bp/day, t=-0.92, HAC t=-0.80, 95% CI [-0.0029, +0.0012], MDE(|t|=2)=+0.0021

## Manifests

- SYN-1: `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`
- incumbent: `{"arm": "incumbent", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm incumbent --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R02 --cost-seed 4242", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": null, "exec_weights_dir": null, "generated": "2026-06-10T18:44:08", "genome": null, "genome_hash": null, "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": null, "nightly_manifest": null, "sigma_source": null, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 1.877, "arm": "incumbent", "cost_bps_of_traded": 3.318, "cost_drag_bps_of_start_nav": 35.376, "final_value_cost_adjusted": 111667.94454783354, "final_value_raw": 112032.21, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 122, "start_value": 102971.07, "total_cost_dollars": 364.2655, "total_traded_dollars": 1097907.87, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 2.9, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: BAKE-OFF (§4.2). Verdict series = cost-adjusted; raw paired stats printed in BAKEOFF.md. Incumbent appears as score line + harness wiring only.
