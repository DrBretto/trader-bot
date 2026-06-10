# Evidence comparison — SYN-1 @ cost seed 4244 (R14) vs SYN-1 @ cost seed 4242 (R01)

Holdout boundary: 2026-03-11. Delta = SYN-1 @ cost seed 4244 (R14) − SYN-1 @ cost seed 4242 (R01).
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 @ cost seed 4244 (R14) | SYN-1 @ cost seed 4242 (R01) | delta |
|---|---|---|---|
| total_return | +1.23% | +1.23% | -0.00% |
| cagr | +4.93% | +4.93% | -0.00% |
| sharpe | +1.6541 | +1.6541 | -0.0000 |
| max_drawdown | +1.32% | +1.32% | -0.00% |
| win_rate | +51.56% | +51.56% | +0.00% |
| realized_round_trips | 4 | 4 | 0 |
| cumulative_transaction_costs | +0.0000 | +0.0000 | +0.0000 |
| avg_gross_exposure | +25.72% | +25.72% | +0.00% |

Paired daily diff: n=64, mean=-0.00 bp/day, sd=+0.00 bp/day, t=-0.84, HAC t=-0.87, 95% CI [-0.0000, +0.0000], MDE(|t|=2)=+0.0000

## Holdout only

| metric | SYN-1 @ cost seed 4244 (R14) | SYN-1 @ cost seed 4242 (R01) | delta |
|---|---|---|---|
| total_return | +1.30% | +1.30% | -0.00% |
| cagr | +7.67% | +7.67% | -0.00% |
| sharpe | +2.4980 | +2.4980 | -0.0000 |
| max_drawdown | +0.59% | +0.59% | -0.00% |
| win_rate | +54.55% | +54.55% | +0.00% |
| realized_round_trips | 0 | 0 | 0 |
| cumulative_transaction_costs | +0.0000 | +0.0000 | +0.0000 |
| avg_gross_exposure | +25.71% | +25.71% | +0.00% |

Paired daily diff: n=44, mean=-0.00 bp/day, sd=+0.00 bp/day, t=-1.05, HAC t=-1.33, 95% CI [-0.0000, +0.0000], MDE(|t|=2)=+0.0000

## Manifests

- SYN-1 @ cost seed 4244 (R14): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R14 --cost-seed 4244 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R14/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4244, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:45:01", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.008, "cost_drag_bps_of_start_nav": 2.529, "final_value_cost_adjusted": 104332.26988641941, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 26.0701, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.4, "window": "full"}`
- SYN-1 @ cost seed 4242 (R01): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: SEED SENSITIVITY: raw_value column ASSERTED byte-identical to R01 (strategy path is seed-free); only the post-hoc cost overlay differs. Cost-adjusted endpoints: R14=104332.27 vs R01=104329.46.
