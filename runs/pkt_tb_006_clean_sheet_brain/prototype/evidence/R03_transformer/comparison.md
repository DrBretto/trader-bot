# Evidence comparison — SYN-1 (R01, CAST in slot) vs ridge-in-slot (R03, RT-2)

Holdout boundary: 2026-03-11. Delta = SYN-1 (R01, CAST in slot) − ridge-in-slot (R03, RT-2).
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 (R01, CAST in slot) | ridge-in-slot (R03, RT-2) | delta |
|---|---|---|---|
| total_return | +1.23% | +0.10% | +1.13% |
| cagr | +4.93% | +0.38% | +4.54% |
| sharpe | +1.6541 | +0.2559 | +1.3981 |
| max_drawdown | +1.32% | +0.45% | +0.87% |
| win_rate | +51.56% | +53.12% | -1.56% |
| realized_round_trips | 4 | 8 | -4 |
| cumulative_transaction_costs | +0.0000 | +15.8651 | -15.8651 |
| avg_gross_exposure | +25.72% | +8.24% | +17.48% |

Paired daily diff: n=64, mean=+1.77 bp/day, sd=+16.19 bp/day, t=+0.87, HAC t=+0.89, 95% CI [-0.0002, +0.0006], MDE(|t|=2)=+0.0004

## Holdout only

| metric | SYN-1 (R01, CAST in slot) | ridge-in-slot (R03, RT-2) | delta |
|---|---|---|---|
| total_return | +1.30% | +0.17% | +1.12% |
| cagr | +7.67% | +1.00% | +6.67% |
| sharpe | +2.4980 | +0.5635 | +1.9345 |
| max_drawdown | +0.59% | +0.42% | +0.17% |
| win_rate | +54.55% | +50.00% | +4.55% |
| realized_round_trips | 0 | 3 | -3 |
| cumulative_transaction_costs | +0.0000 | +7.7987 | -7.7987 |
| avg_gross_exposure | +25.71% | +9.68% | +16.03% |

Paired daily diff: n=44, mean=+2.55 bp/day, sd=+16.83 bp/day, t=+1.00, HAC t=+1.13, 95% CI [-0.0002, +0.0007], MDE(|t|=2)=+0.0005

## Manifests

- SYN-1 (R01, CAST in slot): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`
- ridge-in-slot (R03, RT-2): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R03 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R03/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out_rt2 --nightly-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/store/nightly_rt2_ridge_slot --exec-mode learned", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "learned", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out_rt2", "generated": "2026-06-10T18:44:12", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 1, "code_sha": "3edd62e80aa1", "generated": "2026-06-10T18:03:53", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.215, "arm": "syn1", "cost_bps_of_traded": 3.341, "cost_drag_bps_of_start_nav": 4.368, "final_value_cost_adjusted": 103156.66853378685, "final_value_raw": 103201.7, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 14, "start_value": 103085.44, "total_cost_dollars": 45.0315, "total_traded_dollars": 134790.86, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.4, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: ORIENTATION: E2 = R01 − R03 (with-CAST minus ridge twin in slot, executive re-fit RT-2). E1 read: evidence/e1_reads.json['transformer'] (with=syn1_frozen_base, without=ridge_in_slot_rt2).

## E1 primary read

HAC t=-0.28 on n=1507 pooled fold-days, mean=-0.0000, CI95=[-0.0000, +0.0000], MDE=+0.0000
