# Evidence comparison — SYN-1 (R01) vs GBM-Cond OFF, trust renorm (R05)

Holdout boundary: 2026-03-11. Delta = SYN-1 (R01) − GBM-Cond OFF, trust renorm (R05).
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 (R01) | GBM-Cond OFF, trust renorm (R05) | delta |
|---|---|---|---|
| total_return | +1.23% | +8.95% | -7.72% |
| cagr | +4.93% | +40.12% | -35.19% |
| sharpe | +1.6541 | +2.0504 | -0.3963 |
| max_drawdown | +1.32% | +8.22% | -6.90% |
| win_rate | +51.56% | +59.38% | -7.81% |
| realized_round_trips | 4 | 0 | 4 |
| cumulative_transaction_costs | +0.0000 | +0.0000 | +0.0000 |
| avg_gross_exposure | +25.72% | +109.32% | -83.60% |

Paired daily diff: n=64, mean=-12.05 bp/day, sd=+94.53 bp/day, t=-1.02, HAC t=-0.94, 95% CI [-0.0037, +0.0013], MDE(|t|=2)=+0.0026

## Holdout only

| metric | SYN-1 (R01) | GBM-Cond OFF, trust renorm (R05) | delta |
|---|---|---|---|
| total_return | +1.30% | +10.71% | -9.41% |
| cagr | +7.67% | +79.12% | -71.45% |
| sharpe | +2.4980 | +3.4830 | -0.9850 |
| max_drawdown | +0.59% | +3.86% | -3.27% |
| win_rate | +54.55% | +59.09% | -4.55% |
| realized_round_trips | 0 | 0 | 0 |
| cumulative_transaction_costs | +0.0000 | +0.0000 | +0.0000 |
| avg_gross_exposure | +25.71% | +109.28% | -83.57% |

Paired daily diff: n=44, mean=-20.78 bp/day, sd=+95.36 bp/day, t=-1.45, HAC t=-1.68, 95% CI [-0.0045, +0.0003], MDE(|t|=2)=+0.0025

## Manifests

- SYN-1 (R01): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`
- GBM-Cond OFF, trust renorm (R05): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R05 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R05/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:21", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 0, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "d8ec87414698", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.0, "arm": "syn1", "cost_bps_of_traded": null, "cost_drag_bps_of_start_nav": 0.0, "final_value_cost_adjusted": 112722.48, "final_value_raw": 112722.48, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 0, "start_value": 103466.89, "total_cost_dollars": 0.0, "total_traded_dollars": 0.0, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.5, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: ORIENTATION: E2 = R01 − R05 (with-member minus member-dropped). E1 read: evidence/e1_reads.json['ensemble_gbm_drop'] (with=syn1_frozen_base, without=gbm_off_renorm).

## E1 primary read

HAC t=+2.48 on n=1507 pooled fold-days, mean=+0.0001, CI95=[+0.0000, +0.0002], MDE=+0.0001
