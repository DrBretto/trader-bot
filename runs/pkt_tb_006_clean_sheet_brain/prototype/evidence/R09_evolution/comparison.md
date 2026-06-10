# Evidence comparison — SYN-1 (R01, champion genome) vs B0 DEFAULT_GENOME (R09)

Holdout boundary: 2026-03-11. Delta = SYN-1 (R01, champion genome) − B0 DEFAULT_GENOME (R09).
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 (R01, champion genome) | B0 DEFAULT_GENOME (R09) | delta |
|---|---|---|---|
| total_return | +1.23% | +4.62% | -3.39% |
| cagr | +4.93% | +19.46% | -14.53% |
| sharpe | +1.6541 | +2.1659 | -0.5119 |
| max_drawdown | +1.32% | +4.04% | -2.72% |
| win_rate | +51.56% | +54.69% | -3.12% |
| realized_round_trips | 4 | 23 | -19 |
| cumulative_transaction_costs | +0.0000 | +21.2231 | -21.2231 |
| avg_gross_exposure | +25.72% | +65.27% | -39.55% |

Paired daily diff: n=64, mean=-5.27 bp/day, sd=+39.12 bp/day, t=-1.08, HAC t=-0.92, 95% CI [-0.0016, +0.0006], MDE(|t|=2)=+0.0011

## Holdout only

| metric | SYN-1 (R01, champion genome) | B0 DEFAULT_GENOME (R09) | delta |
|---|---|---|---|
| total_return | +1.30% | +5.12% | -3.82% |
| cagr | +7.67% | +33.09% | -25.42% |
| sharpe | +2.4980 | +3.1268 | -0.6288 |
| max_drawdown | +0.59% | +2.63% | -2.04% |
| win_rate | +54.55% | +56.82% | -2.27% |
| realized_round_trips | 0 | 15 | -15 |
| cumulative_transaction_costs | +0.0000 | +13.3271 | -13.3271 |
| avg_gross_exposure | +25.71% | +72.08% | -46.37% |

Paired daily diff: n=44, mean=-8.57 bp/day, sd=+44.58 bp/day, t=-1.27, HAC t=-1.23, 95% CI [-0.0022, +0.0005], MDE(|t|=2)=+0.0014

## Manifests

- SYN-1 (R01, champion genome): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`
- B0 DEFAULT_GENOME (R09): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R09 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R09/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:39", "genome": {"abstain_threshold": 0.1, "cash_floor": 0.1, "conviction_temp": 1.0, "dd_brake_strength": 0.5, "dd_brake_threshold": 0.1, "event_weight_cap": 1.0, "feature_gate": [1, 1, 1, 1, 1, 1, 1, 1], "gross_target": 0.6, "max_symbol_weight": 0.08, "member_gate": [1, 1, 1], "no_trade_band": 0.01, "record_weight_eps": 0.25, "risk_aversion_lambda": 2.0, "trust_halflife_days": 21.0, "trust_prior": [0.0, 0.0, 0.0], "vol_target_ann": 0.1}, "genome_hash": "c09995de90f5", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 1.554, "arm": "syn1", "cost_bps_of_traded": 2.567, "cost_drag_bps_of_start_nav": 4.704, "final_value_cost_adjusted": 107778.92771650846, "final_value_raw": 107827.4, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 101, "start_value": 103046.68, "total_cost_dollars": 48.4723, "total_traded_dollars": 188798.01, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 5.0, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: ORIENTATION: E2 = R01 − R09 (champion minus B0; the EA proposal B2 read). E1 read: evidence/e1_reads.json['evolution'] (with=syn1_frozen_base, without=b0_default_genome).

## E1 primary read

HAC t=-0.96 on n=1507 pooled fold-days, mean=-0.0001, CI95=[-0.0003, +0.0001], MDE=+0.0002
