# Evidence comparison — SYN-1 (R01) vs LLM neutral-constants retrain (R10, RT-3)

Holdout boundary: 2026-03-11. Delta = SYN-1 (R01) − LLM neutral-constants retrain (R10, RT-3).
Paired daily-difference stats on identical dates are the required
form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).

## Full period

| metric | SYN-1 (R01) | LLM neutral-constants retrain (R10, RT-3) | delta |
|---|---|---|---|
| total_return | +1.23% | +2.67% | -1.44% |
| cagr | +4.93% | +10.91% | -5.99% |
| sharpe | +1.6541 | +2.3541 | -0.7000 |
| max_drawdown | +1.32% | +2.08% | -0.76% |
| win_rate | +51.56% | +59.38% | -7.81% |
| realized_round_trips | 4 | 4 | 0 |
| cumulative_transaction_costs | +0.0000 | +1.4146 | -1.4146 |
| avg_gross_exposure | +25.72% | +37.16% | -11.44% |

Paired daily diff: n=64, mean=-2.22 bp/day, sd=+13.17 bp/day, t=-1.35, HAC t=-1.17, 95% CI [-0.0006, +0.0001], MDE(|t|=2)=+0.0004

## Holdout only

| metric | SYN-1 (R01) | LLM neutral-constants retrain (R10, RT-3) | delta |
|---|---|---|---|
| total_return | +1.30% | +2.86% | -1.57% |
| cagr | +7.67% | +17.56% | -9.89% |
| sharpe | +2.4980 | +3.3574 | -0.8594 |
| max_drawdown | +0.59% | +1.14% | -0.55% |
| win_rate | +54.55% | +61.36% | -6.82% |
| realized_round_trips | 0 | 0 | 0 |
| cumulative_transaction_costs | +0.0000 | +1.0154 | -1.0154 |
| avg_gross_exposure | +25.71% | +40.51% | -14.80% |

Paired daily diff: n=44, mean=-3.52 bp/day, sd=+15.57 bp/day, t=-1.50, HAC t=-1.54, 95% CI [-0.0008, +0.0001], MDE(|t|=2)=+0.0005

## Manifests

- SYN-1 (R01): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R01/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out --exec-mode linear_twin", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "linear_twin", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out", "generated": "2026-06-10T18:44:04", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 1, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "8f4184844e8b", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "e620dc0e34a6", "generated": "2026-06-10T17:35:57", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.062, "arm": "syn1", "cost_bps_of_traded": 3.332, "cost_drag_bps_of_start_nav": 2.801, "final_value_cost_adjusted": 104329.4649138139, "final_value_raw": 104358.34, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 4, "start_value": 103091.25, "total_cost_dollars": 28.8751, "total_traded_dollars": 86670.31, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.9, "window": "full"}`
- LLM neutral-constants retrain (R10, RT-3): `{"arm": "syn1", "code_sha": {"git": "747c58f8f640", "prototype_wiring": "3edd62e80aa1"}, "command": "python /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py --arm syn1 --window full --out /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R10 --cost-seed 4242 --genome /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/runs_battery/R10/genome.json --exec-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out_rt3 --nightly-dir /Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/store/nightly_rt3_llm_neutral --exec-mode learned", "cost_model": {"seed": 4242, "version_sha": "bf8b77716f74"}, "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1", "exec_mode": "learned", "exec_weights_dir": "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype/exec_out_rt3", "generated": "2026-06-10T18:44:44", "genome": {"abstain_threshold": 0.2443899204544654, "cash_floor": 0.1031998699462734, "conviction_temp": 4.0, "dd_brake_strength": 1.0, "dd_brake_threshold": 0.08001844143783443, "event_weight_cap": 0.8795265838631618, "feature_gate": [0, 1, 0, 1, 0, 1, 1, 1], "gross_target": 0.5157883065754484, "max_symbol_weight": 0.08175373267203596, "member_gate": [1, 1, 1], "no_trade_band": 0.02596656629933275, "record_weight_eps": 0.38223385230068074, "risk_aversion_lambda": 3.477283678899133, "trust_halflife_days": 41.740361368232826, "trust_prior": [-0.7976769586105441, 1.8441695577251007, -1.429998600799269], "vol_target_ann": 0.08648335911431486}, "genome_hash": "c5bd382bfa6a", "n_trading_dates": 84, "native_start_portfolio_date": "2026-03-11", "nightly_audit": {"dir": "nightly_audit", "n_files": 130}, "nightly_manifest": {"cast_n_seeds": 3, "code_sha": "3edd62e80aa1", "generated": "2026-06-10T18:03:55", "window": ["2026-02-02", "2026-06-09"]}, "sigma_source": {"book_vol_key": "book_vol_hat_proxy", "mode": "trailing21", "sigma_col": "sigma_hat_proxy"}, "snapshot_range": ["2026-02-03", "2026-06-10"], "start_portfolio_date_used": "2026-02-03", "summary": {"actions_per_decision_date": 0.154, "arm": "syn1", "cost_bps_of_traded": 2.973, "cost_drag_bps_of_start_nav": 2.919, "final_value_cost_adjusted": 105810.36274766688, "final_value_raw": 105840.46, "first_date": "2026-02-04", "last_date": "2026-06-10", "n_decision_dates": 65, "n_executed_actions": 10, "start_value": 103091.59, "total_cost_dollars": 30.0973, "total_traded_dollars": 101239.69, "window": "full"}, "variant_params_hash": "3023921c7bc7", "wall_clock_s": 3.6, "window": "full"}`

Forced metric choices: round trips = executed SELL/REDUCE count; cumulative costs = raw-minus-costadj value drag; gross exposure from timeline ending_value/ending_cash (stats.run_metrics docstring).

NOTE: ORIENTATION: E2 = R01 − R10 (with-LLM minus llm_* neutralized + RT-3 re-fit). E1 read: evidence/e1_reads.json['llm'] (with=syn1_frozen_base, without=llm_neutral_rt3).

## E1 primary read

HAC t=-1.08 on n=1507 pooled fold-days, mean=-0.0000, CI95=[-0.0000, +0.0000], MDE=+0.0000
