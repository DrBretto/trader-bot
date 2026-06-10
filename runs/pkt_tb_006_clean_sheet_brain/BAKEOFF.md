# BAKEOFF.md — SYN-1 vs incumbent (PKT-TB-006, mechanical assembly)

Generated 2026-06-10T19:02:16 by `prototype/assemble_evidence.py` driving
`prototype/report_evidence.py` + `prototype/stats.py` (pure pre-registered
functions). Every number below traces to a cited file. The incumbent appears
ONLY as its score line and harness wiring (packet rule; ASSIGNMENT_BRIEF §1).

## 1. Window, seeding, harness identity

- Replay window: snapshots 2026-02-03 -> 2026-06-10; first/last daily
  values 2026-02-04 -> 2026-06-10 (65 decision dates). Holdout
  boundary 2026-03-11 (TOURNAMENT §4.1); holdout-only is the VERDICT read.
- Harness: identical engine + snapshots for both arms — code SHA git `747c58f8f640`,
  prototype wiring `3edd62e80aa1`; cost model `transaction_costs`
  version `bf8b77716f74`, post-hoc cost overlay seed 4242 (both arms;
  paired stats require identical slippage draws, §4.1). Incumbent run command and
  wiring: `runs_battery/R02/manifest.json`.
- SYN-1 configuration: the ONE frozen config (FREEZE_SYN1.md) — EA champion genome
  `ea/genome_2026-02-06.json` (hash `8f4184844e8b`), exec-mode `linear_twin`
  (frozen ladder rung), sigma source `trailing21`. Variant params hash
  `3023921c7bc7`.
- Rung decisions stated with the results (FREEZE_SYN1.md): champion genome adopted
  (adoption gate: margin +0.876 > 1.0 x cross-fold sd 0.638); executive = linear twin
  (MLP 5-seed val 1.688e-4 did NOT beat twin 1.701e-4); fine-tune none (delta exactly 0,
  de-claimed); Transfer-B CAST weighted / GBM uniform; Transfer-A conjunctive gate DEAD,
  R3-only screening ships; LLM ships (Stage-1 pass w/ chattiness partial-fail note);
  G1 dictionary `0 (measured)` per placebo FAIL.

## 2. §4.2 numbers — cost-adjusted (verdict series)

### Full period (context, never the verdict)

| metric | SYN-1 (R01) | incumbent (R02) | delta (SYN-1 − incumbent) |
|---|---|---|---|
| total_return | +1.23% | +8.46% | -7.23% |
| cagr | +4.93% | +37.67% | -32.74% |
| sharpe | +1.6541 | +3.2709 | -1.6168 |
| max_drawdown | +1.32% | +2.78% | -1.46% |
| win_rate | +51.56% | +54.69% | -3.12% |
| realized_round_trips | 4 | 74 | -70 |
| cumulative_transaction_costs | +0.0000 | +353.2172 | -353.2172 |
| avg_gross_exposure | +25.72% | +38.13% | -12.41% |

- **Paired daily diff, full period (cost-adjusted):** n=64, mean=-10.96 bp/day, sd=60.29 bp/day, t=-1.45, HAC t=-1.32, 95% CI [-0.002724, +0.000533], MDE(|t|=2)=+0.001662 (+16.62 bp/day)
- **Paired daily diff, full period (RAW, alongside):** n=64, mean=-11.45 bp/day, sd=60.04 bp/day, t=-1.53, HAC t=-1.38, 95% CI [-0.002776, +0.000486], MDE(|t|=2)=+0.001665 (+16.65 bp/day)

### Holdout only (>= 2026-03-11) — the verdict read

| metric | SYN-1 (R01) | incumbent (R02) | delta (SYN-1 − incumbent) |
|---|---|---|---|
| total_return | +1.30% | +4.98% | -3.68% |
| cagr | +7.67% | +32.07% | -24.39% |
| sharpe | +2.4980 | +2.8869 | -0.3889 |
| max_drawdown | +0.59% | +1.95% | -1.36% |
| win_rate | +54.55% | +59.09% | -4.55% |
| realized_round_trips | 0 | 42 | -42 |
| cumulative_transaction_costs | +0.0000 | +132.1051 | -132.1051 |
| avg_gross_exposure | +25.71% | +36.50% | -10.78% |

- **Paired daily diff, holdout (cost-adjusted) — PRIMARY:** n=44, mean=-8.28 bp/day, sd=59.50 bp/day, t=-0.92, HAC t=-0.80, 95% CI [-0.002857, +0.001202], MDE(|t|=2)=+0.002071 (+20.71 bp/day)
- **Paired daily diff, holdout (RAW, alongside):** n=44, mean=-8.52 bp/day, sd=59.35 bp/day, t=-0.95, HAC t=-0.83, 95% CI [-0.002876, +0.001172], MDE(|t|=2)=+0.002066 (+20.66 bp/day)

## 3. Verdict (§4.2 rule, computed by stats.beats_ties_loses)

**BRAIN vs INCUMBENT: TIES** — holdout paired t=-0.92 (rule: BEATS iff
t>=+1.0 AND dSharpe>0; LOSES iff t<=-1.0; TIES iff |t|<1.0), holdout
dSharpe=-0.39, holdout dReturn=-3.68% (endpoint context).

Honesty line (§4.2, printed with the verdict): at t=+-1 the per-comparison false-call probability under the null ~ 16% one-sided; holdout MDE at t=2 ~ 5 bp/day ~ dSharpe_ann ~ 4.0 — the bake-off verdict is a sign-grade read; full-period paired stats are context, never the verdict.
This holdout's computed MDE at |t|=2 is +20.71 bp/day on n=44 days.

## 4. Turnover + cost drag, both arms (manifest summaries)

| arm | executed actions | actions/decision-date | traded $ | cost $ | drag bps of start NAV | cost bps of traded |
|---|---|---|---|---|---|---|
| SYN-1 (R01) | 4 | 0.062 | 86,670 | 28.88 | 2.80 | 3.33 |
| incumbent (R02) | 122 | 1.877 | 1,097,908 | 364.27 | 35.38 | 3.32 |

(Sources: `runs_battery/R01/manifest.json`, `runs_battery/R02/manifest.json`.)

## 5. Equity curve (sampled every 5th common date, cost-adjusted)

| date | SYN-1 cost-adj | incumbent cost-adj |
|---|---|---|
| 2026-02-04 | 103,062.37 | 102,960.02 |
| 2026-02-12 | 103,372.10 | 103,411.23 |
| 2026-02-20 | 103,610.56 | 106,919.11 |
| 2026-03-03 | 103,487.77 | 107,281.58 |
| 2026-03-11 | 102,991.49 | 106,374.52 |
| 2026-03-19 | 102,603.37 | 104,568.93 |
| 2026-03-27 | 102,441.41 | 104,487.43 |
| 2026-04-09 | 103,021.08 | 106,524.39 |
| 2026-04-17 | 103,190.20 | 111,373.17 |
| 2026-04-28 | 103,345.81 | 110,307.67 |
| 2026-05-06 | 103,742.93 | 112,486.03 |
| 2026-05-26 | 104,530.50 | 112,382.90 |
| 2026-06-03 | 104,332.37 | 112,629.41 |
| 2026-06-10 | 104,329.46 | 111,667.94 |

(Full series: `runs_battery/R01/daily_series.csv`, `runs_battery/R02/daily_series.csv`.)

## 6. Seed sensitivity (R13/R14 vs R01)

- ASSERTION RUN: R13 and R14 `raw_value` columns are byte-identical to R01's
  (checked by `assemble_evidence.py`; the strategy path consumes no slippage seed).
- Cost-overlay spread across master seeds 4242/4243/4244:

| run | cost seed | final raw | final cost-adj | total cost $ | drag bps of start NAV |
|---|---|---|---|---|---|
| R01 | 4242 | 104,358.34 | 104,329.46 | 28.88 | 2.80 |
| R13 | 4243 | 104,358.34 | 104,331.95 | 26.39 | 2.56 |
| R14 | 4244 | 104,358.34 | 104,332.27 | 26.07 | 2.53 |

Comparison artifacts: `evidence/R13_seed4243/`, `evidence/R14_seed4244/`.

## 7. Findings logged by this assembly

- R13 raw_value column byte-identical to R01 (ASSERTED) — only the cost overlay (seed 4243) differs.
- R14 raw_value column byte-identical to R01 (ASSERTED) — only the cost overlay (seed 4244) differs.
- FINDING (R07): daily_series.csv byte-identical to R01 — the arm ran --sigma-source trailing21, which IS the deployed FREEZE convention; the E2 contrast is degenerate (diff ≡ 0, paired t undefined). R07 stays in the record as a consumed look (it measured the null contrast by construction); the RiskNet E2 read is sourced from contingency replay R19 (--sigma-source risknet), candidate-minus-deployed orientation.
- Holdout paired n=44 daily diffs (45 holdout dates), not the §4.1 ~62-day estimate: the full-window snapshot store yields 65 trading dates 2026-02-04->2026-06-10 (`runs_battery/R01/manifest.json` n_decision_dates; snapshot gap 2026-05-11->05-22 journaled in RUN_JOURNAL.md). The printed MDE is computed on the actual n.
- Metric artifact (forced choice, stats.run_metrics docstring): cumulative_transaction_costs reads 0.00 for R01 in both slices because its only fills occur at the first daily mark — the drag is embedded in v[0] of the cost-adjusted series. The traceable dollar cost is the manifest total_cost_dollars ($28.88 R01 / $364.27 R02), printed in §4 below.

## 8. Manifests

- SYN-1: `runs_battery/R01/manifest.json` (config_hash ed35d0a7cb616e40)
- incumbent: `runs_battery/R02/manifest.json` (config_hash 5bf392338d9f4c19)
- Full comparison JSON/MD: `evidence/R01_vs_R02_bakeoff/`
- Ledgers: `prototype/holdout_looks.jsonl` (15 looks),
  `prototype/validation_looks.jsonl` (78 decisions)
