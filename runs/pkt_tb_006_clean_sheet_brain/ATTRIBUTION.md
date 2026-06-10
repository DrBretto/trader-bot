# ATTRIBUTION.md — per-organ matrix (PKT-TB-006, mechanical assembly)

Generated 2026-06-10T19:02:16 by `prototype/assemble_evidence.py`. Verdicts computed
exclusively by `stats.organ_verdict` (§4.3 three-valued rule); E1 numbers from
`prototype/evidence/e1_reads.json` (pooled F1-F6 OOF daily utility difference, HAC
t Newey-West 10 lags, ~1,507 fold-days); E2 sign reads from the holdout-only
(>= 2026-03-11) cost-adjusted paired comparisons in `prototype/evidence/<run_id>/`.

## Summary table

| organ | arm | E1 HAC t | E1 mean bp/day | E1 95% CI (daily) | E2 holdout dSharpe | E2 paired t | verdict |
|---|---|---|---|---|---|---|---|
| transformer | R03 | -0.28 | -0.06 | [-0.000045, +0.000034] | +1.93 | +1.00 | **0 (measured)** |
| ensemble_cast_drop | R04 | +0.17 | +0.09 | [-0.000094, +0.000112] | +2.04 | +1.22 | **0 (measured)** |
| ensemble_gbm_drop | R05 | +2.48 | +1.11 | [+0.000023, +0.000199] | -0.98 | -1.45 | **indeterminate** |
| ensemble_event_drop | R06 | +0.36 | +0.06 | [-0.000026, +0.000038] | +0.54 | -0.32 | **0 (measured)** |
| risknet | R19 | +1.39 | +0.43 | [-0.000018, +0.000104] | +0.00 | +0.87 | **indeterminate** |
| executive | R08 | -1.02 | -0.53 | [-0.000153, +0.000048] | -0.94 | -1.47 | **indeterminate** |
| evolution | R09 | -0.96 | -0.86 | [-0.000262, +0.000090] | -0.63 | -1.27 | **0 (measured)** |
| llm | R10 | -1.08 | -0.12 | [-0.000034, +0.000010] | -0.86 | -1.50 | **indeterminate** |
| gdelt | R11 | -1.79 | -0.52 | [-0.000108, +0.000005] | -1.00 | -1.59 | **indeterminate** |
| infotropy_b | R12 | -0.40 | -0.11 | [-0.000063, +0.000042] | -0.57 | -1.19 | **0 (measured)** |
| infotropy_a | — (E1-only) | +0.88 | +0.09 | [-0.000010, +0.000028] | n/a | n/a | **0 (measured)** |

E2 reads on n=44 holdout days carry MDE ~+20.71 bp/day at |t|=2 —
sign-confirmation only (§4.3 hierarchy is fixed: E1 primary, E2 sign).

## Assignment-item mapping (ASSIGNMENT_BRIEF §1 -> scorecard organs)

- **Item 1** (ensemble of genuinely different ML model types (incl. a transformer doing real work)): transformer (R03) + ensemble members cast/gbm_cond/event_head (R04-R06) + risknet (R19; R07 degenerate)
- **Item 2** (evolutionary algorithm as the balancing organ): evolution (R09 + B0/B1/adoption gate)
- **Item 3** (LLM doing real sentiment analysis): LLM (R10, RT-3)
- **Item 4** (GDELT as a load-bearing differentiated data source): GDELT (R11, RT-4 + shift gate + placebo)
- **Item 5** (learned meta-evaluator (the executive)): meta-evaluator (R08 + diversity floor + ladder)
- **Item 6** (infotropy angle examined): infotropy (R12 Transfer-B; Transfer-A fold-only read)

## Multiplicity (§4.5, verbatim)

"7 scorecard organs were each given one pre-registered arm. At the E2 holdout sign bar (t >= 1) the per-arm false-positive rate under the null is ~ 16%; expected false 'carries weight' sign-confirmations across 7 organs ~ 1.1; family-wise P(>=1 false positive) ~ 70%. That is why E2 is sign-confirmation only. At the E1 primary bar (HAC |t| >= 2 on ~1,500 pooled fold-days) the per-arm FPR is ~ 5%; expected false positives across 7 organs ~ 0.35. The holdout MDE is ~ 5 bp/day (dSharpe_ann ~ 4 at t=2); no claimed effect is that large, and no holdout number below it is treated as certified. This battery consumed 15 holdout looks and 78 validation-fold decisions (ledgers below)."

## Per-organ detail


### transformer (R03; final-line key `transformer`)

- **E1 (primary, §4.3):** HAC t=-0.28 on n=1507 pooled fold-days, mean=-0.06 bp/day, 95% CI [-0.000045, +0.000034], MDE=+0.000040 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=-1.21; F2: t=+0.73; F3: t=-0.30; F4: t=+1.05; F5: t=-1.20; F6: t=-0.27
- **E2 (sign confirmation):** holdout dSharpe=+1.93, paired mean=+2.55 bp/day, paired t=+1.00, n=44 (`evidence/R03_transformer/comparison.json`)
- orientation: E2 = R01 − R03 (with-CAST minus ridge twin in slot, executive re-fit RT-2)
- **verdict: 0 (measured)** — E1 |t|=0.28 < 1.0 (stats.organ_verdict)
- provenance: retrain RT-2 (ridge + executive re-fit); exec rung shipped: mlp (1.497e-4 > twin 1.254e-4) (`exec_out_rt2/ladder.json`)
- SPECIAL PRINT: CAST ships regardless — item-1 transformer presence is mandated; the scorecard prints the measured attribution (§4.3 row 1).

### ensemble_cast_drop (R04; final-line key `ensemble`, member `cast`)

- **E1 (primary, §4.3):** HAC t=+0.17 on n=1507 pooled fold-days, mean=+0.09 bp/day, 95% CI [-0.000094, +0.000112], MDE=+0.000105 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+0.26; F2: t=-0.10; F3: t=+0.54; F4: t=+1.01; F5: t=-0.92; F6: t=-1.19
- **E2 (sign confirmation):** holdout dSharpe=+2.04, paired mean=+2.59 bp/day, paired t=+1.22, n=44 (`evidence/R04_ensemble_cast/comparison.json`)
- orientation: E2 = R01 − R04 (with-member minus member-dropped)
- **verdict: 0 (measured)** — E1 |t|=0.17 < 1.0 (stats.organ_verdict)
- provenance: retrain none (member gate off, K17); exec rung shipped: linear_twin (base) (`exec_out/ladder.json`)

### ensemble_gbm_drop (R05; final-line key `ensemble`, member `gbm_cond`)

- **E1 (primary, §4.3):** HAC t=+2.48 on n=1507 pooled fold-days, mean=+1.11 bp/day, 95% CI [+0.000023, +0.000199], MDE=+0.000090 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+1.45; F2: t=+0.53; F3: t=n/a; F4: t=+1.61; F5: t=+1.35; F6: t=+1.89
- **E2 (sign confirmation):** holdout dSharpe=-0.98, paired mean=-20.78 bp/day, paired t=-1.45, n=44 (`evidence/R05_ensemble_gbm/comparison.json`)
- orientation: E2 = R01 − R05 (with-member minus member-dropped)
- **verdict: indeterminate** — E1 t=+2.48 >= +2.0 but E2 holdout sign disagrees/unavailable (stats.organ_verdict)
- provenance: retrain none (member gate off, K17); exec rung shipped: linear_twin (base) (`exec_out/ladder.json`)

### ensemble_event_drop (R06; final-line key `ensemble`, member `event_head`)

- **E1 (primary, §4.3):** HAC t=+0.36 on n=1507 pooled fold-days, mean=+0.06 bp/day, 95% CI [-0.000026, +0.000038], MDE=+0.000033 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+0.70; F2: t=-1.40; F3: t=-0.71; F4: t=+0.43; F5: t=-0.62; F6: t=+1.73
- **E2 (sign confirmation):** holdout dSharpe=+0.54, paired mean=-0.76 bp/day, paired t=-0.32, n=44 (`evidence/R06_ensemble_event/comparison.json`)
- orientation: E2 = R01 − R06 (with-member minus member-dropped)
- **verdict: 0 (measured)** — E1 |t|=0.36 < 1.0 (stats.organ_verdict)
- provenance: retrain none (member gate off, K17); exec rung shipped: linear_twin (base) (`exec_out/ladder.json`)

### risknet (R19; final-line key `ensemble`, member `risknet`)

- **E1 (primary, §4.3):** HAC t=+1.39 on n=1507 pooled fold-days, mean=+0.43 bp/day, 95% CI [-0.000018, +0.000104], MDE=+0.000062 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+1.54; F2: t=+1.30; F3: t=-0.08; F4: t=+1.61; F5: t=+0.51; F6: t=+2.01
- **E2 (sign confirmation):** holdout dSharpe=+0.00, paired mean=+0.00 bp/day, paired t=+0.87, n=44 (`evidence/R19_ensemble_risknet/comparison.json`)
- orientation: E2 = R19 − R01 (candidate minus deployed, matching the e1_reads DIRECTION note; positive flatters the CANDIDATE, not the shipped SYN-1 config. R19 is the contingency replay repairing degenerate R07, whose --sigma-source trailing21 IS the deployed convention -> null contrast by construction)
- **verdict: indeterminate** — 1.0 <= |E1 t|=1.39 < 2.0 (stats.organ_verdict)
- provenance: retrain none (sigma swap, contingency); exec rung shipped: linear_twin (base) (`exec_out/ladder.json`)
- SPECIAL PRINT: planned arm R07 was DEGENERATE (series byte-identical to R01; --sigma-source trailing21 == deployed convention -> it measured the null contrast by construction). R07 stays in the record as a consumed look (`evidence/R07_ensemble_risknet/`); the E2 read above comes from contingency replay R19 (--sigma-source risknet, reserved slot per TOURNAMENT §4.4), candidate-minus-deployed orientation.

### executive (R08; final-line key `meta-evaluator`)

- **E1 (primary, §4.3):** HAC t=-1.02 on n=1507 pooled fold-days, mean=-0.53 bp/day, 95% CI [-0.000153, +0.000048], MDE=+0.000103 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=-0.35; F2: t=-0.90; F3: t=+1.30; F4: t=+0.85; F5: t=-1.39; F6: t=-0.79
- **E2 (sign confirmation):** holdout dSharpe=-0.94, paired mean=-5.50 bp/day, paired t=-1.47, n=44 (`evidence/R08_meta_evaluator/comparison.json`)
- orientation: E2 = R01 − R08 (learned executive minus fixed rule)
- **verdict: indeterminate** — 1.0 <= |E1 t|=1.02 < 2.0 (stats.organ_verdict)
- provenance: retrain none (executive bypass); exec rung shipped: equal_trust bypass (tau=1/M, f=0.7) (`runs_battery/R08/manifest.json`)
- SPECIAL PRINT (diversity floor, §4.6.5): mean pairwise solo-book corr = 0.941 >= 0.90 -> item-5 trust attribution expected ~0 by construction; the equal-trust tie is the honest outcome (`store/step10_checks.json`).
- SPECIAL PRINT (rung): linear twin shipped — MLP 5-seed val util 0.0001688 did NOT beat twin 0.0001701 (`exec_out/final_diagnostics.json`, `exec_out/ladder.json`).
- SPECIAL PRINT (proposal §7 kill criteria, per-fold per §4.6.10): static-trust falsifier FIRES — std tau < 0.02 in every fold (max 0.0029); calibration corr = -0.281 <= 0 (kill criterion fires); challenger parity at the ladder (twin >= MLP). All three reported, none reinterpreted.

### evolution (R09; final-line key `evolution`)

- **E1 (primary, §4.3):** HAC t=-0.96 on n=1507 pooled fold-days, mean=-0.86 bp/day, 95% CI [-0.000262, +0.000090], MDE=+0.000180 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=-0.22; F2: t=-1.40; F3: t=+0.05; F4: t=+0.67; F5: t=-1.64; F6: t=-0.91
- **E2 (sign confirmation):** holdout dSharpe=-0.63, paired mean=-8.57 bp/day, paired t=-1.27, n=44 (`evidence/R09_evolution/comparison.json`)
- orientation: E2 = R01 − R09 (champion minus B0; the EA proposal B2 read)
- **verdict: 0 (measured)** — E1 |t|=0.96 < 1.0 (stats.organ_verdict)
- provenance: retrain none (genome swap); exec rung shipped: linear_twin (base) (`exec_out/ladder.json`)
- SPECIAL PRINT (EA proposal pre-committed sentences, verbatim rules): (1) carries-weight test: champion vs B0 E1 t=-0.96 (champion does NOT clear cross-fold ΔU>0 with paired t) -> does not carry weight on E1; (2) optimizer test: champion fitness 1.0919 > B1 best 0.8383 (K=366 budget-matched random search) -> NOT ceremonial as an optimizer; (3) organ test: adoption gate PASSED (margin 0.8758 > 1.0 x cross-fold sd 0.6380; champion shipped) — B2/E2 holdout read printed above. (`ea/ea_manifest_2026-02-06.json`, `ea/b1_2026-02-06.json`, FREEZE_SYN1.md)

### llm (R10; final-line key `LLM`)

- **E1 (primary, §4.3):** HAC t=-1.08 on n=1507 pooled fold-days, mean=-0.12 bp/day, 95% CI [-0.000034, +0.000010], MDE=+0.000023 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+0.99; F2: t=-1.12; F3: t=-1.06; F4: t=+0.69; F5: t=-1.32; F6: t=-1.97
- **E2 (sign confirmation):** holdout dSharpe=-0.86, paired mean=-3.52 bp/day, paired t=-1.50, n=44 (`evidence/R10_llm/comparison.json`)
- orientation: E2 = R01 − R10 (with-LLM minus llm_* neutralized + RT-3 re-fit)
- **verdict: indeterminate** — 1.0 <= |E1 t|=1.08 < 2.0 (stats.organ_verdict)
- provenance: retrain RT-3 (LLM neutral); exec rung shipped: mlp (1.769e-4 > twin 1.620e-4) (`exec_out_rt3/ladder.json`)
- SPECIAL PRINT (Stage-1): variance PASS, tone-proxy PASS (corr 0.116 < 0.8), truncation PASS (1.5%); PARTIAL FAIL — event-flag chattiness (fires 98% of days); the pre-registered kill condition 'misses known scheduled events' is NOT met (8/8 hit), so the organ shipped with the partial-fail logged (FREEZE_SYN1.md).
- SPECIAL PRINT (model-cutoff gate, §4.6.1): single pinned model `anthropic.claude-3-haiku-20240307-v1:0` (cutoff 2023-08) on all 686 calls; 0 fallback days (`bedrock_spend.jsonl`). PASS.
- FINDING carried from RT-3 (`validation_looks.jsonl`): the shipping GBM never split on llm_* (zeroing them changes nothing, max |dP|=0); the LLM difference flows entirely through EventHead + executive z gating.

### gdelt (R11; final-line key `GDELT`)

- **E1 (primary, §4.3):** HAC t=-1.79 on n=1507 pooled fold-days, mean=-0.52 bp/day, 95% CI [-0.000108, +0.000005], MDE=+0.000058 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=-2.32; F2: t=+0.90; F3: t=-0.29; F4: t=+1.82; F5: t=-0.96; F6: t=-3.29
- **E2 (sign confirmation):** holdout dSharpe=-1.00, paired mean=-4.86 bp/day, paired t=-1.59, n=44 (`evidence/R11_gdelt/comparison.json`)
- orientation: E2 = R01 − R11 (with-GDELT minus block-ablated + RT-4 re-fit)
- **verdict: indeterminate** — 1.0 <= |E1 t|=1.79 < 2.0 (stats.organ_verdict)
- provenance: retrain RT-4 (GDELT ablated); exec rung shipped: mlp (1.342e-4 > twin 1.260e-4) (`exec_out_rt4/ladder.json`)
- SPECIAL PRINT (shift gate, §4.6.2): PASS — record-count ratio 0.942 in [0.5,2.0]; tone shift 0.40 sd; worst bucket energy_oil 4.83 sd < 5.0 (`gdelt_shift_gate.json`). 'No signal' is distinguishable from 'feed broke'.
- SPECIAL PRINT (G1 placebo consequence, §4.6.3): FAIL — real training-fold rank-IC -0.0014 at percentile 28 of 50 permutations (need >95th) -> the G1 dictionary is reported `0 (measured)` REGARDLESS of this block arm (`placebo_result.json`). The champion genome already gates G1_themes + G3_tone OFF on both sides of the contrast.
- §0.7 caveat (printed with all E1 GDELT numbers): the 64-ETF universe and every curated mapping were authored in 2026 with full knowledge of 2015-2026 history; freezing protects only holdout claims (ATTACK_SKEPTIC §0.7).
- FINDING carried from RT-4 (`validation_looks.jsonl`): with G1-G5 ablated the EventHead's event-mass gate is 0 -> permanent abstain; GDELT structurally carries the EventHead activity gate, so the ablated brain is CAST+GBM with renormalized trust.

### infotropy_b (R12; final-line key `infotropy`)

- **E1 (primary, §4.3):** HAC t=-0.40 on n=1507 pooled fold-days, mean=-0.11 bp/day, 95% CI [-0.000063, +0.000042], MDE=+0.000054 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+0.11; F2: t=-0.26; F3: t=-1.39; F4: t=+1.50; F5: t=-1.41; F6: t=-0.80
- **E2 (sign confirmation):** holdout dSharpe=-0.57, paired mean=-3.58 bp/day, paired t=-1.19, n=44 (`evidence/R12_infotropy_b/comparison.json`)
- orientation: E2 = R01 − R12 (record-weighted minus uniform; contrast = {CAST record-weighting, executive w_rec loss weighting, trust-tilt record weighting} — shipping GBM already uniform per FREEZE §9.2)
- **verdict: 0 (measured)** — E1 |t|=0.40 < 1.0 (stats.organ_verdict)
- provenance: retrain RT-5 (uniform weights); exec rung shipped: mlp (3.345e-4 > twin 3.177e-4) (`exec_out_rt5/ladder.json`)
- SPECIAL PRINT (rungs, FREEZE §9.2): record-weighting shipped per-learner on purged-validation wins — CAST weighted (4/6 fold wins), GBM uniform (3/6, mean -0.0044); the R12 contrast is exactly {CAST record-weighting, executive w_rec, trust-tilt record weighting}.
- Transfer-A fold verdict printed in parentheses on the final line (next section).

### infotropy_a (E1-only fold read; no replay per battery plan)

- **E1:** HAC t=+0.88 on n=1507 pooled fold-days, mean=+0.09 bp/day, 95% CI [-0.000010, +0.000028], MDE=+0.000019 (`evidence/e1_reads.json`)
- per-fold HAC t: F1: t=+1.61; F2: t=+1.18; F3: t=n/a; F4: t=+1.10; F5: t=-1.48; F6: t=-1.52
- **verdict: 0 (measured)** — E1 |t|=0.88 < 1.0
- SPECIAL PRINT (FREEZE §9.1): the conjunctive gate is DEAD (0/36 family x fold wins); R3-only screening ships for EventHead/GBM routing. This read is the shipped R3-only screen vs a no-screen twin, fold-level only (D5).

## Battery provenance

- Battery: 14/14 planned replays + 1 contingency (R19, risknet repair) executed
  (`prototype/holdout_looks.jsonl`, 15 looks; budget <=20), 5/5 planned retrains
  RT-1..RT-5 (`prototype/retrains.jsonl`); remaining contingency arms R15-R18/R20 not run
  (`battery_plan.json`).
- Exec-rung ladder per retrain (`exec_out*/ladder.json`): base RT-1 linear_twin FROZEN;
  RT-2/3/4/5 ship MLP per the pre-registered rung rule (recorded in
  `validation_looks.jsonl` rt_ladder_rungs_summary).
- Validation-look ledger: 78 decisions (`prototype/validation_looks.jsonl`).
- Gate readouts (§4.6) and the §4.7 final line: `prototype/evidence/SCORECARD.md` /
  `scorecard.json`.

## Final line (§4.7)

```
BRAIN vs INCUMBENT: TIES by -3.68%, dSharpe -0.39 on holdout (paired daily t=-0.92, n=44, mean=-8.28 bp/day, sd=59.50 bp/day); ASSIGNMENT SCORECARD: transformer=0 (measured) ensemble=cast:0(measured)/gbm_cond:-0.98(indeterminate)/event_head:0(measured)/risknet:+0.00(indeterminate) evolution=0 (measured) LLM=-0.86 (indeterminate) GDELT=-1.00 (indeterminate) meta-evaluator=-0.94 (indeterminate) infotropy=no-transfer; COST: $9.50/mo — CAST OOF seeds 3->2, deploy ensemble 5->3, 8-day gradient minibatches (semantics-preserving), LLM Tier-2 window reduced to 2024-08-15->2026-01-28 ($3.10 cap)
```
