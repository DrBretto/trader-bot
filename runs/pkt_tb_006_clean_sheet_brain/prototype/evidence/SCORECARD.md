# PKT-TB-006 assignment scorecard (mechanical assembly)

Generated 2026-06-10T19:02:16.

## Bake-off (§4.2)

**TIES** — holdout paired t=-0.92, n=44, dSharpe=-0.39, dReturn=-3.68%

at t=+-1 the per-comparison false-call probability under the null ~ 16% one-sided; holdout MDE at t=2 ~ 5 bp/day ~ dSharpe_ann ~ 4.0 — the bake-off verdict is a sign-grade read; full-period paired stats are context, never the verdict

## Organ verdicts (§4.3)

| organ | arm | E1 HAC t | E1 n | E2 holdout dSharpe | verdict | reason |
|---|---|---|---|---|---|---|
| transformer | R03 | -0.28 | 1507 | +1.93 | 0 (measured) | E1 |t|=0.28 < 1.0 |
| ensemble_cast_drop | R04 | +0.17 | 1507 | +2.04 | 0 (measured) | E1 |t|=0.17 < 1.0 |
| ensemble_gbm_drop | R05 | +2.48 | 1507 | -0.98 | indeterminate | E1 t=+2.48 >= +2.0 but E2 holdout sign disagrees/unavailable |
| ensemble_event_drop | R06 | +0.36 | 1507 | +0.54 | 0 (measured) | E1 |t|=0.36 < 1.0 |
| risknet | R19 | +1.39 | 1507 | +0.00 | indeterminate | 1.0 <= |E1 t|=1.39 < 2.0 |
| executive | R08 | -1.02 | 1507 | -0.94 | indeterminate | 1.0 <= |E1 t|=1.02 < 2.0 |
| evolution | R09 | -0.96 | 1507 | -0.63 | 0 (measured) | E1 |t|=0.96 < 1.0 |
| llm | R10 | -1.08 | 1507 | -0.86 | indeterminate | 1.0 <= |E1 t|=1.08 < 2.0 |
| gdelt | R11 | -1.79 | 1507 | -1.00 | indeterminate | 1.0 <= |E1 t|=1.79 < 2.0 |
| infotropy_b | R12 | -0.40 | 1507 | -0.57 | 0 (measured) | E1 |t|=0.40 < 1.0 |
| infotropy_a | — (E1-only) | +0.88 | 1507 | n/a | 0 (measured) | E1 |t|=0.88 < 1.0 |

## Multiplicity (§4.5, verbatim)

"7 scorecard organs were each given one pre-registered arm. At the E2 holdout sign bar (t >= 1) the per-arm false-positive rate under the null is ~ 16%; expected false 'carries weight' sign-confirmations across 7 organs ~ 1.1; family-wise P(>=1 false positive) ~ 70%. That is why E2 is sign-confirmation only. At the E1 primary bar (HAC |t| >= 2 on ~1,500 pooled fold-days) the per-arm FPR is ~ 5%; expected false positives across 7 organs ~ 0.35. The holdout MDE is ~ 5 bp/day (dSharpe_ann ~ 4 at t=2); no claimed effect is that large, and no holdout number below it is treated as certified. This battery consumed 15 holdout looks and 78 validation-fold decisions (ledgers below)."

## Evidence gates (§4.6)

- 1. LLM model-cutoff: model=anthropic.claude-3-haiku-20240307-v1:0, cutoff=2023-08 (predates every scored window; Tier-2 starts 2024-08-15), fallback-days=0.0% -> PASS (686 calls, single pinned model, 0 fallback; bedrock_spend.jsonl)
- 2. GDELT distribution-shift gate: PASS (prototype/gdelt_shift_gate.json (worst bucket energy_oil 4.83 sd < 5.0))
- 3. Permuted-dictionary placebo: real training-fold rank-IC=-0.0014 at percentile 28.0 of 50 placebos -> FAIL -> G1 dictionary reported 0 (measured)
- 4. Break-even IC: purged-validation weekly rank IC=+0.0851 vs functionality bar 0.02 (cost break-even 0.006) -> PASS — forecast pathway carries edge (per-fold 0.061-0.113; store/step10_checks.json)
- 5. Diversity floor: mean pairwise solo-book corr=+0.941 (threshold 0.90) -> TRIGGERED (>= 0.90) — item-5 trust attribution expected ~0 by construction; the equal-trust tie is the honest outcome (§4.6.5; store/step10_checks.json)
- 6. Train-vs-harness gap: NOT PROVIDED
- 7. Fine-tune delta: pre=+0.000131 post=+0.000131 delta=+0.000000 — de-claimed (~0)

## Final line (§4.7)

```
BRAIN vs INCUMBENT: TIES by -3.68%, dSharpe -0.39 on holdout (paired daily t=-0.92, n=44, mean=-8.28 bp/day, sd=59.50 bp/day); ASSIGNMENT SCORECARD: transformer=0 (measured) ensemble=cast:0(measured)/gbm_cond:-0.98(indeterminate)/event_head:0(measured)/risknet:+0.00(indeterminate) evolution=0 (measured) LLM=-0.86 (indeterminate) GDELT=-1.00 (indeterminate) meta-evaluator=-0.94 (indeterminate) infotropy=no-transfer; COST: $9.50/mo — CAST OOF seeds 3->2, deploy ensemble 5->3, 8-day gradient minibatches (semantics-preserving), LLM Tier-2 window reduced to 2024-08-15->2026-01-28 ($3.10 cap)
```
