# PKT-TB-006 assignment scorecard (mechanical assembly)

Generated 2026-06-10T19:02:16.

## Bake-off (§4.2)

**TIES** — holdout paired t=-0.92, n=44, dSharpe=-0.39, dReturn=-3.68%

at t=+-1 the per-comparison false-call probability under the null ~ 16% one-sided; holdout MDE at t=2 ~ 5 bp/day ~ dSharpe_ann ~ 4.0 — the bake-off verdict is a sign-grade read; full-period paired stats are context, never the verdict

**Sample-construction disclosure + n-boundary sensitivity (Phase-D repair, Skeptic F3;
full mechanism in BAKEOFF.md §3.1):** the verdict series structurally excludes Mondays —
the production snapshot cadence is Tue–Sat, Monday-dated snapshot dirs carry no
prices/inference artifacts (verified in cache/s3/daily/), and the harness admits only
price-bearing dirs as decision dates — identically for BOTH arms, by harness
construction, in the battery and in the smoke runs alike. Holdout paired n=44, not the
planned ~62 (which assumed a full Mon–Fri grid and no May snapshot gap); week-boundary
and gap steps are single multi-day observations. Boundary sensitivity, both sentences
side by side: at the observed mean (−8.28 bp/day) and sd (59.50), a hypothetical n=62
same-distribution read gives t ≈ −1.10 → LOSES; the pre-registered rule reads the
actual paired sample (n=44, t = −0.92) → TIES. The realized-n MDE (20.71 bp/day) is ~4x
the pre-registered ~5 bp/day power estimate — do not anchor on 5 bp/day.

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

Gloss (Phase-D repair, Skeptic F3/F13): the verbatim "~ 5 bp/day" MDE was the
pre-registration's power estimate at the planned n~62; the computed MDE on the realized
holdout sample (n=44 — structural Monday exclusion + May snapshot gap, see the bake-off
disclosure above) is 20.71 bp/day, ~4x the estimate. The verbatim text is preserved as
registered; the realized number governs.

## Evidence gates (§4.6)

- 1. LLM model-cutoff: model=anthropic.claude-3-haiku-20240307-v1:0, cutoff=2023-08 (predates every scored window; Tier-2 starts 2024-08-15), fallback-days=0.0% -> PASS (686 calls, single pinned model, 0 fallback; bedrock_spend.jsonl)
- 2. GDELT distribution-shift gate: PASS (prototype/gdelt_shift_gate.json (worst bucket energy_oil 4.83 sd < 5.0))
- 3. Permuted-dictionary placebo: real training-fold rank-IC=-0.0014 at percentile 28.0 of 50 placebos -> FAIL -> G1 dictionary reported 0 (measured)
- 4. Break-even IC: purged-validation weekly rank IC=+0.0851 vs functionality bar 0.02 (cost break-even 0.006) -> PASS — forecast pathway carries edge (per-fold 0.061-0.113; store/step10_checks.json)
- 5. Diversity floor: mean pairwise solo-book corr=+0.941 (threshold 0.90) -> TRIGGERED (>= 0.90) — item-5 trust attribution expected ~0 by construction; the equal-trust tie is the honest outcome (§4.6.5; store/step10_checks.json)
- 6. Train-vs-harness gap: mean |daily utility| = 3.45 bp/day (training-convention simulator: frozen deployed config walked over store/nightly inputs with ea.fold_utility math) vs 14.29 bp/day (harness replay, R01 cost-adjusted) on 18 identical pre-holdout steps 2026-02-04→2026-03-06 — **gap = 122.1% of mean |daily utility| (> 25%) ⇒ reported as relaxation-gaming per §4.6.6** (delivered late, Phase-D repair of Skeptic F11: gap_diagnostic.py → evidence/gap_diagnostic.json). Window caveat: pre-holdout slice only (holdout untouched); the training window itself ends 2026-02-06, so the simulator is read out-of-sample on 16 of 18 steps — this compares conventions (training utility math vs harness fills), not in-sample fit. Mechanism note (verifiable, gap_diagnostic.json): the gap is dominated by realized-book divergence — the training-convention walk trims the inherited seed book to ~3% gross under the genome's no-trade band, while the harness book retained an inherited ~22% SCHD position (the seed portfolio held SCHD in two equal lots; the adapter's per-symbol trim reduced only one lot's worth — R01 result.json final_holdings)
- 7. Fine-tune delta: pre=+0.000131 post=+0.000131 delta=+0.000000 — de-claimed (~0)

## Final line (§4.7)

```
BRAIN vs INCUMBENT: TIES by -3.68%, dSharpe -0.39 on holdout (paired daily t=-0.92, n=44, mean=-8.28 bp/day, sd=59.50 bp/day); ASSIGNMENT SCORECARD: transformer=0 (measured) ensemble=cast:0(measured)/gbm_cond:-0.98(indeterminate)/event_head:0(measured)/risknet:+0.00(indeterminate) evolution=0 (measured) LLM=-0.86 (indeterminate) GDELT=-1.00 (indeterminate) meta-evaluator=-0.94 (indeterminate) infotropy=no-transfer; COST: $9.50/mo — CAST OOF seeds 3->2, deploy ensemble 5->3, 8-day gradient minibatches (semantics-preserving), LLM Tier-2 window reduced to 2024-08-15->2026-01-28 ($3.10 cap)
```
