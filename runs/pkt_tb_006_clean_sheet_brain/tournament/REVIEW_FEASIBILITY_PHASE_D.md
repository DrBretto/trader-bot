# REVIEW — FEASIBILITY AUDITOR (panel role 12) — Phase D evidence pass

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — 2026-06-10
**Charge:** audit COST_WORKSHEET.md against actuals; battery budget conformance; the
buildability postmortem on dissent D1; deployment-readiness cost caveats.
**Method:** every number below recomputed independently from `prototype/bedrock_spend.jsonl`
(707 rows), `prototype/holdout_looks.jsonl` (15), `prototype/retrains.jsonl` (5),
`runs_battery/R*/manifest.json` wall-clocks, BAKEOFF.md, ATTRIBUTION.md, FREEZE_SYN1.md,
RUN_JOURNAL.md, against my Phase B audit (ATTACK_FEASIBILITY.md) and TOURNAMENT.md §3/§4.4.

---

## 1. Bedrock ledger arithmetic — VERIFIED, with one labeling defect

1.1. The ledger is internally sound: 707 rows; cumulative column strictly monotonic; max
drift between per-row `cost_usd` sums and the cumulative column = 4.4e-16 (float noise);
recomputing every row's cost from its token counts at list price ($0.25/$1.25 per MTok,
Haiku 3) reproduces the total **exactly**: $2.872972. One pinned model on all rows
(`anthropic.claude-3-haiku-20240307-v1:0`); zero fallback days — the §4.6.1 gate read in
ATTRIBUTION.md is confirmed by the ledger itself.

1.2. **Labeling defect (immaterial, direction conservative):** COST_WORKSHEET §2 prints
"687 calls, 6.876M input tok, 0.868M output tok = $2.873". The true decomposition is
**686 scoring calls + 1 Phase-A verification + 20 schema-repair retries = 707 ledger
rows**. The stated token counts are the non-repair rows exactly (6,876,401 in / 868,455
out = $2.8047), while the stated dollar figure is the FULL ledger including repairs
($2.872972, repairs = 196.5k in / 15.3k out = $0.0683). So the line's tokens and dollars
come from two different row subsets. The cap-relevant number ($2.873) is correct to the
cent and ties to the ledger; the worksheet should have said "686 scoring calls + 20
schema repairs + 1 verification". No verdict impact.

1.3. Per-call mean: worksheet's $0.00418/call = $2.872972 / 687 — i.e. repair cost
amortized over scoring calls. That is the **right** effective rate to project deployed
cost with (it silently prices the observed 2.9% repair rate in). Plain scoring-call mean
is $0.004088. My Phase B estimate was (10k in + 1.5k out) ⇒ $0.004375/call — **7% high**.
Measured per-call means: 10,004 in / 1,250 out. My input estimate was within 0.1%; output
estimate 20% high. Phase B Bedrock arithmetic holds up.

1.4. Cap conformance: **$2.873 spent of the $3.10 hard cap = 92.7% consumed, $0.227
headroom.** PASS, but TIGHT — and it passed only because the pre-authorized lever fired:
the naive full Tier-2 window projected ~$3.9 (ledger-measured rate × 758 days confirms
~$3.87), was killed at $0.754 cumulative, and relaunched as the latest contiguous fit
2024-08-15→2026-01-28 (journaled, logged in validation_looks.jsonl, carried to the final
line). The D7 conditional rule (Tier-2 starts only if cumulative < $3.00) was honored
(cum $0.726 at launch). This is exactly the failure mode my Phase B §B.1 flagged for
Gamma ("one bug discovered after Tier-2 starts forces a re-run that breaches the cap")
— it materialized as a window-cost mis-projection instead of a bug, and the pre-registered
ladder absorbed it. Process worked as designed.

## 2. Deployed-shape projection (+$0.21/mo marginal) — CREDIBLE; envelope verdict PASS

2.1. **Bedrock line $0.14/mo:** 22 nights × 1.5 headroom × measured-effective
$0.00418/call = $0.138. The projection now rests on MEASURED token counts (10.0k/1.26k)
rather than my Phase B estimates, and the 1.5× headroom covers the observed 2.9%
schema-repair rate roughly 17× over. Sound.

2.2. **Lambda line +$0.06/mo:** +65 s/night (25 s GDELT 8 zips + 40 s LLM funnel/call +
5 s SYN-1 inference + 5 s CSVs) × 22 × 2.94 GB = 4,204 GB-s ≈ $0.070 at list; the
worksheet's +$0.06 comes from rounding ($0.19−$0.13). A cent low; immaterial. The
component budgets are now evidence-backed rather than guessed: (a) the 8-zip nightly pull
was actually exercised — top-up ran 124/124 days, 0 failures, at verified 2026 file sizes
(5–7 MB GKG); 25 s is plausible for ~30–55 MB; (b) SYN-1 inference: a FULL 84-date replay
including brain inference completes in **3.9 s wall** (R01 manifest) ⇒ single-night
inference is ~50 ms; the 5 s budget is ~100× headroom; (c) the LLM call itself measured
seconds per night in backfill.

2.3. **Other lines:** S3 +$0.01 (25 MB + ~70 req/day — trivially right); ECR +$0 (no new
heavy deps — confirmed: prototype runs on torch/sklearn already in the image); absolute
total 6.20+0.14+1.06+0.12+2.00+0.01 = $9.53 ≈ the printed $9.5. Marginal sum
0.06+0.14+0.01 = **+$0.21** — arithmetic checks.

2.4. **Verdict: PASS.** Marginal +$0.21/mo; absolute ≈$9.5/mo on the conservative $6
Lambda carry (≈$3.6 at honest list price), inside the ≤$10 target with ≥$5.5 headroom to
the ~$15 hard fail. The Phase C one-time bill ($2.873 Bedrock + $0 AWS + Mac-only
compute, no resources created) is its own clean PASS. The one standing asterisk is
unchanged from Phase B: the $6/mo Lambda baseline is ~37× its list-price arithmetic and
was never reconciled — the absolute column is an upper bound and the marginal column
remains the decision-grade number (finding 4.1).

## 3. Battery budget conformance — CONFORMANT

3.1. **Caps:** 15 replay looks (R01–R14 + contingency R19) vs cap ≤20; 5 retrain cycles
(RT-1…RT-5) vs cap ≤9. Planned battery was 14/5; the +1 replay used a reserved
contingency slot per §4.4 to repair the degenerate R07 (its planned sigma swap WAS the
deployed convention, so it measured the null by construction — the consumed look stays in
the record, correctly). Ledger counts match the journal and ATTRIBUTION.md exactly.

3.2. **Wall-clock vs my Phase B estimates: massively under, safe direction.** I estimated
~0.5 h of replays and 2–3 h base training + 2–3 h battery retrains (≈5–7 h foreground
total for the synthesis). Actuals: all 15 replays totaled **54.5 s** (3–5 s each, 84
dates, disk-cached snapshots — the cache I demanded in B.3 was built and is why);
RT-2…RT-5 completed in minutes (ledger timestamps 18:02:29→18:06:38); the final-grade base
cycle measured ≈80 s warm / 35–50 min cold-cache. My estimates were conservative by 1–2
orders of magnitude — they were sized against CAST-45k × 3 seeds × 6 folds pre-shrink,
and the 16k resize + rungs + caching compounded. No budget pressure ever materialized;
nothing was shrunk *because of* wall-clock at battery time.

3.3. **Shrink rungs vs the ladder:** CAST OOF seeds 3→2 and deploy ensemble 5→3 are
ladder rungs 1–2 verbatim (§4.4). The LLM Tier-2 window reduction is the D7 conditional
mechanism, logged. The **8-day gradient minibatches** rung is NOT a named rung on the
§4.4 ladder (nearest named rung: early-stop patience); it is adjacent in kind
(seeds/epochs-class, semantics-preserving, never an organ), was logged at the time, and
appears on the final line — I classify it a **minor off-ladder reduction, disclosed,
no protocol breach** (the ladder's governing constraint "seeds/epochs before organs,
never the bake-off" was respected). EA stayed at G=14, K=366 ≤ 400. All reductions
appear on the final line as required.

3.4. **My Phase B GDELT byte estimate was badly high:** I carried 40–55 GB for the
prototype-density deep pull; actual cache is **3.4 GB** for 4,131 days + top-up. Pre-2023
GKG files are far smaller than the 2026 files I probed (5.4 MB), and I extrapolated the
2026 size backwards. Consequence: none ($0 cash, background job) — but the correction
belongs on the record since my "does not fit a session" framing drove shrink-ladder
design. The deep pull as actually shaped would have fit comfortably in the foreground.

## 4. Buildability postmortem — dissent D1: NOT VINDICATED

4.1. My Phase B recommendation (D.2) was ETT in the transformer slot; synthesis chose
CAST-Small (TR's gradient-information arithmetic governing over my cost view). D1's
vindication clause, per its exact wording in TOURNAMENT.md §3: *"if in Phase C CAST-Small
**fails to beat its ridge twin on purged validation Spearman** while **the Infotropy-A
family screen shows strong R3 lift on event features**, the event-interaction hypothesis
earns a follow-on packet with ETT at TR's G1 size."* Both conditions FAILED to occur:
(a) CAST-Small **beat** its ridge twin decisively on purged validation Spearman (.106 vs
−.009 — twin at noise; RUN_JOURNAL wave 2+3); (b) the Infotropy-A read shows **no strong
R3 lift** — the conjunctive gate died 0/36 family×fold and the shipped R3-only screen
read E1 t=+0.88, `0 (measured)` (ATTRIBUTION.md). **The clause is not triggered. No ETT
follow-on is earned under it, and I do not claim one.**

4.2. Stating it plainly against the temptation to self-vindicate: yes, the transformer's
final attribution was `transformer=0 (measured)` (E1 HAC t=−0.28) — superficially "the
expensive organ bought nothing, as the cost auditor feared." But the evidence PATTERN
points away from my dissent, not toward it: CAST learned genuine validation structure its
linear twin could not (the twin-beating), the failure was translation of that structure
into attributable utility through the executive at this sample size, and the
event-feature side (ETT's food) showed *zero* R3 lift — so the data say "no
transformer-shaped attributable edge on ANY food here," not "chips on the wrong food."
Had ETT taken the slot, the honest expectation from these reads is the same measured
zero, at a cheaper attribution arm. My cost objection was answered structurally
(45k→16k + shrink rungs) and then mooted empirically: CAST training never became the
budget item I feared (finding 3.2). D1's adjudication stands as correct on the evidence.

## 5. Deployment-readiness cost caveats (for any follow-on productionization packet)

5.1. **Reconcile the Lambda baseline before any envelope claim matters.** The $6/mo
carry vs ~$0.14/mo list-price arithmetic (Phase B anomaly, still open) means the
absolute worksheet is an upper bound resting on an unexplained bill. One hour against
Cost Explorer closes it; until then only the marginal +$0.21 is decision-grade.

5.2. **The pinned-Haiku contingency as written is not buildable as written.** The
fallback `gpt-4o-mini` is not a Bedrock model: invoking it means a new provider secret,
new egress, and a code path that was never built or tested in Phase C. The in-family fix
— IAM widening to a newer Haiku-class ID (~$0.55/mo, in envelope) — should be the plan
of record; the frozen-artifacts rule (never re-score) stands regardless.

5.3. **Repair-storm exposure.** 2.9% schema-repair rate is priced in via headroom, but
production throttling or prompt/schema drift makes retries open-ended; carry the
per-night call cap + fail-soft-dark into the Lambda, and alarm on >2 repairs/night.

5.4. **GDELT feed watch.** The shift gate passed at 4.83 sd of the 5.0 limit on its worst
bucket (energy_oil) — one bucket from `indeterminate (feed-shift)`. Run the gate nightly
in production and mask on breach; recheck the +25 s / 8-zip budget annually (2026 GKG
files already run 5–7 MB and grow).

5.5. **Cost is not the objection — the verdict is.** The recurring +$0.21/mo buys organs
that measured 0/indeterminate (LLM and GDELT negative-leaning; RT-3 found the shipping
GBM never split on a single `llm_*` feature). The envelope PASSES; the reason not to
productionize SYN-1 is TIES (holdout paired t=−0.92), not dollars.

5.6. **Operational gotchas inherited from Phase C/D:** (a) S3 snapshot gaps
(2026-05-11→05-22) silently cut holdout n from ~62 to 44 — production should alarm on
snapshot gaps or every future evaluation loses power the same way; (b) the 80 s
warm-cache monthly retrain presumes the 3.4 GB gdelt_cache + 709 MB store stay resident
on the Mac — deleting the "deletable" prototype caches makes every month a 35–50 min
cold run; (c) llm-feature training history is only 2024-08→2026-01 (the cap-fit window)
— extending it costs ~$1 more Bedrock one-time, if anyone ever cares.

---

## Statement for the committee report

The cost record is clean and the envelope holds: the Bedrock ledger reconciles exactly to
list price ($2.873 across 707 rows — 686 scoring calls, 20 schema repairs, 1 verification;
the worksheet's "687 calls" line mixes a non-repair token count with a full-ledger dollar
figure, an immaterial labeling defect in the conservative direction), the $3.10 hard cap
was met with 7.3% headroom only because the pre-authorized Tier-2 window reduction fired
exactly as designed, and the deployed projection of +$0.21/mo marginal (≈$9.5/mo absolute
on the still-unreconciled $6 Lambda carry, ≈$3.6 at list) is now grounded in measured
token counts, a measured 3.9 s full-replay inference path, and an actually-exercised
8-zip nightly GDELT pull — verdict **PASS**, with PASS-TIGHT on the Phase C Bedrock cap.
The battery ran 15 replays / 5 retrains against caps of 20/9, every shrink rung taken was
ladder-sanctioned and final-lined except the disclosed, semantics-preserving 8-day
minibatch rung (minor off-ladder, no breach), and my Phase B wall-clock estimates proved
conservative by 1–2 orders of magnitude (and my deep-GDELT byte estimate ~12× high) — the
budget machinery was never stressed. On dissent D1, I state plainly: the vindication
clause is **not triggered** — CAST-Small beat its ridge twin on purged validation
Spearman (.106 vs −.009) and the Infotropy-A screen showed no R3 lift — so although the
transformer's final attribution is the measured zero I priced as likely, the evidence
says "no transformer-shaped attributable edge on any food at this sample size," not
"wrong transformer," and no ETT follow-on is earned; the adjudication that overruled me
stands. If this brain is ever productionized, the binding caveats are the unreconciled
Lambda baseline, the unbuildable-as-written gpt-4o-mini fallback (widen IAM to a newer
Haiku instead, ~$0.55/mo), nightly shift-gate + repair-rate alarms, and the fact that the
recurring spend buys organs that measured zero — the envelope is not the reason to
hesitate; the TIES verdict is.

*— Feasibility Auditor, Phase D review, 2026-06-10*
