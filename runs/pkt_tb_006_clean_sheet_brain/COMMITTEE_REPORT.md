# COMMITTEE_REPORT — PKT-TB-006 Clean-Sheet Trader's Brain

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN-V1-20260610
**Committee thread:** Claude (Fable 5) orchestrator + 12-role panel via real subagents, 2026-06-10
**Branch:** `ai/clean-sheet-traders-brain` (registration `4d68974`; pre-registration `eda53a8`; freeze `4afc606`)
**Field set per PKT-TB-002 requirement.**

---

## Brief Rundown

The committee designed, built, and proved a complete new trading system — SYN-1 — from a
clean sheet, under an absolute sealed-incumbent rule that held through design registration
(verified in git: no panel role read any incumbent strategy file before commit `4d68974`;
the Skeptic's post-hoc anchoring audit found none). Three architects drafted independent
whole systems from three priors (forecast-first / allocation-learner / information-edge),
fed by five blind specialist proposals; three attackers tore them down; synthesis grafted
one build candidate (SYN-1) and pre-registered every bake-off criterion before a line of
prototype code. The brain was then actually built — GDELT re-ingested at event/theme depth
(4,131 days, 2015→present), a real LLM organ run on Bedrock Haiku over 686 scored days
($2.873 of a $3.10 cap), four ensemble members trained walk-forward, a learned executive,
a 27-gene evolutionary balancing layer with random-search and default-genome controls —
and run head-to-head against the incumbent on the identical replay harness with E2 holdout
discipline, then dissected organ-by-organ by leave-one-out attribution (15 holdout looks
of a budgeted 20; 5 retrains of 9).

## Final Verdict

**SYN-1 TIES the incumbent on the pre-registered holdout rule and does not demonstrably
beat it; every point estimate favors the incumbent.** Holdout paired daily t = −0.92
(n = 44 identical dates, mean −8.28 bp/day, sd 59.50), ΔSharpe −0.39, endpoint Δreturn
−3.68% (context). TIES is the mechanical |t| < 1 verdict; the dossier prints the boundary
honestly: at the planned n≈62 the same distribution reads t≈−1.10 → LOSES. The Skeptic's
summary stands as the committee's: **"TIES is a statement about power, not parity."**

**Assignment scorecard:** no organ achieved a confirmed positive attribution. Transformer,
CAST-drop, EventHead-drop, evolution, and Infotropy-B all read **0 (measured)** at E1
power; GBM-Cond is the lone suggestive carrier (E1 HAC t = +2.48, the only organ clearing
the +2 bar, but its E2 holdout sign flipped — pre-registered rule: indeterminate);
LLM, GDELT, and the meta-evaluator read indeterminate with negative lean; **infotropy =
no-transfer** (Transfer-A's conjunctive gate died by its own falsifier; Transfer-B's
replay arm read 0). An honest "built it; it does not beat the incumbent; the organs do
not demonstrably carry weight in this configuration" is the packet's explicitly
sanctioned passing outcome, and it is this committee's finding.

## Key Tensions

1. **Transformer slot (D1, Alpha-CAST vs Gamma-ETT):** resolved for CAST on
   gradient-information arithmetic. Outcome: CAST genuinely learned (purged Spearman .106,
   weekly rank IC .105 ≈ 5× the functionality bar, ridge twin at noise) yet measured 0 in
   the brain. The Feasibility Auditor's D1 vindication clause was checked against its
   exact wording and did **not** fire — the evidence says "no transformer-shaped edge
   cashed out at this sample size," not "wrong food."
2. **Perception vs expression (the run's central tension, named by the Training
   Realist):** the μ-stack carries real, cost-clearing forecast signal (stack weekly rank
   IC .085 ≫ the .006 cost break-even), but the honestly-sample-budgeted pipeline produced
   an ultra-conservative champion genome (26% gross, 4 trades/65 days, 0 holdout round
   trips) that throttles signal→utility conversion for every organ at once. The measured
   zeros are mostly a statement about the spend, not the signal.
3. **Evolution's paradox:** the EA is a real optimizer (champion fitness beats B0 by
   1.37× the adoption-gate bar and beats budget-matched random search — "not ceremonial
   as an optimizer," the pre-committed sentence) whose evolved disposition nonetheless
   reads 0-vs-B0 out-of-sample on E1/E2. It optimized the fitness surface it was given.
4. **Diversity floor (predicted in Phase B, triggered in Phase C):** member solo-books
   correlate 0.941 ≥ the 0.90 floor — the pre-registered statement applies: trust
   attribution ≈ 0 by construction. The executive's three §7 kill criteria all fired
   (static trust per fold, negative calibration, twin parity) and the **linear twin
   shipped as the gate** per the ladder rule.

## Non-Obvious Findings

- **The gap diagnostic caught a real wiring defect:** the seed portfolio held SCHD in two
  equal 738-share lots; the adapter's per-symbol map saw one, so SYN-1's executed replay
  book carried an unintended ~22% SCHD position throughout (R01 avg gross 25.7% partly
  explained). All SYN-1 arms shared the defect identically (arm-vs-arm comparisons remain
  internally consistent); the bake-off pair is materially caveated by it. Reported, not
  patched post-verdict — a fix-and-rerun would be an unregistered second look.
- **§4.6.6 train-vs-harness gap = 122% ⇒ "relaxation-gaming" prints**, but the mechanism
  note shows it is dominated by the SCHD book divergence, not cost-noise gaming.
- **GDELT's live daily-counts endpoint is dead** (404; tone fields were placeholder zeros
  in the live pipeline); the working unit is the 15-minute GKG v2 file. The committee's
  backfill restored full holdout coverage and the distribution-shift gate PASSED — so the
  GDELT verdict is about signal, not feed breakage.
- **The permuted-dictionary placebo failed** (real theme→sector dictionary at the 28th
  percentile of 50 permutations) — and the EA independently gated G1-themes OFF. Two
  separate mechanisms agreed the theme dictionary carries nothing.
- **The LLM organ is not a tone proxy** (corr 0.116 vs the 0.8 kill bar) — it produces
  genuinely distinct signal at $0.14/month — but that signal did not convert to measured
  utility. Its event flags fire on 98% of days (chatty, logged partial fail).
- **Verdict series excludes Mondays by harness construction** (production night-analysis
  runs Tue–Sat), for both arms identically: holdout n = 44, not ~62; realized MDE ≈ 21
  bp/day ≈ 4× the planning estimate.
- **EventHead only acts when GDELT exists** — ablating G1–G5 silences it completely
  (its activity gate is GDELT event mass): the organs are more entangled than the
  arm-per-organ frame assumes.

## What Changed From Prior Assumptions

- The packet said 65 symbols; the verified universe is **64**.
- S3 is not the deep archive (rolling-window snapshots only from 2026-01-31; 365-day
  lifecycle); deep history lives in local training parquets + free refetch.
- The harness applies **no transaction-cost model** in its fill path; the pre-registered
  "same cost model, seeded RNG" was implemented as an identical post-hoc overlay on both
  arms (journaled before any replay).
- The orchestrator's early wiring note (native 03-11-seeded verdict pair) was superseded
  by the registered text (both arms replay the full window; verdict read holdout-only) —
  supersession recorded in the journal.
- Planning MDE (~5 bp/day) was ~4× optimistic for the bake-off pair (Monday exclusion +
  May snapshot gap + realized sd).
- Phase B cost/wall-clock fears never bit: full final training cycle ≈ 80 s warm-cache;
  battery replays totaled ~55 s; deployed marginal cost ≈ **+$0.21/mo** (absolute ≈ $9.5,
  inside the ≤$10 target; Bedrock actuals audited to the penny).

## Implementation Shape

Everything lives in `runs/pkt_tb_006_clean_sheet_brain/` (no production paths touched, no
deploys, no new AWS resources; Infotropy Book repo read-only). `prototype/` is a complete,
seeded, manifested, re-runnable system: data layer (S3 snapshot cache, OHLCV, CBOE, FRED,
COT), GDELT rich backfill (3.4 GB local cache), LLM organ (frozen prompt/schema, spend
ledger), feature store (alignment-tested, no-look-ahead joins), four members + twins,
solo-book/ledger/executive stack, 27-gene EA with B0/B1 controls, replay adapter +
holdout-guarded runners, battery with in-runner look ledgering, mechanical evidence
assembly, 151 tests green. Dossier: ASSIGNMENT_BRIEF, designs (3 + transfer + 4
proposals), TOURNAMENT (pre-registration), BAKEOFF, ATTRIBUTION, COST_WORKSHEET,
SCORECARD, three Phase-D reviews, RUN_JOURNAL. **If the operator wants any of this
productionized, that is a follow-on packet; this committee recommends against
productionizing SYN-1 as-is on the evidence above.**

## Dissent

- **Skeptic (review F3, carried):** the n=44/TIES-vs-LOSES boundary must not be read as
  parity — at the planned sample the same numbers read LOSES; nothing in this dossier
  licenses "any mandated organ demonstrably carries weight." Also carried: F7 (the
  shipped brain does not consume RiskNet+ heads at decision time; the risknet line
  characterizes a non-shipped candidate), F11-as-discharged (gap diagnostic delivered
  late, by repair), and one VIOLATION downgraded to discharged-by-repair.
- **Training Realist:** the zeros split three ways — true zeros (transformer, event,
  infotropy at E1 power), structural zeros (executive trust, by the 0.94 correlation
  floor), and underpowered reads (evolution, LLM, GDELT). GBM should be read as
  "E1-positive, confirmation pending," not as a failure. A longer holdout alone does not
  fix this (certifying ~1 bp/day forward needs years); the fix is structural —
  decorrelated members (corr ≤ 0.7 acceptance gate), forecast-space attribution
  pre-registered beside utility-space, an equivalence band instead of sign-confirmation
  for sub-MDE effects.
- **Feasibility Auditor:** D1 not vindicated (stated plainly against his own Phase B
  position); envelope is not the reason to hesitate — the verdict is. The gpt-4o-mini
  fallback as written is unbuildable on Bedrock-only IAM; widen to a newer Haiku-class ID
  (~$0.55/mo) in any follow-on.
- No other dissent; the panel converged on this report.

## Panel effectiveness (one row per declared panelist)

| # | Role | Subagent(s) | Effectiveness note |
|---|---|---|---|
| 1 | Analyst | a13dbefde6f9f5d8f | High — the brief's live verifications (64 symbols, GDELT dark, snapshot format break) shaped every later decision; zero seal incidents. |
| 2 | Skeptic | a01bfd1e9409cf9ac, ae914b55f6ea329b8 | Highest-leverage role of the run: killed the look-ahead gate pre-build, forced the E1-primary multiplicity discipline, and caught the Monday-exclusion disclosure gap post-build. |
| 3 | Architect-Alpha | a018912d47aae4ff9 | Strong — contributed the ensemble diversity discipline, CAST, and the leak-correct infotropy scoping that shipped. |
| 4 | Architect-Beta | aff54d45791a09ead | Strong — the book-space executive chassis and exact counterfactual ledger are SYN-1's spine; its boldest ideas (policy library) died honestly on buildability. |
| 5 | Architect-Gamma | a4da0c0e324093437 | Mixed — GDELT backbone and info-health block shipped; its ETT/record-gate centerpiece carried the tournament's biggest kill (look-ahead as written). |
| 6 | Infotropy Canon Liaison | ab17a414a5a736a55 | Did the job the packet demanded: honest grading (2 transfers / 3 restatements / 1 no-transfer); both transfers were then measured and read no-transfer — the verdict is evidence-backed, not forced mysticism and not silent omission. |
| 7 | Meta-Evaluator Designer | aa09e33072b30b784 | Good architecture with built-in kill criteria — which fired, exactly as designed; the auditability schema (meta_decision.json) is what later exposed the SCHD defect. |
| 8 | Evolution Engineer | aeb20fd360cc68910 | The B0/B1/adoption-gate battery is the reason "evolution = 0 (measured)" is a defensible sentence rather than a shrug; the disposition genome did real optimization that didn't generalize. |
| 9 | LLM Sentiment Engineer | a032bf2416ea57752 | Measured, not guessed: live Bedrock verification, funnel volumes from a real GKG parse, cost within 7% of estimate; Stage-1 battery did its filtering job. |
| 10 | Data Edge Scout | a71ae19d80ece78bc | The depth-audit table (GDELT dark over the holdout; LLM features have no deep history) set the build order and the D3 adjudication; source kills were honest. |
| 11 | Training Realist | aa8cfeddac8591f8f, a3c91babd444a2da2 | The effective-sample arithmetic governed the build (every forced simplification honored, verified at review); its Phase-D reading of the zeros is the report's interpretive core. |
| 12 | Feasibility Auditor | a34851f7bdd79ea2c, a703ee26d57f706d0 | Cost worksheets audited to the penny; battery caps designed and respected; owned its wrong D1 call publicly — the panel's honesty norm held. |

Synthesis chair: af7dfa4520e28c37a. Build executors (non-panel): a8748ddb59a2fafe8,
aaa8d5cebfa2b2fd3, a67b742ec68f2ac6c, a03ac50b9300a44d3, aa983ab8c666edf4c,
a7aa37727747a6b1b, afb4ae348574a5775, ad6d8b710f11f23d8, a027e033f479c6e87,
a439d31c950f3e4da, a0ac2f070f432347d, aa7207e79a4bccae6, ace2e9e895638ce87,
aa9b4e9df7fbfcf78.

## Protocol conformance summary

Sealed-incumbent rule: held (git-verifiable ordering; anchoring audit clean). Holdout
looks: **15 of ≤20** (14 planned + 1 disclosed contingency repair), ledgered in-runner.
Retrains: 5 of ≤9. Validation looks: 78, ledgered. Every variant logged; seeds fixed and
manifested; one configuration frozen before the first holdout replay (`4afc606`).
Executed deviations, all journaled with provenance: post-hoc cost overlay (implements the
registered "same cost model"); full-window verdict slice (the registered text, superseding
an earlier orchestrator note); 8-day gradient mini-batches (semantics-preserving
wall-clock measure); §4.6.6 delivered by post-review repair. Known defect carried, not
patched: SCHD double-lot adapter bug (above).

## Acceptance test answer (the packet's first-time-reader question)

The new system, organ by organ, is specified in BUILD_SPEC/FREEZE_SYN1 and measured in
ATTRIBUTION.md. It does **not** beat the incumbent on the holdout: TIES at paired t
−0.92 (n=44), ΔSharpe −0.39, with LOSES at the boundary under the planned sample. What
each organ individually buys, in numbers, is the scorecard line below — currently:
nothing demonstrably, with GBM-Cond the one suggestive positive. Monthly cost of the
deployed design ≈ $9.50 absolute (+$0.21 marginal). Built it; it does not win — that is
the real answer.

---

BRAIN vs INCUMBENT: TIES by -3.68%, dSharpe -0.39 on holdout (paired daily t=-0.92, n=44, mean=-8.28 bp/day, sd=59.50 bp/day); ASSIGNMENT SCORECARD: transformer=0 (measured) ensemble=cast:0(measured)/gbm_cond:-0.98(indeterminate)/event_head:0(measured)/risknet:+0.00(indeterminate) evolution=0 (measured) LLM=-0.86 (indeterminate) GDELT=-1.00 (indeterminate) meta-evaluator=-0.94 (indeterminate) infotropy=no-transfer; COST: $9.50/mo — CAST OOF seeds 3->2, deploy ensemble 5->3, 8-day gradient minibatches (semantics-preserving), LLM Tier-2 window reduced to 2024-08-15->2026-01-28 ($3.10 cap)
