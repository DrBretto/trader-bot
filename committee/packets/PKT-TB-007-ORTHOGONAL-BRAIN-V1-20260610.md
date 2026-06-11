# PKT-TB-007-ORTHOGONAL-BRAIN-V1-20260610 (UNOFFICIAL — operator-authorized in-thread)

**Packet Owner:** TraderBot committee thread (continuity from PKT-TB-006; the operator
explicitly authorized this thread to author and execute its own follow-on packet without a
fresh conversation, 2026-06-10: "we let you retain continuity and build and execute your own
unofficial committee packet... with the better framing and goals. you may improvise where it
brings value. but not to be lazy.")
**packet_kind:** committee (design + build + proof; unsettled scope on orthogonality scheme
and expression mechanism → committee required)
**Owning root:** trader-bot (standalone; NOT LAB-010-spine-governed)
**Builds on:** runs/pkt_tb_006_clean_sheet_brain/ (all assets reusable: data layer, GDELT
2015→present, 671 LLM artifact days, trained members + OOF matrices, harness adapter,
battery machinery, evidence protocol discipline)

## Objective

Build the most genuinely intelligent full trading brain achievable on this substrate, framed
the way the TB-006 evidence says the question should have been framed: the incumbent's
chassis (regime picker, base ranking engine, exposure behavior) is mechanically sound and
STAYS; the brain's job is to supply **orthogonal senses the chassis does not have** and to
demonstrably convert them into decisions. Beat the incumbent **at matched exposure** on the
identical harness — so the verdict measures intelligence, not risk appetite (the confound
that made TB-006's TIES uninformative about signal).

## Operator directives (this thread, 2026-06-10 — binding)

1. "Genuine orthogonal ML models and whatever we can to get as intelligent a true full brain
   as possible."
2. "If the regime picker is a good approach to start with that you couldn't beat in
   isolation, then yeah, I say you can start there." — augmentation sanctioned; the regime
   picker is substrate, not the enemy.
3. "The 64 is arbitrary, so re-picking which stocks to use is also on the table — if some
   are more exploitable than others, that's a good thing."
4. **The one hard requirement:** "it uses the various different model types on purpose
   because I want all of them present, because they CAN be used for different orthogonal
   looks. I don't want them to be redundant, I want you to find a configuration where each
   can bring as much value as possible, wherever it is."
5. The ML models were never supposed to pick the regime: "they were supposed to get some
   OTHER feel of the market that the regime picker doesn't pick up on."
6. Timeline honesty caveat (operator-volunteered): the live record was rebuilt/repaired at
   points as honestly as possible but "can't be 100% trusted" — some future-bleed is
   unavoidable. Consequence: paired same-harness comparisons only (contamination hits both
   arms identically); the caveat is printed with every full-record read.
7. Token/compute budget explicitly unconstrained for this run. Improvisation licensed where
   it brings value; laziness is not.

## Design constraints carried as HARD from TB-006 evidence (not re-derivable opinions)

- **C1 — Exposure parity is structural, never statistical.** The brain's arm must hold the
  incumbent's gross/risk budget by construction (tilt/reallocate, never de-risk). A verdict
  contaminated by an exposure gap is a failed verdict.
- **C2 — Orthogonality is engineered and gated.** Members get different targets, horizons,
  and disjoint feature partitions BY DESIGN; a pre-registered decorrelation acceptance gate
  (pairwise signal corr ceiling ≤ 0.7 on validation folds) must pass BEFORE the bake-off; a
  member that fails is redesigned or its redundancy is reported, never papered over.
  (TB-006: 0.941 solo-book corr made trust attribution structurally unmeasurable.)
- **C3 — Conversion is the showcase.** The expression layer must be able to SPEND signal:
  any evolutionary/learned balancing optimizes a fitness aligned with the pre-registered
  verdict metric (TB-006's Sharpe-shaped defensive fitness produced a 26%-gross brain graded
  on offense). "How much to listen to which organ" remains the trained intelligence.
- **C4 — Certifiability by construction.** The verdict pair differs ONLY by the brain's
  tilt (C1), so paired daily sd is small and bp/day effects become certifiable. Use the full
  replayable artifact record (2025-08-04→present, ~170+ paired days net of the Tue–Sat
  cadence) with the 2026-03-11 holdout split retained for E2 discipline; print the TB-004
  backfill caveat (2025 inference artifacts are backfilled heuristics) with every full-record
  read — both arms inherit it identically.
- **C5 — Known defects fixed at the root:** per-symbol lot aggregation in any adapter
  (TB-006 SCHD double-lot); Monday/cadence exclusion disclosed up front; cost overlay
  convention carried unchanged (seed 4242 paired).
- **C6 — Evidence discipline unchanged:** committee/EVIDENCE_PROTOCOL.md; pre-registration
  before build and before any holdout read; every variant logged; battery caps; look
  ledgers; three-valued organ verdicts; mechanical report assembly.

## The integration substrate (verified entry points)

- The incumbent's decision engine natively accepts **external per-symbol ranking scores**
  blended at a configured weight (`ranking_scores` / `ranking_blend`; the deployed config
  runs its own ranking model at blend 0.35). This is the primary candidate socket: the brain
  as the chassis's cross-sectional sense organ. Phase 0 verifies the exact mechanics (how
  scores flow into buy selection and sizing) before any design relies on them.
- Secondary candidate socket: `Strategy.post_decision` intent reshaping at matched gross
  (TB-006 adapter, lot-fix applied).
- The committee chooses the expression mechanism(s) in Phase B with the mechanics in hand.

## Assignment (every item load-bearing)

1. **Orthogonal multi-model brain:** ≥4 genuinely different model types, each present ON
   PURPOSE with a named LOOK the others (and the regime picker) do not have — e.g. (kinds-of,
   not a checklist): cross-sectional relative strength (transformer — exists at IC ~.105),
   conditional/macro tabular (GBM — TB-006's one utility-positive organ), event/news
   (LLM+GDELT in their honest value-maximizing role), vol/dispersion structure, positioning,
   breadth/internals, alternative horizons. Each member: distinct target + distinct feature
   partition + the C2 gate.
2. **Expression layer** that converts the orthogonal senses into decisions through the
   chassis at matched exposure (C1, C3), with per-organ contribution observable.
3. **Evolutionary balancing** on verdict-aligned fitness (which organs to trust, how hard to
   tilt) — meaningful, with the B0/B1 control battery carried from TB-006.
4. **LLM organ in its value-maximizing role** — TB-006 proved it is not a tone proxy and
   costs ~$0.14/mo; find where its signal actually pays (event-risk conditioning? tilt
   veto? salience weighting?) or report honest zero with the arm to prove it.
5. **Universe exploitability:** rank the 64 by predictability/convertibility (per-symbol
   IC, spread cost, tilt capacity); the brain may concentrate where it is exploitable
   (within the replayable universe for the verdict); a tiered note may propose expansion
   symbols for a future packet (non-verdict).
6. **Per-organ attribution:** leave-one-organ-out battery per TB-006 discipline; the
   operator's requirement is non-redundancy — each organ's verdict line printed.

## Verdict (pre-registered fully in Phase B; frame fixed here)

- Pair: incumbent-as-deployed vs incumbent+brain, identical harness, identical seeds,
  identical windows; paired daily t on cost-adjusted identical dates; full-record AND
  holdout-only reads; the E2 verdict from the holdout read.
- Because C1+C4 shrink the MDE, the BEATS bar is **stricter than TB-006**: the committee
  pre-registers numeric BEATS/TIES/LOSES with BEATS requiring at minimum paired t ≥ +2.0 on
  the primary read. Exact thresholds, arms, budgets, seeds: TOURNAMENT_007 §pre-registration.

## Committee panel (10 roles; declare before Phase 0; real subagents)

1. **Analyst** — substrate audit: ranking-socket mechanics end-to-end (scores→selection→
   sizing), artifact depth/quality 2025-08→ incl. the backfill + repaired-timeline caveats
   mapped date-by-date, what TB-006 assets are reusable as-is.
2. **Skeptic** (mandatory) — owns anti-leakage (the overlay trains on records the incumbent's
   own behavior generated — circularity audit), the multiplicity ledger, and the
   "quant's sneer" pass on orthogonality claims.
3. **Orthogonality Engineer** — the new central role: designs the member set (targets,
   horizons, feature partitions), the C2 gate, and the redundancy measurement.
4. **Expression Architect** — the socket choice and the tilt mechanics at matched gross;
   owns C1's structural enforcement and the conversion path (C3).
5. **Universe Selector** — assignment item 5.
6. **LLM/Event Role-Finder** — assignment item 4.
7. **Evolution Engineer** — assignment item 3, verdict-aligned fitness.
8. **Training Realist** — sample budgets, window splits, fine-tune discipline, the
   repaired-timeline consequence for training data.
9. **Feasibility Auditor** — cost envelope (≤$10/mo deployed shape), battery budget, build
   wall-clock; TB-007 should be CHEAPER than TB-006 (assets exist).
10. **Synthesis Chair** — separate convergence step; pre-registration author.

## Phased work

- **Phase 0:** Analyst + Universe Selector audits (parallel).
- **Phase A:** specialist proposals (roles 3,4,6,7) + two competing brain designs from
  opposed priors (e.g. minimal-socket purist vs full-tilt maximalist) — independent.
- **Phase B:** Skeptic + Training Realist + Feasibility attack; synthesis to ONE build;
  TOURNAMENT_007 pre-registration committed BEFORE build.
- **Phase C:** build (reuse TB-006 prototype; new: retargeted decorrelated members, socket
  integration, lot fix, verdict-aligned EA).
- **Phase D:** battery, mechanical evidence, panel review, COMMITTEE_REPORT_007.

## Write surface

- `runs/pkt_tb_007_orthogonal_brain/` (everything; prototype may import from
  `runs/pkt_tb_006_clean_sheet_brain/prototype/` read-only or copy modules in)
- `docs/plans/2026-06-10-orthogonal-brain.md`
- NOT: production paths, no deploys, no new AWS resources; bounded Bedrock spend with a cap
  set in Phase B (default $3.00, ledgered).

## Stop condition

Dossier complete; brain built; bake-off + per-organ battery run under pre-registered
criteria; report converged with the PKT-TB-002 field set + final line:
`BRAIN+CHASSIS vs INCUMBENT: <BEATS|TIES|LOSES> by <Δreturn, ΔSharpe> on holdout (<paired-stat>); ORTHOGONALITY: <gate result, pairwise corr>; ORGANS: <per-organ attr tags>; COST: $<x>/mo — <reductions>`.
Committed on `ai/orthogonal-brain`. No production integration, no deploy.

## Hard constraints

C1–C6 above; the operator directive 4 (all model types present on purpose, non-redundant)
is the assignment's non-negotiable core; honest zeros remain passing outcomes; the
incumbent appears as chassis + score line, never as an evaluation subject; if any referenced
substrate is missing, stop and surface.
