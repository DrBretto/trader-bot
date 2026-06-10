# PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN-V1-20260610

**Packet Owner:** TraderBot orchestrator (Claude OG session, 2026-06-10)
**packet_kind:** committee (greenfield design; widest unsettled scope in the program → committee required)
**Work shape tags:** packet_kind=committee+discovery+design; work_shape=clean-sheet-system-design, evidence-grading; question_shape=can-a-new-design-beat-the-incumbent
**Owning root:** trader-bot (standalone program; NOT LAB-010-spine-governed)

## Objective

Design, build, and prove a NEW trading system — a "trader's brain" — that beats the
incumbent canon strategy head-to-head, and that meets the operator's actual
assignment, which outranks raw P&L: an ML showcase in which an ensemble of genuinely
different model types (at least one transformer), an evolutionary algorithm doing the
balancing, an LLM doing real sentiment analysis, and differentiated data (GDELT
named explicitly) **each demonstrably carry weight**, with a learned, genuinely
intelligent evaluator making the final allocation calls. The incumbent is not
sacred, not the starting point, and not the frame: this committee invents.

**The named failure mode this packet exists to break:** every prior attempt to get a
better model set has collapsed into a report about the existing system. A return
whose design section reads as a critique, refactor, or incremental extension of the
incumbent is a FAILED return, regardless of quality. The sealed-incumbent rule below
is the structural countermeasure.

> pain signals (operator, 2026-06-10): the current ML models "aren't actually doing
> anything" — regime classification comes out accurate but that accuracy never
> cashes out into decisions; the learning capacity is aimed at the wrong target.
> What he wants trained: how much to spend, how much to listen to which model — with
> every model bringing value and an intelligent evaluator making the final call. And
> it must be differentiated: "I need it to actually BRING value so it's different
> from what others have."

## The assignment (design requirements — every one is load-bearing)

A compliant design MUST contain, with measurable marginal contribution from each:

1. **An ensemble of genuinely different ML model types** — different architectures
   doing different jobs, not two networks voting on one label. At least one
   **transformer** doing real work.
2. **An evolutionary algorithm as the balancing organ** — evolution tunes/evolves
   how the brain weighs its parts (allocator parameters, model trust weights,
   ensemble composition — the panel decides where evolution bites hardest).
3. **An LLM doing real sentiment analysis** — reading actual text (news, GDELT-fed
   or otherwise sourced free) and emitting structured signal the brain consumes
   nightly. Not a veto garnish, not a narrative paragraph: an input organ whose
   removal measurably hurts.
4. **GDELT as a load-bearing differentiated data source** — event structure, actors,
   themes, tone dynamics — used in a way generic retail bots are not. If the panel
   finds GDELT signal genuinely absent after honest effort, that finding must be
   evidence-backed (attribution ≈ 0 in the built system), never a silent omission.
5. **A learned meta-evaluator — the brain's executive** — that takes all model
   outputs and signals and makes the final decision-grade calls: position sizing,
   capital deployment, how much to trust which model today. Trained on
   decision-grade targets (realized forward utility of choices), NOT on regime
   labels. Regime classification may exist as an internal organ, but the
   intelligence being showcased is allocation, not labeling.
6. **Infotropy angle examined** — a panel role reads the Infotropy canon and
   attempts to derive a concrete, computable mechanism (feature, evaluator
   structure, information-flow framing) that gives the brain an edge. Honest
   no-transfer is an acceptable verdict; forced mysticism is not.

## Operating envelope (hard)

- **Cost ceiling:** runs on AWS at roughly the incumbent's cost — target ≤$10/month,
  hard fail above ~$15/month. The dossier MUST include an itemized monthly cost
  worksheet (Lambda GB-seconds, Bedrock tokens/night, S3, ECR, data transfer) for
  the deployed design. A design that needs $100/month is a failed design.
- Daily cadence; nightly batch + morning execution shape (or justify a deviation
  that stays in the envelope).
- Training: local monthly (launchd pattern) on the operator's Mac; Lambda handlers
  stay thin; free data sources only (network allowlist applies); no new persistent
  paid infrastructure.
- Sample-size reality: daily bars × 65 symbols × ~10y public history is the
  training universe; the brain's own live record is only ~10 months. Designs must
  state their sample budget and why their learnable components won't just memorize.

## Sealed-incumbent rule (structural anti-anchoring)

- **Phase A designers work blind to the incumbent's strategy.** Allowed reads:
  this packet; `committee/EVIDENCE_PROTOCOL.md`; the assignment brief and data/infra
  inventory (Phase 0 output); `config/universe.csv`; `docs/DEPLOY.md` (infra shape);
  cost model interface (`src/utils/transaction_costs.py` signatures); replay harness
  interface (`src/utils/three_line_replay/replay_engine.py` signatures only).
- **Forbidden until designs are registered:** `src/steps/decision_engine.py`,
  `src/signals/`, regime fusion, `optimizer/fitness.py`, `summary.md`'s strategy
  sections, prior committee returns under `runs/`, and the incumbent's parameter
  bundles. "Registered" = candidate designs committed to the run dir.
- The run journal must show the registration commit BEFORE any incumbent-reading
  appears. After registration, the incumbent may be read freely — it becomes the
  bake-off opponent, nothing more.

## Committee panel (12 roles — breadth is the point; declare all before Phase 0)

1. **Analyst** (mandatory) — compiles the Phase 0 assignment brief + data/infra
   inventory (without leaking incumbent strategy logic); maintains the assignment
   scorecard; structures the dossier.
2. **Skeptic** (mandatory) — hunts hidden incumbent-anchoring in every design;
   attacks unlearnable targets, leakage in training design, complexity that exists
   to impress rather than decide; owns the "would this survive a quant's sneer"
   pass.
3. **Architect-Alpha: prediction-stack prior** (ad hoc) — designs from "forecast
   well, then act": supervised heads, meta-labeling, cross-asset sequence
   transformer.
4. **Architect-Beta: allocation-learner prior** (ad hoc) — designs from "skip
   forecasting, learn the action": learned sizing/allocation policies trained on
   realized forward utility.
5. **Architect-Gamma: information-edge prior** (ad hoc) — designs from the data in:
   the system exists to digest differentiated information (GDELT event structure,
   LLM-read text, entropy/flow measures) into decisions others can't make.
6. **Infotropy Canon Liaison** (ad hoc) — reads
   `/Users/drbretto/Desktop/Projects/Infotropy Book/shared-canon/` (read-only;
   write nothing in that repository) and attempts the canon→mechanism transfer per
   assignment item 6.
7. **Meta-Evaluator Designer** (ad hoc) — owns the executive: gating/stacking
   architecture, its decision-grade training targets, how it expresses "how much to
   listen to which model today," and how its judgment is auditable after the fact.
8. **Evolution Engineer** (ad hoc) — owns where the EA bites: what is evolved, what
   is gradient-trained, generation cadence, and how evolution stays meaningful
   rather than ceremonial.
9. **LLM Sentiment Engineer** (ad hoc) — owns the LLM organ: what text it reads
   nightly, where the text comes from free, the structured output schema, model
   choice (e.g. Bedrock Haiku-class), prompt/caching design, and the per-night token
   cost line for the worksheet.
10. **Data Edge Scout** (ad hoc) — owns differentiated data beyond GDELT and the
    historical-depth problem (signals need history deep enough for honest holdout
    reads).
11. **Training Realist** (ad hoc) — owns the sample-budget audit of every learnable
    component; forces simplification where data cannot support the ambition; designs
    the pretraining/fine-tuning split (public history vs own record).
12. **Feasibility Auditor** (ad hoc) — owns the cost worksheet and the envelope;
    kills or shrinks anything that breaches it BEFORE build effort is spent.

## Phased required work

- **Phase 0 — Brief.** Analyst compiles the assignment brief + inventory (data
  sources with historical depth, S3 layout, compute envelope, harness and cost-model
  interfaces). Committed before design work.
- **Phase A — Independent designs.** Architects Alpha/Beta/Gamma each draft a
  COMPLETE system (models, data, brain, evolution's role, LLM organ, cost sketch,
  training plan) independently — no cross-reading until all three are committed.
  Liaison (6) and roles 7–10 contribute mechanism proposals to all three blindly
  (they serve the assignment, not a faction).
- **Phase B — Tournament + synthesis.** Skeptic, Training Realist, and Feasibility
  Auditor attack all three; panel converges on ONE build candidate (synthesis
  allowed — graft the best organs; document what was taken from which design and
  what was killed and why). Pre-register the bake-off success criteria HERE, before
  any building: metrics, holdout reads, paired-stats form, and the assignment
  scorecard — per `committee/EVIDENCE_PROTOCOL.md`.
- **Phase C — Build.** Prototype the brain in the run dir: training pipeline,
  models, meta-evaluator, EA balancing, LLM organ (real calls on a bounded budget;
  cached/sampled where cost demands), GDELT ingestion. Everything seeded,
  manifested, re-runnable.
- **Phase D — Bake-off + attribution.** Head-to-head vs the incumbent on the
  IDENTICAL harness: same replay engine, same cost model, same data snapshots, same
  holdout discipline (E2 per EVIDENCE_PROTOCOL). Then leave-one-out attribution over
  the brain's organs (each model type, the LLM, GDELT, evolution, the
  meta-evaluator) — assignment item compliance is "every organ shows positive
  marginal value," and a zero-attribution organ is reported as such, with the
  honest implication for the showcase goal.

## Write surface

- `runs/pkt_tb_006_clean_sheet_brain/` — everything: brief, designs, tournament
  record, prototype code, manifests, dossier, bake-off evidence.
- `docs/plans/2026-06-10-clean-sheet-traders-brain.md`
- NOT: any production path (`src/`, `config/`, `training/`, `optimizer/`,
  `frontend/`), no deploys, no AWS resource creation beyond reading existing S3
  data and bounded Bedrock calls for the LLM organ. The Infotropy Book repository
  is strictly read-only.

## Stop condition

Dossier complete; prototype built; bake-off run with pre-registered criteria;
attribution matrix done; committee report converged; committed on
`ai/clean-sheet-traders-brain`. **No production integration, no deploy** — if the
brain wins, productionization is a follow-on packet authored from this return. If
compute forces shrinkage, shrink seeds/epochs before dropping organs or the
bake-off itself; state every reduction on the final line.

## Acceptance test

A first-time reader can answer: "What is the new system, organ by organ; does it
beat the incumbent on the holdout, by how much, with what confidence; and what does
each organ — transformer, ensemble members, evolution, LLM sentiment, GDELT,
meta-evaluator, infotropy mechanism (if adopted) — individually buy, in numbers?
And what would it cost per month on AWS?" A return that instead describes the
incumbent's strengths and weaknesses FAILS this test by definition. An honest
"built it, it loses to the incumbent by X" with a clean dossier PASSES — that is a
real answer; another evaluation report is not.

## Return artifacts

- `runs/pkt_tb_006_clean_sheet_brain/ASSIGNMENT_BRIEF.md` (Phase 0)
- `runs/pkt_tb_006_clean_sheet_brain/designs/DESIGN_{ALPHA,BETA,GAMMA}.md` + `INFOTROPY_TRANSFER.md`
- `runs/pkt_tb_006_clean_sheet_brain/TOURNAMENT.md` (incl. pre-registered bake-off criteria)
- `runs/pkt_tb_006_clean_sheet_brain/prototype/` (code + manifests)
- `runs/pkt_tb_006_clean_sheet_brain/BAKEOFF.md` + `ATTRIBUTION.md` + `COST_WORKSHEET.md`
- `runs/pkt_tb_006_clean_sheet_brain/COMMITTEE_REPORT.md` (required fields as in PKT-TB-002)
- Commits on `ai/clean-sheet-traders-brain`

## Hard constraints

- **Sealed-incumbent rule is absolute in Phase A.** Violation = restart the design
  phase with fresh framing, and say so in the report.
- **Do not repeat the named failure:** no incumbent evaluation content anywhere in
  the dossier. The incumbent appears exactly twice: as the bake-off opponent's
  score line, and (post-registration) in harness-wiring notes.
- Assignment items 1–6 are MANDATORY design content — a "simpler is better" verdict
  that deletes the showcase is out of scope (the operator has explicitly ranked
  assignment over P&L). Within the items, the panel's enumeration governs; this
  packet's examples are kinds-of.
- Budget: cost worksheet mandatory; >$15/month projected = redesign before build.
  Bounded real-LLM spend during the run: keep Bedrock usage to prototyping scale
  and record actual dollars in the worksheet.
- All evidence per `committee/EVIDENCE_PROTOCOL.md`: pre-registration before
  holdout reads, every variant logged, holdout looks counted, fixed seeds,
  manifests sufficient to re-run any verdict.
- Free data only; allowlist applies; nothing paid gets signed up for.
- The canon record is never restated; no production writes; no deploy.
- If any referenced path is missing, stop and hand back the discrepancy.

## Methods audit

Triggers matched from `Infotropy Book/shared-methods/`:
- `EVIDENCE_LADDER.md` — work_shape=evidence-grading (via `committee/EVIDENCE_PROTOCOL.md`).
- `NOVELTY_TRIAGE.md` — packet_kind=discovery, work_shape=intake-classification:
  Phase B's tournament triages three whole-system candidates plus organ proposals.
- `ANTI_BULLSHIT_ASSESSMENT_DESIGN_METHOD.md` — work_shape=assessment-design: the
  pre-registered bake-off + attribution battery is an assessment that must survive
  vagueness pressure (the Skeptic owns this audit).
Off-trigger inclusion: `INFOTROPIC_ANALYTICAL_POSTURE.md` —
off_trigger_justification: the Infotropy Canon Liaison needs the canon's own
analytical posture to attempt an honest mechanism transfer (assignment item 6).
Lakatos firewall: excluded — no programme-shift verdict surface.

## Final line requirement

COMMITTEE_REPORT.md final line:
`BRAIN vs INCUMBENT: <BEATS|TIES|LOSES> by <Δreturn, ΔSharpe> on holdout (<paired-stat>); ASSIGNMENT SCORECARD: transformer=<attr> ensemble=<attr> evolution=<attr> LLM=<attr> GDELT=<attr> meta-evaluator=<attr> infotropy=<attr|no-transfer>; COST: $<x>/mo — <reductions taken, if any>`.
