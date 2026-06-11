# REVIEW_SKEPTIC_007_PHASE_D — Skeptic's evidence review (PKT-TB-007, Phase D)

**Role:** Skeptic (panel role 2, mandatory). **Date:** 2026-06-11.
**Sources:** TOURNAMENT_007 §4 (pre-registration), RUN_JOURNAL.md (all three chair
adjudications), FREEZE_ORB1.md (+ addendum), BAKEOFF_007.md, ATTRIBUTION_007.md,
COST_WORKSHEET_007.md, evidence_007/ (scorecard + 8 comparisons + final line),
holdout_looks_007.jsonl, validation_looks_007.jsonl, ledgers/ (all six), my own Phase B
attack (ATTACK_SKEPTIC_007). Underlying series spot-checked and **recomputed** where
load-bearing: `designs/c2_gate_matrix.json` (full matrix), `store/surrogate_007/
anchor_007_rerun.json`, `runs_battery_007/D01_INC` + `D03_APRIORI` daily series (my own
regression, §F8), git commit timeline (`0b52981 → b305034 → 40f656b → 85b0572 → 0bd78fe`).

Every claim below was verified against the artifact named, not the prose that cites it.

---

## Findings

### F1 — Chair adjudication 1 (C2 G-book instrument defect): **CLEAN** — legitimate instrument critique, not a quiet gate-relaxation

I recomputed the call from `designs/c2_gate_matrix.json` directly:

- Signal space: max pooled |ρ| = **0.202** (M1–M4), all ten directional pairs ≤ .202.
  Matches every quotation of it.
- Book space: all ten pairs pool at **.881–.969**. The decisive internal control is
  **M2–M4: signal ρ = +0.004, book ρ = +0.932**. Two statistically independent signals
  cannot produce 93%-correlated books through organ redundancy; only a common factor
  loading can do that. The instrument measures the market beta of long-only unit-gross
  books, not what the gate was registered to detect. The arithmetic is sound.
- Market-residualized book diagnostic: max **+0.498** (M1–M4) ≤ .70, next +0.373 — the
  corrected form of the same instrument passes. (Note .498 is half the bar, not
  negligible; it is printed, which is all I asked of it.)
- Counterfactual mechanical reading: 9 pairs fired the replication trigger; the ladder
  (≤4 rungs system-wide, then demote by seniority) would have demoted essentially the
  entire roster on a demonstrated instrument artifact. Refusing to fire 10
  instrument-driven breaches as 10 redundancies was the correct call.
- Process: the executor did NOT self-relax — the artifact's own `GATE` field still reads
  `REMEDIATION REQUIRED` and `ladder_disposition: ESCALATED, no rungs burned`; the chair
  adjudicated in a journaled commit (0b52981) BEFORE the freeze, with a pre-stated
  vindication path that Phase D then actually answered (challenger tilts produced sub-bp
  marginal books, not near-identical competing books).

Residual honesty point, correctly handled: the registered text made BOTH spaces binding,
so this is a post-registration criteria change — and it is reported as exactly that ("a
pre-registration defect") everywhere the gate result is quoted, including the final line.

### F2 — Chair adjudication 2 (anchor remediation branches): **CLEAN** — pre-stated, then applied at its harshest branch with no argument-from-consequences

Commit order verified: branch rules journaled 01:28 (0b52981), re-run artifact 01:33–01:36,
re-run commit 01:38 (b305034). The realized Pearson 0.2818 fired the **< 0.5 ⇒ B0 ships
outright** branch mechanically; the EA production sequence never ran (ea_cycles 0/3); the
restriction prints in `ea/ship_decision_007.json` and every downstream surface. The re-run
was ledgered as a replay arm AND a remediation rung (1/4). Window 2026-02-03→03-06,
pre-holdout only, guard structurally unable to touch ≥ 03-11. The rule destroyed the
packet's own headline machinery and was honored anyway — that is what pre-stated rules
are for. One critique of the rule's construction (not its application): the specific
representative genome VALUES were not in the adjudication-2 text; they were chosen at
re-run time (see F3.ii).

### F3 — Chair adjudication 3 (value-blind a-priori deviation battery): **CONCERN** — attestation overdrawn as worded; substantively credible on the evidence

This is the one that matters, and the timeline has a real wrinkle:

- **The wrinkle.** The attestation says no panel role and not the chair "has read the
  genome's pre-holdout or holdout P&L as of this entry." But the REGISTERED FM5
  mean-equivalence leg of the anchor re-run computed and stored exactly that genome's
  pre-holdout mean paired delta — `anchor_007_rerun.json` contains
  `mean_real = +3.15e-05` (**+0.32 bp/day, n=20**) — at 01:33, seven minutes BEFORE
  adjudication 3 was journaled (40f656b, 01:40:28). The chair demonstrably handled this
  artifact (the < 0.5 branch was fired from its Pearson) and the wave-3 journal entry
  quotes the equivalence gap, which is a function of mean_real. The attestation as
  worded cannot be literally airtight. It should have read: "no full-window or holdout
  read exists; the registered anchor equivalence check necessarily exposed a 20-day
  pre-holdout mean (+0.32 bp/day), disclosed."
- **Why I nonetheless grade it credible rather than violated:**
  (i) **No full-window or holdout P&L of any non-neutral genome existed anywhere before
  the battery.** Verified from the replay ledger and the runs directory: the only
  pre-battery tilt replays are SMOKE_TILT and ANCHOR_RERUN, both pre-holdout (ending
  03-06). The first live-window run of the a-priori genome is D03 at 01:50, after the
  freeze. Nothing holdout-grade was peekable.
  (ii) **Genome provenance is mechanical and predates the values' only readable P&L.**
  tilt_gain 0.5 is mid-range AND the wave-1 smoke value committed Jun 10 22:24 (before
  any anchor read existed); disp_gain 1.0 mid-range; trust {M1:+1} is roster-mechanical;
  caps at B0 mids. One value is NOT mechanical: dead_zone 0.05 is the smoke carryover,
  not the mid of [0, 0.3] — flagged for the record, though it traces to the 22:24 commit
  all the same.
  (iii) **The leak hypothesis predicts the wrong sign.** The only peekable number was
  mildly POSITIVE (+0.32 bp/day pre-holdout), and the dossier leads with the NEGATIVE
  full-window deviation read. A chair selecting a genome to flatter a peeked number got
  the opposite of flattery and printed it.
- **Labeling and quarantine held** (F4, F5). The holdout firewall for fitness/selection
  held throughout (nothing ≥ 03-11 touched any training, gate, mask, or genome choice).

Net: the deviation battery was the right call honestly executed, with one overdrawn
sentence of attestation that a hostile reader could quote. The committee report should
carry the corrected wording, not the original.

### F4 — Labeling discipline: **CLEAN**

All 8 `evidence_007/*/comparison.json` manifests carry an explicit label field (REGISTERED
vs the full deviation label); every numeric block in BAKEOFF §3 and ATTRIBUTION sits under
a header carrying *(deviation: a-priori genome, instrument-failed EA)*; the final line
labels the deviation read inline; `holdout_looks_007.jsonl` has 9 entries, each labeled
and timestamped before its run completes (cross-checked against replay_arms completion
timestamps); the scorecard separates `registered_verdict` / `deviation_verdict` with the
label embedded. My grep for unlabeled deviation numbers found none.

### F5 — Deviation read leaking into registered framing: **CONCERN (minor, one sentence)**

The registered verdict cell (BAKEOFF §1) is clean of deviation numbers. But ATTRIBUTION's
Evolution section uses the deviation read to grade the chair's own rule: "its read came
back negative-pointing, which is information FOR the instrument-failure rule… an EA…
would have shipped tilt_gain > 0 into a window where the tilt **lost**." "Lost" is
directional wording hung on an uncertifiable t ≈ −1.3 read — and per F8 the mechanism is
mostly an exposure leak, not selection. Strike or soften to "into a window where the
tilt's point estimate was negative (mostly an exposure-leak artifact, F8)." Self-serving
adjudication-vindication prose is exactly where deviation reads must not do work.

### F6 — Protocol conformance: **CLEAN**, two notes

- Git ordering: pre-registration (2122e09) → build waves → adjudications (0b52981) →
  anchor re-run (b305034) → adjudication 3 (40f656b) → freeze (85b0572) → battery
  (0bd78fe). Correct.
- Registered E2: exactly 1/1 in `ledgers/looks_holdout.json`, orchestrator-fired with the
  env guard. Caps: replays 10/18, retrains 0/9, EA 0/3, remediation 1/4. B0-ship applied
  mechanically. No deviation arm upgraded anywhere; the registered cell leads every
  surface.
- Note (a): wave-1 B0_EXPR and SMOKE_TILT replays predate the ledger's initialization and
  are absent from replay_arms.json; cap math is unaffected (12/18 if counted). Bookkeeping
  gap only.
- Note (b) — **the real price of adjudication 3, for the committee report:** the deviation
  battery read and published holdout-subset statistics for the a-priori genome family
  (D03–D09). The 2026-03-11→06-10 holdout is now **spent** for this genome/roster family;
  any successor packet evaluating a similar tilt must treat it as seen and obtain fresh
  holdout. This obligation should be stated explicitly in the committee report.

### F7 — D03 prose tone: **CLEAN**, one inconsistency noted

Nowhere does the dossier lean kinder than "negative-pointing, uncertifiable." If anything
BAKEOFF's "the value-blind M1 tilt **cost** an uncertifiable ~2 bp/day" is HARSHER than
the registered TIES(straddling) cell permits (the cell bans directional wording) — a
letter violation in the unflattering direction, inside a labeled deviation read, which I
do not propose to punish. The inconsistency: ATTRIBUTION's "the realized sign happened to
be negative" (noise framing) and BAKEOFF's "cost" (causal framing) cannot both be right;
per F8 the noise framing is closer to the truth for the selection component, and the
causal framing is right only about the exposure leak.

### F8 — β-parity / exposure: **CONCERN — the deviation read is substantially an exposure read; disclosed, but under-quantified and direction-masked**

I recomputed from the raw daily series (D01_INC vs D03_APRIORI, cost-adjusted):

- Regressing daily paired deltas on the incumbent's daily return: **slope −0.1175** — the
  tilt arm realized ~12% LESS market exposure than the incumbent, consistent with the
  one-signed average-gross shortfall visible in the comparison tables (34.68% vs 39.07%).
- The window's incumbent mean return was **+14.4 bp/day**. Exposure component of the
  deviation mean = slope × mean = **−1.70 bp/day = ~86% of the −1.98 bp/day** headline
  deviation mean. Exposure-stripped residual: **−0.29 bp/day, t ≈ −0.22** — a dead zero.
  (Holdout: residual ≈ −1.2 of −3.1.)
- Mechanism: the cash-neutral projection is exact ex-ante (residuals ~1e-12, as printed),
  but quantization, no-short clamps, min_order floors and no_tilt_capacity days leak
  gross one-directionally downward, and the window rose. "Matched exposure" held by
  construction and failed in realization — which the dossier DOES disclose (trigger
  fired, mean |Δβ| .069, gross gap 4.4pp), but (a) it reports absolute gaps, masking that
  the gap is one-signed, and (b) it never converts the gap into bp/day.
- Consequences: the answer to the packet's question is **yes — the deviation read is
  partially (≈86% of its point estimate) an exposure read.** This cuts BOTH ways: the
  −1.98 bp/day is even weaker evidence that M1 selects badly (selection residual ≈ 0),
  AND the M4-A positive becomes more mechanically suspect (F9). No verdict cell changes —
  TIES(straddling) stands either way — but the honest gloss on D03 is "≈ −1.7 bp/day
  implementation/exposure leak + ≈ −0.3 bp/day uncertifiable selection residual."
  **Repair requested:** add this decomposition (or the executor's own version of it) to
  BAKEOFF §3.1 / the committee report. It is the single most interpretation-changing
  number not currently in the dossier.

### F9 — M4-A +1.84/+2.18: **CLEAN as printed**, strengthen the mechanism

The verdict discipline held: only the full-window E1 HAC t (+1.84 < +2) carries the
three-valued tag; the two rails (multiplicity ≈25–30% family-wise; width-of-losing-tilt)
block a discovery claim. F8 sharpens the second rail: M4's damp shrinks the very tilt
whose dominant realized effect was a downward exposure leak in a rising window, so a
width cut partially restores gross and scores positive mechanically — fold that into the
caution. The holdout HAC +2.18 in the BAKEOFF table is quotable out of context; recommend
a parenthetical "(confirmation window; not a verdict statistic)" beside it.

### F10 — Acceptance test (first-time reader): **PASS**

The final line gives, in order: the registered degenerate TIES with its cause (brain
ships neutral; evolution instrument-failed), the labeled deviation read with negative
point estimates and MDEs, orthogonality PASS with the instrument defect named, per-organ
tags with cautions inline, the LLM retirement with receipts, EA instrument-failed, and
cost with reductions. A first-time reader gets the honest answer: *no certified
improvement; the one real tilt read pointed negative and uncertifiable; the orthogonality
engineering demonstrably worked at the signal level; the evolution instrument failed its
own certification and the system shipped neutral by rule.* No text drifts into evaluating
the incumbent. What a first-time reader would currently miss: the F8 exposure
decomposition and the one-signedness of the gross gap — hence the repair request.

---

## Statement for the committee report

The Skeptic verified the three chair adjudications against the artifacts and the git
timeline and grades them: adjudication 1 CLEAN (the G-book instrument demonstrably
saturates on the market factor — a ρ=.004 signal pair books at .93; the corrected
market-residualized diagnostic passes at .498; the executor escalated rather than
self-relaxed, and the call is disclosed as a pre-registration defect, not buried);
adjudication 2 CLEAN (branches journaled before the re-run, harshest branch fired
mechanically, EA never ran, 0/3 cycles); adjudication 3 CONCERN-but-credible (the
attestation is overdrawn as worded — the registered anchor equivalence check had already
computed and stored the genome's 20-day pre-holdout mean, +0.32 bp/day, before the
adjudication was journaled — but no full-window or holdout read of any non-neutral genome
existed before the freeze, the genome values trace to mechanical mid-range/smoke
provenance committed hours earlier, and the only peekable number was positive while the
dossier leads with the negative result; labeling discipline is airtight across all
surfaces and the deviation read never touches the registered verdict). Protocol
conformance is otherwise clean (look budgets, single E2, caps, B0-ship rule). The
Skeptic's own recomputation adds the dossier's one missing load-bearing number: ~86% of
the deviation read's −1.98 bp/day is a one-signed realized exposure leak (slope −0.118 on
a +14.4 bp/day window; selection residual −0.29 bp/day, t −0.22), so the deviation read
is substantially an exposure read — which weakens it as evidence in BOTH directions and
changes no verdict cell. Two repairs requested (add the exposure decomposition to the
dossier; strike the "window where the tilt lost" vindication sentence) and one carried
obligation (the 2026-03-11→06-10 holdout is spent for this genome family). **Verdict: the
return PASSES the packet's acceptance test** — the registered degenerate TIES is honest,
mechanically derived, and a first-time reader gets the true answer from the final line.

*— Skeptic, panel role 2, Phase D evidence review, 2026-06-11*
