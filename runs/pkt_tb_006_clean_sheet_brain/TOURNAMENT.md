# TOURNAMENT.md — Phase B record + PRE-REGISTERED bake-off criteria

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — **Synthesis Chair, 2026-06-10**
**Inputs:** packet; ASSIGNMENT_BRIEF.md; EVIDENCE_PROTOCOL.md; NOVELTY_TRIAGE.md (methods
audit binding); DESIGN_{ALPHA,BETA,GAMMA}.md; INFOTROPY_TRANSFER.md; the four mechanism
proposals; ATTACK_{SKEPTIC,TRAINING_REALIST,FEASIBILITY}.md.
**Status:** this document is committed BEFORE any build (per EVIDENCE_PROTOCOL and packet
Phase B). §4 is the pre-registration of record. The incumbent appears in this document only
as the bake-off opponent and harness wiring — no incumbent-evaluation content anywhere.
**Novelty-triage note (methods audit):** the three whole-system candidates are classified
**bounded experiments** converging to one build; the organ proposals are components under
them, not separate process objects; no canonical-implication item arose. Anti-nerf honored:
each candidate was attacked at its strongest stated form before any kill.

---

## 1. Candidate scoreboard

Six assignment items + envelope + buildability, with each attacker's verdict condensed.
(Skeptic = SK, Training Realist = TR, Feasibility Auditor = FA.)

| Criterion | ALPHA "Forecast First" | BETA "Bookwright" | GAMMA "Information Funnel" |
|---|---|---|---|
| **1. Ensemble of genuinely different types** | **STRONGEST** (SK §5): diversity guaranteed by disjoint input pathways (price-only / tabular-exogenous / events-only / risk / trigger). TR: most honest training plan; CAST oversized at 45k (UNSUPPORTED; 22k MARGINAL) | WEAKEST (SK B-K1): same loss + overlapping state ⇒ correlated books ⇒ τ-unidentifiability; TR: P1 26k on one scalar loss/day = UNSUPPORTED ("credit densification" = the program's load-bearing hand-wave) | Close second (SK): genuinely different targets; TR: ETT 28k vs ~2–3k effective = worst ratio in the program; ISRE lacks any simple twin |
| **2. Evolutionary balancing** | Weakest bite (SK): proposal + 3 genes, standard risk knobs only | **STRONGEST bite** (SK): slot-selector library — EA selects loss shaping/composition. FA: the largest single training multiplier in any design (×18 P1 trainings); kill for Phase C | Second-best bite (SK): funnel-gate genes (streams must pay their way) — graft regardless |
| **3. LLM sentiment organ** | **STRONGEST** (SK, narrowly): proposal adopted in full, most explicit feature routing, transformer kept text-free so the LLM arm is cleanly removable | Strong; cleanest adjudication of the Scout/LLM training conflict (trainable only where real, masks mandatory) | Highest ceiling (annotator) but breaks its own token budget (SK G-K2 ⇒ routine fail-soft dark) and feeds a contaminated gate; FA: breaches the $5 Bedrock cap headroom as designed |
| **4. GDELT load-bearing** | Adopts G1–G5 fully; routing explicit | Weakest (SK; Beta's own §15.8 concedes): buried as P4 + context columns | **STRONGEST** (SK): structural backbone, sleeve-resolved, most ablation paths, only design naming the feed-shift hazard |
| **5. Learned meta-evaluator** | MetaTrigger insertion adds an approximation layer with a genome singularity + stated train/serve mismatch (SK A-K2) | **STRONGEST** (SK): book-space convex blend — exact counterfactual ledger, fewest approximation layers; TR: MARGINAL→SUPPORTED, best executive instance | Adopted proposal + the information-health block — "the best one-line executive improvement on the table" (SK); LLM-as-member has undefined ledger pre-2024 (G-K6) |
| **6. Infotropy mechanism** | **STRONGEST** (SK): only leak-correct split of Transfer A (train-time family screen; inference R2-only); eps on genome; clean separate A/B arms | Middle; best eps *reasoning* (0.25 floor — TR endorses) | Weakest as written: Transfer A per-event inference gate is the tournament's biggest kill (look-ahead / train-serve impossibility, SK G-K1); eps=0.05 starves inaction learning |
| **Envelope (≤$10/mo)** | PASS — +$0.22 marginal, ≈$9.5 absolute (FA) | PASS — +$0.22 marginal, ≈$9.9 absolute (carry artifact) | PASS — +$0.30 marginal, ≈$9.4–9.6 after FA corrections |
| **Phase C buildability** | ≈6–8 h foreground (FA); CAST cycles dominant | ≈8–11 h; most bespoke machinery to debug; 20-arm battery | ≈6–9 h compute but highest engineering variance (dictionary/funnel); R3-as-written infeasible per-event (FA B.4) |
| **Leakage audit** | PASS (SK A-K5) + shared items | PASS, best of three (SK B-K5) | FAIL as written (G-K1) until the gate is restated |
| **Anchoring (sealed-incumbent)** | none found | none found | none found |

**Attacker bottom lines:** SK — "no design survives intact; Beta's spine is the strongest
chassis; import Alpha's diversity-by-input partitioning and infotropy-A scoping, plus
Gamma's GDELT backbone and information-health block." TR — "most honest training plan:
Alpha; most surgery needed: Gamma; Beta's central training claim is the largest single
overstatement; put the transformer chips on the CAST slot; one well-fed transformer beats
two starving ones." FA — "no design fails the envelope; the discriminator is buildability;
spine from the shared proposals; transformer slot: ETT (cost view)."

---

## 2. Convergence decision — the ONE build candidate: **SYN-1**

SYN-1 is a synthesis, not a winner-take-all. Shape: **supervised, input-partitioned
perception members (Alpha's diet) feeding a book-space convex-blend learned executive
(Beta's chassis) that is information-health-aware (Gamma's gift), balanced by a
disposition-genome EA with funnel gates (proposal + Gamma), fed by the GDELT-rich funnel +
LLM organ (Scout + LLM Engineer via Gamma's backbone framing), with the two Infotropy
transfers in Alpha's leak-correct scoping.** Utility-gradient training lives in the
executive ONLY; members stay supervised (FA D.6 — halves the bespoke machinery).
Full buildable spec: `prototype/BUILD_SPEC.md`.

### 2.1 Graft table — what was taken from which design/proposal

| # | Organ / mechanism | Taken from | Form in SYN-1 (post-forced-simplification) |
|---|---|---|---|
| 1 | **Transformer member: CAST-Small** | Alpha E1, resized by TR A1/§2.1 | 1 encoder layer, d_model 32, 4 heads, FFN 64, shared GRU(14→32), dense next-day auxiliary head (w=0.3), per-day Gaussian-rank target (TR A2), wd 1e-3, ≈16k params. Ridge twin on identical features is the qua-transformer attribution referee; **transformer ships in the brain regardless** (assignment item 1 mandates presence); attribution is measured, honest zero reportable |
| 2 | **Tabular member: GBM-Cond** | Alpha E2, capped by TR A3; absorbs Gamma's GBM ranker (FA D.3: one GBM only) | HistGradientBoostingClassifier ≤150 iters, 15 leaves, binding early stop; P(5d excess>0) → probit z → cross-sectional z; macro/vol-surface/COT/GDELT/LLM features with masks |
| 3 | **Event member: EventHead** | Alpha E3 + SK A-K3 fix | 27-bucket elastic net (675 coefs) with **LLM mask-interaction columns**; abstains on quiet nights; absorbs Gamma's event-direction job at linear scale |
| 4 | **Risk instrument: RiskNet+** | Alpha E4 + Gamma's correlation/book-vol target (FA D.4) | Ridge heads: per-symbol 5d vol, beta, and forward 5d equal-weight-book vol; HAR-style features; ≈150 coefs; not a trust-head member |
| 5 | **Executive** | PROPOSAL_META_EVALUATOR, via **Beta's book-space convex-blend adaptation** + **Gamma's information-health block** (6 features) + SK §0.5 per-fold ledger standardization + TR A5 fine-tune restriction + TR §5 LOFO executives | Members' μ/σ → fixed-rule solo books; executive = shared-φ trust head + sizing head (~560 params) blending BOOKS convexly; exact counterfactual ledger; entropy floor 0.05; linear-gate ladder; trains by differentiable decision replay with **slippage noise injection** (SK B-K2) |
| 6 | **Evolution** | PROPOSAL_EVOLUTION machinery verbatim + **Gamma's funnel-gate genes** + Alpha's `record_weight_eps`/`event_weight_cap` genes; TR S2 LOFO-executive fitness | (μ+λ) GA, P=28, G=14, K≤400; 27-gene genome; B0 default-in-population; adoption gate >1 cross-fold sd; B1 random-search control; B2 = the E2 attribution read; embargo 22d (TR S1) |
| 7 | **LLM organ** | PROPOSAL_LLM_SENTIMENT in full, with **Alpha's routing** (features → GBM-Cond, EventHead, executive context; CAST text-free) | Pinned Bedrock Haiku, ≤120 clusters/night, one call, temp 0, fail-soft never-fabricate; Tier-1 backfill MANDATORY ($0.40, covers replay window + entire holdout); Tier-2 conditional; Phase C Bedrock hard cap **$3.10** (FA); model-cutoff rule §4.6.1 |
| 8 | **GDELT** | Scout G1–G5 via Gamma's backbone framing | Top-up 2026-02-05→present = build task #0; deep backfill at prototype scale (2/day pre-2023, 4/day 2023→, FA B.1); trailing-90d z; `visible_from` joins; distribution-shift gate §4.6.2; permuted-dictionary placebo §4.6.3 program-wide; dictionary frozen BEFORE any validation-fold model selection (TR G4) |
| 9 | **Data edge** | Scout: S1 CBOE (adopt), S3 FRED (adopt), S2 COT (context-only, publication-keyed, shuffle placebo), S0 breadth (folded in) | Per Scout's falsifiers |
| 10 | **Infotropy Transfer B** (anti-HILL record weights) | Liaison B, Alpha's genome form, **Beta's floor reasoning**, TR S3 | Sample weights on CAST/GBM/executive losses; gene `record_weight_eps ∈ [0.20, 0.50]`, default 0.25; mandatory uniform-vs-weighted A/B twin |
| 11 | **Infotropy Transfer A** (record-grade gate) | Liaison A in **Alpha's leak-correct scoping**, R3 per-family (FA B.4) | Train-time R1∧R2∧R3 feature-FAMILY screen on event features entering EventHead/GBM; inference-time R2-only price-echo down-weight ×0.25; R3-only twin falsifier |
| 12 | **Floor baselines + guards** | Beta: P2-style linear floor pair + dumb-twin guard (validation-only, zero holdout looks), turnover budget line (≤6%/day one-way), shared-fetch Lambda lever; TR §3.4 solo-book-correlation statistic + SK B-K1 diversity floor | Pre-registered §4.6 |

### 2.2 Kill table — every killed component, with the attack that killed it

| # | Killed component | Source design | Killed by | Reason (one line) |
|---|---|---|---|---|
| K1 | **P1 transformer policy (26k, end-to-end on utility)** | Beta | TR B1/§2.2 + FA D.6 | One scalar loss/day ⇒ ≤~7k constraint-equivalents vs 26k params = UNSUPPORTED; inseparable from per-member differentiable-replay machinery |
| K2 | **Per-member differentiable replay** | Beta | FA D.6 | Utility-gradient training lives in the executive only; members supervised — halves bespoke machinery to debug |
| K3 | **Policy library + slot-selector genes** | Beta | FA D.6 (over SK's ranking — see Dissent D2) | Largest single training multiplier (36 P1 trainings/cycle); Phase C unaffordable |
| K4 | **ETT event-theme transformer (28k) in the transformer slot** | Gamma | TR G1/§2.3 + SK G-K1/G-K3 | ~2–3k effective constraint-equivalents vs 28k params (worst ratio in program); richest token features exist on ~120 windows; fed by a contaminated gate as written |
| K5 | **Per-event inference-time record-grade gate (R1·R2·R3 > τ at 03:00 UTC)** | Gamma §6.4 | SK G-K1 (the tournament's biggest kill) + FA B.4 | R1 needs future returns; R3 per-event is combinatorially infeasible; as-written = look-ahead or undefined at decision time. Replaced by graft #11 |
| K6 | **LLM event-token annotator extension** | Gamma §6.3 | SK G-K2 + FA B.1 | +1.8k output tokens vs max_tokens 1500 ⇒ guaranteed truncation ⇒ routine fail-soft dark; breaches Bedrock cap headroom; consumer (ETT) dead anyway |
| K7 | **ISRE GRU regime encoder (12–14k)** | Gamma | TR §3.6/G2 | HAR-RV+EWMA captures most of vol/corr forecastability with ~3 coefs; no twin pre-registered; replaced by RiskNet+ ridge heads |
| K8 | **Member-5 COT contrarian** | Gamma | TR §1.3 + Scout's own skepticism | ~13 holdout obs ⇒ unevaluable as an organ; COT survives as context features in GBM-Cond + executive z |
| K9 | **LLM-as-μ-member (trust-head citizen)** | Gamma §3.4 | SK G-K6 | No ledger history for 80% of pretraining; executive pretraining undefined for that member. LLM is an input organ (Alpha routing) |
| K10 | **MetaTrigger standalone organ (E5)** | Alpha | SK A-K2 + TR §3.5 | Genome singularity at p_min=0.5; trains on a blend it never serves; label ≈ vol (duplicates RiskNet); 4th redundant cost brake. Its cost-vs-edge features fold into the executive sizing head (meta-labeling claim preserved via sizing-head attribution) |
| K11 | **CAST-45k variant** | Alpha | TR A1 | 45k vs 7–12k honest effective; 22k/16k rung is the registered form; 45k not trained in Phase C (wall-clock); logged as a declined challenger |
| K12 | **Gamma GBM ranker** | Gamma | FA D.3 | Same sklearn family as GBM-Cond; two GBMs = one redundant LOO arm |
| K13 | **S4 Treasury auctions** | Scout (kept small) / Gamma (adopted) | Scout's own falsifier arithmetic + Alpha/Beta concurrence | Episodic; <10 holdout firings ⇒ no honest holdout verdict possible |
| K14 | **Infotropy C, D, E as infotropy claims** | Liaison | Liaison's own RESTATEMENT grades (all three architects concur) | Standard quant in canon vocabulary; their content exists under standard names; never sold as infotropy |
| K15 | **Infotropy F (possibility-space/exergy)** | Liaison | Liaison NO-TRANSFER | Rename of vol-squeeze; canon's own STOP rule forbids it |
| K16 | **Per-arm seed replication of the battery** | (all) | FA B.3 ruling | ×3 replication = 12–20 h buying nothing the protocol needs; internal seed-ensembles + 2 seed-sensitivity replays instead |
| K17 | **Member-drop arms via executive retrain (M−1)** | Alpha §11 | FA battery arithmetic | Trust renormalization (Beta's form) is the registered drop mechanism — entropy floor keeps every member's ledger live, so renorm is valid and free |
| K18 | **Alpha gene `p_min`** | Alpha §6 | Dies with K10 | MetaTrigger gone; gene removed from genome |

---

## 3. Dissent record

Real disagreements among panel members, the adjudication, and what Phase C/D evidence would
vindicate the overruled position. No theater: items where all attackers agreed (e.g. the
holdout-power arithmetic, the GDELT top-up as task #0) are not listed as dissents.

**D1 — The transformer slot (the tournament's sharpest conflict).**
*Positions:* FA D.2: take Gamma's ETT — cheapest to train, cheapest to attribute, food on
which a transformer is least replaceable. TR §2.4: CAST-Small > ETT-Small > P1 by honest
probability of genuine positive attribution; "put the transformer chips on the CAST slot";
ETT only at challenger scale. SK: at 45k Alpha's was the weakest transformer instance, but
SK endorsed the inverted ladder and ranked Alpha's ensemble structure strongest.
*Adjudication:* **CAST-Small takes the slot.** The assignment-decisive question is "which
transformer has the best honest chance of measured positive attribution," and that is
governed by gradient-information supply (TR's arithmetic: CAST's cross-sectional supervised
target delivers ~12 independent constraints/day vs ETT's ~8 halved by event-conditioning,
on top of ETT's contaminated gate and ~120-window annotation food). FA's cost objection is
answered structurally: the 45k→16k resize plus 1-layer cut roughly halves the dominant
training item, and the shrink ladder (seeds 3→2, deploy 5→3) stands behind it.
*Vindication for the FA/ETT position:* if in Phase C CAST-Small fails to beat its ridge
twin on purged validation Spearman while the Infotropy-A family screen shows strong R3 lift
on event features, the event-interaction hypothesis earns a follow-on packet with ETT at
TR's G1 size (8–10k, vocab-floored). That outcome would mean the transformer chips were on
the wrong food, exactly as FA argued.

**D2 — Evolution's bite: library vs disposition genome.**
*Positions:* SK §5 ranked Beta's slot-selector library the strongest EA bite ("the only
design where the EA decides something gradients also wanted to decide"). FA D.6 killed it
as the largest training multiplier.
*Adjudication:* **killed for Phase C buildability** (K3). The EA retains real bite via
funnel gates (which streams pay their way), trust priors/halflife, member gates, allocator
risk genes, and the two infotropy genes — plus the pre-committed B0/B1/B2 ceremonial-proof
battery, so "EA matters" remains a measured claim, not a ceremony.
*Vindication for SK's position:* if Phase D reads `evolution = 0 (measured)` or the
adoption gate fails while B1 random search matches the GA, the "EA needs loss-shaping genes
to bite" hypothesis is the first follow-on candidate, and the library is its design.

**D3 — LLM features as training inputs.**
*Positions:* Data Edge Scout §1.4: "never a trained-on feature" (history too shallow). LLM
Engineer: Option A backfill exists precisely to train on them.
*Adjudication:* **for the Engineer, with Beta's guard** (all three architects independently
ruled the same way): LLM features are trainable only where real output exists (Tier-2
window, ~520 days), always behind `llm_available` masks, with mask-interaction columns in
linear models (SK A-K3); no deep-history component conditions on them; CAST stays text-free.
*Vindication for the Scout:* if the LLM arm reads positive on E1 but the effect is
concentrated in a single macro episode of the Tier-2 window (pre-registered sub-check:
drop the largest-|contribution| month and the E1 t falls below 1), the Scout's
shallow-history warning is confirmed and the dossier says so.

**D4 — Record-weight floor eps.**
*Positions:* Liaison sketch 0.05; Alpha gene [0.05, 0.50]; Beta fixed 0.25 (inaction-
learning argument); TR S3: floor ≥0.2 binding (effective-sample cost); SK G-K5 concurs.
*Adjudication:* gene range **narrowed to [0.20, 0.50], default 0.25** — Beta's reasoning,
Alpha's genome form, TR's floor. Alpha's wider lower range is overruled.
*Vindication for the low-eps position:* if the A/B twin shows record-weighting helps and
the EA repeatedly pins eps at the 0.20 boundary across months, the range is loosened in a
follow-on (never mid-run).

**D5 — Infotropy arm form.**
*Positions:* SK 3.4(iii): A and B must have separate arms (one combined arm is
unattributable). FA B.3: honest-minimum battery has one combined infotropy arm.
*Adjudication:* **both honored at zero extra replay cost**: Transfer A is measured at
training-fold level (gated vs R3-only twin on walk-forward folds — an E1 read, no replay,
no holdout look); Transfer B gets the replay arm (uniform-weights retrain). The scorecard's
`infotropy=<attr>` prints B's replay attribution with A's fold-level verdict in parentheses.

**D6 — Member-drop mechanics.**
*Positions:* Alpha §11: drop member + retrain executive at M−1. Beta §14: member OFF, trust
renormalized, same genome.
*Adjudication:* **Beta's form** (K17): the entropy floor guarantees live counterfactual
ledgers for all members, so renormalization is a valid counterfactual and saves 3 retrain
cycles. *Vindication for Alpha's form:* if a drop arm shows the renormalized executive
pathologically misallocating (trust mass pinned at the floor on a remaining member whose
ledger is positive), one retrained-executive arm is run from the contingency budget and
both are reported.

**D7 — Bedrock cap.**
*Positions:* LLM Engineer: $5.00 packet cap. FA: synthesis Phase C bill ≤$3.10.
*Adjudication:* **$3.10 hard cap** (pilot 0.05 + Tier-1 0.40 + Tier-2 2.05 + 0.60 retry/
iteration headroom), logged call-by-call; Tier-2 conditional on cumulative spend <$3.00
before it starts. The Engineer's $5 stands as the absolute packet ceiling if the operator
explicitly authorizes an overrun; default is $3.10.

---

## 4. PRE-REGISTERED BAKE-OFF CRITERIA (committed before any build)

Everything in this section is fixed now. Changes after this commit are protocol violations
and must be reported as such in COMMITTEE_REPORT.md.

### 4.1 The bake-off pair, harness, and windows

- **Pair:** the incumbent canon line vs SYN-1 (the brain). The incumbent runs as-is via the
  existing replay harness; SYN-1 plugs in as a strategy emitting trade-intent dicts. The
  incumbent appears in the dossier exactly as: its score line, and harness-wiring notes.
- **Identical harness:** `src/utils/three_line_replay/replay_engine.py` (`run_variant`),
  same `S3Cache` snapshots, same fill rule (next-session open from `daily/D+1/`), same
  cost model `src/utils/transaction_costs.py` (version recorded in manifest), same seeded
  slippage RNG, same VUG 6:1 handling, same harness cluster cap.
- **Replay window:** full-rolling-window snapshots **2026-01-31 → 2026-06-10** (~89 trading
  days; the earliest snapshot in the full-window format). Both runs replay the full window.
- **Reads:** **full-period AND holdout-only (≥ 2026-03-11, ~62 trading days)** deltas
  reported per EVIDENCE_PROTOCOL. The VERDICT is read holdout-only (E2 is the adoption bar).
- **One pre-registered configuration:** SYN-1 contributes exactly ONE configuration
  (champion-or-B0 genome + record-weighting decision + every model rung decision), frozen
  on pre-holdout validation only and committed to the run dir BEFORE the first holdout
  replay. Every variant tried along the way is logged (EVIDENCE_PROTOCOL §4).
- **Slippage seeds:** seed 4242 for ALL paired comparisons (identical dates, identical
  slippage draws — required for paired stats); alternate master seeds 4243, 4244 for the
  two seed-sensitivity replays only.

### 4.2 Metrics, primary verdict metric, and BEATS/TIES/LOSES (defined now)

Reported per comparison (one JSON + one MD, full-period AND holdout-only):
**total return, CAGR, Sharpe, max drawdown, win rate, realized round trips, cumulative
transaction costs, average gross exposure** — plus the **paired daily-difference series
stats (mean, sd, t on identical dates)**, which are the required form; endpoint deltas
alone are inadmissible and never headline.

- **PRIMARY VERDICT METRIC (named in advance):** the paired daily-difference t-statistic of
  (SYN-1 − incumbent) daily returns on identical holdout dates, with holdout ΔSharpe
  (annualized) as the effect size.
- **BEATS** iff holdout paired t ≥ +1.0 **and** holdout ΔSharpe > 0.
- **LOSES** iff holdout paired t ≤ −1.0.
- **TIES** iff |holdout paired t| < 1.0 — regardless of endpoint sign (the Skeptic's rule,
  adopted verbatim).
- Honesty line printed with the verdict: at t=±1 the per-comparison false-call probability
  under the null ≈ 16% one-sided; the holdout MDE at t=2 is ≈ 5 bp/day ≈ ΔSharpe_ann ≈ 4.0
  — the bake-off verdict is a sign-grade read, and the full-period paired stats are printed
  beside it as context (never as the verdict).

### 4.3 The assignment scorecard — per-organ arms, primary reads, verdict rules

**Primary read (every organ): E1 cross-fold paired statistic** — the with-organ vs
without-organ daily utility difference series on identical dates over the pooled F1–F6
out-of-fold panel (~1,500 fold-days, 2020-02→2026-02), HAC-adjusted t (Newey–West, 10
lags). **E2 holdout replay = sign confirmation only.** This removes the "lean on E1 where
the holdout is short" discretion (Skeptic §0.1/§4.2): the hierarchy is fixed now.

**Three-valued verdict rule (numeric, per organ):**
- **POSITIVE:** E1 HAC t ≥ +2.0 AND E2 holdout paired daily mean Δ ≥ 0.
- **ZERO (measured):** E1 |t| < 1.0. (If E1 t ≤ −2.0: reported as NEGATIVE attribution —
  the organ hurts — which also fails the assignment item, honestly.)
- **INDETERMINATE:** anything else (1.0 ≤ |E1 t| < 2.0, or E1 t ≥ 2.0 with E2 sign
  disagreement). Printed with the 95% CI and the pre-stated MDE so "underpowered" is a
  computed label, not an escape hatch.
- Scorecard print format: `organ=<E2 holdout ΔSharpe> (<verdict tag>)`, with tags from
  {positive, 0 (measured), 0 (gate-honest), negative, indeterminate}. "0 (gate-honest)"
  is reserved for organs the EA gated off with the gate state reported (Evolution proposal
  §1 note) — the reader cannot choose the flattering reading (Skeptic §4.10).

| Organ (final-line key) | Leave-one-out / neutralization arm | Pre-committed kill interpretation |
|---|---|---|
| **transformer** | PRIMARY: CAST-Small → ridge twin swapped into the slot (identical features), executive re-fit (retrain RT-2), replay R03. Secondary: CAST member OFF, trust renormalized (R04) | Verdict ZERO/NEGATIVE ⇒ "the transformer does not do real work in this brain at this sample size"; the transformer still ships (item-1 presence is mandated); scorecard prints the measured zero |
| **ensemble** (each member) | Member OFF, trust renormalized, same genome (R04 CAST, R05 GBM-Cond, R06 EventHead); RiskNet+ → trailing-21d-realized replacement (R07; survives only if paired t > −1 AND its removal degrades, i.e. organ kill = removal does NOT hurt at t ≥ 1) | A member at ZERO ⇒ that pathway adds nothing; reported per member. All three at ZERO ⇒ item 1 fails honestly |
| **evolution** | Champion-genome brain (R01) vs B0 DEFAULT_GENOME brain (R09) = the EA proposal's B2 read; plus the in-run B1 budget-matched random-search control and the adoption gate, reported verbatim per the proposal's three pre-committed verdict sentences | Champion ≤ B0 on E1/E2 ⇒ `evolution=0` (B0 ships); champion ≤ B1 fitness ⇒ "ceremonial as an optimizer" printed even if champion beats B0 |
| **LLM** | PRIMARY: full retrain with all `llm_*` at neutral constants (RT-3), replay R10. Secondary (contingency): feature-neutralized without retrain (R15) | ZERO ⇒ "LLM organ does not carry weight"; also dies pre-integration if Stage-1 fails (variance degenerate; corr with V2TONE ≥ 0.8 ⇒ expensive tone proxy; event flags miss known scheduled events) |
| **GDELT** | G1–G5 block ablated everywhere + retrain (RT-4), replay R11; sub-arm (contingency R16): G1 theme-dictionary alone; plus the permuted-dictionary placebo (§4.6.3, training-fold, no replay) | ZERO ⇒ evidence-backed no-signal per item 4 (with the feed-shift gate state §4.6.2 printed beside it — "no signal" vs "feed broke" must be distinguishable) |
| **meta-evaluator** | Equal-trust (τ=1/M) + fixed f=0.7 + same vol cap baseline (R08); plus the proposal's §7 kill criteria: static trust (std τ < 0.02 per fold), trust collapse, calibration corr ≤ 0, challenger parity | ZERO ⇒ learned executive adds nothing over the fixed rule; the solo-book pairwise-correlation statistic (§4.6.5) printed beside it — if corr ≥ 0.9 the expected verdict is the equal-trust tie "by construction," stated as such |
| **infotropy** | Transfer B: uniform-weights retrain (RT-5), replay R12 — this is the scorecard number. Transfer A: gated vs R3-only twin on training folds (E1 only, no replay), printed in parentheses | Both ZERO ⇒ `infotropy=no-transfer` on the final line, per packet — honest no-transfer is an acceptable verdict |

### 4.4 Battery budget (pre-registered; Feasibility's caps binding)

- **Hard caps: ≤20 replay runs, ≤9 retrain cycles** for the entire Phase C/D battery
  including the bake-off pair. Planned battery = **14 replays, 5 retrains** (the honest
  minimum band, FA B.3):

| Run | What | Retrain? |
|---|---|---|
| R01 | SYN-1, the ONE pre-registered config (bake-off + base of every paired arm) | RT-1 (base) |
| R02 | Incumbent (bake-off opponent) | — |
| R03 | Ridge-in-slot (transformer qua transformer) | RT-2 (ridge + executive re-fit) |
| R04 | CAST OFF, trust renorm | — |
| R05 | GBM-Cond OFF, trust renorm | — |
| R06 | EventHead OFF, trust renorm | — |
| R07 | RiskNet+ → trailing-21d realized | — |
| R08 | Executive → equal-trust + fixed f=0.7 | — |
| R09 | B0 default genome (evolution B2 read) | — |
| R10 | LLM neutral-constants retrain | RT-3 |
| R11 | GDELT G1–G5 ablated retrain | RT-4 |
| R12 | Infotropy-B uniform-weights retrain | RT-5 |
| R13, R14 | Seed-sensitivity: full brain at master seeds 4243, 4244 | — |
| R15–R20 | CONTINGENCY (≤6): LLM-neutralized-no-retrain; G1-only sub-arm; S1/S3/S2 data-edge block arm; D6's retrained-executive drop arm; 2 reserved | ≤4 contingency retrains |

- **Seeds (fixed, listed now):** slippage seed 4242 (all paired runs), 4243/4244
  (sensitivity); component master seeds = first 8 hex digits of
  `sha256("PKT-TB-006-SYN1::" + component + "::" + train_window_end)` as int; CAST/executive
  deploy-ensemble seeds {11, 13, 17, 19, 23}; OOF training seeds {101, 102, 103}; EA seed =
  `sha256("PKT-TB-006-EA" + train_window_end)[:8]` per the proposal. Internal seed-ensembles
  only; **no per-arm seed replication** (FA B.3 ruling, K16).
- **Training-cycle budget:** base full cycle ≈ 2–3 h (CAST-Small dominant) + battery
  retrains ≈ 2–3 h + replays ≈ 0.5 h. Shrink ladder if breached (in order, per packet —
  seeds/epochs before organs, never the bake-off): CAST OOF seeds 3→2; deploy ensemble
  5→3; CAST early-stop patience tightened; EA G 14→8 (K≈230, adoption gate unchanged);
  deep-GDELT panel-start 2019-01 at 2/day. Every reduction taken is stated on the final line.

### 4.5 Multiplicity line (printed verbatim with the scorecard)

> "7 scorecard organs were each given one pre-registered arm. At the E2 holdout sign bar
> (t ≥ 1) the per-arm false-positive rate under the null is ≈ 16%; expected false 'carries
> weight' sign-confirmations across 7 organs ≈ 1.1; family-wise P(≥1 false positive) ≈ 70%.
> That is why E2 is sign-confirmation only. At the E1 primary bar (HAC |t| ≥ 2 on ~1,500
> pooled fold-days) the per-arm FPR is ≈ 5%; expected false positives across 7 organs
> ≈ 0.35. The holdout MDE is ≈ 5 bp/day (ΔSharpe_ann ≈ 4 at t=2); no claimed effect is that
> large, and no holdout number below it is treated as certified. This battery consumed
> <N_holdout> holdout looks and <N_validation> validation-fold decisions (ledgers below)."

### 4.6 Attack-mandated evidence gates (all pre-registered)

1. **LLM model-cutoff rule (Skeptic §0.2, verbatim adoption):** any LLM artifact scoring
   window W must come from a model whose public training-data cutoff predates start(W).
   Pinned model `anthropic.claude-3-haiku-20240307-v1:0` (cutoff 2023-08) and fallback
   `gpt-4o-mini` (cutoff ~2023-10) both predate every scored window (Tier-2 starts
   2024-01). If the pinned model dies mid-program, existing backfill/holdout artifacts are
   **frozen, never re-scored**; `model_used` is stored per artifact and carried into
   attribution; fallback-model days are flagged and excluded from attribution windows if
   they exceed 10% of any read window.
2. **GDELT distribution-shift gate (Skeptic §0.3):** before any GDELT-fed model trains,
   compare the top-up window (2026-02-05→2026-06-10) to the reference window
   (2025-09-01→2026-01-31): (a) daily GKG record-count median within [0.5×, 2.0×] of
   reference; (b) |mean tone shift| ≤ 2× the reference daily-tone sd; (c) no G1 bucket's
   mean mention-share shifted by > 5× its reference sd. Any breach ⇒ GDELT features still
   enter (masked) but item-4 attribution is labeled `indeterminate (feed-shift)` — feed
   breakage must not masquerade as signal or no-signal.
3. **Permuted-dictionary placebo (Skeptic G-K4, program-wide):** 50 seeded random
   permutations of the theme→sleeve dictionary (same cardinality); the real G1 features'
   training-fold rank-IC (via EventHead/GBM) must exceed the 95th percentile of the placebo
   distribution, or the G1 dictionary is reported `0 (measured)` regardless of the block
   arm. Dictionary + bucket_map + ACTOR_MAP frozen and committed BEFORE any validation-fold
   model selection (TR G4); the §0.7 universe/curation-hindsight caveat is printed with all
   E1 GDELT numbers.
4. **Break-even IC line (Skeptic A-K4):** at the turnover budget (≤6%/day one-way, blended
   ~3 bps half-spread), cost drag ≈ 0.45%/yr; with long-only transfer coefficient ≈ 0.4 and
   ~400 effective bets/yr, cost break-even weekly rank IC ≈ 0.006, and the pre-registered
   **functionality bar is sustained purged-validation weekly rank IC ≥ 0.02** for the
   μ-stack. Below it, the pre-registered conclusion prints: "the forecast pathway adds no
   forecast edge; the brain is a vol-targeted diversified book with learned sizing" — and
   the bake-off proceeds to report whatever that brain does.
5. **Diversity floor (Skeptic B-K1 + TR §3.4/S5):** the dossier prints the mean pairwise
   correlation of member solo-book daily returns on validation folds. If ≥ 0.90, the
   pre-registered statement is that item-5 trust attribution is expected ≈ 0 by
   construction and the equal-trust tie is the honest outcome (members are already
   input-partitioned — the structural remedy is in place; the statistic verifies it).
6. **Train-vs-harness gap diagnostic (Skeptic B-K2):** executive training injects ±2 bps
   uniform slippage noise into its cost term; the dossier reports the gap between training
   utility and harness replay utility on identical dates; a gap > 25% of mean |daily
   utility| is reported as relaxation-gaming.
7. **Validation-look ledger (Skeptic §0.4):** every model-selection decision that reads
   F5/F6 (ladder rungs, A/B twins, EA champion, gate states, the final configuration) is
   logged (component, decision, date) in `prototype/validation_looks.jsonl`; the count is
   printed beside the bake-off verdict.
8. **Off-register constants (Skeptic B-K4):** every decision-relevant constant in SYN-1 is
   enumerated in BUILD_SPEC §16 with provenance (a-priori vs validation-tuned). Anything
   tuned later is appended to the ledger with its look count.
9. **Fine-tune discipline (TR A5/S4):** fine-tune cutoff = holdout_start − h − 1 =
   **2026-03-03** by formula; ≤20 effective fine-tuned parameters (the exact set in
   BUILD_SPEC §7.6); the fine-tune's pre/post validation delta is reported and de-claimed
   if ≈ 0; the dossier may not narrate fine-tuning as adapting to "real" fills (same
   simulator generated the live record).
10. **Trust-ledger fold-age standardization (Skeptic §0.5):** ledger stats standardized
    per-fold; static-trust falsifier read per-fold, never pooled.
11. **False-precision check (Skeptic §4.11):** no pre-registered bar smaller than the
    holdout noise floor is read on the holdout; any such bar is anchored to the E1 read.

### 4.7 Mapping to the packet's required final line

`BRAIN vs INCUMBENT: <BEATS|TIES|LOSES> by <Δreturn, ΔSharpe> on holdout (<paired-stat>);
ASSIGNMENT SCORECARD: transformer=<attr> ensemble=<attr> evolution=<attr> LLM=<attr>
GDELT=<attr> meta-evaluator=<attr> infotropy=<attr|no-transfer>; COST: $<x>/mo —
<reductions taken, if any>`

- `<BEATS|TIES|LOSES>` from §4.2's numeric rule; `<Δreturn, ΔSharpe>` = holdout-only
  endpoint deltas (context values; the verdict came from the paired t); `<paired-stat>` =
  `paired daily t=<x.xx>, n=<62>, mean=<bp/day>, sd=<bp/day>`.
- Each `<attr>` = the organ's E2 holdout ΔSharpe with its §4.3 verdict tag, e.g.
  `transformer=+0.21 (indeterminate)` or `GDELT=0 (measured)`; `infotropy` prints Transfer
  B's arm with Transfer A's fold verdict in parentheses, or `no-transfer` if both ZERO.
- `COST` = the deployed monthly worksheet total (BUILD_SPEC §14); reductions = every shrink
  rung taken (§4.4).

---

## 5. Holdout-look and validation-look ledgers — initialization

**Holdout-look ledger (`prototype/holdout_looks.jsonl`):**
- **Looks consumed to date: 0.** No replay touching dates ≥ 2026-03-11 has been executed by
  any panel role. (The incumbent's own production history exists but no panel read of
  holdout-window comparative results has occurred.)
- **Budget: ≤ 20 looks total** (one per replay run, §4.4): 2 bake-off + 12 planned arms/
  sensitivity + ≤6 contingency. Each look is logged as
  `{run_id, date_executed, config_hash, purpose}` at execution time. The EA, all model
  training, all ladder decisions, and the dictionary placebo consume **zero** holdout
  looks by construction (fitness data ends 2026-02-06; embargo 22d).
- No holdout replay executes before: (i) this file's registration commit, (ii) the single
  SYN-1 configuration freeze commit, (iii) all 14 planned arms wired and runnable.

**Validation-look ledger (`prototype/validation_looks.jsonl`):**
- **Looks consumed to date: 0** (no SYN-1 model has been trained). Every F5/F6-reading
  decision from here forward is appended; the final count prints with the scorecard (§4.5).

---

*End of TOURNAMENT.md. Buildable spec: `prototype/BUILD_SPEC.md`. Next phase action:
commit this pre-registration, then GDELT top-up backfill (build task #0).*
