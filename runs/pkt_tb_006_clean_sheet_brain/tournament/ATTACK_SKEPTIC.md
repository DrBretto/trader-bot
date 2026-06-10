# ATTACK_SKEPTIC — Phase B tournament attack on Designs Alpha / Beta / Gamma

**Panel role 2 of 12 (Skeptic, mandatory) — PKT-TB-006 — 2026-06-10**
**Inputs:** packet, ASSIGNMENT_BRIEF.md, EVIDENCE_PROTOCOL.md, DESIGN_{ALPHA,BETA,GAMMA}.md,
INFOTROPY_TRANSFER.md, the four mechanism proposals, ANTI_BULLSHIT_ASSESSMENT_DESIGN_METHOD.md.
**Anchoring check:** incumbent files read AFTER registration commit 4d68974, solely for the
isomorphism check; per the packet's hard constraint this dossier contains no incumbent
description — anchoring appears only as verdict lines (§1.5, §2.5, §3.5).
**Frame honored:** the assignment outranks raw P&L. No attack proposes deleting a mandated
organ; every attack forces an organ into its most defensible, honestly-attributable form.

---

## 0. Shared findings (arithmetic done once; binds all three designs)

### 0.1 The holdout cannot certify what the scorecard promises (the headline arithmetic)

Holdout = 2026-03-11 → present ≈ **62 trading days**. Paired daily-difference t on an
ablation arm: `t = mean(d)/(sd(d)/√62)`; in annualized Sharpe of the difference book,
`ΔSharpe_ann ≈ t · √(252/62) ≈ t × 2.02`. Therefore:

- **Minimum detectable effect at t=2: ΔSharpe_ann ≈ +4.0.** No organ in any design claims
  anything within an order of magnitude of that.
- At the common kill bar (paired t < 1 ⇒ dead), an organ "survives" at t ≥ 1 ⇒ point estimate
  ΔSharpe_ann ≈ +2.0 — far above any plausible true effect — while under the null
  **P(t ≥ 1) ≈ 0.16** per arm (one-sided).
- **False positives across the scorecard:** 7 mandated organs ⇒ expected false "carries
  weight" verdicts ≈ 7 × 0.16 ≈ **1.1**; P(≥1 false positive) ≈ 70%. Across the full arm
  tables (Alpha 12, Beta 9+, Gamma 7+sub-arms) family-wise FP probability is 85–90%.
- **Power:** an organ with TRUE ΔSharpe_ann = +0.5 (large, real) has expected t ≈ 0.25 ⇒
  P(t ≥ 1) ≈ **23%**. The battery is ≈16% FPR vs ≈23% TPR: on the holdout alone it is
  **nearly uninformative**.

Consequence (non-negotiable for synthesis): holdout LOO arms are **sign-confirmation reads**,
not verdicts. Every organ's PRIMARY attribution read must be pre-registered as the cross-fold
E1 paired statistic, with the E2 holdout read reported as confirmatory sign + CI. All three
designs gesture at this ("lean on E1 where the holdout is too short") but none pre-registers
which read is primary per organ — that discretion is a wiggle path; close it in TOURNAMENT.md.
The bake-off verdict has the same problem (BEATS at t=2 needs ΔSharpe ≈ 4): **"TIES" must be
defined now** — proposed: |paired t| < 1 ⇒ TIES regardless of endpoint sign; endpoint deltas
alone are inadmissible per EVIDENCE_PROTOCOL and must not headline.

### 0.2 LLM backfill leakage — graded honestly (severity: LOW now, HIGH-latent)

The feared leak: the LLM "reading" 2026-02 news in backfill already knows what happened.
Checked against the actual model arithmetic:

- Pinned model `anthropic.claude-3-haiku-20240307-v1:0` — public training-data cutoff
  **August 2023**. Backfill windows: Tier-2 2024-01→2026-01, Tier-1 2026-01-31→present,
  holdout 2026-03-11→. **Every scored window post-dates the cutoff.** The pinned model cannot
  know the outcomes of the news it scores; fallback `gpt-4o-mini` (cutoff ~Oct 2023) has the
  same property. Hindsight leakage through the pinned chain is ≈ **nil — by lucky arithmetic,
  not by design**.
- **The latent leak is the deprecation path.** All three designs carry the flag that the
  pinned Haiku is deprecated upstream and name a Haiku-4.5-class IAM widening as the remedy.
  A 2025-cutoff model re-scoring ANY 2024–2026 window has read the actual news and knows what
  markets did — instant hindsight contamination of training features and of the holdout
  attribution, laundered through "we upgraded the model." No design pre-registers against it.
- **Required fix (condition for all three):** pre-register in TOURNAMENT.md: *any LLM artifact
  scoring window W must come from a model whose public training cutoff predates start(W); if
  the pinned model dies mid-program, existing backfill/holdout artifacts are frozen, never
  re-scored; `model_used` per artifact is the audit key.* The prompt line "do not use
  memorized knowledge" is an instruction, not enforcement; it counts for nothing here.
- Residual (small): mixed-model artifacts (dark nights falling to gpt-4o-mini) create
  backfill-vs-live distribution shift that can masquerade as signal; carry `model_used` as a
  mask-class feature or exclude fallback days from attribution windows.

### 0.3 GDELT-dark holdout — all three designs stake item 4 on an unverified backfill

The brief verifies `historical_gdelt.parquet` ends 2026-02-04: **the entire holdout currently
has zero real GDELT.** All three correctly make the 2026-02-05→present top-up build-task #0 —
but only Gamma (§13.2) admits the next hazard: if the 15-min files' density/composition
differs from the historical era (a feed whose daily endpoint already died once), GDELT
attribution becomes a measurement artifact. Required for all: a pre-registered
**distribution-shift check** (doc-count and tone-moment comparison, late-2025 vs 2026 top-up)
gating GDELT feature admission, before any GDELT-fed model trains.

### 0.4 Validation-look ledger is missing everywhere

All three count holdout looks. None counts **pre-holdout validation looks** — yet the
simplification ladders, A/B infotropy twins, library grids, EA champion selection, member
gates, and the "single pre-registered configuration" choice ALL read the same 2025–2026
validation window (F5/F6), the regime adjacent to the holdout: dozens of looks at the data
most predictive of holdout behavior. Not a protocol violation — but unstated, it lets a mined
validation winner pose as a single clean candidate. Required: TOURNAMENT.md states the
validation-decision count, and the dossier reports it next to the bake-off.

### 0.5 Trust-ledger fold-age drift (shared via PROPOSAL_META_EVALUATOR)

Walk-forward expert opinions come from expanding-window training: later folds' opinions are
mechanically better (more data). A trust head trained over that history partly learns "trust
everyone more in later years" — a nonstationarity artifact that inflates apparent trust-head
intelligence and pollutes ledger features with fold-age. Affects all three identically. Cheap
pre-registered fix: per-fold standardization of ledger stats (or train-window size as an
explicit nuisance covariate), and the static-trust falsifier read per-fold, not pooled.

### 0.6 Fine-tune-on-own-record is a near-empty organ (shared via the proposal)

The ~130 pre-holdout live-record days were generated by a different decision process; the new
brain's counterfactual books never traded on them. Everything the fine-tune consumes (prices,
opens, costs) is already in the replay substrate; the only new information is real
morning-fill timing, which the 20-pair alignment test already audits. Net effect: the regime
closest to the holdout is consumed twice, for a calibration claim unattributable at 62 days.
Defensible form: keep the alignment audit; report the fine-tune's pre/post validation delta
and de-claim it if ≈0. All three adopted it as written.

### 0.7 Universe/curation hindsight (mild, shared, must be stated)

The 64-ETF universe and every curated mapping (bucket_map, theme→sector, ACTOR_MAP,
THEMES_FIN) were authored in 2026 with full knowledge of 2015–2026 history (semis buckets,
ARKK, NVIDIA→semis). No walk-forward split removes that researcher degree of freedom;
freezing before the holdout read protects only holdout claims. Every E1 number carries this
caveat; Gamma's dictionary placebo (§3) should be grafted program-wide.

---

## 1. DESIGN_ALPHA — "Forecast First"
### 1.1 Kill-list

**A-K1. CAST's effective-sample claim contradicts its own derivation (arithmetic).** §2/§10
state "~20–35k effective samples vs 45k weights." The design's own decomposition — "~8 truly
distinct behaviors/day × ~580 non-overlapping 5-day windows" — multiplies to **4,640**, not
20–35k. At its own honest count, CAST is ~10 parameters per effective sample (22k fallback:
~5:1); the stated range is unsupported by the stated argument. Defenses (b)–(f) are real, but
the ladder is inverted for showcase reasons (45k default, 22k/ridge fallbacks). Flip the
burden: **22k (or ridge) is the default rung; CAST-45k ships only if it beats the smaller
rung on purged validation Spearman.**

**A-K2. MetaTrigger: a division-by-zero on its own genome range, a stated train/serve
mismatch, in a redundant slot.** The scale `(2·p_win − 1)/(2·p_min − 1)` is singular at
`p_min = 0.5`, and the pre-registered gene range is `p_min ∈ [0.50, 0.70]` — the EA can
select the singularity (§2 E5, §6). E5 trains on the equal-trust blend but serves a τ-tilted
one — by §13.6's own admission, miscalibrated exactly when the executive deviates most. And
its function (does the bet clear costs) is the fourth turnover/cost brake in the pipeline
(§3 step 6 says costs are priced three times *before* the trigger). Kill the standalone organ
in its current form: (i) fold `p_win` in as one feature of the executive's sizing head
(meta-labeling claim preserved via that feature's attribution), or (ii) fix the range to
[0.55, 0.70] and pre-register the gated-book-corr threshold that declares it broken.

**A-K3. EventHead's linear zero-fill conflates "absent" with "neutral."** E3 is an elastic
net; LLM features are 0 before 2024 with an `llm_available` mask, but a linear model cannot
gate coefficients on a mask — no interaction term exists in the stated design matrix. LLM
coefficients are effectively fit on ~520 days inside a model whose other coefficients are fit
on ~2,790; covariance is mis-estimated and a 0-sentiment night is indistinguishable from a
pre-2024 night. Fix: mask-interaction columns or a separate post-2024 head. E2 (trees)
handles this natively; E3 as written does not.

**A-K4. No pre-registered skill floor that maps to cost break-even.** The prior rests on
5-day cross-sectional rank skill in 64 of the most arbitraged instruments on earth. By the
fundamental law with the design's own breadth (~8 clusters × ~50 wk ≈ 400 independent
bets/yr) and a long-only transfer coefficient ≈ 0.3–0.5: net IR 0.5 needs sustained weekly
rank IC ≈ **0.06–0.08** — institutional-grade alpha. §13.1 admits IC ~0.02 makes the stack
pointless, yet no minimum OOS Spearman that clears costs is pre-registered. Demand a stated
break-even IC line in TOURNAMENT.md with its pre-registered conclusion ("stack = expensive
vol-targeted book") attached, before the holdout is read.

**A-K5. Leakage audit — pass, with the shared items.** Target construction, purge/embargo,
COT publication keying, GDELT `visible_from`, ledger D−h−1 lag, MetaTrigger walk-forward
labels: all correctly specified. Alpha is the only design that splits Infotropy-A correctly
(train-time R1∧R2∧R3 screen; inference-time R2-only down-weight — R2 needs only past data).
Remaining exposure = shared items §0.2–§0.7; the 12-arm table is the largest multiple-looks
surface of the three (§0.1 applies hardest here).

### 1.2 Complexity-to-impress audit

- **RiskNet (ridge-by-default, MLP only on QLIKE win)** is the anti-showcase exemplar — keep.
- **MetaTrigger** is the organ most plausibly tied by a 10x simpler twin (fixed cost-vs-edge
  ratio filter); defensible form per A-K2.
- **CAST** is mandated showcase; defensible form = the inverted ladder (A-K1). Of the three
  transformer instances, Alpha's has the **worst capacity-to-sample ratio on the least
  differentiated food** (price bars) — its own §13.3 concedes Gamma's transformer has more
  honest work. Weakest instance of the transformer organ.
- The 5-expert + trigger + executive stack is three approximation layers before capital moves
  (§13.2, admitted). A 10x simpler twin (ridge cross-section + vol target + band) plausibly
  ties on P&L; the ladders would detect it — provided validation looks are counted (§0.4).

### 1.3 Quant's sneer (three hardest questions; can the design answer?)

1. *"Your edge is weekly relative-value ranking of SPY-complex ETFs. State the IC you need to
   clear 1–8 bps spreads long-only, and your evidence you can sustain it."* — *Cannot answer
   as written*: no break-even IC computed or pre-registered (A-K4).
2. *"Five organs with thresholds, ladders, and A/B twins, all tuned on the same pre-holdout
   window. How many validation decisions did your final configuration absorb?"* — *Cannot
   answer*: no validation-look ledger (§0.4).
3. *"Your precision gate is trained on a blend you never serve. Bound the miscalibration."* —
   *Partial*: §13.6 admits and monitors (corr plot); no pre-registered break threshold (A-K2).

### 1.4 Anti-bullshit audit of Alpha's attribution battery

Strong: numeric kill bars on most arms; the transformer-qua-transformer ridge arm is the
best-designed falsifier in any design; infotropy arms have clean A/B form. Vagueness found:
(i) RiskNet arm kill = "no holdout degradation" — directionless; specify (paired t > −1 vs
the trailing-vol replacement = survives). (ii) LLM arm runs "retrain + neutralized variants"
without designating a primary — designate retrain now. (iii) §13.4 pre-builds the
"underpowered, honest but unsatisfying" escape — close with the three-valued verdict rule
(§4 item 3). (iv) No multiplicity line for 12 arms.

### 1.5 Anchoring verdict

**Alpha — anchoring: none found.** No structural isomorphism beyond generic quant furniture
(drawdown brake, no-trade band, disagreement features) that all three designs and the blind
proposals independently contain.

### 1.6 Conditions for survival
1. Invert the CAST ladder (22k/ridge default; 45k must win to ship); correct the
   effective-sample claim to its own 4.6k derivation.
2. MetaTrigger: fold into the executive sizing head OR fix the p_min range and pre-register
   the gated-book-corr break threshold.
3. EventHead mask-interaction (or post-2024 head).
4. Pre-register break-even IC, plus the §4 shared register (E1-primary reads, TIES,
   multiplicity, validation looks, LLM cutoff rule, GDELT shift gate).

---

## 2. DESIGN_BETA — "Bookwright"
### 2.1 Kill-list

**B-K1. Ensemble diversity collapse → τ-unidentifiability (the structural risk Beta
under-states).** All policy families train on the same utility objective over heavily
overlapping state (P1, P2, P3 all see the price panel). Utility-trained long-only policies
converge to vol-targeted momentum/quality tilts (§15.2 admits the closet-baseline half). The
unstated consequence: **candidate books with pairwise rank-corr ~0.9 make the trust simplex
unidentifiable** — any τ yields the same blended book, the trust head learns noise, and the
flat-trust kill (std < 0.02) fires not because the executive failed but because the ensemble
gave it nothing to arbitrate. Item-1 AND item-5 attributions then read ≈0 *by construction*,
not by market verdict. Alpha guarantees diversity by disjoint inputs; Beta only hopes for it
via hypothesis-class differences. Demand: a pre-registered **diversity-floor diagnostic**
(mean pairwise rank-corr of candidate books on validation below a stated bound, e.g. 0.8)
that must pass before the ensemble is presented as an ensemble; the pre-registered remedy on
failure is input partitioning (e.g. P3 loses the price block), not a silent ship.

**B-K2. Cost-model gaming in the differentiable replay.** Members and executive train through
a smooth cost term with `E[slip] = 0` and linear spreads, then are scored on a harness with
seeded ±2 bps slippage and discrete banded intents. Gradient ascent will exploit the training
model's exact linearity — parking weight changes at band edges, harvesting "edge" that is an
artifact of the smooth relaxation; the straight-through band estimator makes this worse.
Demand: train-time slippage noise injection, plus a pre-registered **train-loss vs
harness-replay utility gap** diagnostic on identical dates (a material gap = the policy is
gaming the relaxation — report it).

**B-K3. Credit-densification is an overclaim doing load-bearing work.** §5/P1 funds 26k
params on "~189k symbol-day credit terms," but the loss is book-level: distinguishing two
*policies* still rests on ~600 quasi-independent book windows over ~6 regimes (§15.1 admits
this; the sample table then half-retracts by listing both numbers). Per-symbol gradient terms
share the book outcome — they are not independent observations. Force the table to carry ~600
as P1's decision-relevant effective sample, and make the no-attention twin the default rung
(attention must beat it to ship), the same inverted-burden posture as A-K1.

**B-K4. Off-register knobs.** P3's policy is `topk(ŷ, k=12)·rank-weights, capped` — k, the
rank-weight scheme, and the cap are decision-relevant constants appearing in no genome,
library grid, or pre-registration. Same for P4's "salient-bucket pooling." Each is a free
parameter someone chose while looking at *something*; list them in TOURNAMENT.md with
provenance (a priori vs validation-tuned — if tuned, count the looks per §0.4).

**B-K5. Leakage audit — pass, best of the three.** Correct: `visible_from` joins, ledger lag,
LLM trainable-only-where-real with masks (cleanest adjudication of the Scout/LLM conflict),
HAC/non-overlap stats named, dumb-twin guards consuming no holdout look, fold geometry
single-sourced from the EA layout. Shared items §0.2–§0.7 apply. Minor: `cash_rate` in P3's
target is undefined — pin it. Credit: the ≤6%/day one-way turnover budget is the only
pre-registered *bound* (vs description) in any design.

### 2.2 Complexity-to-impress audit

- The **policy library × slot-selectors** is sold as "evolution evolves reward shaping."
  Arithmetic: λ_dn ∈ {2,6} is a binary switch per slot — composition selection over small
  grids, honest and useful, but the dossier must not dress a binary switch as continuous
  reward-shaping evolution. Keep, described as composition selection (still the strongest EA
  bite of the three — §5).
- **P2 linear floor policies** are the anti-impress exemplar of the whole tournament — keep.
- The **book-space convex blend executive** is strictly simpler than the proposal it adapts
  and makes counterfactuals exact — the strongest meta-evaluator instance (§5); its weakness
  is entirely B-K1 (nothing to arbitrate if books collapse together).
- The 10x-simpler twin (P2-slow at fixed f) is already a pre-registered guard — good; but the
  guard "measures, does not prevent" (§15.2). Beta is the design most likely to produce an
  honest-zero scorecard on items 1 and 5 simultaneously (B-K1).

### 2.3 Quant's sneer

1. *"Regress your learned book's weights on {trailing vol, 21d momentum, equal-weight}. If
   R² > 0.9 you built a closet vol-targeted momentum fund with 30k parameters of ceremony.
   What's the number?"* — *Partial*: the dumb-twin guard is adjacent; the weight-variance
   regression itself is not pre-registered. Add it as a named diagnostic.
2. *"Your policies trained inside a smooth cost model they can game. Show me the
   train-vs-harness utility gap on identical dates."* — *Cannot answer as written* (B-K2).
3. *"~600 effective windows fund the members, executive, EA, and library composition — all
   reading the same 6 folds. What's left for selection that isn't fold-mining?"* — *Partial*:
   B1 control, adoption gate, fold subsampling are real; missing the validation-look count.

### 2.4 Anti-bullshit audit of Beta's attribution battery

Strongest battery of the three: primary/secondary designated for the LLM and transformer
arms; the no-attention twin reads transformer attribution against the right null; two
non-learned floors pre-registered at no holdout cost; turnover stated as a bound. Vagueness
found: (i) "`evolution=<attr>` may be ≈0 by construction of honesty" (§15.7) — true, but
pre-register the reporting (`evolution=0 (gate-honest)` vs `=0 (measured)`) or a motivated
reader spins either way. (ii) The diversity floor (B-K1) and weight-variance regression
(sneer 1) are missing from the battery. (iii) No multiplicity line (§0.1). (iv) Off-register
constants (B-K4).

### 2.5 Anchoring verdict
**Beta — anchoring: none found.**

### 2.6 Conditions for survival
1. Pre-registered diversity-floor diagnostic + named remedy (input partitioning) — B-K1.
2. Slippage noise in training + train-vs-harness gap diagnostic — B-K2.
3. Sample table carries ~600 for P1's decision capacity; no-attention twin becomes the
   default rung — B-K3.
4. Register every off-genome constant with provenance — B-K4; pin `cash_rate`.
5. The §4 shared register.

---

## 3. DESIGN_GAMMA — "Information-Funnel Brain"
### 3.1 Kill-list

**G-K1. The record-grade ingestion gate, as written, is either look-ahead or undefined at
decision time (the single biggest kill in the tournament).** §6.4: "only events with
`record_grade > record_grade_tau` become tokens," where `record_grade = R1·R2·R3`. Walk the
data flow: **R1 (persistence) is computed from `regime_stat(s, [t+1, t+d])` — future returns
relative to the event** — and R3 requires a with/without model ablation, a training pass.
Neither exists at 03:00 UTC on the night a token must be admitted. So either (a) the nightly
funnel cannot run as specified, or (b) the gate is applied only in historical construction —
and then the ETT's *training inputs are selected conditional on forward returns*: attention
is learned over an event field pre-filtered to events that "worked," a curated-input
look-ahead that inflates every backtest number the ETT touches and cannot be reproduced live
(train/serve impossibility). Contrast: Alpha scopes the same Liaison mechanism correctly
(train-time family screen; inference-time R2-only — R2 needs only past returns); Beta scopes
it as a walk-forward family screen. Gamma's per-event, evolution-thresholded, inference-time
version is the over-reach. **Condition: restate the gate** as (i) a training-time
feature-family screen (Beta's form), and/or (ii) a *learned record-grade predictor* — trained
walk-forward to predict past events' eventually-realized grades from ingestion-time
observables — whose output gates tokens live. Until restated, every ETT number in Gamma is
presumptively contaminated.

**G-K2. The LLM-annotator extension breaks its own call budget (arithmetic).** §6.3 adds
per-cluster annotation tuples "an extra ~15 tokens/cluster in the same JSON, negligible
cost." The adopted call config (PROPOSAL_LLM_SENTIMENT §2) is `max_tokens: 1500`, and the
bucket JSON alone budgets 1.0–1.5k output tokens. Up to 120 clusters × ~15 tokens ≈ +1,800
tokens ⇒ **guaranteed truncation ⇒ schema failure ⇒ repair-retry ⇒ fail-soft dark** — the
marquee extension as specified makes the organ emit neutral zeros routinely. Fix on the
record: annotate top ~20 clusters only, raise max_tokens (~3k; still ≈$0.01/night), re-run
the cost line, add a truncation-rate falsifier to Stage-1.

**G-K3. ETT capacity and food mismatch.** 28k params over (its own table) ~600 effectively
independent windows ≈ 47:1 — the worst ratio on the table, worse than CAST. Compounding it:
the "richest token features" (LLM annotations) exist only where backfill exists (~520 days,
2024-01→), so the marquee transformer pretrains 2015–2023 *without* the food that justifies
it, then serves with it — train/serve distribution shift concentrated in the showcase organ.
§13.5 admits the capacity half but not the food half. Condition: bag-of-events MLP becomes
the default rung (invert the burden as in A-K1/B-K3), PLUS a pre-registered **2024+-only ETT
twin** so the dossier can say whether the annotation channel is signal or shift.

**G-K4. The theme→sector dictionary needs a placebo, not just a freeze.** §13.3 names the
curation-overfit trap honestly, but freezing before holdout protects only holdout claims; all
E1/cross-fold GDELT evidence — which §0.1 makes PRIMARY — remains exposed to 2026-hindsight
curation (semis/China mappings "work" partly because we know 2020–2025). Condition:
pre-register a **permuted-dictionary placebo arm** (random theme→sleeve assignment, same
cardinality; G1 must beat the placebo distribution on training folds) before any GDELT-fed
model ships. Cheap, decisive, the single best answer Gamma could give a sneering quant.
Adopt program-wide (Alpha/Beta inherit the same dictionaries via the proposals).

**G-K5. Anti-HILL eps = 0.05 starves inaction learning.** Gamma applies Transfer B to the
*executive's utility* at the Liaison's sketch floor (eps=0.05). Beta's §10 argument applies
with full force: at eps≈0 the executive barely trains on round-trip days — exactly the states
where the right action is staying small — and will systematically overdeploy in chop.
Condition: raise the floor or put eps on the genome (Alpha's `record_weight_eps ∈ [0.05,
0.5]` is the right form); reconcile in synthesis.

**G-K6. Member-4 (LLM-as-μ-expert) has no ledger history for most of pretraining.** The
executive trains over 2015→2026 walk-forward member opinions; mu_LLM exists from 2024 at
best. How trust-ledger stats, shared-φ inputs, and the entropy floor behave for an expert
absent 80% of history is unspecified (mask? bench until 2024? synthetic prior?) — the
executive's pretraining is undefined for M=4/5. Specify before build.

**G-K7. Multiple looks + vague kill phrasing.** "t respecting protocol form" (§12) is not a
threshold — the protocol mandates *reporting* paired t, not a pass bar. State the bar per arm
on the E1 cross-fold read (§0.1). Seven arms + sub-arms on 62 days: §13.7 acknowledges the
hazard, but acknowledgment is not a decision rule.

### 3.2 Complexity-to-impress audit

- The funnel (~140-dim `field_state` + token stream + 5 members) is the largest surface of
  the three; its decorative tier is named honestly by the design itself (G4 novelty/JSD —
  §13.6 "pretty but may be noise") and is gated/ablatable — fine.
- **ISRE is the most defensible single organ in the entire tournament**: dense, high-SNR
  vol/correlation targets, modest params, a real sizing job. Graft candidate.
- The **information-health block** (executive knows when its differentiated food is thin or
  dark) is cheap, attributable, and the best one-line executive improvement on the table.
  Graft candidate.
- **LLM-as-annotator** is the highest-ceiling and most fragile LLM instance (G-K2, G-K3);
  defensible form: top-20 clusters, raised max_tokens, 2024+-twin falsifier. As written it is
  the showcase organ most likely to silently zero itself.
- Member-5 contrarian behind an EA gate with a drop-and-report rule — correctly humble, keep.

### 3.3 Quant's sneer

1. *"News→daily-returns is a published-null graveyard — academia mined GDELT itself for a
   decade. Your G1 differs from the corpses how, and where's the placebo?"* — *Cannot answer
   as written*: sleeve-resolution + structure is plausible but the placebo arm doesn't exist
   (G-K4).
2. *"It's 03:00 UTC. An event token is at the gate. Compute its record_grade using only what
   you have right now."* — *Cannot answer*: R1 needs next week's returns (G-K1). This is the
   question that kills the design in a quant room until restated.
3. *"Your transformer's richest inputs exist for ~520 of ~2,790 training days. What did
   attention learn from the other 2,270, and why does it transfer?"* — *Weak*: masks and
   graceful degradation are specified; only the 2024+-twin (G-K3) turns this into evidence.

### 3.4 Anti-bullshit audit of Gamma's attribution battery

Strong: evolution-tuned infotropy thresholds make "evolution turned it off" a measurable
verdict; the tone-proxy corr < 0.8 test is kept; minimum-detectable-effect promised per arm.
Vagueness found: (i) "respecting protocol form" (G-K7); (ii) the E1-vs-E2 "lean on" clause
(§12) is exactly the post-hoc evidence-shopping the method doc warns about — fix with
per-organ primary-read pre-registration; (iii) infotropy A+B share one ablation arm
(`record_grade_tau→0` AND `eps→1` together, §12 table) — two mechanisms, one arm,
unattributable; split as Alpha/Beta do; (iv) no multiplicity line; (v) the §6.4 gate text and
§12 arm disagree about what the gate even is at inference (G-K1) — an assessment cannot be
pre-registered for a mechanism that isn't pinned down.

### 3.5 Anchoring verdict
**Gamma — anchoring: none found.** One independent convergence noted for the record: the
information-health/dispersion-style caution conditioning rhymes with generic risk-throttle
furniture any allocation system grows; Gamma's is derived blind, on a different substrate
(news-field health, not price internals) — graded benign.

### 3.6 Conditions for survival

1. Restate the record-grade gate (training-time family screen and/or learned walk-forward
   grade predictor); until then all ETT results presumptively contaminated — G-K1.
2. Fix the annotator call budget (top-20, max_tokens raised, cost line re-run, truncation
   falsifier) — G-K2.
3. Bag-of-events as default rung + 2024+-only ETT twin — G-K3.
4. Permuted-dictionary placebo pre-registered before any GDELT model trains — G-K4.
5. eps on genome or raised floor — G-K5; specify Member-4 ledger handling — G-K6.
6. The §4 shared register, plus split infotropy A/B arms.

---

## 4. Consolidated anti-bullshit register (what TOURNAMENT.md must pin before build)

Per ANTI_BULLSHIT_ASSESSMENT_DESIGN_METHOD: an assessment passes only if a motivated reader
cannot wiggle out. The batteries are unusually strong on numeric kill bars and pre-committed
verdict sentences (the EA proposal's three verdicts are the gold standard). Remaining wiggle
paths, consolidated — each a pre-registration line item:

1. **TIES undefined** on the final line → define: |paired t| < 1 on holdout ⇒ TIES; endpoint
   sign reported, never headlined (endpoint-only already inadmissible per protocol).
2. **Per-organ primary read** (E1 cross-fold paired stat) vs confirmatory E2 — fixed in
   advance; "lean on E1 where holdout is short" discretion removed.
3. **Three-valued attribution verdicts** with "indeterminate" defined (CI contains 0 and the
   pre-stated MDE); "underpowered" ceases to be a free escape hatch.
4. **Multiplicity line**: scorecard prints expected false positives at the chosen bars
   (≈1.1 of 7 organs at t≥1) next to the attributions.
5. **Validation-look ledger** alongside the holdout-look ledger (§0.4).
6. **LLM model-cutoff rule** (§0.2): cutoff predates scored window; artifacts frozen on model
   death; `model_used` carried into attribution.
7. **GDELT distribution-shift gate** (§0.3) before feature admission.
8. **Dictionary/bucket-map placebo** (G-K4), program-wide.
9. **Off-register constants** enumerated with provenance (B-K4 and equivalents in all three).
10. **Honest-zero phrasing** pre-fixed: `=0 (gate-honest)` vs `=0 (measured)` vs
    `indeterminate` — the reader cannot choose the flattering reading.
11. **False-precision check**: any bar smaller than the §0.1 noise floor (e.g. holdout
    ΔSharpe ≥ +0.10) is re-anchored to the E1 cross-fold read or labeled sign-check-only.

---

## 5. Cross-design ranking (per assignment item — feeds synthesis grafting)

| Item | Strongest | Weakest | Why |
|---|---|---|---|
| 1. Ensemble of genuinely different types | **Alpha** — diversity guaranteed structurally by disjoint input pathways (price-only / tabular-exogenous / events-only / risk / trigger); remove one and a pathway goes dark | **Beta** — same loss + overlapping state ⇒ correlated books ⇒ τ-unidentifiability (B-K1); diversity is hoped for, not built in | Gamma close second: genuinely different *targets* (direction/vol-corr/rank/LLM) |
| 2. Evolutionary balancing | **Beta** — slot-selector library: evolution actually selects loss-shaping/composition, the only design where the EA decides something gradients also wanted to decide | **Alpha** — proposal + 3 genes; bites only on standard risk knobs | Gamma's funnel-gate genes (streams pay their way) are the second-best bite and should be grafted regardless |
| 3. LLM sentiment organ | **Alpha** (narrowly over Beta) — adopts the proposal in full with the most explicit feature-routing table and keeps the transformer text-free so the LLM arm is cleanly removable | **Gamma** — highest ceiling (annotator) but breaks its own token budget (G-K2) and feeds a contaminated gate (G-K1) as written | Graft note: the annotator idea is worth saving at top-20 scale with the twin falsifier |
| 4. GDELT load-bearing | **Gamma** — structural backbone, sleeve-resolved, most ablation paths, only design that names the feed-shift hazard | **Beta** — buried as P4 + context columns; its own §15.8 concedes under-exploitation | All three correctly make the top-up backfill task #0 |
| 5. Learned meta-evaluator | **Beta** — book-space convex blend: exact counterfactual ledger, fewest approximation layers, stage/capacity/target separation argued best | **Alpha** — MetaTrigger insertion adds an approximation layer with a stated train/serve mismatch and a genome singularity (A-K2) | Graft Gamma's information-health block into whichever executive ships — cheap and genuinely additive |
| 6. Infotropy mechanism | **Alpha** — only leak-correct split of Transfer A (train-screen vs inference R2-only); eps on genome; clean separate A/B arms | **Gamma** — Transfer A as written is the tournament's biggest leak (G-K1); eps=0.05 starves inaction; A+B share one arm | Beta middle: correct family-level scoping and the best eps *reasoning* (floor=0.25 rationale) — graft Beta's reasoning with Alpha's genome form |

Headline: **no design survives intact. Beta's spine (allocation executive + EA bite +
evidence battery) is the strongest chassis; it must import Alpha's diversity-by-input
partitioning and infotropy-A scoping, plus Gamma's GDELT backbone, ISRE, and
information-health block — Gamma's inventions entering only after G-K1/G-K2 are fixed.**

---

## 6. Skeptic's bottom line

- Three kills that MUST be fixed before any build: **G-K1** (record-grade gate look-ahead /
  train-serve impossibility), **G-K2** (annotator breaks its own token budget ⇒ silent organ
  death), and the **§0.1 power/multiplicity re-anchoring** (E1-primary per organ, TIES
  defined, multiplicity printed) — without the third, the assignment scorecard is noise
  dressed as measurement.
- One leakage class graded and closed: **§0.2** — Haiku backfill is clean today only because
  the pinned model's Aug-2023 cutoff happens to predate every scored window; the
  deprecation/upgrade path is a live hindsight trap; pre-register the cutoff-before-window rule.
- To their credit, these are unusually bullshit-resistant batteries (numeric kill bars,
  pre-committed verdicts, every-variant logging, honest self-attack sections). The remaining
  wiggle is concentrated in §4's eleven line items — close them in TOURNAMENT.md and the
  batteries survive vagueness pressure.

*End of ATTACK_SKEPTIC.*
