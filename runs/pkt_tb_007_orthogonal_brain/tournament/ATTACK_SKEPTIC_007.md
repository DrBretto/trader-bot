# ATTACK_SKEPTIC_007 — Phase B attack pass over DESIGN_MIN and DESIGN_MAX

**Role:** Skeptic (panel role 2, mandatory). **Date:** 2026-06-10.
**Sources read:** packet PKT-TB-007; ANALYST_AUDIT.md; UNIVERSE_EXPLOITABILITY.md;
DESIGN_MIN.md; DESIGN_MAX.md; all four Phase-A proposals; TB-006
REVIEW_SKEPTIC_PHASE_D.md (my prior findings, carried); EVIDENCE_PROTOCOL.md. Every
number below is recomputed from the cited text's own figures or derived with the
arithmetic shown. Specialty per packet: anti-leakage/circularity, multiplicity, sneer.

---

## 0. Ruling summary (details and arithmetic in the numbered sections)

- **KILL (MIN):** P1-as-primary (DESIGN_MIN §8) — the registered primary is the EA's own
  fitness, on the EA's own training folds, in the EA's own simulator, against a baseline
  fabricated by running a 2026-trained model on 2020–2026 prices. False-BEATS rate under
  a true-zero brain: **~12–25%** (§2.1). A primary should hold ~2.5%.
- **KILL (MAX):** M6 GAP-GRU as a battery/LOO organ (§4.1.1); the 30-param executive
  gate (§4.1.2 — twin test <50% power at 6–8 effective regime episodes); the hybrid
  secondary arm (§4.1.3 — confounded, certifies nothing).
- **RENAME/RESCOPE (MAX):** TIES-POSITIVE (DESIGN_MAX §9) — null firing rate 26–48%
  (§2.2). Keep the cell, change the claim it licenses.
- **LEAKAGE MUST-FIX (both):** tier masks / support sets selected on the same F1–F6 OOF
  the members trained on and the EA optimizes on, then re-read by fold-space verdicts
  AND by the rotation gate's "unseen" folds (§1.3–1.4). The Universe Selector's own
  numbers imply the selected core's displayed per-name ICs shrink from 0.08–0.15 toward
  ≈0.05 (§1.3) — both designs' Grinold/NetEdge arithmetic inherits this.
- **VERDICT HIERARCHY IMPOSED:** live real-engine E1 is the only admissible registered
  primary; the only admissible fold-space headline is the pooled UNSEEN-fold rotation
  series; full ruling with permitted final-line claims in §2.3.

---

## 1. Circularity and leakage — the actual data flows

### 1.1 Organ targets: clean. The conditioning surfaces are not.

The targets themselves (y5_rank, D+5→D+21 rank, dispersion, bucket exceedance — DESIGN_MIN
§2.2 / DESIGN_MAX §2) are functions of market prices, cross-sectionally demeaned. A $100k
bot does not move ETF prices; the incumbent's behavior does not enter the target values.
That part of the circularity audit passes for both designs. The chassis leaks in through
three other surfaces, and both designs treat all three as free: (1) **the tilt support
set** (DESIGN_MIN §3 / DESIGN_MAX §3: chassis buy set ∪ held ∪ tilt core) — what the
brain may express is conditioned on chassis output; (2) **the surrogate's base arm**
(both adopt PROPOSAL_EVOLUTION §5) — the fold-era "incumbent" is synthesized by the
deployed MLP walked backwards (§1.2); (3) **the trust ledger** — organ_trust genes
optimized against (2) restricted to (1), then re-read by fold-space verdicts (§1.3–1.4).

### 1.2 The chassis surrogate: legitimate counterfactual or time-travel? Both, and the design must say which half it is using.

PROPOSAL_EVOLUTION §5 (adopted by DESIGN_MIN §7 and DESIGN_MAX §8): "base ranking from the
deployed MLP run on historical features" over F1–F6 (2020-02→2026-02). The deployed model
(`models/ranking_expanded_unconditioned`, written 2026-03-21 — ANALYST_AUDIT A4) was
trained on data that overlaps the fold era. Two distinct problems:

- **As a fixed-policy counterfactual** ("what would this policy have done in 2020") the
  walk-back is legitimate — IF the policy were exogenous to the era. It is not: the MLP
  was fit on these years. Its fold-era scores are partially memorized, so the synthetic
  incumbent is **in-sample-flattered** — sharper selection than any policy that existed
  in 2020 (nothing existed; the whole system is 2025+).
- **What it biases in the EA's fitness:** (i) the base book over-holds retrospective
  winners → the tilt's residual alpha on that support is depleted relative to live, AND
  the support set is tilted toward names the model retro-"knew" — sign of the net bias
  on mean(Δr) is **unknown and unmeasured in either design**; (ii) an in-sample-sharp
  base arm understates selection noise → surrogate paired sd understated → fitness IRs
  inflated → the adoption gate's 1.0σ bar softer than it looks; (iii) trust weights are
  tuned against a stronger-than-real member zero, mismeasuring M1's fold-space marginal
  value in a direction that does not transfer.
- **Cheap falsifier neither design includes — DEMANDED:** compute the deployed MLP's
  per-fold rank-IC over F1–F6 vs its live-era (2026-01-31→03-10) rank-IC. Fold ≫ live
  ⇒ memorization measured, surrogate bias quantified; match ⇒ the time-travel objection
  collapses to a footnote. One script, pre-holdout only, zero new looks. Plus one
  surrogate sensitivity arm: base arm at blend 0 (health-only ranking) — champion trust
  weights unstable across base-arm specs ⇒ the fitness measures the fiction, not organs.
- **Surrogate-incumbent fidelity gap, named:** the real incumbent's base_score is
  0.65·health + 0.35·MLP with regime multiplier and threshold-by-regime
  (ANALYST_AUDIT A1). PROPOSAL_EVOLUTION §5 describes the surrogate base as deployed-MLP
  + top-N + vol-sizing. If the 0.65 health blend and regime-threshold behavior are not
  in the surrogate, the fitness is paired against a *different selection policy* than
  the verdict's incumbent. The anchor check is the only thing standing between this and
  the verdict — see §1.6 for why that is thin.

### 1.3 The exploitability-tier masks: selection on the training data, with the inflation computable from the Universe Selector's own table

The tilt core (10 names) was selected by per-symbol OOF IC over F1–F6
(UNIVERSE_EXPLOITABILITY §4, rule (i): 90% CI fully > 0). DESIGN_MIN §3 freezes these
tiers as caps; DESIGN_MAX §7 re-derives masks from the NEW members' own F1–F6 OOF at
member acceptance — a *fresher* version of the same leak. Then both designs' fold-space
reads (MIN's P1; MAX's forecast battery and EA fitness) are evaluated **on the same
folds, restricted to the selected names.**

The inflation, from the table's own numbers: CAST mean per-symbol IC +0.041,
cross-symbol sd of estimates 0.046, per-estimate se ≈ 0.079/1.645 ≈ 0.048. Note first
that the **cross-symbol variance of the estimates (0.046²) is no larger than the noise
variance of one estimate (0.048²)** — the displayed per-name variation is statistically
consistent with noise around a common IC ≈ 0.04 (Friedman p=0.012 says SOME true
tiering exists, but small). Selection bound: a name enters the core iff
IC_hat > ~0.079; conditional on true IC 0.04, E[IC_hat | selected] = 0.04 +
0.048·λ(0.81) ≈ **0.106** (inverse Mills). The selected names' displayed ICs (ITA .148,
FXE .134, …, core table mean ≈ 0.10) are exactly what pure selection noise around a
uniform 0.04 would display. Honest expected true edge of the core: **≈ 0.04–0.06, not
0.08–0.15.**

Consequences both designs must absorb:
- PROPOSAL_EXPRESSION §4's Grinold ceiling used blend-IC ≈ 0.08 → ann IR 1.2–1.8 →
  expected live t ≈ 0.6–0.9. At shrunk IC 0.04–0.06 the ceiling is **IR ≈ 0.6–1.1,
  expected live t ≈ 0.3–0.7.** Both designs quote the unshrunk ceiling (DESIGN_MIN §8
  preamble, DESIGN_MAX §1).
- Every fold-space utility read on the masked support re-realizes the selection bias;
  the live read does not (selection precedes the live window) — one more reason the
  verdict must live there (§2.3).
- **Fix demanded:** (a) cross-fit — select tiers on F1–F3, evaluate fold-space
  statistics on F4–F6 and vice versa — or (b) freeze the Phase-0 table, print the
  shrinkage arithmetic next to every fold-space number, and let the live-engine read
  carry the verdict. MAX's §7 refresh-at-acceptance is the worst of the three options
  and is rejected as written.

### 1.4 The rotation gate's "unseen" folds are not unseen — contamination via global masks and gates

PROPOSAL_EVOLUTION §2.2 (adopted by both): rotation r holds fold f out of the EA's
fitness. But in both designs the tier masks (§1.3), the C2 matrix, and MIN's acceptance
gates (§2.3) are computed ONCE on pooled F1–F6 — **including fold f** — and held fixed
across rotations. The rotation champion evaluated on "unseen" fold f trades a support
set and an organ roster chosen with fold f's data in hand. The gate certifies
gene-setting generalization only, not pipeline generalization, while both designs lean
on it as the system-level instrument (DESIGN_MIN §6: "the one meta-question that
matters"). **Fix demanded:** each rotation re-derives masks, acceptance outcomes, and
(cheaply approximated) C2 outcomes from its own 5 training folds; if judged too
expensive, the gate is relabeled "EA-gene generalization only" in the pre-registration
and may not be cited as evidence the brain generalizes.

### 1.5 MIN's acceptance gates and both designs' organ gates: thresholds are a mix of formalities and coin flips, none derived

DESIGN_MIN §2.3, attacked row by row:
- M1: "rank-IC ≥ 0.04" — CAST's known IC is 0.105; a bar at 38% of an already-measured
  value cannot fail. A formality, not a gate.
- M2: "t ≥ +1.5" — known TB-006 E1 is +2.48; same objection, and 1.5 is underived.
- M3: "R² > 0 vs trailing-mean baseline" — no significance requirement; a noise fit
  clears it ≈ 50% of the time. **Coin-flip gate.**
- M4: "AUC ≥ 0.55" — at n_eff ≈ 4,500, se(AUC) ≈ 0.01 → a real ~5σ bar. But the
  conjunct "uplift over vol-only control > 0" is again a 50% null pass.
- M5: "per-episode sign tally > 50%" on n ≈ 15–25 — binomial: at n=20, P(>10 | p=0.5)
  ≈ **41%**. The mandatory static-trust test the packet asks about has essentially no
  power: the rule would be ENABLED on a coin flip 4 times in 10 under the null,
  contradicting MIN's own ships-disabled expectation. (Same gate in
  PROPOSAL_ORTHOGONALITY §5, inherited by MAX.)
- MAX's M4-A/B (DESIGN_MAX §2): ΔAUC > +0.01 on F5–F6 only (~500 OOF days, n_eff ≈ 100
  → se(ΔAUC) ≈ 0.02–0.03) → under the null the LLM columns get admitted ≈ **30% of the
  time**. Raise to 2·se or pre-commit M4-A.

**Ruling:** every gate threshold null-calibrated (permutation or analytic), false-pass
rate printed in the pre-registration; known-pass gates (M1, M2) labeled formalities so
the ledger does not count them as evidence; where a gate decides verdict-arm
membership, indeterminate maps to DISABLED (fix M5: enable only at binomial p < 0.05,
≥15/20).

### 1.6 The anchor checks: carrying the entire surrogate's validity at near-zero power

- **MIN's anchor (DESIGN_MIN §7):** ~26–27 live dates; per-day Δr Pearson ≥ 0.8 AND
  "sign agreement of the mean ΔU." The sign clause is vacuous: with true effect 1–2
  bp/day and paired sd 6–15 bp/day, se(mean) at n=26 ≈ 1.2–3 bp/day → the real-harness
  mean's sign is near a coin flip EVEN WHEN THE SURROGATE IS PERFECT (t ≈ 0.5–1); a
  junk-but-correlated surrogate passes ~50% of the time, a correct one fails ~30–40%.
  The Pearson clause has real power (Fisher-z se ≈ 0.21) but 2–3 large days can carry it.
- **MAX's anchor (DESIGN_MAX §8):** sign agreement on the 2025 artifact window — 122
  days on which the brain can express 2 of 10 core names (PROPOSAL_EXPRESSION §4.1),
  i.e. ΔU ≈ 0 by construction, on a four-epoch repaired record (ANALYST_AUDIT B4) where
  surrogate-vs-real disagreement is as likely artifact-vintage noise as
  misspecification. **A sign test on a mean engineered to be ≈ 0 is a coin toss.**
  Building the epoch shim to run it buys negative information per unit work.
- **Ruling:** adopt MIN's window (live pre-holdout, no shim); criteria become per-day
  Pearson ≥ 0.8 AND regression slope of real-on-surrogate Δr ∈ [0.7, 1.3] (magnitude
  calibration, which sign agreement is not); failure = stop-and-fix finding; the
  check's own power limits printed. It gates admissibility of every surrogate-space
  statistic, so it runs before the first EA generation, as both designs already say.

### 1.7 Smaller leakage items (both designs)

- **Deployed-MLP training window vs the holdout firewall:** the model was written
  2026-03-21 — after holdout start 2026-03-11. If its training data extends past 03-11,
  the incumbent arm's holdout behavior is generated by a model trained on early holdout
  days. Both arms share it, so the paired Δ cancels — but E2's "untouched holdout"
  sentence must carry the disclosure. **Verify the training-data end date and print it.**
- **C2 matrix vs member zero:** M1-vs-member-zero correlation is measured on data
  in-sample for member zero — the 0.8 bar faces a flattered opponent. Report-only.
- Model-cache hazard + separate-process discipline: correctly handled in both designs
  (ANALYST_AUDIT A4). No finding. Repaired-timeline epochs: both print the directive-6
  caveat; MIN's refusal to run the 2025 leg (§7) also removes the widest repair (the
  2026-04-01 signals rebuild) from the verdict surface entirely. Credit where due.

---

## 2. The verdict hierarchy under the ceiling arithmetic

### 2.1 DESIGN_MIN §8 P1: the primary verdict is the optimizer's loss re-read as evidence — REJECTED as registered

The chain, in MIN's own words: fitness = floored paired IR of champion-vs-B0 on F1–F6
(PROPOSAL_EVOLUTION §1, adopted §4); P1 = "paired tilt series (champion vs B0,
surrogate, OOF organ outputs) pooled over F1–F6," HAC t ≥ +2.0, registered as
**primary**. The champion is the (deduped median of the) top of a ~400-genome search
that maximized approximately this statistic on exactly these folds. P1 is not a test;
it is the search objective. Its null under the search is the max-of-K order statistic
PROPOSAL_EVOLUTION §2.1.3 itself computed at ≈1.4σ_fold of pure noise uplift — above
the 1.0σ adoption gate, which the proposal acknowledges and then leaves at 1.0σ.

False-BEATS arithmetic for "BEATS = P1 ∧ P2 ∧ P3" under a true-zero champion that
cleared the gates (the TB-006 precedent: champion cleared the gate at 1.37× and read
0-vs-B0 on replay):
- P1: pooled t ≈ in-sample IR_ann × √(n_eff/252) ≈ 1.09 × IR_ann at n_eff ≈ 300; the
  search inflates in-sample IR by the documented ≈1.4σ noise uplift. P(P1 | gates
  passed, true zero) is not 2.5%; it is plausibly 30–60%, and the design never
  estimates it.
- P2 ("mean Δ > 0 AND 90% CI not entirely below 0", n=66): the CI clause only excludes
  t ≤ −1.645, so P2 ≈ P(t > 0) = 50% under the null. P3 ("point estimate ≥ 0", n=43):
  50% under the null — and P2/P3 share 43 of 66 days, so jointly ≈ 35–40%, not 25%.
- **Net: P(false BEATS) ≈ 0.3–0.6 × 0.35–0.40 ≈ 12–25%** — a one-in-five false-positive
  headline. MIN §10.6 concedes the critique and prints it; printing it does not
  discharge it when the cell is named BEATS.

What a hostile reader says, verbatim quotable: *"They certified their improvement in
their own simulator, on the data the optimizer trained on, against a baseline the
simulator fabricates by running a 2026 model on 2020 prices, and confirmed it with
two coin-flip checks."* Every clause is, as registered, true.

### 2.2 DESIGN_MAX §9 TIES-POSITIVE: a flattering tag with a 26–48% null rate

The cell: t ∈ (0, 2.0) AND 90% CI excludes −2 bp/day, n=66. Null arithmetic at the
design's own sd range: sd = 6 bp/day ⇒ se = 0.74, CI excludes −2 iff t > −1.06, so the
cell fires iff 0 < t < 2 → **P ≈ 48% under a true-zero brain**; sd = 15 ⇒ se = 1.85,
needs t > +0.56 → **P ≈ 27%.** The tag a reader will parse as "real but small" attaches
to a true-zero brain roughly one run in three. What it actually certifies is a
**harmlessness bound** plus a positive coin flip — a legitimate cell whose LABEL is the
escape hatch, not its math. **Ruling:** keep the cell, rename: `TIES (positive point
estimate; certified not-worse-than −2 bp/day at 90%)`. The words "real," "works,"
"small but positive signal" are banned from its prose; any "positive" wording requires
one-sided t ≥ +1.645, pre-registered as a distinct cell if wanted.

### 2.3 The minimum honest hierarchy (imposed; the synthesis adopts this or argues to the Chair in writing)

1. **Registered primary = live real-engine E1**: full live window, n≈66 paired deltas,
   tilt verdict arm, paired t. The packet floor stands: **BEATS ⇔ E1 t ≥ +2.0 AND E2
   (one holdout read, n≈43) point estimate ≥ 0.** E2 never upgrades. Yes, this makes
   BEATS nearly unreachable (expected t ≈ 0.3–0.7 after §1.3 shrinkage); that is the
   truth of a 66-day record, and the verdict must say it rather than relocate the
   primary to where the truth is softer. A surrogate-space number may not be the basis
   of the registered BEATS/TIES/LOSES; per EVIDENCE_PROTOCOL, E1/E2 are replay-engine
   counterfactuals — a fast-sim fold read is not an E-grade at all.
2. **The only admissible fold-space headline: the pooled unseen-fold rotation series.**
   Concatenate each rotation champion's paired ΔU on its held-out fold (6 folds, ~1,500
   days, genuinely out-of-search once §1.4 is fixed); report its HAC t as
   **"procedure-level evidence (surrogate space)"** — it certifies the balancing
   PROCEDURE generalizes, never that the brain beats the incumbent. MIN's P1 is
   admissible only as a labeled training diagnostic, never quotable in the final line.
3. **Permitted final-line claims:** BEATS — "beats incumbent at matched exposure, live
   engine, holdout-confirmed" (the only cell licensing "beats"); TIES (positive point
   estimate; bounded) — "no certifiable difference; loss bounded at 2 bp/day; point
   estimates favor the brain; procedure-level fold evidence t=X (surrogate space)" —
   the expected honest outcome, a passing one per the packet; TIES (straddling) — "no
   certifiable difference at available power; MDE printed"; LOSES — live t ≤ −2.0 or
   holdout 90% CI entirely < 0.
4. **Every surrogate-space number in the dossier carries the suffix "(surrogate space)"
   in the same sentence — no exceptions, including the EA gates' values.** The TB-006
   lesson (my F4: "TIES means underpowered, not parity") generalizes: power statements
   and space statements travel WITH the number, not in a footnote.

---

## 3. Multiplicity — counting the looks the designs actually imply

### 3.1 DESIGN_MIN's count (its §8 ledger undercounts)

Registered by MIN: ≤12 battery arms, E2=1, EA gates "do not consume battery slots."
Actually implied: 5 acceptance gates (selection events deciding verdict-arm COMPOSITION
— one of up to 2³ = 8 rosters once the M1/M2 formalities are set aside) + the C2 matrix
(10+ directional pairs × 2 spaces, remediation ladder ≤3 rungs each = up to ~12
retrain-and-regate events with **no stated retrain cap** — TB-006 capped at 9, used 5)
+ anchor check + EA adoption/rotation gates + P1/P2/P3 + ≤12 arms + R-LLM.
**Finding: the acceptance gates are selection events feeding the verdict arm and MIN's
ledger does not carry them.** Cheap fix: ledger every gate outcome with its null pass
rate (§1.5), and pre-register the mapping from every gate-outcome vector to a roster,
so roster choice is mechanical, not a Phase-C judgment call. **Consistency kill:** §2.1
promises every gate-failer runs as a challenger arm; §8 budgets "≤2 challenger arms."
Three failures (M3+M4+M5 is live) breaks the promise. Pick one.

### 3.2 DESIGN_MAX's count and the expected false positives

14–17 replay arms (+3 contingent) + M4-A/B (~30% null-pass as specified, §1.5) +
executive twin (2 comparisons) + 6 rotations + R-LLM kill + M6 demotion (3 criteria) +
tier-mask freeze (a 64×6-cell selection event, §1.3) + the per-organ forecast battery
(BH-FDR(10%) applied — correct; MIN should graft exactly this). Expected false
positives among utility-altitude organ tags: 6 LOO arms at fold-pool t ≥ +2 one-sided
≈ 2.3% each → P(≥1 spurious "positive" organ) ≈ 1 − 0.977⁶ ≈ **13%**; across all 14–17
arms at |t| ≥ 2 ≈ **30–35%** chance of one spurious certifiable-looking line. With
per-arm MDE 2–4 bp/day against true organ contributions of 0.5–3 bp/day (MAX's own
§11.2), the most likely battery output is 6 INDETERMINATEs plus a ~1-in-3 chance of a
false flourish. MAX prints the MDEs (good); it must also print this family-wise
expectation so a single positive LOO line cannot be narrated as a discovery.

### 3.3 The EA's internal multiplicity (both designs)

Production K ≤ 400 + 6×190 rotation evals + B1 ≈ 1,900 genome looks, governed by the
adoption gate at 1.0σ — which PROPOSAL_EVOLUTION §2.1.3 itself shows sits BELOW the
≈1.4σ expected pure-noise max-of-K uplift. Both designs adopt this unchanged. **Demand:**
raise the adoption gate to the computed noise ceiling (≈1.4–1.5σ from the actual K at
run time), or formally designate the rotation gate as the sole binding control and say
in the pre-registration that the adoption gate is decorative. Either is honest; the
current text implies a control it does not have.

---

## 4. Per-design kill lists

### 4.1 DESIGN_MAX

1. **M6 GAP-GRU — KILL as an organ; keep at most as a forecast-altitude line.**
   (a) *The lever does not exist in the harness.* The replay fills intents at next open,
   one fill per day (PROPOSAL_EXPRESSION §1b mechanics). "Entry-timing refinement on
   funded tilts" (DESIGN_MAX §2) names no representable action: within-day timing is not
   in the simulator's vocabulary, and the design never specifies what the intents DO
   differently at m6_timing_gain > 0 (delay a tilt leg one day? reorder buy cash
   priority?). An organ whose expression is unspecified cannot be attributed.
   (b) *The LOO arm is pre-ordained indeterminate:* 1d IC ~0.03 × σ1 ~120 bp × the small
   budget fraction it can move ≈ sub-1 bp/day vs per-arm MDE 2–4 bp/day — a battery slot
   spent printing a foregone conclusion.
   (c) *Target overlap:* y1_z's single bar IS the first bar of M1's y5 window — the
   disjointness claim is true for M2, false for M6; correct the text even if the C2
   gate would catch it.
   (d) The stated motive — "adds the one production-native model type still absent" —
   is presence-for-presence's sake, the failure directive 4 warns against.
   **Survival condition:** specify the exact intent-level mechanics, show the surrogate
   represents them, re-cost the LOO arm honestly; otherwise M6 ships as an OOF forecast
   line ("1d microstructure IC = x (surrogate space), unexpressed — no honest
   expression channel at this fill model") and no genome gene.
2. **The 30-param state-conditioned executive gate (§4.2) — CUT from this packet's
   build.** Conditioning value certifies only across regime VARIATION, of which the
   fold era holds ~6–8 effective episodes (PROPOSAL_ORTHOGONALITY §6 organ-trust row —
   the same arithmetic MAX §4.2 accepts when it rejects attention). The twin test
   demands the gate beat static trust by 1× cross-fold sd on 6 fold scores: power
   against a plausible true uplift of 0.5σ is ≈ 31%. Expected output:
   `executive_gate≈0 (measured)` — MAX §4.2 nearly says so itself. Cost: ~0.5 day, 2
   ledger looks, and one more §1.4-class circularity surface (gradient-trained on the
   same surrogate objective on the same folds; its twin win, if any, is in-search-
   space). The TB-006 lesson was not "build a smaller executive"; it was "the trust
   question needs samples that do not exist yet." Survival condition: per-rotation
   train/test (§1.4) AND a pre-registered expected zero like R-LLM — at which point it
   is a falsifier arm, not an executive, and should be named one.
3. **The hybrid secondary arm (§3) — KILL; keep the perm-only ablation, demoted.** The
   hybrid differs from the verdict arm by the permutation channel AND the live
   channel_mix gene → its delta vs incumbent is not attributable to selection. The
   perm-only arm does isolate the WHICH channel — but at its own sd (~+30 bp/day while
   a swapped position diverges, PROPOSAL_EXPRESSION §1a) its 66-day MDE is tens of
   bp/day: it certifies nothing directional. Its honest products are descriptive
   (guard-infeasible rate, Kendall distances, swap inventory). Keep ONE perm-only arm
   with a pre-committed sentence: "may not be cited as evidence of selection skill at
   any t < 2." Worth its sd? As description yes, as evidence no.
4. **The √k motivation (§1.1) — corrected, because it is the design's load-bearing
   sales arithmetic.** IR_k = IR₁·√(k/(1+(k−1)ρ̄)). At the C2 gate's own ceiling
   ρ̄ = 0.7: ×1.15 at k=6; at a realistic post-gate ρ̄ = 0.3–0.5: ×1.3–1.55. "Scales
   toward √k" (×2.45) overstates the one lever claimed to "move expected t upward" by
   ~2×. And only 2–3 of six organs are return-directional (M3 gain, M4 veto, M5
   episodic, M6 timing) — directional k_eff ≈ 3. Expected t moves from ≈0.5 to ≈0.7,
   not toward 2.0. The maximalist case must be re-argued on per-organ certifiability
   alone (its point 2, real), not headline-t scaling (its point 1, wrong).

### 4.2 DESIGN_MIN

1. **P1-as-primary — KILL** (§2.1 above; the design's central registered structure).
   Replace per §2.3. MIN survives this amputation intact: P2/P3 become the E1/E2
   verdict, the rotation gate's unseen-fold pooled series becomes the powered
   supporting read.
2. **Acceptance-gate thresholds — null-calibrate or they are post-hoc-able** (§1.5).
   The set mixes two formalities (M1, M2), two coin flips (M3's R²>0, M4's uplift>0),
   and one real bar (M4's 0.55). A roster chosen by uncalibrated gates is a roster
   chosen by hand with extra steps. Fix is one page of permutation nulls.
3. **The anchor check's sign clause — vacuous; replace with slope calibration** (§1.6).
   MIN moved the anchor to the right window (credit: deletes the epoch shim and the
   25-name fiction) but hung the surrogate's validity — under MIN's own P1, the
   PRIMARY's validity — on n=26 with a coin-flip criterion. Under the imposed
   hierarchy (§2.3) the anchor's load drops to "EA-training validity," which 26 days
   with Pearson+slope can honestly carry.
4. **Challenger-arm budget contradiction** (§3.1): "every gate-failer runs as a
   challenger" vs "≤2 challenger arms." State the rule that resolves three failures.
5. **VIXY exclusion (§3) — sustained, against MAX §7.** "The σ-projection prices VIXY
   structurally" re-imports risk-appetite through a side door: the projection bounds
   Σ|Δw·σ̂|, but VIXY's convexity and 18 bp RT make its tiny permitted legs pure cost.
   MIN's fiat exclusion is the cleaner registration (its §10.2 "weakness" is the design
   being right). MIN's §10 self-assessment is otherwise accurate, except §10.6, which
   understates: "a critic can say…" — the critic is the registered protocol itself (§2.1).

### 4.3 Exists-to-impress inventory (packet asked directly)

MAX: M6 (sixth-type box-tick), the executive gate (structure as costume — its twin test
is designed to lose), the hybrid arm (a second headline-shaped number), the √k
paragraph. MIN: the "n_eff ≈ 300, MDE 1.2–2 bp/day" dressing of P1 (a powered-sounding
frame on the training objective). Neither LLM section is decorative: both engage R5
honestly and identically; the R-LLM falsifier arm discharges directive 4 at one slot.

---

## 5. Cross-design verdict per assignment item + synthesis instruction

| # | Assignment item | Verdict |
|---|---|---|
| 1 | Orthogonal multi-model brain | **MIN's gated 5-organ set wins.** M4 GDELT-only per the measured R5 (M4-A/B kept only with the bar raised per §1.5). M6 killed as organ (§4.1.1). Directive-4 defense: MIN §2.1's "present = built, measured, attributed" is the honest reading and should be quoted to the operator verbatim. |
| 2 | Expression layer | **MIN's single tilt channel carries the verdict** (both designs agree on the mechanics). Graft from MAX: ONE perm-only descriptive arm with the pre-committed no-claims sentence (§4.1.3). Kill the hybrid. |
| 3 | Evolutionary balancing | PROPOSAL_EVOLUTION adopted by both — adopt, with: adoption gate raised to the computed noise ceiling or demoted to decorative in writing (§3.3); per-rotation mask/gate re-derivation (§1.4); unseen-fold pooled series promoted to the quotable fold-space statistic (§2.3.2). |
| 4 | LLM value-maximizing role | Designs are identical and correct (R5 + one falsifier arm + monitoring + re-test trigger). Keep exactly one R-LLM arm. |
| 5 | Universe exploitability | Tilt core adopted with the shrinkage arithmetic of §1.3 printed beside it; MAX's refresh-at-acceptance rejected; MIN's frozen Phase-0 tiers + VIXY exclusion adopted; cross-fitting demanded for any fold-space read on the masked support. |
| 6 | Per-organ attribution | MIN's LOO set + MAX's BH-FDR forecast-battery discipline + family-wise false-positive expectation printed (§3.2). Most LOO lines will read indeterminate at 2–4 bp/day MDE — pre-register that expectation so it reads as arithmetic, not failure. |

**Synthesis instruction:** build MIN's chassis (5 gated organs, single tilt channel,
live-era window, no epoch shim, 12-gene EA) with these grafts/amendments: the imposed
E1-live-primary verdict frame with the renamed bounded-TIES cell (§2.2–2.3); MAX's
BH-FDR forecast battery; one perm-only descriptive arm; null-calibrated gate thresholds
(§1.5); per-rotation mask re-derivation (§1.4); the deployed-MLP fold-vs-live IC check
and blend-0 sensitivity arm (§1.2); slope-calibrated anchor (§1.6); adoption-gate
noise-ceiling fix (§3.3); ledgered acceptance gates with a pre-registered
outcome→roster map and a retrain cap (§3.1); the MLP training-window disclosure (§1.7).
Kill: M6, the executive gate, the hybrid arm, MIN-P1-as-primary, MAX's 2025-leg anchor,
the TIES-POSITIVE label, MAX §7 mask refresh.

---

## 6. Conditions for survival — the pre-build MUST-FIX list (each blocks pre-registration if unresolved)

1. **L1 — Tier/support cross-fitting or freeze-and-disclose** (§1.3); per-rotation
   re-derivation of masks+gates so the rotation gate's unseen folds are unseen (§1.4).
2. **L2 — Surrogate time-travel quantified:** deployed-MLP per-fold IC vs live IC,
   printed; blend-0 base-arm sensitivity run; surrogate base replicates the 0.65/0.35
   blend + regime threshold or discloses the divergence (§1.2).
3. **L3 — Anchor check re-specified:** live window, Pearson ≥ 0.8 + slope ∈ [0.7,1.3],
   sign-of-mean clause deleted, power limits printed (§1.6).
4. **L4 — Verdict hierarchy per §2.3:** live-engine E1 primary; surrogate-space numbers
   suffixed inline and ineligible for the final line's BEATS/TIES/LOSES; unseen-fold
   pooled series the only fold-space headline.
5. **L5 — All gate thresholds null-calibrated, false-pass rates printed**; M5 enable
   bar at binomial p < 0.05; M4-A/B bar ≥ 2·se (§1.5).
6. **L6 — Multiplicity ledger extended** to acceptance gates, C2 remediation retrains
   (capped), and the family-wise battery false-positive expectation (§3).
7. **L7 — Deployed-MLP training-window end date verified** against the 2026-03-11
   firewall, disclosed with the paired-cancellation argument (§1.7).

Expected honest outcome of the synthesized design, stated now so nobody is surprised in
Phase D: live E1 t ≈ 0.3–0.7 (post-shrinkage ceiling, §1.3) → bounded-TIES with positive
point estimates, a procedure-level fold result of real interest, two to three powered
forecast-altitude organ verdicts, and LOO indeterminates at printed MDEs. That is a
passing outcome under the packet and the strongest certifiable product this record can
produce. A design that promises more is selling something the arithmetic does not own.

*— Skeptic, panel role 2, Phase B attack pass, 2026-06-10*
