# ATTACK — TRAINING REALIST (role 8) — PKT-TB-007 Phase B

**Author:** Training Realist, 2026-06-10.
**Inputs:** packet; ANALYST_AUDIT; UNIVERSE_EXPLOITABILITY (+json); PROPOSAL_{ORTHOGONALITY,
EXPRESSION, EVOLUTION_007, LLM_ROLE}; DESIGN_MIN; DESIGN_MAX; my TB-006 conventions
(ATTACK_TRAINING_REALIST §0 axioms N1–N4; REVIEW_TRAINING_REALIST_PHASE_D §§2–9).
This is arithmetic, not opinion. Where a design's number disagrees with mine, both appear.

## 0. Carried conventions (recomputed where the substrate changed)

- **N1:** 64 ETFs ≈ **12 effective names** for market-neutral targets (Phase 0 confirms
  the structure independently: ENB 2.94 for LEVEL, mega-clone cluster ENB 1.14/14 names;
  the 10-name tilt core alone is ENB 4.48 — N1's block count was right).
- **N2:** ~**580** independent 5d windows full-history; ~558 on the GDELT panel;
  h=16-day windows (the M2 target) → ~**147** independent time windows on the
  ~2,344 valid-D+21 days; 1d targets → no overlap divisor, but IC observations cluster
  in vol regimes (haircut ×1.4–2, see §1.4).
- **N3:** ~**6–8 regime observations** for anything encoding regime-conditional
  disposition. Unchanged. This governs the EA genome AND MAX's executive gate.
- **N4 (new windows):** live verdict n=66 paired deltas, holdout n=43. At the
  Expression Architect's tilt sd 6–15 bp/day: live MDE@t2 = 1.5–3.7 bp/day, holdout
  1.8–4.6. Per-LOO-arm MDE ~2–4 bp/day. Fold-pool E1 (n=1,507, n_eff≈300): per-organ
  MDE ~0.2–1.8 bp/day — still the only powered utility instrument.
- **Master conversion:** effective ≈ rows × (12/64) ÷ h_overlap. Daily cross-sectional
  IC sd at breadth 12 ≈ 1/√11 ≈ **0.30** (the constant behind every gate-power number
  below).

Both designs adopted my conventions correctly in their tables. The attack therefore moves
one level up: not "are the effective-n claims honest" (mostly yes) but "what do those n
actually buy at the gates, the meta-layer, and the verdict reads."

---

## 1. The new targets, audited

### 1.1 M2 — D+5→D+21 rotation (both designs)

**Effective-n verification.** ~150k symbol-days with valid D+21 → 150k × (12/64) ÷ 16 =
**1,758** ✓ (both designs claim ~1,750; the 16-divisor is correct — the window is 16
trading days, consecutive targets share 15/16). Capacity ~600 leaf values → 0.34
params/eff — SUPPORTED. The 4× cut vs TB-006's GBM correctly pays the overlap tax.

**But the gate runs on ~147 time windows, not 1,758.** The acceptance statistic (mean
daily cross-sectional rank-IC) is already cross-sectionally aggregated; its se is
0.30/√147 ≈ **0.025**. The MIN gate (overlap-corrected t ≥ +1.5) therefore demands
mean IC ≥ 0.037. Power table:

| true rotation IC | source of the hypothesis | P(pass t≥1.5) |
|---|---|---|
| 0.000 (null) | — | 6.7% |
| 0.025 (half the 5d GBM receipt) | signal decay with horizon | **31%** |
| 0.049 (full 5d GBM receipt carried) | no decay | 68% |

**Is rotation alpha believed at this granularity? It is UNMEASURED.** Every OOF receipt
cited for M2 — whole-book rank-IC +0.049 (se 0.018, 6/6 folds), E1 HAC t +2.48, per-name
skill on XRT +0.111 / INDA +0.095 / XLY +0.086 — was measured at the **D→D+5** target
(UNIVERSE_EXPLOITABILITY §0 methods; TB-006 OOF matrices). No artifact in either run
measures anything at D+5→D+21. The design's own orthogonality argument cuts against its
evidence claim: **a target that shares zero bars with the old one also shares zero
receipts with it.** Residual predictability at D+5→D+21 exists only through genuine
return autocorrelation at the 3-week scale — exactly the thing nobody has measured here.
MIN §2.3's "Expected shipping set: M1+M2 certain" is wrong as written: M2's ship
probability under honest decay priors is the 31–68% band above, not certainty. Neither
design needs to change the member — the gate is the first measurement and that is the
correct posture — but the pre-registration must carry M2 as OPEN, and the synthesis must
not build anything downstream that breaks if M2 demotes (currently true in both: M1
alone still feeds the tilt).

### 1.2 M3 — dispersion HAR, ~558 effective

558 ÷ ~10 OLS coefs — SUPPORTED for the **fit**. Sufficient for WHAT claimed effect:

- **The forecast gate (R² > trailing-21d-mean baseline) is near-unfalsifiable.**
  Dispersion is vol-clustering-family with daily autocorrelation ~0.7+; HAR lags 1/5/21
  beat a trailing mean almost by construction. P(pass | the member is just an AR
  repackaging) ≈ high. The gate admits the member; it does not certify the INCREMENT the
  vol-surface block (COR3M/VVIX/SKEW) adds. Cheap fix in §4.
- **The utility effect it can claim is second-order.** M3 modulates tilt gain. If the
  dispersion forecast correlates ~0.5 with realized and swings the gain ±30%, the paired
  uplift ≈ 0.3 × 0.5 × (mean tilt PnL 1–3 bp/day) ≈ **0.15–0.45 bp/day** — below the
  live LOO MDE (2–4), at the edge of the fold-pool MDE band (0.2–1). Pre-register M3's
  utility line as likely `indeterminate`; its certifiable line is the forecast gate.
- 558 effective is also the ceiling for any "second market-level scalar" — the proposal's
  rejection of breadth/seasonality members on this arithmetic is correct and MAX's
  +3-calendar-dummies-inside-M3 compromise is the right shape (capacity +3 vs 558).

### 1.3 M4 — exceedance target, event-rate × bucket arithmetic

Target = 1{|5d bucket excess| > trailing 80th pct} → base rate 20% **by construction**.
75k bucket-days × (8/27 effective buckets) ÷ 5 ≈ 4,500 effective ✓; effective positives ≈
900. AUC se (Hanley-McNeil at AUC≈0.5) ≈ √(0.0833/900 + 0.0833/3,600) ≈ **0.011**;
exceedances cluster in vol episodes (the proposal's own printed caveat), haircut ×1.5 →
se ≈ **0.016**. So the 0.55 admission bar is ~3σ above null — the one genuinely powered
admission gate in the set. L1-sparsified ~60–200 active coefs vs 4,500 — SUPPORTED.

**Where the arithmetic breaks: MAX's M4-A/B decision.** The LLM-columns decision runs on
F5–F6 only (the LLM-covered folds): ~500 days ÷ 5 ≈ 100 windows × 8 effective buckets ≈
800 effective, ~160 effective positives → per-variant AUC se ≈ 0.026; the A−B difference
of two nested models is correlated but its se is still ≳ **0.02**. MAX's ship-B bar is
ΔAUC > **+0.01 — half its own measurement noise.** P(ship B | LLM columns truly add
nothing) ≈ Φ(−0.01/0.02) ≈ **31%**. That is a coin flip dressed as a measurement, and it
re-opens a door the Role-Finder closed with 46 screens and 0 FDR survivors. Forced:
either raise the bar to 2×se (≈ +0.04–0.05) or adopt MIN's position (ship M4-A, log B as
a challenger). The pre-registered LLM zero already discharges directive 4.

### 1.4 M6 — GAP-GRU, the MAX claim of 35.4k effective ÷ 1 overlap

**First, the double-count check the committee asked for: MAX did NOT skip the
cross-sectional cut.** 189k × (12/64) = 35.4k — the ~5× breadth tax is already paid in
the printed number. Attacking it again would be double-counting. The honest haircuts
that REMAIN:

- **Time clustering of 1d IC observations** (no overlap divisor, but flow-signature ICs
  cluster in vol regimes): ×1.4–2 → honest fitting sample ≈ **18–25k**. Still 7–10×
  the 2.5k params. The fitting verdict SUPPORTED stands; M6 is genuinely the densest
  target in the system, and its falsifier is the most powered: se(mean 1d IC) ≈
  0.30/√2,950 × 1.4 ≈ 0.008, so the IC<0.02 demotion bar separates null from threshold
  at ~2.5σ.
- **Regime fragility is governed by N3, not by n.** Overnight-vs-intraday and gap-fill
  effects are the most era-dependent microstructure family on record (the overnight
  anomaly's published strength decays post-2015); 18–25k effective observations of a
  sign that flips by regime still yield ~6–8 independent observations of WHETHER it
  holds. The dense n certifies the fit, not the persistence. MAX's per-fold falsifier
  partially covers this; the pre-registration should require sign-consistency across
  ≥4/6 folds for the IC gate, same as everything else.
- **The utility channel is the real zero.** Standalone: IC 0.03 × σ1 (~100 bp) ≈ 3.6 bp
  gross vs 6–18 bp RT — negative, and MAX correctly never trades it standalone. But then
  M6's only expression is 1-day re-timing/ordering of tilts other organs fund: effect ≈
  IC_1d × σ1 × tilted-NAV-share ≈ 0.03 × 100bp × 8% ≈ **0.2 bp/day on the tilt sleeve ≈
  ≤0.1 bp/day at book level** — below the floor of even the fold-pool instrument
  (MDE ≥ 0.2) and 20–40× under the live LOO arm MDE. **M6's utility verdict is
  structurally unresolvable by any instrument in the program at this T_max.**

**M6 ruling:** ADMIT at forecast altitude only — the arithmetic is honest, it is the
cheapest member (GRU ≤1h train), demotion costs nothing, and a powered 1d-IC line is a
real addition to the certifiable product. But (binding, §6): drop its live LOO replay
arm (it cannot resolve; it spends a battery slot and adds sd surface), print its utility
line as `structurally unresolvable at this T_max — forecast-altitude verdict only`, and
strike the √k-more-shots framing for M6 in MAX §1: an organ whose book-level ceiling is
~0.1 bp/day does not move the expected headline t at all.

### 1.5 M5 — ~15–25 crowding episodes

0 trained params — SUPPORTED because capacity = 0, both designs, correct. **But the MIN
acceptance gate is miscalibrated in the permissive direction:** "per-episode sign tally
> 50%" on n≈20 episodes is passed by a COIN FLIP 41% of the time (P(≥11/20 | p=0.5) =
0.41). As written, a noise rule has a 2-in-5 chance of wiring itself into the verdict
arm's tilt. MIN's own expectation ("likely ships DISABLED") contradicts its own gate.
Forced: binomial significance, e.g. ≥14/20 (p≈0.058) or one-sided p ≤ 0.10, else
disabled-with-tally-printed. At n=15–25 the honest expected outcome remains
indeterminate-disabled, which is fine for a 0-param organ.

---

## 2. The C2 gate's statistical reality

Pooled statistic: n_eff ≈ 300 → corr se ≈ **0.06** (the proposal's number, confirmed).
Per-fold statistic: n_eff ≈ 50/fold → se ≈ 1/√47 ≈ **0.146**.

**The pooled gate (≤0.7) is powered and safe.** P(pooled ρ̂ > 0.7 | true ρ = 0.5) =
Φ(−3.3) ≈ 0.04%. False alarms on the pooled bar are a non-issue; and in the direction
that matters, a true-0.78 pair sneaks under 0.7 only ~9% of the time. Fit for purpose.

**The per-fold ceiling (no fold > 0.8) is where the noise lives.** Per-cell breach
probabilities: P(fold ρ̂ > 0.8 | true 0.5) ≈ 2.0%; | true 0.6 ≈ 8.5%; | true 0.7 ≈ 25%.
Cell counts: MIN ≈ 7 gated pairs × 6 folds × 2 spaces = 84 cells (+~30 scalar-row
cells); MAX ≈ 21 pairs → ~**252 cells** + scalar rows.

- A genuinely orthogonal MIN set with most pairs at true ρ ≈ 0.3–0.4 and one or two
  pairs honestly at 0.55–0.65 (M1-vs-M2 is the candidate) expects **~1 spurious
  fold-level breach**; for a single pair at true 0.6, P(≥1 breach across its 12 cells)
  ≈ **65%**.
- **The M1-vs-member-zero row is near-certain to breach.** M1 shares 6/10 features and
  the momentum family with the deployed MLP; if its true pooled corr sits ≈ 0.7 (the
  realistic case), P(some fold ρ̂ > 0.8) = 1−(1−0.25)^12 ≈ **97%**. As written, the
  ladder then fires on the program's most valuable member — re-partitioning M1 in
  response to a noise draw.
- MAX at 252 cells expects **2–4 spurious breaches** even with everything genuinely
  engineered — ladder churn is near-certain, and each remediation rung retrains the
  junior and re-rolls the gate: iterate-until-the-noise-passes is selection on the gate
  itself, a forking path the multiplicity ledger currently does not count.

**Recommended multiplicity-honest gate form (binding for both designs):**

1. **Pooled |ρ| ≤ 0.7 stays the binding gate, unchanged** (se 0.06 — powered).
2. **Per-fold ceiling becomes a replication trigger, not a single-cell trip-wire:** the
   ladder fires only if a pair breaches 0.8 in **≥2 of its 12 fold×space cells**, or in
   1 cell by ≥2×fold-se (ρ̂ > 0.8 + 0.29). Under true ρ = 0.5 the double-breach
   probability is ~1.5%/pair (vs 21% single-cell); at true 0.6 it is ~24% — the trigger
   still catches genuine fold-collapse (true 0.85 in one regime ⇒ both spaces breach
   together) while ignoring lone noise cells.
3. **Every remediation rung is a look-ledger entry**, the system-wide rung budget is
   pre-registered (≤4 total), and a pair that exhausts its rungs demotes — no re-rolls
   beyond the ladder as written.
4. The M1-vs-member-zero row keeps its pooled 0.8 bar but gets **no per-fold ceiling**
   (per-fold se 0.146 against a 0.8 bar on a true-0.7 quantity is pure noise); instead
   its per-fold values are PRINTED, and M1's separate acceptance condition (forecast IC
   delta vs member zero > 0) carries the redundancy burden — that statistic is powered.

---

## 3. The executive question, settled by arithmetic

**MAX's 30-param state-conditioned gate trains on how many effective state-episodes?**
MAX claims ~300 (1,507 ÷ 5). That is the count of utility WINDOWS; the gate's question —
"does organ k's relative paired utility rotate with state?" — is an interaction effect
on a utility-class target whose full-history t is **0.5–0.8** (TB-006 §3.2, verified by
the TB-006 outcome: the 121-param linear twin beat the MLP; learned-vs-fixed E1 = −1.02).
The state observables flip faster than regimes, but the persistence structure that makes
state-conditioned trust WORTH anything is regime-scale: the honest effective sample sits
between **N3 (~6–8)** and ~300, and the signal those samples carry (per-organ
state-conditional utility differences) is a difference of quantities each individually
unresolvable at full-history power. 30 params at 0.1 params/window passes the ⅓ rule
NOMINALLY and fails it in substance: the rule assumes the target carries signal at the
window grain, and this one does not.

The twin control (beat static trust AND frozen-init copy at 1× cross-fold sd) converts
the gate from a risk into a pre-paid measurement — the same defense TB-006's executive
had, and it lost there on members that were 0.941-correlated; with engineered-orthogonal
members the question is more real, but the SAMPLE did not grow. Expected outcome: static
trust ships and the scorecard prints `executive_gate ≈ 0 (measured)`. MAX half-admits
this; the cost is build time, one gene (`gate_strength`) on 6 fold-samples, and two more
look-ledger entries.

**Executive ruling: MIN.** No learned meta-evaluator; the rotation gate certifies the
one meta-question (does the balancing PROCEDURE generalize) with an auditable yes/no.
If the synthesis keeps MAX's gate it must be priced as a falsifier with expected-zero
pre-registered (like R-LLM), and it MUST inherit the TB-006 S2 LOFO discipline — six
leave-one-fold-out gates, EA fitness on fold f using the gate trained without f — which
MAX does not currently specify; without it `gate_strength` selection is partially
in-sample against the same folds the gate trained on (the exact §5 hole I forced closed
in TB-006).

**EA genome sizes (12 vs 20) vs 6 fold-samples.** Both exceed the fold count; both are
"MARGINAL by design" as always; what changed since TB-006 is the countermeasure set, so
grade the countermeasures:

- Tightened ranges with neutral mid-range directly attack the boundary-pinning I
  diagnosed (champion at temp 4.0 = range edge). Adopted by both. ✓
- λ_reg shrinkage toward B0 taxes maximal-distance (= edge) genomes. ✓
- Median-of-top-8 deduped kills the argmax-of-392 selection. ✓
- **The rotation gate's null-pass rate, computed:** under a pure noise-miner, per-rotation
  unseen-fold ΔU is mean-zero ⇒ P(≥4/6 non-negative) = 22/64 = **34.4%**; AND pooled
  mean > 0 (~50%, positively correlated with the first) ⇒ joint null pass ≈ **25–30%**.
  Combined with the adoption gate (>1σ — which TB-006's own arithmetic showed a noise
  champion clears at the expected 1.4σ max-of-K uplift), the joint false-pass is maybe
  15–25%. Real protection, not certification. **Binding upgrade, costs nothing:** the
  pooled-unseen clause becomes mean > 1×se of the six ΔU_f (null pass ≈ 16% standalone,
  ≈ **8–12% joint**), and the pre-registration prints these null-pass numbers so the
  committee reads "rotation PASS" at its true evidential weight.
- Verdict: **the 12-gene MIN genome survives** under the upgraded gate. **MAX's 20
  survives only after cutting the two parity-tightener genes** — they are
  verdict-invisible by construction (both arms inside the C1 band regardless), i.e.
  pure K_eff inflation: dimensions the search can wander in with zero possible paired
  payoff, raising the max-of-K noise uplift for nothing. MIN already cut them; MAX
  carries them with no defense. `m6_timing_gain` and `channel_mix` are bounded,
  B0-anchored, and cheap — keep. MAX lands at 18.
- **New audit line (both designs, binding):** the champion manifest prints the fraction
  of genes within 5% of a range edge; >⅓ pinned ⇒ a pre-registered noise flag printed
  next to the adoption-gate result (the TB-006 lesson, instrumented).

---

## 4. Acceptance gates (MIN) vs all-ship (MAX)

Survivorship inside the verdict arm is the gate's PURPOSE (the shipped brain is the
gate-passers, with failures measured as challenger arms) — that structure is right and
MAX's all-ship alternative pays for inclusiveness with measured dilution: T_max split
six ways ⇒ per-organ marginal 0.5–3 bp/day against per-LOO MDE 2–4 — MAX's own §11.2
concedes most LOO verdicts read indeterminate. **Gates win.** But are MIN's thresholds
calibrated to anything? Test: P(pass | null) and P(pass | the organ at its receipt-size
effect):

| Gate | se of the statistic | P(pass \| null) | P(pass \| receipt-size effect) | Verdict on the number |
|---|---|---|---|---|
| M1 IC ≥ 0.04 | ~0.020 | 2.3% | 99.9% (receipt 0.105) | defensible — 2×se, receipt far above |
| M2 t ≥ +1.5 (IC ≥ 0.037) | ~0.025 | 6.7% | 68% at 0.049; **31% at half-decay 0.025** | round number; implicitly demands rotation ≈ 75% of the 5d alpha |
| M3 R² > trailing mean | — | ~high (AR repackaging passes) | ~certain | too weak — certifies nothing the lags don't give free |
| M4 AUC ≥ 0.55 + ΔAUC>0 vs vol-net | ~0.016 | ≈0.1% | 50% at true 0.55 | bar = the effect size ⇒ coin flip at the boundary |
| M5 tally > 50% | binomial n≈20 | **41%** | — | broken (§1.5) |

The pattern: thresholds set AT plausible effect sizes are coin flips for genuinely
effective organs (M2, M4), and one gate (M5) barely beats a coin for null organs. None
of the numbers traces to a power analysis. **Principled replacement (binding):**
each organ's gate becomes the same two-part form —

1. **Existence:** effect > 0 at one-sided p ≤ 0.05 with the correct overlap/cluster
   correction (M1: IC ≥ 1.64×0.020 ≈ 0.033; M2: IC ≥ 1.64×0.025 ≈ 0.041 ⇔ t ≥ 1.64;
   M4: AUC ≥ 0.5 + 1.64×0.016 ≈ 0.526 AND ΔAUC over the vol-net > 1.64×se_Δ;
   M5: binomial p ≤ 0.10; M3: the INCREMENT gate — vol-surface block coefficients
   jointly add R² over the lags-only HAR at p ≤ 0.05, killing the AR-repackaging pass).
2. **Materiality:** point estimate clears the cost-conversion floor already built in
   Phase 0 — blend-IC × σ5 > RT over the tilt set (the NetEdge ratio > 1 convention)
   for directional members; for M3/M4/M6 a named conversion sentence with the same
   shape. This replaces the round numbers with the two questions that matter (real?
   clears cost?), and BH-FDR(10%) across the five admission tests goes in the
   multiplicity ledger (it is a 5-test family, currently uncounted).
3. The pre-registration **prints the power column above** so a gate failure reads at
   its true weight (an M2 fail at 31% power against the honest decay prior is "not
   measurable yet," not "rotation is dead").

---

## 5. The surrogate + the anchor

**What 27 live dates actually bound (MIN).** Pearson ≥ 0.8 on n=27 daily Δr pairs:
Fisher se = 1/√24 ≈ 0.204, so observed 0.8 ⇒ true ρ ∈ [0.64, 0.89] at 90%. The check
does discriminate a broken surrogate (true 0.5 passes ~2%) from a decent one — it is
worth having. What it CANNOT do: (i) **magnitude calibration** — a surrogate reading 2×
the real effect passes with ρ = 1.0, and the EA's fitness scale (and the λ_reg/s_min
constants tuned in fitness units) inherit the bias; (ii) **regime coverage** — 27 dates
= one regime (Feb–Mar 2026); fold-era fidelity over 2020/2022 is then an ASSUMPTION
(geometry preserved by construction), not a measurement, and must be printed as such;
(iii) **the sign clause is my own TB-006 §7 finding repeated** — se of the 27-day mean
ΔU at sd 6–15 bp/day is 1.2–2.9 bp/day against a champion true edge of ~1–2 bp/day ⇒
P(sign agree | surrogate perfect) ≈ 0.64–0.77. A correct surrogate fails the sign clause
1-in-3; the stop-and-fix would fire on noise. **Binding:** replace sign-agreement with
an equivalence band — |mean ΔU_surrogate − mean ΔU_real| ≤ 2×se_real — plus the Pearson
bar; and add the magnitude line: regression slope of real-on-surrogate daily Δr within
[0.5, 2.0], printed.

**Is the 2025 epoch shim anchor-grade (MAX)? No.** The leg trades 25 names of which
**2/10 tilt-core names exist** (TLT, USO); the tilt overlay the surrogate would be
checked against there is a different, near-degenerate book. The era's inference
artifacts are one-hot heuristic backfills (regime collapse + ensemble overrides inert
— the chassis itself behaves differently), the prices convention needs the shim at all,
and all four repair epochs sit under it. Validating the surrogate against a window where
the system-under-test is structurally different is anchoring the wrong function — and
the reward is a sign read whose se on ~122 days at the 2-name tilt's sd is no sharper
than the live leg's. MIN's relocation is correct. The shim is ~0.5 day of build for a
context read with near-zero verdict information; build it only if the committee
separately wants the labeled 2025 context read for a named question, and never let an
anchor PASS there be cited as surrogate validation.

**The bigger pre-registration hole sits next to the surrogate (MIN P1).** MIN's primary
powered read is "champion vs B0 pooled over F1–F6" — **the exact statistic the EA
maximized, read on the exact folds it trained on.** The champion is the
median-of-top-8 of K≤400 genomes selected on those fold scores; with K_eff ≈ 30–60
independent-ish trials the expected max-of-K selection uplift is √(2 ln K_eff) ≈
2.6–2.9 fold-noise sd. P1's t ≥ +2.0 can be reached by selection alone — the rotation
gate gates SHIPPING but does not de-bias the READ. **Binding fix, zero extra compute:
P1 = the pooled unseen-fold paired series from the six rotation runs** (each fold's Δr
produced by the rotation champion that never scored it) — a true out-of-sample pooled
read covering all six folds at the same n_eff ≈ 300 and the same MDE (1.2–2 bp/day),
with the production champion's own-fold number printed beside it as context, never as
the verdict. MAX, by keeping the live-era E1 as primary with the TIES-POSITIVE cell,
does not have this hole — it pays instead with a pre-registered expected TIES
(expected t 0.6–0.9 at the Grinold ceiling). The synthesis should take MIN's powered
fold-space primary WITH the LOFO fix, and MAX's graded TIES-POSITIVE cell for the live
confirmation read.

---

## 6. Forced simplifications (binding, per my TB-006 convention)

**MIN**
- **FM1. M2 embargo 21 → ≥22 td** (h_max+1; MAX's 26 acceptable). MIN §7 carries
  "21-td embargo verbatim" — TB-006 honored S1 at 22; a D+21-consuming label purges one
  day short at 21.
- **FM2. P1 re-based to the pooled unseen-fold rotation read** (§5). Champion-on-own-
  training-folds is printed as context only. The single most load-bearing item in this
  list.
- **FM3. Acceptance gates re-based to existence (p≤0.05, corrected) + materiality
  (cost-conversion floor), §4;** M3's gate becomes the increment-over-lags test; M5's
  becomes binomial p≤0.10; the power column is printed in TOURNAMENT_007; the 5-test
  admission family enters the multiplicity ledger under BH-FDR(10%).
- **FM4. C2 fold-ceiling → replication trigger** (≥2 of 12 cells, or 1 cell at
  ρ̂ > 0.8+2×fold-se); ladder rungs ≤4 system-wide, each a look-ledger entry; M1-vs-
  member-zero loses its per-fold ceiling (printed instead), its redundancy burden moves
  to the powered IC-delta acceptance condition (§2).
- **FM5. Anchor: sign clause → equivalence band ± slope-in-[0.5,2.0]** (§5); the
  pre-registration states fold-era surrogate fidelity is assumed-by-geometry, certified
  only on the live leg.
- **FM6. M2 carried as OPEN, not "certain," in the pre-registration** (§1.1), with the
  decay-prior power band (31–68%) printed next to its gate.

**MAX**
- **FX1. Cut parity_tol_gross/risk genes** (verdict-invisible K_eff inflation); genome
  20 → 18.
- **FX2. M4-A/B bar +0.01 < its own se (~0.02): raise to 2×se or ship M4-A** and log B
  as a challenger (§1.3).
- **FX3. M6 ships at forecast altitude only:** keep the powered IC falsifier and the
  fold-pool LOO; DROP the live LOO replay arm; utility line pre-printed as
  `structurally unresolvable at this T_max`; strike the √k-headline-t claim for M6;
  add ≥4/6 fold sign-consistency to its IC gate (§1.4).
- **FX4. The 2025 shim is not anchor-grade** (2/10 core names, one-hot inference era,
  4 repair epochs): adopt MIN's live-leg anchor with FM5's form; the shim is optional
  context, never surrogate validation (§5).
- **FX5. Executive gate inherits S2-LOFO discipline** (6 leave-one-fold-out gates; EA
  fitness on fold f uses the no-f gate) and is pre-registered as a falsifier with
  expected outcome `static ships`; else cut it entirely (the ruling is cut, §3).
- **FX6. C2 at 21 pairs adopts the same FM4 replication trigger** — at 252 cells the
  single-cell trip-wire expects 2–4 noise firings.

**BOTH (program-wide)**
- **FB1. Rotation gate pooled-unseen clause: mean > 1×se of the six ΔU_f** (joint null
  pass ~8–12%, printed in the pre-registration as the gate's operating characteristic).
- **FB2. Boundary-pin audit line in the champion manifest:** fraction of genes within
  5% of a range edge; >⅓ ⇒ pre-registered noise flag printed beside the adoption gate.
- **FB3. Every effective-n claim in TOURNAMENT_007 recomputed with the §0 master
  conversion** and every gate carries its P(pass|null) / P(pass|design-effect) pair —
  the TB-006 S5 discipline extended from samples to gates.

**What the SYNTHESIS should take:**
- From **MIN:** the acceptance-gate STRUCTURE (gate-failers as measured challengers —
  directive 4 done honestly) with FM3's recalibrated bars; the tilt-only verdict arm;
  the 12-gene genome; the live-leg anchor + shim deletion; no learned meta-layer.
- From **MAX:** the 26-td M2 embargo; the TIES-POSITIVE graded cell for the live read;
  the regime-label trap analysis (labels untrainable: absent pre-2025-08, 74%/day flip
  rate — condition nothing on them, cross-tab in the report only); the per-name tier
  masks frozen from the NEW members' own OOF (tiers-not-ranks, consistent with the
  Friedman evidence); M6 as a forecast-altitude measurement organ IF build budget
  allows, under FX3.
- From **neither:** a powered primary read on the folds the champion trained on (FM2
  replaces it); an LLM column path into M4 below 2×se (FX2); any gene the verdict
  cannot see.

---

## 7. Headline verdicts

- **DESIGN_MIN: EXECUTABLE WITH SURGERY.** The sample arithmetic is honest end-to-end
  (every table number reproduces under my conventions), the organ capacities are at or
  under accepted ratios, and the no-learned-meta call is the one the arithmetic
  supports. Its two real flaws are statistical, not architectural: P1 reads the
  optimizer's own training folds (FM2), and three of five acceptance gates are round
  numbers that coin-flip real organs or admit null ones (FM3, FM5). Fixed, this is the
  design I would certify.
- **DESIGN_MAX: EXECUTABLE AT FORECAST ALTITUDE, OVERBUILT AT UTILITY ALTITUDE.** Its
  honest additions are cheap and measured (M6's fitting arithmetic is right; the
  graded verdict ladder is more honest than MIN's BEATS path); but its load-bearing
  thesis — more orthogonal organs raise expected t — fails arithmetic at this T_max
  for exactly the marginal members it adds (M6 ceiling ~0.1 bp/day book-level;
  executive expected zero at between-6-and-300 effective episodes; M4-B decided at
  half its own se; 2025 sign-anchor on a 2-name tilt book). Strip FX1–FX5 and what
  remains of MAX over MIN is: one extra powered forecast line (M6), a better embargo,
  and a better live-read verdict cell — which is the honest size of the maximalist
  increment.

— end —
