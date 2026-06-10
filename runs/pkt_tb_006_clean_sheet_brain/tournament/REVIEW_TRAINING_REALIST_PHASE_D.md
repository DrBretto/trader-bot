# PHASE D REVIEW — TRAINING REALIST (panel role 11)

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — Phase D evidence review
**Author:** Training Realist, 2026-06-10
**Inputs:** ATTRIBUTION.md, BAKEOFF.md, prototype/evidence/SCORECARD.md + e1_reads.json,
FREEZE_SYN1.md, BUILD_SPEC.md §5/§7/§16, RUN_JOURNAL.md, my Phase B audit
(ATTACK_TRAINING_REALIST.md). All arithmetic below is recomputed from the cited
artifacts, not transcribed from prose.

---

## 1. Compliance audit — were the forced simplifications honored?

Verified against FREEZE_SYN1.md, BUILD_SPEC.md §16 (registered constants), and manifests.

| Binding | Status | Evidence |
|---|---|---|
| **A1** CAST ≤22k, 45k demoted | **HONORED** — CAST-Small shipped at 16,483 params; CAST-45k never trained (kill K11, BUILD_SPEC §5.1 "NOT trained") | FREEZE members row; RUN_JOURNAL Phase C wave 2+3 |
| **A2** wd 1e-3, per-day Gaussian-rank target, dense aux head | **HONORED** — all three in BUILD_SPEC §5.1 (aux next-day return-z head, weight 0.3, dropout 0.20); optional label smoothing skipped, which my §3.1 explicitly permitted | BUILD_SPEC §5.1, §16 |
| **A3** GBM ≤150 iters / 15 leaves, binding early stop | **HONORED** — max_iter 150, max_leaf_nodes 15, early stopping on purged folds | BUILD_SPEC §5.2, §16 |
| **A4** MetaTrigger vol-only logistic rung | **MOOT (superseded by stronger remedy)** — MetaTrigger killed outright (K10), partly on my §3.5 arithmetic (label ≈ vol, duplicates RiskNet); cost-vs-edge features folded into the executive sizing head | TOURNAMENT §2.2 K10/K18 |
| **A5** fine-tune ≤~20 effective params | **HONORED, tighter than asked** — exactly 7 params (b_m ×3, T, ψ gain/bias, f_max bias), ψ hidden frozen | BUILD_SPEC §7.6 |
| **S1** embargo 22 td (h_max+1) | **HONORED** | BUILD_SPEC §10.1/§16 |
| **S2** LOFO executives for EA fitness | **HONORED, and extended** — six LOFO executives built; the EA fitness walk uses fold-f-excluded executives, and the E1 organ reads themselves walk LOFO LinearGate twins (e1_reads.json spec) — the leak-shaped hole I found in §5 is closed in both places | BUILD_SPEC §7.5/§8; e1_reads.json `base_side` |
| **S3** record-weight eps floor 0.2 | **HONORED** — gene range [0.20, 0.50], default 0.25; champion eps 0.382; A/B falsifier ran (Transfer-B: CAST weighted 4/6 wins, GBM uniform) | FREEZE genome row + §9.2 |
| **S4** cutoff = holdout_start − h − 1 by formula; no "real fills" narration | **HONORED** — cutoff 2026-03-03 by formula; §7.6 carries the "never narrated as adapting to real fills" clause verbatim | BUILD_SPEC §7.6/§16 |
| **S5** print solo-book pairwise corr, holdout MDE, recomputed effective samples | **HONORED** — corr 0.941 printed as gate 5; MDE printed with the verdict and per-organ CIs/MDEs in e1_reads.json | SCORECARD gates 4–5; ATTRIBUTION header |
| (G4 graft) dictionary freeze before validation-fold selection | **HONORED** — dedicated freeze commit before any fold-driven model selection | RUN_JOURNAL dictionary-freeze entry |

**Compliance verdict: clean.** Two deviations to note, neither a protocol violation:
(a) holdout n=44, not the ~62 my §0-N4 assumed — snapshot gap 2026-05-11→05-22 plus
window math, journaled, affecting both arms identically; (b) one reporting nit in §3
below (blanket MDE print). The build honored the sample-budget discipline in full. The
outcome that follows is therefore not an execution artifact — it is what the arithmetic,
honestly applied, produces.

## 2. The two instruments and their actual power (recomputed)

Everything in §§3–7 hangs on this. Two instruments measured each organ:

- **E1 (fold pool, n=1,507 OOF days, HAC t):** per-organ MDE at |t|=2 recomputed from
  e1_reads.json: transformer 0.40 bp/day, cast-drop 1.05, event-drop 0.33, gbm-drop 0.90,
  risknet 0.62, executive 1.03, evolution 1.80, llm 0.23, gdelt 0.58, infotropy_b 0.54.
  **E1 resolves effects down to ~0.2–1.8 bp/day.** This is the only powered instrument
  in the battery.
- **E2 (holdout, n=44 paired days):** per-arm sd back-computed from the paired t's
  (sd = mean·√44/t): R03 ≈ 17 bp/day, R10 ≈ 16, R11 ≈ 20, R08 ≈ 25, R05 ≈ 95. Per-arm
  MDE at t=2: **~5 bp/day for the well-behaved arms, ~29 bp/day for R05**. The ~5 bp/day
  figure matches my §0-N4 estimate for similar books; R05's 95 bp/day sd does not,
  because member-drop arms are not small perturbations (see §5).
- **Reporting nit:** ATTRIBUTION's header quotes a single E2 MDE of +20.71 bp/day for
  all organ reads. That number is the *bake-off pair's* MDE (sd 59.5 bp/day). The organ
  arms' own MDEs range ~5–29 bp/day. Directionally honest (all are far above any
  plausible organ effect), but the blanket print is borrowed arithmetic; a per-arm MDE
  column would have been correct.

Consequence: every plausible organ-level true effect (0.1–1 bp/day) sits **5×–100× below
the E2 MDE**. E2 was never going to adjudicate anything at organ scale, which is exactly
why the pre-registered hierarchy made it sign-confirmation only — and §7 shows even the
sign read is weaker than that.

## 3. The central paradox, resolved: IC 0.105 with zero measured attribution

CAST's purged-validation weekly rank-IC is **0.105** (per-fold 0.061–0.113), 5× the 0.02
functionality bar and ~17× the 0.006 cost break-even (gate 4 PASS). The ridge twin sits
at noise (−0.009). Yet R03 (CAST vs ridge-in-slot, executive re-fit) reads E1 t=−0.28 ≈ 0:
**swapping in a forecaster with ~11 IC points more signal produces statistically zero
extra utility.** Both numbers are correct. The reconciliation is the expression channel,
not the signal:

- The champion genome is maximally conservative: gross 0.516 × vol_target 0.086,
  no_trade_band 0.026, abstain 0.244, conviction_temp 4.0. SYN-1 executed **4 trades in
  65 days** and **0 round trips in the entire holdout** at ~26% average gross.
- The diversity floor TRIGGERED at 0.941 mean pairwise solo-book correlation: the four
  members produce one book in four costumes. Trust reallocation among near-clones moves
  nothing (my §3.4/S5 prediction, verified to the letter — see §6).
- A brain that almost never trades cannot convert forecast-rank differences into utility
  differences. The IC→utility conversion is throttled upstream of every organ.

So the transformer's "0 (measured)" is a **true zero of the marginal-utility channel in
this configuration**, NOT a finding that CAST learned nothing. Item 1's transformer
demonstrably does real forecasting work (the IC gate is the receipt); the shipped system
then declines to spend that work. Attribution measured the spend, as pre-registered.

## 4. Classifying the zeros (the honest taxonomy)

Three distinct kinds of "≈0" appear in the scorecard; they license different conclusions.

**(a) True zeros at E1 power — measured, believe them.** transformer (CI ±0.4 bp/day),
ensemble_cast_drop (±1.0), ensemble_event_drop (±0.3), infotropy_b (±0.5), infotropy_a
(±0.2). With n=1,507 and MDEs under ~1 bp/day, these CIs genuinely exclude any effect
that would matter. Within SYN-1's architecture these pathways' marginal utility is zero.
The qualifier "within this architecture" is load-bearing per §3 — the zeros are verdicts
on the plumbing as configured, not on the organs' information content.

**(b) Structural zeros by construction — predicted before measurement.** executive trust
attribution (diversity floor 0.941 ≥ 0.90 → equal-trust tie expected "by construction,"
pre-registered) and risknet's original R07 arm (the planned swap WAS the deployed
convention; null contrast by construction, honestly logged, repaired via R19). These
zeros carry no information about whether a learned executive or a vol organ *could* help;
they certify the question was unanswerable on this membership/wiring.

**(c) Underpowered reads dressed as zeros — do not over-read.** evolution (E1 CI
[−2.6, +0.9] bp/day, MDE 1.8): the "0 (measured)" print is mechanically correct at
|t|<1, but the CI admits a ~2 bp/day harm or a ~1 bp/day gain. Meanwhile the optimizer
test is separately positive (champion fitness 1.0919 > B1 budget-matched random-search
best 0.8383) and the adoption gate passed (+0.876 > 1.0× cross-fold sd 0.638). Honest
summary: the EA is a real optimizer whose chosen disposition ("barely trade") cannot be
shown to add or subtract utility at the resolution available. Note the EA optimizing a
drawdown-penalized, min-over-cost-scenario fitness on **6 fold-scores** (my N3: 6–8
regime observations) choosing extreme conservatism is not a malfunction — it is the
predictable fixed point of honestly small sample budgets. Negative-leaning indeterminates
(llm E1 −1.08, gdelt −1.79) also belong here: bounded at "at best nothing, plausibly a
small drag" (gdelt CI [−1.08, +0.05] bp/day), with gdelt's per-fold reads violently
heterogeneous (F1 −2.32, F4 +1.82, F6 −3.29) — regime-dependent noise around a small
negative lean, below the −2 certification bar, and subject to the §4.5 multiplicity
caveat in both directions.

## 5. The GBM disagreement (E1 +2.48 vs E2 −1.45): arithmetic, not mystery

E1: dropping GBM-Cond costs +1.11 bp/day pooled, **sign-positive in all five folds where
the read exists** (t = +1.45, +0.53, n/a, +1.61, +1.35, +1.89), HAC t=+2.48 — the only
organ to clear the pre-registered +2.0 primary bar. E2: holdout paired mean −20.78
bp/day, t=−1.45, back-computed sd ≈ 95 bp/day.

Two computations settle what this means:

1. **The sign flip is a coin flip.** If the E1 effect (+1.11 bp/day) is exactly true, the
   44-day E2 mean has standard error 95/√44 ≈ 14.3 bp/day; P(observed mean < 0) =
   Φ(−1.11/14.3) ≈ **47%**. The E2 "disagreement" was a near-even-money event under the
   hypothesis that E1 is right. It is not evidence against E1; it is evidence that E2
   cannot see effects of this size.
2. **Why R05's sd is 95 bp/day when R03's is 17:** member-drop arms are not infinitesimal
   perturbations. With trust renormalized over near-clone members, removing GBM shifts
   the blend enough to move forecasts across the no_trade_band/abstain thresholds —
   threshold nonlinearity converts a small forecast change into a structurally different
   trade sequence. The −20.78 bp/day E2 mean (≈ −9% over the window, on a book that
   itself returned +1.3%) is the dropped-GBM variant *luckily catching more of an
   up-tape*, i.e., path variance, not organ signal.

Honest reading: **GBM-Cond is the one organ with suggestive evidence of carrying
weight.** Suggestive, not certified: one of 7 pre-registered arms (family-wise expected
false positives at the E1 bar ≈ 0.35; a lone +2.48 across 7 arms is ~p≈0.09 family-wise),
and the holdout neither confirms nor denies. The mechanical "indeterminate" print is
correct; the committee should read it as "E1-positive, confirmation pending," not as a
contradiction.

## 6. Phase B predictions — scoreboard (both directions)

Claiming the hits requires owning the misses.

- **VERIFIED (exact):** §3.4/S5 — "if solo-book correlation exceeds ~0.9, trust learning
  is arithmetic cosmetics and the expected result is the equal-trust tie." Measured:
  0.941; equal-trust tie; all three §7 kill criteria fired (std τ ≤ 0.0029 every fold,
  calibration corr −0.281, MLP lost the ladder to the 121-param linear twin 1.688e-4 vs
  1.701e-4).
- **VERIFIED (exact):** §4 — "expect the fine-tune delta to be ≈0." Delta exactly 0 on
  all 5 seeds, de-claimed per the pre-registered rule.
- **VERIFIED:** §3.2 — "the utility target cannot support rich structure learning; expect
  ties with equal-trust; the executive supports tiny gates only." Linear twin shipped;
  learned-vs-fixed E1 = −1.02 (zero-to-slightly-negative).
- **HALF-VERIFIED:** §2.1 — "at 15–22k CAST has a genuine fighting chance of positive
  attribution vs ridge." CAST beat ridge decisively at the *signal* level (IC 0.105 vs
  −0.009) and showed zero at the *utility* level. I predicted the fight at the wrong
  altitude: the forecaster won; the genome declined to cash the winnings. Phase B never
  priced the interaction "honest sample budgets → ultra-conservative champion genome →
  attribution throttled for every organ at once." That interaction is this packet's
  single most important empirical finding.
- **WRONG (factor of 4):** §0-N4 estimated bake-off holdout MDE ~5 bp/day assuming
  paired sd 15–25 bp/day for "two similar long-only books." The bake-off books are not
  similar (gross 26% vs 36%, 4 vs 122 executed actions); measured sd 59.5 bp/day, MDE
  20.7 bp/day. My figure was right for the organ arms (R03/R10/R11 sds 16–20 bp/day),
  wrong for the headline pair. Also n=44, not ~62 (snapshot gap, not foreseeable).

## 7. Design lesson: the E2 sign gate is a coin flip at organ scale

For a true 1 bp/day effect read through a 17 bp/day-sd arm, P(holdout sign agrees) =
Φ(1/(17/√44)) ≈ 0.65; at 0.5 bp/day it is ≈ 0.58. The sign-confirmation step therefore
converts genuine E1 positives into "indeterminate" 35–45% of the time while a null organ
passes it 50% of the time — it adds almost no discrimination, only noise-driven verdict
downgrades (GBM is the live example). The pre-registration was right to make E1 primary;
a follow-on should go one step further and replace E2 sign-confirmation for sub-MDE
effects with an explicit equivalence band ("holdout consistent with E1 estimate ± MDE"),
reserving sign language for effects the window can actually see.

## 8. The bake-off TIES, in sample-budget terms

TIES (t=−0.92) with endpoint −3.68% on a 44-day up-tape window, against an incumbent
running ~11 pp more gross and 42 round trips to SYN-1's 0. At matched cost machinery,
most of the endpoint gap is exposure on a rising tape — beta, not demonstrated alpha —
and the window cannot distinguish "incumbent has edge" from "incumbent had more beta in
an up-window" (MDE 20.7 bp/day vs observed mean −8.3). The committee should resist both
available over-readings: the clean-sheet brain did not "lose" (sub-MDE, sub-|t|=1), and
it did not "honorably tie" either — it abstained its way to a tie, which is what the
arithmetic of §4(c) predicts a 6-sample disposition optimizer will do.

## 9. What a follow-on actually needs (with the arithmetic)

1. **Stop trying to certify organ attribution on forward windows.** To certify a 1 bp/day
   organ effect at t=2 through a 17 bp/day-sd arm needs n ≈ (2·17/1)² ≈ **1,150 trading
   days (~4.6 years)**. No feasible forward window does this. The fold-pool E1 read is
   the only powered organ instrument; extend it (new folds F7, F8 as time accrues) and
   give the forward window the job it can do: whole-brain disgrace-detection at the
   ~5–20 bp/day scale.
2. **GBM confirmation arm, pre-registered now:** re-run the R05 contrast on the next
   accrued fold(s) before believing +2.48. One number decides it.
3. **Structurally decorrelated members before re-asking item 5.** With corr 0.941 the
   executive question was unanswerable by construction. A follow-on must enforce a
   member-acceptance gate (e.g., solo-book correlation vs existing members ≤ 0.7 on
   validation folds) and buy decorrelation structurally: different targets (vol-timing,
   event-window-only, cross-asset), different horizons (h=21 sleeve), not different
   architectures on the same target — same-target architectures provably converge to the
   same book (this packet's R03/R04 are the receipts).
4. **Attribute in forecast space alongside utility space.** The IC-vs-utility decoupling
   (§3) means utility-only attribution under a conservative genome measures the genome,
   not the organs. Pre-register a forecast-level attribution read (per-organ IC delta,
   fold-pooled) next to E1 so the two altitudes are never conflated again.
5. **If utility expression is wanted, the genome's throttle must be a measured choice:**
   an attribution-mode replay with no_trade_band/abstain relaxed (pre-registered,
   measurement-only, never deployed) would let organ differences express — at the cost of
   measuring a system nobody ships. Cheaper and cleaner: item 4.

---

## 10. Statement for the committee report

The build honored every binding simplification from my Phase B audit (A1–A5, S1–S5, plus
the G4 dictionary freeze), so this scorecard is the honest output of the arithmetic, not
an execution failure — and what it licenses is precise. It licenses: that the transformer
and perception stack learned real, cost-clearing signal (purged weekly rank-IC 0.105
against a 0.006 break-even, ridge twin at noise) while contributing zero *measured
utility* through a champion genome that traded 4 times in 65 days — a true zero of this
configuration's expression channel, certified to ±0.4–1 bp/day by the only powered
instrument (E1, n=1,507), and not a zero of the organs' information content; that
GBM-Cond alone shows suggestive positive marginal utility (E1 +2.48, sign-consistent in
all five defined folds, ~p≈0.09 family-wise) whose holdout "sign flip" was a 47%
coin-flip event under the E1 estimate and refutes nothing; that the learned executive
measurably adds nothing over equal trust *and could not have* on members whose solo books
correlate 0.941 — the question was structurally unanswerable, exactly as pre-registered;
that LLM and GDELT are architecturally load-bearing (the EventHead dies without GDELT)
but their measured utility is bounded at zero-to-slightly-negative, so continued spend on
them is a structural bet, not a measured edge; that evolution is a genuine optimizer
(beats budget-matched random search) whose chosen disposition cannot be utility-graded at
available resolution; and that infotropy showed no transfer, twice, cleanly. It does NOT
license "the organs don't work," "the incumbent is better" (44 up-tape days at MDE ~21
bp/day against an 11-pp gross gap is a beta read), or any claim that a longer holdout
would fix attribution — certifying 1 bp/day organ effects forward needs ~4.6 years. The
follow-on that would actually move the needle is structural, not temporal: enforce
member decorrelation at acceptance (solo-book corr ≤ 0.7, bought with different targets
and horizons, not different architectures on the same target), pre-register a
forecast-space attribution read beside E1 so genome throttling can never again masquerade
as organ failure, extend the fold pool as time accrues with the GBM confirmation arm
registered first, and retire holdout sign-confirmation for sub-MDE effects in favor of an
equivalence band.

— end —
