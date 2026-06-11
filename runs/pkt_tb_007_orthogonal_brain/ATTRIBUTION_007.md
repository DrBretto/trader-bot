# ATTRIBUTION_007 — per-organ attribution & receipts (PKT-TB-007, Phase D)

Companion to BAKEOFF_007.md. Verdicts are mechanical (§4.5 three-valued rule:
`positive` t ≥ +2; `zero (measured)` CI ⊂ ±MDE; else `indeterminate at available
power`). The registered roster is [M1] with B0 shipped, so **every live utility number
on this page is from the deviation battery and carries its label**:
**(deviation: a-priori genome, instrument-failed EA)**. Machine record:
`prototype/evidence_007/scorecard_007.json`; per-comparison manifests under
`prototype/evidence_007/`.

---

## M1 — CAST-XS transformer (fast 5d relative strength) · SHIPS, sole roster member

**Acceptance receipt (registered, near-formality — labeled, does not count as
evidence):** pooled OOF weekly rank-IC **0.1076** (se .0195, p 1.6e-8), forecast-IC delta
vs member zero **+0.042**, NetEdge ratio on the masked core **2.60** (bar 1.0).
BH-FDR(10%) family ledgered.

**Live utility read — THE first real read of the M1 tilt** *(deviation: a-priori
genome, instrument-failed EA)*: D03-vs-D02 (≡ D03-vs-D01; D02 hash-equal). Full: n=66,
**−1.98 bp/day**, t −1.26, HAC t −0.96, sd 12.81 bp/day, MDE 4.13 bp/day. Holdout: n=44,
**−3.13 bp/day**, t −1.34, HAC t −1.09, MDE 5.75 bp/day. ΔSharpe −0.15 full / −0.42
holdout. 90% CI straddles zero in both periods.

**Verdict: indeterminate at available power** — with negative point estimates printed.
The honest sentence: a strong forecast receipt (IC .108) converted, through the one
expression channel at a-priori scale, into an uncertifiable ~−2 bp/day on this window.
That is exactly the Grinold-ceiling arithmetic the pre-registration printed (expected
live t 0.3–0.7 at BEST under shrinkage; the realized sign happened to be negative), plus
the known bandwidth ceiling: 14/67 days had zero cash-neutral funding capacity, 11 more
quantized away below min_order.

## M2 — ROT-GBM slow rotation (D+5→D+21) · challenger, status OPEN (FM6)

**Acceptance receipt:** rotation-target rank-IC .037, t 1.45 vs bar 1.64 (p .074);
materiality NetEdge .55 < 1 on XRT. Power pair: 62% at no-decay IC, **29% at half-decay
⇒ a fail reads "not measurable yet," never "rotation is dead"** (pre-committed §4.8.2).

**Marginal read** *(deviation: a-priori genome, instrument-failed EA)*: D04 (trust
{M1:+1, M2:+1}) vs D03: full −0.05 bp/day, HAC t −0.14, MDE 0.77 bp/day; holdout −0.12
bp/day, HAC t −0.22. **Verdict: indeterminate at available power.** Adding M2 at equal
trust moved the book by less than a tenth of the arm's own MDE; M2 stays OPEN, nothing
downstream breaks (M1 alone fed the tilt as registered).

## M3 — DISP-HAR dispersion timing · B-disp constant (no gene)

**Gate receipt:** existence PASS (vol-surface increment F-test p ≈ 2e-11) — the
dispersion structure is real; materiality FAIL (**0.011 bp/day** < bar 0.2 bp/day) — at
this T_max it cannot pay. Disposition per the registered rule: dispersion enters
conviction as the **B-disp sigmoid constant** (disp_t = 0.5); the LOO control IS the
constant arm. Utility line as pre-printed: `indeterminate` — no separate replay arm
exists or is owed. **Verdict: 0 (gate-honest).**

## M4 — EVT-NET event decoupling · challenger (M4-A variant)

**Acceptance receipts:** pooled OOF AUC **0.616** (bar .526, p ≈ 0) BUT ΔAUC vs the
vol-only control **−0.002** (bar +0.002) — GDELT events priced as a noisy vol proxy
(echoes TB-006). **A/B one-look** (G8, ledgered): ΔAUC(B−A) on F5–F6 = **−.020** vs bar
+.011 ⇒ **M4-A ships as challenger; the LLM-columns zero is printed** (pre-committed
§4.8.5).

**Marginal read** *(deviation: a-priori genome, instrument-failed EA)*: D05 (event_damp
0.5 mid-range, trust M4:+1) vs D03: full **+0.08 bp/day**, t +2.91, **HAC t +1.84**,
MDE 0.09 bp/day; holdout +0.13 bp/day, HAC t +2.18. **Verdict: indeterminate at
available power** (E1 HAC t < +2.0), and two rails print with it:

1. **Multiplicity (§4.6.5):** ≈25–30% pre-registered chance of one spurious
   certifiable-looking line across the battery — a single positive line is not
   narratable as a discovery.
2. **Width-of-a-losing-tilt caution:** M4's pathway only shrinks tilt width on high
   p_exceed names. The base tilt's point estimate is negative on this window, so ANY
   width reduction tends to score positive mechanically. This read cannot separate
   "M4 sees decoupling risk" from "M4 dampened a losing tilt." A future window where the
   base tilt's point estimate is positive is the discriminating test.

## M5 — POS-Z positioning rule · DISABLED

**Gate receipt:** episode sign tally **12/19**, binomial one-sided p = **.18** (bar .05;
TR's D3 band (.05,.10] not reached either — no "would have enabled under the TR bar"
line). Tally printed in `gates/acceptance_007.json`.

**Marginal read** *(deviation: a-priori genome, instrument-failed EA)*: D06 (rule ON)
is **byte-identical to D03** — M5 fired **zero** episodes in the live window
(`store/m5_signal.npz`: no nonzero day ≥ 2026-01-31). **Verdict: zero (measured),
degenerate-structural** — the rule's live utility is unmeasured because the phenomenon
did not occur, which is the honest answer for a ~15–25-episode-per-decade signal on a
4-month window.

## M6 — GAP-GRU · forecast altitude only (never a roster member)

**The registered sentence:** pooled OOF 1d rank-IC **0.0224** (bar 0.02, se .0066,
clustering-haircut se .0092), **sign-positive 6/6 folds** — forecast-altitude **PASS**;
*structurally unresolvable at this T_max — forecast-altitude verdict only.* No genome
gene, no tilt path, no replay arm consumed. The GRU model class is present on purpose,
measured with the most powered falsifier in the program, and honestly unexpressed (a
next-open fill model gives a 1-day re-timing organ no representable action).

## LLM — retired from the decision path, with receipts

46 pre-registered screens, **0 BH-FDR(10%) survivors**; monitoring emission stays
($0.14/mo, ~22 scored days/mo toward the 900-day re-test trigger); M4-B was its one
data-driven re-entry door (closed by the A/B look above).

**R-LLM falsifier arm (§4.4.9)** *(deviation: a-priori genome, instrument-failed EA)*:
disagreement-z → tilt-width conditioner (g ∈ [0.78, 1.24] realized on 42 active days) vs
the unconditioned a-priori arm: full mean **+0.001 bp/day**, t **+0.01**; **MDE 0.21
bp/day** printed for this ~66-day arm; pre-registered expectation O(0.1–0.5) bp/day —
the expected zero, realized. **KILLED at the pre-registered bar (E1 t < +2.0).**
Scorecard line (pre-committed §4.8.6): *LLM organ: 0 (measured) — retired by
role-finding (46 screens, 0 BH-FDR survivors); falsifier arm killed at its
pre-registered bar; $0.14/mo monitoring + 900-day re-test trigger live.*

## Evolution — instrument-failed (printed verbatim everywhere)

ANCHOR DISPOSITION: **B0 ships outright (Pearson 0.2818 < 0.5, chair pre-stated rule)**.
The surrogate's carried tilt never feeds back into chassis state; fidelity DEGRADED with
scale (.50 → .28), eliminating the cost-rng noise-floor theory. No certified fitness
instrument ⇒ the EA production sequence (GA, rotation gate, B1, adoption gate,
boundary-pin audit) **did not run**; ea_cycles 0/3; every surrogate number remains
"(surrogate space), diagnostics only." Scorecard: **evolution≈0, honestly**
(pre-committed §4.8.1). The deviation battery exists precisely because this failure left
the registered pair degenerate. [Revised per Skeptic Phase-D review repair 2: the original
sentence here claimed the tilt "lost" in this window — a directional claim an uncertifiable
read does not license, used to grade the chair's own rule; struck. Per the review's F8
decomposition, ~86% of the deviation read's deficit is a one-signed exposure leak and the
exposure-stripped selection residual is −0.29 bp/day (t ≈ −0.22), i.e. zero. The
instrument-failure rule stands on the anchor arithmetic alone (Pearson 0.282 < 0.5).]

## C2 — orthogonality gate

PASS on the binding signal-space leg: **max pooled |ρ| = 0.202 ≤ 0.70** (Z0–M1 row
0.196; all directional pairs ≤ .202; TB-006 baseline was .941). The registered G-book
instrument saturates on the market factor — ruled mechanically inapplicable (chair
adjudication 1); pre-registration defect reported; ladder NOT fired; market-residualized
book diagnostic 0.498 ≤ 0.70 printed. Phase D adds the adjudication's own vindication
check: the challenger tilts produced sub-bp marginal books, not near-identical competing
books — no evidence the call was wrong.

## Expression channel — observed mechanics (context, not verdict)

41/67 days non-neutral at a-priori scale; neutral days: 14 no_tilt_capacity (the
ledgered cash-neutral bandwidth ceiling on force-sell/all-cash days), 11
dead_zone_quantized (sub-min_order tilt), 1 no_organ_file. 246 executed actions vs the
incumbent's 121 on near-identical total traded dollars ($1.0654M vs $1.0663M) — the tilt
reshapes, it does not gross up. Projection exactness held every active day (max
|cash_resid|, |β_resid| ≈ 1e-12); realized path-level exposure parity is the disclosed
item in BAKEOFF §3.1.
