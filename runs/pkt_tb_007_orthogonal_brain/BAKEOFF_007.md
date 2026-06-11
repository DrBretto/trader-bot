# BAKEOFF_007 — Phase D verdict (PKT-TB-007-ORTHOGONAL-BRAIN)

**Executor:** Phase D battery (role: mechanical assembly). 2026-06-11.
**Pre-registration:** TOURNAMENT_007 §4 (frozen 2026-06-10). **Frozen config:** prototype/FREEZE_ORB1.md
(B0 ships, chair adjudications 1–3). **Battery wall-clock: 37.2 s** (9 replay arms, one OS process each)
+ ~5 s E2 subset reads + evidence assembly. Machine record: `prototype/evidence_007/scorecard_007.json`.

---

## 1. THE REGISTERED VERDICT (headline)

**`TIES (degenerate: brain ships neutral; evolution = instrument-failed)`**

The registered verdict pair is ORB-1 at the shipped genome **B0** (hash `3df8053a6c42`,
FREEZE_ORB1 §1 — shipped by the chair's pre-stated anchor-remediation rule: anchor Pearson
0.282 < 0.5 at representative scale ⇒ B0 outright; the EA production sequence never ran
because no fitness instrument survived certification). B0 is structurally neutral
(tilt_gain = 0), so the registered pair's content is the **hash-equality demonstration**,
asserted mechanically at bake-off (`prototype/runs_battery_007/battery_checks_007.json`):

| artifact | D01 incumbent | D02 ORB-1@B0 | equal |
|---|---|---|---|
| sha256(timeline.json) | `4f06fcb1974f…` | `4f06fcb1974f…` | **YES** |
| sha256(daily_series.csv) (raw + cost-adjusted) | `71dc554a2693…` | `71dc554a2693…` | **YES** |
| sha256(cost_overlay.json) | `ce36d76e7635…` | `ce36d76e7635…` | **YES** |

E1 paired deltas on n=66 identical dates (live window 2026-02-02→2026-06-10, seed 4242):
**identically zero** — t not computable, sd = 0. E2 (the ONE registered holdout read,
orchestrator-fired, ledgered in `prototype/ledgers/looks_holdout.json`): n=44 holdout deltas,
**identically zero** (subset of sha-equal series; `runs_battery_007/E2_D01_INC`, `E2_D02_ORB_B0`).

Registered claim-table mapping (§4.2): the pair lands in the TIES region with the
degenerate qualifier. The registered TIES (straddling) permitted claim, quoted verbatim:
*"no certifiable difference at available power; MDE ⟨printed⟩"* — here MDE is n/a: the
paired deltas are identically zero because **the brain ships neutral**, not because noise
swamped an effect. Per pre-committed sentence §4.8.1 the scorecard reads **`evolution≈0`,
honestly** — specifically: **evolution = instrument-failed** (anchor 0.282 < 0.5; B0
shipped; ea_cycles 0/3; fold-era and live-era surrogate remain "(surrogate space),
diagnostics only" for the life of this packet).

Pre-registered expectation vs realized: §4.8.8 expected the bounded-TIES cell with
positive point estimates at t₁ ≈ 0.3–0.7. Realized: the registered pair is degenerate
(upstream instrument failure pre-empted any expressed tilt), and the labeled deviation
read (§3) carries the only performance information in the battery. That is a passing
outcome under the packet; a bigger promise is not owned by this record.

## 2. ORTHOGONALITY (C2)

**PASS on the binding signal-space leg: max pooled |ρ| = 0.202 ≤ 0.70** (TB-006 was
0.941 — the orthogonality engineering worked). The registered G-book instrument (TB-006 §6
long-only unit-gross solo books) saturates on the market factor (all directional pairs
0.88–0.97 including ρ≈0 signal pairs) and was ruled **mechanically inapplicable** — a
pre-registration defect, escalated and adjudicated (chair adjudication 1, commit 0b52981);
the remediation ladder was NOT fired; market-residualized book diagnostic 0.498 ≤ 0.70
printed. Vindication check from the adjudication, now answerable: Phase D's M1-tilt and
challenger tilts did NOT produce near-identical books being separated by the gate's absence
(the challenger margins vs D03 are sub-bp), so no evidence emerged that the call was wrong.

## 3. DEVIATION BATTERY — every number below carries the label **(deviation: a-priori genome, instrument-failed EA)** and none of it touches §1

Chair adjudication 3 (journaled BEFORE any performance read; value-blind attestation in
RUN_JOURNAL): the a-priori genome = tilt_gain 0.5, organ_trust {M1:+1}, disp_gain 1.0,
dead_zone 0.05, caps at B0 mids (hash `952a2f5a565e`). Since roster = [M1], the deviation
pair IS the M1 utility-attribution arm. D02 ≡ D01 bit-for-bit, so D03-vs-D01 below is
identically the D03-vs-D02 read.

### 3.1 D03 ORB-1@apriori vs incumbent — the first real read of the M1 tilt *(deviation: a-priori genome, instrument-failed EA)*

| read | n | mean Δ (bp/day) | paired t | HAC t | sd (bp/day) | MDE@\|t\|=2 (bp/day) | ΔSharpe | ΔReturn |
|---|---|---|---|---|---|---|---|---|
| full period (2026-02-03→06-10) | 66 | **−1.98** | **−1.26** | −0.96 | 12.81 | 4.13 | −0.148 | −1.40% |
| holdout only (≥ 2026-03-11) | 44 | **−3.13** | **−1.34** | −1.09 | 15.55 | 5.75 | −0.424 | −1.41% |

90% CI of mean Δ (HAC): full [−5.38, +1.41] bp/day; holdout [−7.86, +1.60] bp/day.

**Deviation verdict cell (mechanical, §4.2): `TIES (straddling)`** — *"no certifiable
difference at available power; MDE printed"* (4.1 bp/day full, 5.8 bp/day holdout). The
point estimates are **negative** at roughly −1.3t; the holdout 90% CI still straddles
zero, so the LOSES cell does not fire. Nothing here is evidence the tilt works, and the
registered verdict above is unaffected.

### 3.1a Exposure decomposition of the D03 deficit (Skeptic Phase-D review F8 — repair 1)

Regressing the daily paired deltas on the incumbent's daily return: slope −0.118 in a
window whose mean incumbent day is +14.4 bp — the tilted arm ran systematically lower
realized gross (34.68% vs 39.07% average; one-signed, not symmetric noise). That
exposure term accounts for **≈86% of the −1.98 bp/day** point estimate. The
**exposure-stripped selection residual is −0.29 bp/day, t ≈ −0.22 — indistinguishable
from zero.** Honest reading: the deviation read is substantially an exposure read (the
path-divergence channel above, one-signed in an up-window); the M1 tilt's *selection*
content measured zero, not negative. This cuts both ways and certifies nothing; it is
printed so the deficit is not misread as selection failure. It also makes the M4-A
damping arm's positive read (+1.84 HAC) mechanically suspect — damping a
gross-leaking tilt restores exposure.

Expression-channel facts (D03 logs): 41/67 days non-neutral; neutral reasons:
no_tilt_capacity 14 (the ledgered cash-neutral bandwidth ceiling — zero-funding-capacity
days, carried from Phase 0 as promised), dead_zone_quantized 11, no_organ_file 1.
246 executed actions vs incumbent 121; total traded ≈ equal ($1.0654M vs $1.0663M).

**Exposure-parity disclosure (§4.1 — the trigger FIRED):** realized daily gross gap mean
4.39 pp, max 16.28 pp; rolling 21d β gap mean |Δβ| 0.069, max 0.217; **the pre-registered
disclosure trigger (|Δβ| > 0.05 sustained 5 days) fired.** The projection enforces
Σdw = 0 and β-neutrality ex-ante per day against trailing-126d β̂ (per-day beta_resid
≈ 1e-12 in the expression logs); the realized gaps are path divergence (compounding book
state + chassis re-decisions + quantized order conversion), not a projection failure.
This qualifies the "matched exposure" frame of the deviation read; the registered pair is
unaffected (identical books).

### 3.2 Challenger-in arms (marginal vs D03) *(deviation: a-priori genome, instrument-failed EA)*

| arm | trust / config | full: mean bp/day, t (HAC) | holdout: mean bp/day, t (HAC) | MDE full (bp/day) | three-valued organ verdict (§4.5) |
|---|---|---|---|---|---|
| D04 **M2-in** | {M1:+1, M2:+1} | −0.05, −0.12 (−0.14) | −0.12, −0.19 (−0.22) | 0.77 | **indeterminate at available power** |
| D05 **M4A-in** | + M4:+1, event_damp 0.5 | +0.08, +2.91 (+1.84) | +0.13, +3.00 (+2.18) | 0.09 | **indeterminate at available power** (HAC t < +2) |
| D06 **M5-in** | {M1:+1, M5:+1} | 0 identically (timeline sha = D03) | 0 identically | — | **zero (measured), degenerate-structural** |

- **M2** (status OPEN per FM6): adding M2 moved nothing measurable. Read with its
  acceptance receipt (rot-IC .037, t 1.45, 29% power at half-decay): *"rotation not
  measurable yet at this horizon"* — never "rotation is dead" (pre-committed §4.8.2).
- **M4-A**: the one positive-pointing line in the battery (+0.08 bp/day, naive t +2.9,
  HAC t +1.84 full / +2.18 holdout). Two rails printed with it: (i) **multiplicity** —
  across the battery's arms, P(≥1 spurious certifiable-looking line) ≈ 25–30%
  pre-registered (§4.6.5); a single positive line is NOT narratable as a discovery;
  (ii) **width-of-a-losing-tilt caution** — M4's damp only shrinks tilt width, and the
  underlying a-priori tilt has a negative point estimate on this window, so a width cut
  scores positive mechanically; this read cannot separate M4 skill from
  less-of-a-losing-tilt.
- **M5**: fired ZERO episodes in the live window (store/m5_signal.npz) — the arm is
  byte-identical to D03. M5 remains DISABLED (12/19 tally, p = .18, tally printed);
  its live marginal utility is structurally unmeasured on this window.

### 3.3 D07 R-LLM width falsifier (§4.4.9) *(deviation: a-priori genome, instrument-failed EA)*

llm_disag (unweighted 27-bucket sent std, 252d z, visible_from ≤ D) scaling T_t by
clip(1+0.25·tanh z, 0.75, 1.25); realized g ∈ [0.78, 1.24] on 42 active days.
Vs the unconditioned D03: full n=66, mean **+0.001 bp/day**, t **+0.01** (HAC +0.01);
holdout +0.05 bp/day, t +0.17. **Per-arm MDE printed: 0.21 bp/day** (full, |t|=2) on this
~66-day arm; the pre-registered expected effect was O(0.1–0.5) bp/day — the expected zero.
**KILLED at the pre-registered bar (E1 t < +2.0); never touches holdout promotion.**
LLM scorecard line (pre-committed §4.8.6): *LLM organ: 0 (measured) — retired by
role-finding (46 screens, 0 BH-FDR survivors); falsifier arm killed at its pre-registered
bar; $0.14/mo monitoring + 900-day re-test trigger live.*

### 3.4 D08/D09 cost-seed sensitivity *(deviation: a-priori genome, instrument-failed EA)*

Raw series + timeline asserted **byte-identical** to D03 (seeds move only the cost
overlay). Cost-adjusted spread across seeds {4242, 4243, 4244}: total cost $348.46 /
$334.30 / $335.68 on identical $1.0654M traded (3.27 / 3.14 / 3.15 bps of traded);
paired full-period means vs D03: +0.019 / +0.018 bp/day (|t| < 1). The D03 read is not
a cost-seed artifact.

## 4. Organ scorecard (three-valued, §4.5; receipts in ATTRIBUTION_007.md)

| organ | roster state (registered) | live utility read | tag |
|---|---|---|---|
| M1 | **SHIPS** (sole member; acceptance IC .1076, +.042 vs member zero, NetEdge 2.6 — near-formality, labeled, not evidence) | D03-vs-D02 *(deviation)*: −1.98 bp/day, HAC t −0.96, MDE 4.13 | **indeterminate at available power** |
| M2 | challenger, **OPEN** (FM6) | D04-vs-D03 *(deviation)*: −0.05 bp/day, HAC t −0.14 | **indeterminate at available power** |
| M3 | **B-disp constant** (existence PASS p=2e-11; materiality FAIL .011 bp/day < .2) | no gene, no arm; gain rode as constant 0.5 sigmoid input | **0 (gate-honest)** — utility line `indeterminate` as pre-printed |
| M4 | challenger (M4-A; A/B one-look: ΔAUC −.020 < +.011 bar) | D05-vs-D03 *(deviation)*: +0.08 bp/day, HAC t +1.84, cautions printed | **indeterminate at available power** |
| M5 | **DISABLED** (12/19, p=.18, tally printed) | D06 degenerate: 0 live episodes | **zero (measured), structural** |
| M6 | never a roster member | forecast altitude: pooled OOF 1d rank-IC **.0224** (bar .02), sign-positive **6/6** folds — PASS; *structurally unresolvable at this T_max — forecast-altitude verdict only* | **forecast-PASS** |
| LLM | retired with receipts | falsifier killed (t +0.01 < +2.0, MDE 0.21 bp/day) | **0 (measured)** |
| EA | — | **instrument-failed** (anchor .282 < .5 ⇒ B0 ships; production EA not run; 0/3 cycles) | **evolution≈0, honestly** |

## 5. Ledgers, caps, multiplicity, boundary

- **Holdout looks:** registered budget 1/1 consumed (the E2 verdict-pair read,
  `ledgers/looks_holdout.json`). Deviation-battery holdout exposure: **9 replay-look
  entries** in `prototype/holdout_looks_007.jsonl`, every entry written BEFORE its run,
  every one labeled (registered vs deviation).
- **Replay arms: 10 / 18 cap** (`ledgers/replay_arms.json`: anchor re-run + D01–D09).
  **Retrains: 0 / 9. EA cycles: 0 / 3.** Remediation rungs: 1 / 4 (anchor, Phase C).
- **PERM-DESC: CUT, reported not-run** (chair adjudication 3). Pre-committed forfeit
  (§4.8.7): selection intelligence (channel 1a) is forfeited on this window because it
  cannot be certified, not because the question is settled.
- **Multiplicity:** 7 deviation comparisons read; pre-registered family-wise expectation
  quoted in §3.2. No deviation read is averaged with, or substituted for, the registered
  verdict.
- **Boundary:** full window 2026-02-03→2026-06-10 (66 paired deltas), holdout n=44.
  Disclosed exclusions, identical in both arms: Monday/cadence gaps; 2026-03-30..31;
  the 2026-05-11..20 outage hole (7 td, inside the holdout).

## 6. FINAL LINE (registered format, §4.7)

> BRAIN+CHASSIS vs INCUMBENT: TIES (degenerate: brain ships neutral; evolution = instrument-failed) by +0.00% dReturn, +0.00 dSharpe on holdout (paired deltas identically 0, n=44; sha256-equal series); DEVIATION READ (deviation: a-priori genome, instrument-failed EA): TIES (straddling) — full n=66 mean −1.98 bp/day t=−1.26 (HAC −0.96), MDE 4.1 bp/day; holdout n=44 mean −3.13 bp/day t=−1.33, dSharpe −0.42, dReturn −1.41%; ORTHOGONALITY: PASS (signal-space max pooled |rho| = 0.202 ≤ .70; G-book instrument mechanically inapplicable — adjudicated defect); ORGANS: M1:indeterminate-at-available-power(dev-read; ships, acceptance near-formality) M2:indeterminate-at-available-power(challenger, OPEN) M3:B-disp-constant(materiality FAIL .011 bp/day) M4:indeterminate-at-available-power(challenger M4-A; +0.08 bp/day, HAC t +1.84 < +2, width-of-losing-tilt caution) M5:disabled(12/19 p=.18; challenger arm degenerate — 0 live episodes) M6:forecast-PASS(IC .0224, 6/6) LLM:retired(falsifier killed: t=+0.01 < +2.0, MDE 0.21 bp/day) EA:instrument-failed(anchor .282 < .5; B0 shipped; 0/3 cycles); COST: $9.50/mo absolute, +$0.21/mo marginal — reductions: PERM-DESC cut (not run), EA production not run, $0 new Bedrock.

Caveats (§4.1, attached to every read): the operator's repaired-timeline caveat (the live
record was repaired at points; only paired same-harness comparisons are admissible); the
2026-04-01 signals-rebuild epoch covers live-era dirs through ~2026-03-31; L7 deployed-MLP
training-data end 2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms
share the model); Monday/cadence gaps, 2026-03-30..31 and the 2026-05-11..20 outage hole
(7 td, inside the holdout) are excluded identically in both arms. Every read above has an
attached E1/E2 manifest (`prototype/evidence_007/<cmp_id>/comparison.json`); none is
incomplete under §4.7's completeness clause.
