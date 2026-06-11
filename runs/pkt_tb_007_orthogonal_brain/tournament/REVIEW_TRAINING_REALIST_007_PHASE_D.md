# PHASE D REVIEW — TRAINING REALIST — PKT-TB-007 (the cross-packet read)

**Packet:** PKT-TB-007-ORTHOGONAL-BRAIN — Phase D evidence review.
**Author:** Training Realist, 2026-06-11.
**Inputs:** ATTRIBUTION_007.md, BAKEOFF_007.md, prototype/evidence_007/scorecard_007.json,
prototype/gates/acceptance_007.json, prototype/designs/c2_gate_matrix.json, RUN_JOURNAL.md,
my Phase B attack (tournament/ATTACK_TRAINING_REALIST_007.md), TB-006
REVIEW_TRAINING_REALIST_PHASE_D.md, and the battery series themselves
(runs_battery_007/D01_INC). All arithmetic recomputed from artifacts, not transcribed.

---

## 1. Compliance — were my Phase B bindings honored? (brief; the verdict hangs on it)

| Binding | Status | Receipt |
|---|---|---|
| FM1 embargo ≥22 td (MAX's 26 adopted) | HONORED | TOURNAMENT_007 synthesis; RUN_JOURNAL Phase B entry |
| FM2 P1 = pooled unseen-fold rotation read | HONORED-THEN-MOOT — pre-registration re-based; EA never ran (0/3 cycles) | RUN_JOURNAL; scorecard ledgers |
| FM3 gates → existence + materiality + printed power pairs, BH-FDR family | HONORED in full | acceptance_007.json: every gate carries se, p, P(pass\|null)/P(pass\|effect), bh_fdr_family table |
| FM4 C2 per-fold ceiling → replication trigger; M1–Z0 row loses per-fold ceiling | HONORED | c2_gate_matrix.json: `replication_trigger`, Z0–M1 `special_row` |
| FM5 anchor equivalence + slope band (chair widened to [0.5,2.0], dissent recorded) | HONORED | adjudication 2; anchor failed on Pearson regardless (0.282) |
| FM6 M2 carried OPEN with the 31–68% power band | HONORED | acceptance_007.json M2 `power_note`; ATTRIBUTION M2 |
| FB1/FB2 (rotation-gate clause, boundary-pin audit) | MOOT — EA production never ran | ship_decision; ea_cycles 0/3 |
| FB3 power column printed everywhere | HONORED | acceptance_007.json throughout |

**The compliance fact that matters most:** when the fitness instrument failed
(anchor Pearson 0.282 < 0.5 at representative scale), the pre-stated consequence
executed — B0 shipped, the EA did not run, and every deviation number carries its
label. Nothing was silently absorbed. As in TB-006, what follows is the honest output
of the arithmetic, not an execution artifact.

One note on my own machinery: my FM4 replication trigger fired on 9 of 10 directional
pairs — in book space, uniformly, **including pairs whose signal-space ρ is 0.00–0.20**
(c2_gate_matrix: market-residualized book Pearsons 0.018–0.498). An instrument that
assigns ~0.9 to known-orthogonal pairs is measuring the market factor, not redundancy;
the chair's "mechanically inapplicable" call is arithmetically right and the Phase D
vindication check (challenger marginal books are sub-bp, not near-identical competing
books) came back clean. The real orthogonality receipt is signal space: max pooled
|ρ| = 0.202 vs TB-006's 0.941. TB-006 follow-on item 3 (buy decorrelation structurally,
≤0.7) was delivered.

---

## 2. What two packets now establish, at what confidence

### 2.1 Perception: one finding, measured roughly two-and-a-half times

TB-006 CAST: purged weekly rank-IC **0.105** (per-fold 0.061–0.113; ridge twin −0.009).
TB-007 M1 (CAST-XS, independent rebuild, disjoint feature partition): **0.1076**
(se 0.0195, n=305 weeks, p=1.6e-8 — 5.5σ). And the incumbent's own deployed MLP
(member zero) reads **0.0657** on the same statistic. Three architectures, two
independent builds, one panel.

**Is that one finding or two?** It is **one finding about the panel, replicated across
builds** — not two independent findings. The two builds share the four things that
matter for inflation: the 64-ETF panel, the 6-fold purged scheme, the universe, and the
5d cross-sectional target family. They differ in architecture, seeds, feature
partition, and code. So replication has *killed*: implementation bugs, seed luck,
architecture-specific overfit. Replication has *not touched*: fold-scheme optimism,
universe-selection effects, and — the concrete channel I would name — stale-close /
asynchronous-NAV autocorrelation in the international names (FXI, FXE, INDA-class
tickers sit in the tilt core), a classic source of cross-sectional 5d "IC" that is real
in the panel and unharvestable at the close. The program's own G15 sentence already
concedes the honest shrunk core IC is **0.04–0.06**, not 0.105.

**Confidence statement:** that the panel contains genuine 5d cross-sectional structure
at IC ≈ 0.10 (panel-measure) — HIGH (5.5σ, twice, third model concurs). That the
*harvestable* IC is 0.04–0.06 — the program's own stated prior, untested forward. That
it is harvestable at all — UNKNOWN; no test run to date had the power to say.

### 2.2 Conversion: the Grinold arithmetic, computed against the measured −2 bp/day

The pre-registration printed the ceiling before the read (G15 / §4.8.8): post-shrinkage
IR 0.6–1.1 ⇒ expected live t ≈ **0.3–0.7** on n=66. At the measured tilt-overlay sd of
12.81 bp/day, that implies a true mean of **+0.47 to +1.10 bp/day**. Against that:

- Measured: full **−1.98 bp/day** (HAC se 2.06, HAC t −0.96); holdout **−3.13** (HAC t −1.09).
- P(observe ≤ −1.98 | true +1.10) = Φ(−1.49) ≈ **6.8%**; | true +0.47 → Φ(−1.19) ≈ **11.7%**;
  | true 0 → Φ(−0.96) ≈ **16.9%**.
- Likelihood ratio of "true ≈ 0" vs "true ≈ +0.8 (Grinold mid)": ≈ **1.6 : 1**.

So: the measured value **is within the plausible range of a real signal badly
expressed** — a 1-in-9 to 1-in-15 low-tail draw, not a refutation — and the window's
total evidence against the Grinold-honest positive is a Bayes factor under 2:1. The
negative sign is a caution to carry, not a conviction of the OOF IC. What it also is
NOT: confirmation. The same data are 1.6× better explained by zero-or-negative.

The path-friction ledger that a real +0.5–1.1 would have had to survive: 25/67 days
expressed nothing (14 zero-funding-capacity, 11 quantized below min_order — a 37%
capture haircut); realized β path gap mean 0.069 (× market sd 61.8 bp/day ≈ 4.3 bp/day
of pure path noise — most of the 12.8 sd); costs 3.2 bps of traded, seed-stable to
±0.02 bp/day. A 0.5 bp/day true edge net of a 37% expression haircut is ~0.3 bp/day —
**14× below the holdout MDE.**

### 2.3 The three remaining explanations, weighed

**(a) The 5d signal is real but smaller than costs + path frictions at this
expression.** Fully consistent with the data (§2.2). The sharpest sub-version — *the
OOF IC itself is partially inflated by what both builds share* — is the one hypothesis
that explains everything at once: honest IC 0.04–0.06 ⇒ predicted conversion 0.2–0.5
bp/day ⇒ ≈ 0 after the 37% expression haircut ⇒ exactly what both packets show.
Unresolved by replication (shared panel/folds/universe), untestable backward,
**directly testable forward** (§5, item 1). Weight: the live reads tilt here weakly
(LR < 2:1); the structural argument tilts here more strongly.

**(b) Expression-optimization was never performed on a certified instrument.** True as
stated: TB-006's EA optimized a drawdown-penalized fitness on 6 fold-scores and chose
not-trading (misaligned fitness, 4 trades in 65 days); TB-007's EA ran 0/3 cycles
because the surrogate failed certification (0.282). "Conversion fails at optimum" is
therefore **still unmeasured**. But bound the upside honestly: optimization over
{tilt_gain, trust, dead_zone, caps} from a value-blind mid-range start plausibly buys
1.5–2× capture, not a sign flip — if the truth at mid-range is +0.5–1.1 bp/day, the
optimum is maybe +1–2, **still at or below the 4.1 bp/day MDE** of any feasible live
window. (b) is real and worth fixing, but fixing it does not buy a certifiable read;
it buys a better point in a space the instrument still can't grade. And the
instrument-failure rule earned its keep: an EA searched on the uncertified surrogate
would have shipped tilt_gain > 0 into a window where the tilt lost.

**(c) Both utility reads are power-starved by construction.** The dominant established
fact. MDE 4.13 bp/day full / 5.75 holdout vs a predicted effect of 0.3–1.1: the
instrument is **4–14× too coarse** for the hypothesis it was pointed at, on one
~4-month path that happened to be an up-tape (incumbent +10.2% over the window,
+14.9 bp/day mean). To certify 1 bp/day at t=2 through sd 12.8: n = (2·12.8/1)² ≈
**655 td ≈ 2.6 years**; 0.5 bp/day ≈ **2,620 td ≈ 10+ years**. Neither packet ever
fielded a test that could have certified its own prediction — TB-006 by genome
throttle, TB-007 by window arithmetic, both pre-registered as such.

**The honest cross-packet sentence:** "conversion failed twice" overstates. Precisely:
conversion has twice failed to *appear*, at any magnitude the windows could see, with
both point estimates negative, against a predicted magnitude (+0.3–1.1 bp/day) that no
test yet run was powered to confirm. The two opposite expression designs (evolved
sizing at unmatched exposure; a-priori tilt at matched exposure) bracket the expression
space without ever measuring its optimum. No measured edge exists. No proof of no edge
exists either. What the two packets have genuinely bought is the *instrumentation* to
settle it (§5) and a tightly shrunk hypothesis: if there is an edge, it is
0.2–1 bp/day net, and everything above that is now excluded with real confidence
(the holdout 90% CI top is +1.6 bp/day; anything ≥ 4–6 bp/day is flatly refuted).

---

## 3. The gate program verdict: working as designed — with the fail-type decomposition

Six organs in: 1 ships, 2 challengers, 1 constant-arm, 1 disabled, 1 forecast-only.
The question is whether the fails are "too strict" — i.e., reachable with more data —
or genuine no-signal verdicts. Computed per organ:

| organ | what it would need to PASS | reachable? | fail type |
|---|---|---|---|
| M2 existence | IC 0.041; measured 0.0368, se 0.0254 — **missed by 0.17 se** (a coin-flip-distance miss). 80% power at the measured IC needs se 0.0148 ⇒ ~410 16-td windows ≈ **26 years of history** (4.3× the 96 available) | NO — not in any program horizon | data-volume |
| M2 materiality | NetEdge 0.55 < 1.0 **at the point estimate** — the measured rotation effect doesn't pay its own costs even if real | more data doesn't change cost arithmetic | effect-size |
| M3 materiality | 0.011 bp/day vs bar 0.2 — **18× short**; existence is solid (p=2e-11) | NO | effect-size (real, can't pay at this T_max) |
| M4 existence | ΔAUC vs vol-control ≥ +0.0020; measured **−0.0019** (3.1 se below the bar, wrong sign) — more data shrinks the CI around a negative number | NO — replicates TB-006 (GDELT ≈ noisy vol proxy) | no-increment |
| M5 existence | ≥14/19 episode wins (p≤.05); measured 12/19, p=.18. At the observed 63% win rate and ~2 episodes/yr, significance arrives near n≈40 ⇒ **~10 years** | NO at decade scale | data-volume |
| M6 | PASSED (IC .0224 vs bar .02, haircut p=.0074, 6/6 folds) | — | — |

**Verdict: gates working as designed, not too strict.** Two of the four fails are
effect-size/no-signal verdicts that more data cannot reverse (M3 materiality, M4
increment); the two data-volume fails (M2, M5) require 10–26 years to resolve, which
makes "challenger / disabled" operationally equivalent to "no measurable edge at this
granularity" — and the pre-printed sentences keep that honest ("not measurable yet,"
never "rotation is dead"). The calibration also demonstrably mattered in the
permissive direction: the original M5 gate (">50% tally") would have PASSED 12/19 — a
gate a fair coin clears 41% of the time — and wired a noise rule into the verdict arm;
the recalibrated bar kept it out, and M5 then fired zero live episodes anyway. The
counterfactual loose-gate program ships M2+M4+M5 into a tilt whose marginal arms have
MDEs of 0.09–0.77 bp/day and returns three more indeterminates; TB-006 already paid
for that lesson once (0.941-correlated members, zeros everywhere).

---

## 4. The directive-4 paragraph (for the operator, plain English)

Every model type the program could field is in this system on purpose, and almost all
of them now carry a measured number instead of a vibe. The transformer found a real
pattern — twice, in two independently built systems — and it is the only component
that did. The gradient-boosted rotation model sees something too faint to measure and
too small to pay trading costs even if real. The dispersion model sees a real
phenomenon (the statistics are overwhelming) whose tradable value computes to about a
hundredth of a basis point a day — real and worthless, at this book size, honestly
labeled. The event model is measurably a volatility gauge wearing a news costume —
both builds agree. The positioning rule's moment never arrived in the live window: it
fired zero times, which for a once-every-few-months signal is the expected honest
outcome, not a malfunction. The overnight-gap model passed the one test the program
could power, and the program correctly refuses to trade it because no expression
channel exists at this trading cadence. The LLM was given 46 pre-registered chances to
matter and a final live falsifier; it scored zero on all of them and now costs $0.14 a
month to keep monitored. The evolutionary optimizer was not allowed to run because its
measuring stick failed inspection — which protected you from shipping a tilt that this
window says would have lost money. **That configuration is the deliverable.** A
decorative ensemble — everything wired in because more models feel safer — would have
produced exactly the same trades as this system and no knowledge. What you own instead
is a map: one component with real signal whose cash value is still below the noise
floor of any four-month test, and seven measured zeros, each one a retired temptation
with a receipt. Measured zeros compound; decoration doesn't.

---

## 5. The follow-on that would settle it, ranked by information-per-effort

**1. Dual forward shadow (E3-style, costless, pre-registered now).** Two legs on the
same accruing data, zero marginal infra (signals already emit daily; the replay
harness exists), look-budget = one pre-registered read at each maturity, no peeking:
   - **Forecast leg** — M1's forward weekly rank-IC on data neither build has touched.
     Weekly IC sd ≈ 0.30 at this breadth ⇒ distinguishing IC 0.107 from 0 at 2σ needs
     ~31 independent weeks ≈ **7 months**; distinguishing 0.107 from the shrunk 0.05
     needs ~110 weeks ≈ **2.1 years**. This is the direct test of the shared-inflation
     hypothesis (§2.3a) — the one question the two packets structurally could not ask.
   - **Utility leg** — the frozen a-priori tilt vs incumbent as a paper pair, exactly
     the D03 configuration. At sd 12.8 bp/day: certifies |effect| ≥ 2 bp/day in
     (2·12.8/2)² ≈ 164 td ≈ **8 months**; ≥ 1.5 in 291 td ≈ **14 months**; ≥ 1 in
     655 td ≈ 2.6 years. Pre-register the 14-month read with an equivalence verdict:
     CI ⊂ ±1.5 bp/day ⇒ "measured zero at materiality scale," sign language reserved
     for effects the window can see (my TB-006 §7 rule). N for the operator's
     question: **8–14 months settles whether conversion exists at any size that
     matters; if the truth is 0.5 bp/day, nothing feasible certifies it and the
     equivalence read correctly returns "too small to matter."**

**2. M4-A damping confirmation — one pre-registered arm, riding inside #1.** The
+0.08 bp/day (HAC t +1.84 full / +2.18 holdout) line is the battery's only positive,
against a 25–30% pre-registered chance of one spurious certifiable-looking line AND a
mechanical confound that fully explains the sign (damping a losing tilt scores
positive by construction). Honest prior: 15–25% real. The discriminating condition is
already printed: a window where the base tilt's point estimate is **positive**.
Pre-register the D05-equivalent arm in the shadow, verdict valid only on
base-tilt-positive sub-windows; power ≈ 75% at true +0.08 on another n≈66 (se 0.029).
One arm, pre-commit to dropping the thread on a miss. At near-zero cost this is
positive-information even at a 20% prior; running it twice would be noise-chasing.

**3. Expression search on REAL replays (the instrument fix) — correct, and not yet
honest to run.** The fix for the failed surrogate is real-replay fitness, and the
look-discipline question has a hard answer: optimize on pre-holdout live replays only.
That segment today is **22 trading days** (2026-02-03→03-11). Fitness se on 22 days =
12.8/√22 ≈ 2.7 bp/day; a median-of-top-8 champion over K_eff ≈ 30 carries a
√(2 ln 30) ≈ 2.6σ selection uplift ≈ **7 bp/day of pure selection noise** against true
genome differences of well under 1 bp/day. An EA run on that today certifies noise.
Minimum honest training segment ≈ 125–250 td with the continuation untouched — i.e.,
this thread is **downstream of #1 by 6–12 months**, not parallel to it. (The 2025
backfill leg stays disqualified per FX4: 25 names, 2/10 tilt-core, one-hot inference
era.) The deeper engineering fix — the surrogate's carried tilt never feeds back into
chassis state — is worth doing only if #1's forecast leg survives, because optimizing
the expression of an inflated IC is polishing a number.

**4. Fold-pool extension (F7/F8 as time accrues) — forecast-space only.** Forecast
attribution stays valid and cheap and keeps the powered IC instrument current. The
fold-pool *utility* instrument is blocked: it ran through the surrogate, and the
surrogate is the thing that failed certification (0.282). Don't rebuild it ahead of
#1's verdict.

Not worth doing: more live LOO replay arms this window (every marginal MDE is already
printed and the arms are spent); any M2/M5 re-gate before years of accrual (§3); any
LLM re-entry before the 900-day trigger (the door closed with receipts, twice).

---

## 6. Statement for the committee report

Two packets, two opposite expression designs, one disciplined instrument failure, and
the cross-packet ledger now reads: perception is established at panel level (weekly
rank-IC 0.105/0.108 across two independent builds, 5.5σ, with the deployed MLP's 0.066
concurring) but it is one finding about a shared panel measured twice — replication
killed bugs and seed luck, not fold-scheme optimism or stale-close artifacts in the
shared universe, and the program's own shrinkage sentence (honest IC 0.04–0.06)
remains untested forward; conversion has twice failed to appear rather than failed —
the measured −1.98 bp/day (HAC se 2.06) sits within the low tail of the pre-registered
Grinold-honest prediction (+0.47 to +1.10 bp/day; a 1-in-9 to 1-in-15 draw) and favors
zero-or-negative by a likelihood ratio under 2:1, on windows whose MDEs (4.1/5.8
bp/day) were 4–14× the predicted effect by construction, through an expression channel
that ate 37% of days to capacity and quantization; the optimized expression remains
genuinely unmeasured (TB-006 fitness misaligned, TB-007 instrument failed honestly),
but its plausible upside (~1–2 bp/day at best) still sits at the certification floor.
The recalibrated gates returned 1-ship/4-zero-or-challenger/1-forecast-only and the
fails decompose cleanly into two no-signal verdicts (M3 pays 0.011 bp/day against a
0.2 bar; M4's event increment is negative-signed at 3 se, twice now) and two
data-volume verdicts whose honest resolution times (M2 ~26 years, M5 ~10) exceed any
program horizon — gates working as designed, including keeping a 41%-null-pass coin
flip (M5 at the old bar) out of the verdict arm. The operator's question — is there
genuinely no ML edge — has the honest answer: no measured edge, no proof of none, an
excluded region above ~2–6 bp/day, a surviving hypothesis of 0.2–1 bp/day net, and,
for the first time, instruments that can settle it: a costless dual forward shadow
(forecast IC ~7 months, conversion-at-materiality ~8–14 months) with one
pre-registered M4-A confirmation arm riding along, expression optimization deferred
until ~6 months of honest training days exist. The system's remaining cost is $9.50/mo;
the information per dollar of continuing the measurement program is excellent, and the
information per dollar of believing anything stronger than the above is zero.

— end —
