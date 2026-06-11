# DESIGN_MIN — PKT-TB-007 Phase A, Architect-Min (certifiability-first minimalism)

**Date:** 2026-06-10 · **Author:** Architect-Min (independent; DESIGN_MAX not read)
**Inputs:** packet PKT-TB-007 (directives 1–7, C1–C6), ANALYST_AUDIT.md,
UNIVERSE_EXPLOITABILITY.md, PROPOSAL_ORTHOGONALITY.md, PROPOSAL_EXPRESSION.md,
PROPOSAL_EVOLUTION_007.md, PROPOSAL_LLM_ROLE.md.
**Specialist dispositions, up front:** ORTHOGONALITY — adopt the five-axis decomposition
and the C2 gate whole; add per-organ pre-holdout ACCEPTANCE gates before any organ enters
the verdict arm (§2.3). EXPRESSION — adopt channel 1b (tilt) as the ONLY channel; cut 1a
and the hybrid (§3). EVOLUTION — adopt fitness, rotation gate, controls whole; shrink the
genome 17→12 and move the anchor check off the 2025 epoch shim (§4, §7). LLM ROLE — adopt
R5 retirement + the single falsifier arm (§5).

---

## 1. System thesis

The operator's goal is "as intelligent a true full brain as possible," and TB-006's
autopsy says intelligence on this substrate is MEASURED intelligence: a brain whose
claims the verdict machinery can certify, on a record whose paired sd has been driven
down by construction. Every component that cannot produce a certifiable verdict line is
not intelligence — it is parameter decoration that widens the multiplicity ledger,
inflates paired sd, and dilutes attribution for the organs that do work. So the minimal
design buys orthogonality structurally (disjoint targets/horizons/partitions, then the
C2 gate), spends its capacity where receipts already exist (CAST IC .105, GBM E1 +2.48),
admits new senses to the verdict arm only through pre-registered pre-holdout acceptance
gates, expresses everything through one channel whose exposure parity is a projection
(not a statistic), and replaces every unsupported learned component with a fixed rule
the EA can tune within pre-registered bounds. Small is not the goal; CERTIFIED is the
goal, and small is what certified costs on ~66 paired live days.

## 2. Organ set

### 2.1 Decision: keep all five axes; change what "present" means

The Orthogonality Engineer's five-axis decomposition is correct and is adopted: the axes
are genuinely different questions, the partitions are disjoint, and the model-type
assignments are mechanically motivated, which is exactly directive 4 done honestly. The
minimalist change is not to the axis set but to the SHIPPING RULE: an organ is wired into
the verdict arm's tilt only after passing a pre-registered acceptance gate on pre-holdout
data (§2.3). Organs that fail still run — as battery challenger arms with scorecard
lines — so every model type remains present, measured, and attributed; none is present
as unfalsifiable decoration.

**Directive-4 defense, stated for the operator:** "all model types present ON PURPOSE"
is satisfied by all five types being built, trained where trainable, measured against a
named falsifier, and given a verdict line — not by all five being wired into the tilt
regardless of what the measurement said. Wiring a gate-failing organ into the verdict arm
makes the brain dumber and the verdict blurrier; the packet itself sanctions honest zeros
as passing outcomes. If the operator wants gate-failers wired in anyway, that is a
one-line config change (organ gates live in the genome manifest) — the design makes it a
choice, not an accident.

### 2.2 The organ table (architecture, target, partition, params vs effective sample)

Adopted from PROPOSAL_ORTHOGONALITY §2/§6 with two modifications (M4 partition, M5
status). Sample arithmetic carried verbatim (N1 breadth 12/64, ÷h overlap).

| Organ | Type (why this type) | Target | Partition (disjoint) | Params / eff. sample |
|---|---|---|---|---|
| **M1 CAST-XS** | Transformer — cross-asset attention IS the relative-strength question (s's score conditions on the other 63 tokens) | per-day Gaussian rank of open(D)→open(D+5) cross-sectional excess | fast block only: 10 ≤21d price/volume features + sector/class embeddings; NO 63d, macro, event, COT | ≤16.5k / ~7,100 — marginal, accepted on TB-006 precedent (shipped ratio, IC .105, ridge twin at noise; ridge referee re-trained) |
| **M2 ROT-GBM** | GBM — axis-aligned splits + monotone constraints natively express conditional macro interaction at this sample size | per-day Gaussian rank of open(D+5)→open(D+21) excess (horizon-disjoint: zero shared bars with M1) | slow per-symbol block (63d stats) + macro/credit/breadth context; NO ≤21d momentum, vol-surface, event, COT | ~600 leaf values / ~1,750 — supported (4× capacity cut vs TB-006 GBM, paying the 16× overlap tax) |
| **M3 DISP-HAR** | HAR linear — dispersion is long-memory; ~10 coefs is the honest capacity at ~558 effective obs | forward 5d realized cross-sectional dispersion over the tilt set (log) | dispersion lags 1/5/21d + vol-surface block (COR3M, VVIX, SKEW, term slope); NO per-symbol, event, COT | ~10 / ~558 — supported |
| **M4 EVT-NET** | Elastic net — sparse linear transmission; L1 picks which bucket features transmit | per-bucket exceedance: 1{\|5d market-removed bucket return\| > trailing 80th pct} | **GDELT-only** B-matrix columns (G1–G5 + masks, 27 buckets) + trailing bucket-vol control. **LLM columns removed per R5 (§5)** — was the one partition change vs the proposal | ≤~400 nominal, L1 → ~60–150 active / ~4,500 — supported |
| **M5 POS-Z** | Pre-registered z-score rule, 0 trained params — the only honest model class for ~15–25 crowding episodes | none trained — \|z\|>2 on ES/UST10Y/VIX TFF leveraged-net (156w), contrarian sleeve tilt, 21d decay | COT columns exclusively | 0 / ~15–25 episodes — supported BECAUSE capacity = 0; threshold-tuning denied by this row |

Member zero (the deployed RankingMLP, blend 0.35) sits in the C2 matrix as row 0 per the
proposal; M1 must clear ≤0.8 against it or it cannot beat 0.35-blend redundancy.

Five types, five named mechanical reasons: transformer ↔ cross-asset attention; GBM ↔
split-based macro conditioning; linear-AR (HAR) ↔ long memory; sparse linear (enet) ↔
event transmission; rules ↔ sample-starved axis. The LLM is the sixth type and lives in
§5's role. Nothing was cut; the directive is satisfied with the full set, under gates.

### 2.3 Acceptance gates (the minimalist addition — pre-registered, pre-holdout only)

An organ enters the VERDICT arm's tilt iff it passes BOTH (a) the C2 decorrelation gate
(PROPOSAL_ORTHOGONALITY §3, adopted verbatim: pooled |ρ| ≤ 0.7 every pair both spaces,
no fold > 0.8, remediation ladder M1>M2>M4>M3>M5, demotion never paper-over) and (b) its
own evidence gate on F1–F6 OOF:

- M1: purged weekly rank-IC ≥ 0.04 AND forecast-space IC delta vs member zero > 0.
- M2: fold-pool rank-IC at the D+5→D+21 target > 0 with overlap-corrected t ≥ +1.5.
- M3: OOS forecast R² vs trailing-21d-mean baseline > 0 pooled F1–F6.
- M4: OOF exceedance AUC ≥ 0.55 AND AUC uplift over the vol-only control net > 0.
- M5: per-episode sign tally > 50% on F1–F6 episodes (n≈15–25; expected outcome:
  indeterminate ⇒ ships DISABLED with the rule's verdict line reading `indeterminate at
  available power` — present, measured, not spending tilt budget on coin flips).

Gate-failers run as battery challenger arms (organ ON, all else identical) so their
scorecard line is measured, not assumed. Expected shipping set on current evidence:
M1+M2 certain, M3 likely, M4 genuinely open, M5 likely disabled. A 3-organ verdict arm
with 5 measured types is the honest reading of directive 4 on this record.

## 3. Expression — one channel, the tilt (adopt 1b; cut 1a and the hybrid)

**Adopted:** PROPOSAL_EXPRESSION channel 1b as the ONLY expression channel — and the
verdict carrier. **Cut:** the rank-permutation socket (1a) and the hybrid (1c), entirely
(not even as a battery ablation). Reasons: (i) 1a is mute on no-buy days and worth 6–15
bits on the minority of days the chassis buys — its certifiable content over 66 days is
near nil while its machinery (exact counterfactual pipeline replay, score solve-back,
out-of-contract r' values, greedy guard repair) is the largest single build item on the
menu; (ii) its residual exposure channels (vol_adj composition, the 0.80 high-vol gate,
marginal-buy path divergence) are exactly the C1 leaks the packet forbids; (iii) one
swapped marginal buy held for weeks adds ~30 bp/day to paired sd — one bad swap can eat
the entire certifiability budget the tilt channel buys. The selection-intelligence
question 1a answers is real; it is FORFEITED here (see §10) because it cannot be
certified on this window.

**Mechanics (per the proposal, parameters fixed here):**
- `Strategy.post_decision` edits the chassis's own intents at the margin. Support set =
  chassis buy set ∪ held names ∪ **the tilt core** (b-extended: without the core names
  the brain's measured per-name skill — ITA, SOXX, XRT, TLT… — is mostly inexpressible;
  C3 makes this load-bearing, not bloat).
- **Parity projection (C1 by construction):** Δw projected onto {ΣΔw=0} ∩ {ΣΔw·β̂=0} ∩
  {|ΣΔw·σ̂| ≤ ε_σ}; β̂ = trailing 126d OLS beta to SPY, σ̂ = trailing 21d vol, both
  point-in-time from the imported OHLCV cache. No shorting (legs floored at −w_i); no
  tilt on names being fully sold; min_order respected, residual rounding absorbed by the
  largest buy leg. This answers the Universe Selector's disguised-de-risk caveat
  structurally: a TLT tilt must be β-funded, so it cannot lower book beta vs the
  incumbent arm.
- **Tilt strength:** conviction `c_t = σ(a·agree_t + b·disp_t + Σ_k d_k·q_k − θ)`;
  budget `T_t = c_t · T_max` with **T_max = 8% NAV one-sided, pre-registered, never a
  gene above the cap**; dead_zone gene below which T_t = 0.
- **Tier caps (from UNIVERSE_EXPLOITABILITY §4):** core 10 (ITA, SOXX, XRT, TLT, AGG,
  MUB, FXE, USO, FXI, RSP) per-name cap 2.5% NAV; conditional adds (IYR, SHY, KRE)
  1.25%; FXE additionally capped at 1.25% (ADV $14M); **ballast (mega-clone + factor
  sleeve) FIXED 0**; **VIXY EXCLUDED** from the tilt support entirely — its +41 bp/bet
  NetEdge is a vol bet at 18 bp RT that the σ-projection would mostly block anyway;
  what survives the rail is re-risking in disguise. defensive_fraction gene caps the
  TLT/AGG/MUB/SHY share of the budget at ≤0.5 inside the projection rail.
- **Conviction path / neutral recovery:** c_t→0 ⇒ Δw=0 ⇒ post_decision returns the
  incumbent's intent objects unchanged ⇒ bit-for-bit identical series. Mechanically
  checked by arm **B0-EXPR**: conviction forced 0, `sha256(timeline.json)` must equal
  the incumbent arm's. This is the attribution zero point for the whole battery.
- **Lot fix (C5):** the proposal's §3 fix adopted whole — adapter aggregation
  (`held_shares[sym] += shares`, `w_prev[idx] += …`), harness `_execute_intents`
  monkeypatch (BUY increments the held Position share-weighted; SELL iterates all lots),
  applied identically to both arms, with `test_lot_aggregation.py` as specified.
- **Observability:** the proposal's `expression_log/<D>.json` schema adopted verbatim,
  minus the `perm` block (no permutation channel). organ_attribution = per-decision
  recompute with organ k zeroed, τ renormalized — exact w.r.t. the projection used.
- Separate process per arm (model-cache hazard, `replay_engine.py:147-149`), try/finally
  monkeypatch pattern, seed 4242 cost overlay — all per ANALYST_AUDIT.

## 4. Evolution — what the EA governs here

PROPOSAL_EVOLUTION_007 is adopted as the spine: fitness = floored paired IR of the daily
brain-minus-neutral tilt series (√252·mean/max(sd, 2bp), min over cost×{1.0,1.5},
worst-fold floor at full weight, λ_reg=0.05 shrinkage toward B0), B0-in-population, B1
budget-matched random search, adoption gate (champion > 1× cross-fold sd over
FITNESS(B0)≡0), the NEW leave-one-fold-out rotation gate (≥4/6 unseen folds ΔU ≥ 0 AND
pooled unseen mean > 0; both gates AND; B0 ships on any failure), median-of-top-8
champion with L2-dedupe, every-variant logging, deterministic replay. The fitness IS the
verdict's random variable with the market component cancelled — this is the root fix for
the TB-006 defense-graded-on-offense failure and the strongest idea in the proposal set.

**Genome, shrunk 17→12** (minimalist deltas, each with a reason):

| Gene | Count | Disposition |
|---|---|---|
| organ_trust[M1,M2,M4,M5] | ≤4 | kept (genes exist only for organs that pass §2.3; a disabled organ's gene is removed, not frozen) |
| tilt_gain κ | 1 | kept — THE zero point (B0 default 0.0) |
| conviction_temp, dead_zone | 2 | kept (tightened ranges per proposal §2.3) |
| disp_gain (M3's weight b in c_t) | 1 | kept — M3 is a gain organ, not a trust row |
| cap_core, cap_conditional | 2 | kept (ranges [0, 2.5%], [0, 1.25%]) |
| defensive_fraction | 1 | kept |
| event_damp_strength | 1 | kept iff M4 ships; **event_veto_threshold CUT** — two genes for one organ's gate is redundant; damp→0 at the range edge subsumes veto |
| parity_tol_gross, parity_tol_risk | 0 | **CUT — fixed at the C1 hard bounds.** Letting the EA tighten parity adds two genes whose effect on the verdict is invisible by construction (both arms inside the band); invisible-to-the-verdict genes are exactly what this design refuses to carry |
| channel_mix | 0 | **CUT — one channel exists** |

The EA governs, in plain words: how much to listen to each surviving organ, how hard to
tilt overall and per tier, how much budget the defensive sleeve may take, and how hard
event risk damps — twelve numbers, all expressed through one projection, all anchored to
a B0 that reproduces the incumbent bit-for-bit. Universe tiers, T_max, parity bounds, and
the C2/acceptance gates are FIXED inputs evolution cannot touch.

## 5. LLM disposition — adopt R5, with the directive-4 conflict stated honestly

The Role-Finder's R5 verdict is adopted: 46 ledgered screens, 0 BH-FDR(10%) survivors,
the one boundary signal (disagreement→dispersion ρ=0.122) converting to <1 bp/day, and
the confidence conditioner wrong-signed. Concretely:

1. **No llm_* columns in any member partition** (hence M4 goes GDELT-only, §2.2). This
   returns ~80 columns of partition budget and removes TB-006's negative-leaning channel.
2. **The monitoring emission stays live** ($0.14–0.20/mo): the record accrues ~22 scored
   days/mo toward the pre-registered re-test trigger (900 scored days or any future
   LLM-feature packet; promotion bar = same screen surviving BH-FDR(10%)).
3. **One falsifier arm in the battery (R-LLM):** the §5.1 disagreement-width conditioner
   — `llm_disag` z-scored 252d, scaling tilt width by `clip(1+0.25·tanh(z), 0.75, 1.25)`,
   direction-free, C1-safe — vs the unconditioned brain, paired, E1 read only; promote to
   a holdout look only at E1 paired t ≥ +2.0. Expected outcome, pre-registered: not
   promoted.
4. **Scorecard line:** "LLM organ: 0 (measured) — retired by role-finding; strongest raw
   candidate converts to <1 bp/day vs the window MDE; re-test trigger armed."

**Against directive 4, honestly:** the directive says all model types present at maximum
value "wherever it is." The measurement says the LLM's maximum value is as a monitored
sense with one falsifier arm and a re-entry path — "wherever it is" resolves, on this
record, to "not in the decision path yet." The type IS present: it runs nightly, it has
an arm, it has a verdict line, and it has a pre-registered road back in. Wiring it into
the tilt instead would repeat TB-006's measured negative and spend attribution power on
a sense the screens say is dark. If the operator overrules, the conditioner in (3) is the
wired-in form — it is already specced and C1-safe; the cost is one more gene and a
near-certain zero line.

## 6. Meta-level — no learned meta-evaluator; fixed rules + EA, defended

There is no learned meta-evaluator in this design. The combiner is linear in organ
z-scores with a fixed-form logistic conviction; trust is static-per-month EA genes; there
is no MLP executive, no daily trust adaptation, no learned regime-conditional gating.

Defense: (i) TB-006 measured this — the linear twin beat the MLP at the executive level
and daily trust adaptation fired all three of its kill criteria; static trust was never
the binding constraint. (ii) The sample arithmetic is terminal: organ-trust learning has
~6–8 effective samples (N3, fold scores); a learned meta-evaluator on 6 numbers is
astrology with a loss function. (iii) The operator's "as intelligent a true full brain as
possible" is read here as MEASURED intelligence: the brain's intelligence lives in where
its certified bp/day come from, and a meta-layer that cannot produce a certifiable
verdict line subtracts measurement power from the organs that can. The intelligence
budget this design refuses to spend on a meta-learner is spent instead on the rotation
gate — which certifies the one meta-question that matters (does the balancing PROCEDURE
generalize to unseen regimes?) with an auditable yes/no. If accruing live data ever makes
a learned meta-layer supportable (≥ ~30 fold-equivalents), that is a future packet with
its own gate; nothing here forecloses it.

## 7. Training plan

- **Folds:** TB-006 layout carried verbatim — F1–F6 (2020-02→2026-02), 21-td embargo,
  holdout firewall ≥ 2026-03-11, fitness data ends 2026-02-06. All member training, OOF
  matrices, C2 + acceptance gates, and the EA live entirely on F1–F6.
- **Retrains:** M1 on the fast partition (CAST-Small config as shipped, ridge twin
  re-trained as referee), M2 on the D+5→D+21 target with the 4× capacity cut, M3 OLS,
  M4 GDELT-only enet + the vol-only control net (its AUC baseline), M5 computed (no
  training). OOF matrices rebuilt for all; nightly precompute rebuilt after retargeting.
- **Chassis surrogate for fold-era fitness:** the Evolution Engineer's fast paired walk
  is adopted — base arm cached once per fold, tilt arm per genome, half-spread costs,
  matched gross by construction. The surrogate preserves the verdict's GEOMETRY (paired,
  matched-exposure, tilt-channel), which is what fitness alignment requires.
- **Anchor check, MOVED (the one disagreement with the evolution proposal):** the
  proposal anchors the surrogate on the 2025-08→2026-02 artifact window, which requires
  the backfill-era epoch shim and validates against a leg that trades a 25-name universe
  expressing 2 of 10 tilt-core names, on a four-epoch repaired record. This design
  anchors on the **live pre-holdout leg, 2026-01-31→2026-03-10 (~26–27 decision dates,
  native replay plan, no shim):** run champion and B0 through the REAL harness there;
  require per-day Δr_t Pearson ≥ 0.8 between surrogate and real harness AND sign
  agreement of the mean ΔU. Failure = surrogate misspecified = stop-and-fix finding, not
  a tweak. Smaller window, but it validates against the world the brain will actually be
  graded in — and it deletes the epoch shim from the build entirely.
- **Neutral-recovery unit check** (B0 bit-for-bit = incumbent on the real harness) runs
  BEFORE the first EA evaluation, per the proposal.
- **Window strategy (verdict):** the Expression Architect's live-era-primary
  recommendation is adopted whole. Verdict window = 2026-01-31→present (67 decision
  dates, 66 paired deltas); holdout ≥ 2026-03-11 (44 dates, 43 deltas) for the one E2
  read. **The 2025 backfill leg is not run at all** — not as verdict, not as context: it
  expresses 2/10 core names, needs a shim that as-coded replays zero days, and its
  expected contribution to the paired numerator is ~0 (pure t-dilution). The repaired-
  timeline caveat (directive 6) and TB-004 backfill caveat print with every read anyway.

## 8. Pre-registration sketch (numbers this architect would commit)

The ceiling arithmetic is faced, not dodged: at n=66/43 paired days, t ≥ 2.0 demands an
overlay Sharpe ≈ 3.9/4.8 against a Grinold ceiling of ≈ 1.2–1.8. A design that makes the
live paired t the BEATS bar has pre-registered its own TIES. So the primary certifiable
read moves to where the power is, and the live window becomes confirmation:

- **P1 (primary, powered): pooled fold read.** Paired tilt series (champion vs B0,
  surrogate, OOF organ outputs) pooled over F1–F6 (~1,500 days, n_eff ≈ 300;
  MDE@t2 ≈ 1.2–2 bp/day at the projected 6–15 bp/day paired sd). Bar: overlap-corrected
  HAC **t ≥ +2.0**.
- **P2 (live confirmation, real harness): full live window** (66 deltas, incl. pre-
  holdout). Bar: mean Δ > 0 AND 90% CI not entirely below 0. (Sign/CI, not t≥2 — that is
  what 66 days can honestly certify.)
- **P3 (E2, one read): holdout** (43 deltas). Bar: point estimate ≥ 0; CI printed;
  looks-ledger entry = 1.
- **Verdict line:** BEATS = P1 ∧ P2 ∧ P3. LOSES = P1 t ≤ −2.0, OR live full-record
  t ≤ −2.0, OR holdout 90% CI entirely < 0. TIES = everything else, with the honest-
  ceiling sentence printed beside it. ΔSharpe and Δreturn reported with the same
  paired-window convention; exposure-parity diagnostics (realized gross gap, rolling 21d
  β gap, disclosure at |Δβ| > 0.05 sustained 5 days) printed in the same block.
- **Per-organ arms (three-valued verdicts):** LOO drop_M1, drop_M2, drop_M3
  (dispersion-gate → constant), drop_M4 (if shipped), drop_M5 (if enabled), each vs the
  full brain, paired, per-arm MDE printed; organ = `positive` (fold-pool LOO t ≥ +2),
  `zero (measured)` (|effect| CI inside ±MDE), else `indeterminate at available power`.
- **Battery size (cap, ledgered):** incumbent, brain, B0-EXPR (hash check), ≤5 LOO arms,
  R-LLM falsifier, B1 random-search comparison, ≤2 challenger arms for gate-failed
  organs — **≤12 arms total**, every arm logged, no unledgered looks. EA gate reads
  (adoption + rotation) are F1–F6-only and do not consume battery slots.
- **Pre-committed sentences:** evolution carries weight iff both EA gates pass AND E2
  paired Δ ≥ 0; ceremonial-as-optimizer iff champion ≤ B1; the C2 matrix and acceptance-
  gate outcomes commit to `designs/c2_gate_matrix.json` before any bake-off replay.

## 9. Cost sketch + build wall-clock (TB-006 asset reuse assumed throughout)

- **Data/ingest: $0 new.** GDELT cache, OHLCV, CBOE/FRED/COT edges, panel, LLM artifacts
  all import as-is (ANALYST_AUDIT §C). **Bedrock: $0.00 planned** (R5 needs no
  re-scoring; the $3.00 cap stands unspent unless the operator overrules §5).
- **Build items (new code):** tilt adapter + projection + guard logs (~the one
  substantial module), lot fix + unit test, acceptance-gate harness, M2/M3/M4 retarget
  configs, EA fitness swap + rotation driver, anchor-check runner. Deleted vs the
  maximal menu: permutation channel, hybrid arm, epoch shim, parity-tightener plumbing,
  any LLM feature path.
- **Compute (local Mac):** M1 retrain ~2–4 h; M2/M3/M4 + control net < 1 h combined;
  OOF rebuild ~1 h; C2 + acceptance gates minutes; EA production + 6 rotations + B1
  ≈ 20–55 min (cap 90); anchor check 2 real-harness arms × ~27 dates ~30 min; battery
  ≤12 arms × ~67 dates, separate processes, ~2–4 h. **Total ≈ 8–14 h compute, 2–3
  calendar days** including evidence assembly — under TB-006, as the packet demands.
- **Deployed shape (if ever promoted; not this packet):** unchanged Lambda reading a
  static genome JSON; monthly EA local $0 AWS; LLM monitoring $0.14–0.20/mo; total well
  inside the ≤$10/mo envelope.

## 10. Honest weaknesses — where minimalism loses to a bolder design

1. **Selection intelligence is forfeited.** Cutting channel 1a means the WHICH-name-
   enters question — arguably the most chassis-native expression of cross-sectional
   skill — is never exercised. A bolder design that solves 1a's guard problem could find
   value this design structurally cannot see.
2. **VIXY is left on the table.** The single largest NetEdge line (+41 bp/bet) is
   excluded by fiat. A design with a real vol-expression rail could harvest it.
3. **The R5 retirement is power-limited, not proof.** If the LLM's value is real but
   below this screen's MDE, it stays dark here for ~18 months. A bolder design wires the
   conditioner in and accepts the noise.
4. **The verdict arm may ship with 2–3 organs.** If M3/M4 fail their gates, the brain
   that gets graded is thin — honest, but visually far from "full brain," and the
   operator may reasonably feel the directive's spirit was traded for its letter.
5. **Fold-era fitness rests on the surrogate**, and the relocated anchor check is only
   ~27 days. A sign-agreeing surrogate can still be subtly mis-calibrated in magnitude;
   the rotation gate inherits any such bias. (The alternative — paying the epoch shim
   for a longer anchor on a 25-name repaired record — was judged worse, not safe.)
6. **P1-primary is a surrogate-space verdict.** BEATS can be declared with the powered
   read living in the fast sim (gated by P2/P3 real-harness confirmation). That is the
   honest trade at n=66, but a critic can say the headline number was not earned on the
   real engine; the design accepts that critique and prints it.
7. **No meta-layer means regime-conditional trust is unexploited** if it exists; the EA
   sees only fold-level structure. The rotation gate certifies robustness, not
   adaptivity.
8. **min_order quantization at $100k NAV** mutes conviction below ~1.5% NAV total tilt —
   the dead zone is logged but the smallest honest signals never reach the book.

— end —
