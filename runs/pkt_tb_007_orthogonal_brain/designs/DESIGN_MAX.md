# DESIGN_MAX — PKT-TB-007 Phase A architect design (maximal prior)

**Author:** Architect-Max, 2026-06-10. **Prior declared:** maximal measured intelligence —
the operator asked for "as intelligent a true full brain as possible" with every model type
present on purpose; budget explicitly unconstrained. This design takes that directive at
full value and pays for every addition with sample arithmetic, a pre-registered kill, or
both. **Inputs:** packet C1–C6; ANALYST_AUDIT; UNIVERSE_EXPLOITABILITY (+json);
PROPOSAL_ORTHOGONALITY; PROPOSAL_EXPRESSION; PROPOSAL_EVOLUTION_007; PROPOSAL_LLM_ROLE.
DESIGN_MIN not read (independent-priors discipline).

---

## 1. System thesis — why the fuller brain wins

The verdict's headline is power-limited no matter what we build: paired t ≥ 2.0 on n=66
live deltas demands an overlay Sharpe ≈ 3.9 against a Grinold ceiling of 1.2–1.8
(PROPOSAL_EXPRESSION §4 — adopted as the honest frame, not contested). A lean brain and a
maximal brain face the same wall on the same dates. What a fuller brain buys is therefore
NOT a bigger headline t — it is **more independent, individually certifiable senses**:

1. **More orthogonal organs = more uncorrelated shots at positive paired utility.** If k
   engineered-orthogonal organs each carry an independent overlay IR of ~0.5–0.8 (the
   per-organ Grinold share), the combined overlay IR scales toward √k·IR_organ when the C2
   gate holds. Six organs at pairwise ρ ≤ 0.7 (and mostly far below — disjoint targets)
   is the only mechanism on the table that moves the EXPECTED t upward at all; window
   length is fixed, breadth-of-bets is not. Fund-of-uncorrelated-sleeves arithmetic is the
   one lever the lean design forfeits.
2. **Per-organ verdicts are powered even when the headline is not.** Forecast-altitude
   falsifiers (rank-IC, AUC, R² vs baseline) run on 1,500 pooled OOF days with n_eff in
   the hundreds-to-thousands — each organ's "is this sense real" line is certifiable NOW.
   The maximal brain produces six certifiable organ verdicts plus a headline CI; the lean
   brain produces two. The operator's stated requirement is the organ lines (directive 4),
   not the headline alone.
3. **The chassis stays; everything we add is incremental and gated to neutral.** Every
   organ, both expression channels, the executive, and the EA all collapse to bit-for-bit
   incumbent at their zero points (B0-EXPR hash check). Maximalism here adds measurement
   surface, not deployment risk: anything that fails its gate ships as B0/demoted and is
   REPORTED.

The design below is the largest brain for which every component's parameter count is
defensible against its effective sample, every component has a pre-registered kill, and
the verdict pair stays exposure-clean.

## 2. Organ set — six senses, six model types, every one on purpose

Adopt PROPOSAL_ORTHOGONALITY M1–M5 (architecture, targets, partitions, C2 gate, remediation
ladder, §6 sample table) with the amendments below, and ADD one sixth organ (M6) that
passes the same effective-sample honesty. Look-space recap + additions:

| Organ | Type (the named reason) | Axis / the LOOK the chassis lacks | Target | Horizon |
|---|---|---|---|---|
| M1 CAST-XS | **Transformer** — cross-asset attention IS the relative-strength question | WHO beats whom, fast | y5_rank (D→D+5 cross-sectional excess, Gaussian rank) | 5d |
| M2 ROT-GBM | **GBM** — split-based macro conditioning | WHO rotates next, slow + macro-conditioned | rank of open(D+5)→open(D+21) excess (horizon-disjoint, zero shared bars with M1) | 5→21d |
| M3 DISP-HAR | **HAR linear** — long-memory second moment | WHEN dispersion pays (tilt-timing scalar) | fwd 5d cross-sectional dispersion (log) | 5d, market-level |
| M4 EVT-NET | **Elastic net** — sparse linear event transmission | WHICH names are about to decouple (direction-free) | per-bucket \|excess\| exceedance (80th pct) | 5d, bucket-level |
| M5 POS-Z | **Rule, 0 params** — the only honest class at ~15–25 episodes | WHERE positioning is crowded | none (z>2 contrarian sleeve rule) | weekly→21d |
| **M6 GAP-GRU (new)** | **GRU** — sequence model over a 10-day microstructure window; recurrence is the right bias for path-shaped flow signatures, and it adds the one production-native model type (the chassis ensemble's own GRU) still absent from the brain | **HOW the buying happens** — overnight-vs-intraday return decomposition + volume/gap microstructure (institutional-vs-retail flow signature). No chassis component and no other organ reads the within-day decomposition; M1's partition loses gap/range/volume to keep disjointness | y1_z: next-day (D→D+1) cross-sectional excess, z-scored | **1d** |

**M6 honest sample arithmetic (the admission test the packet demands):** ~189k symbol-days
pooled F1–F6; breadth credit 12/64 (N1 convention, same as M1); horizon divisor **1** (no
target overlap at 1d) → effective sample ≈ 189k × (12/64) ÷ 1 ≈ **35,400 — the densest
target in the system** (5× M1's 7,100). Params: GRU(input 8, hidden 24, 1 layer) + linear
head ≈ 2.5k → **0.07 params/eff — SUPPORTED with the most room of any trained member.**
Features (disjoint by construction; removed from M1's partition): gap_open_pct, range_pct,
volume_z_21d + engineered from the imported OHLCV cache (open/high/low/close/volume,
2014→present): overnight_ret_1d, intraday_ret_1d, overnight_minus_intraday_5d_sum,
gap_fill_rate_21d, amihud_illiq_21d_z. Sequence length 10 days.

**M6's named risks, owned up front:** (i) **cost-hostility** — 1d edges (IC 0.03 × σ1
≈ 120bp ≈ 3.6bp gross) lose to 6–12bp RT if traded standalone; therefore M6 NEVER
generates standalone turnover — it enters the combiner only as entry-timing/ordering
refinement on tilts the other organs already fund (marginal cost ≈ 0; enforced: M6's mu
is zeroed for any name with no nonzero tilt from M1/M2/M5 that day). (ii) **C2 risk vs
M1 and member zero** — shares the target-object lever (return rank) though not horizon,
partition, or (vs member zero) any input column; the gate decides, and M6 sits at the
bottom of the trained-member seniority ladder: **M1 > M2 > M4 > M3 > M6 > M5** — first to
re-partition, first to demote. (iii) Falsifier: pooled OOF daily 1d rank-IC < 0.02, or
G-signal vs M1 > 0.7 post-remediation, or LOO utility ≤ 0 ⇒ demoted to challenger,
reported. M6 is the design's declared most-speculative member; its demotion costs nothing
upstream because nothing depends on it.

**Sixth-axis candidates considered and NOT taken** (same arithmetic, applied): seasonality/
calendar conditioning — a second market-level scalar competing with M3 for the same ~558
effective samples (the Orthogonality Engineer's breadth rejection generalizes; calendar
phase enters M3's predictor set as 3 dummies instead: turn-of-month flag, FOMC-week flag,
opex-week flag — capacity +3 coefs, gate-checked); cross-asset lead-lag — shares TWO levers
with M1 (target object, level) and same-day cross-asset structure is literally what M1's
attention computes; 63d rotation — ~46 time windows, unlearnable (already rejected on
arithmetic upstream, concur).

**Amendments to PROPOSAL_ORTHOGONALITY as adopted:**
- M1 partition shrinks by 3 columns (gap/range/volume → M6); M1 keeps return_1/5/21d,
  vol_21d, drawdown_21d, rel_strength_21d, dist_from_52w_high + sector/class embeddings.
  CAST capacity unchanged (≤16.5k params; the MARGINAL 2.3 params/eff verdict carries,
  TB-006 precedent + ridge-twin referee retained).
- M4 ships as a pre-registered two-variant decision (resolves the Orthogonality/Role-Finder
  conflict, §6 below): M4-A = trailing-vol control + GDELT columns only; M4-B = M4-A +
  LLM columns (masked pre-2024-08). Ship M4-B iff pooled OOF AUC(B) − AUC(A) > +0.01 on
  F5–F6 (the only LLM-covered folds); else ship M4-A and print the LLM-columns zero.
  Expected per the Role-Finder: M4-A ships.
- C2 gate matrix grows to 6 members + member zero = 21 directional pairs + scalar rows;
  identical statistics (G-signal, G-book ≤ 0.7 pooled / ≤ 0.8 per-fold; M1 vs member zero
  ≤ 0.8), identical remediation ladder, committed before any bake-off.

System param totals: ~20.3k trained (TB-006-order) vs 35.4k on the densest target; every
row at or under accepted ratios; the only marginal row (M1) is the carried TB-006 precedent.

## 3. Expression — hybrid: tilt is the verdict carrier, the socket is the WHICH channel

Engage PROPOSAL_EXPRESSION; adopt its mechanics wholesale (projection parity, lot fix +
unit test, expression log schema, B0-EXPR hash check, window recommendation). The
architect-level decision it leaves open — does the maximal brain take the socket as a
second expression channel — is decided YES, with this arithmetic:

- **Tilt (1b) carries the verdict.** Paired sd ≈ 6–15 bp/day → MDE 1.5–4.6 bp/day. Nothing
  is allowed to contaminate this: the verdict arm is tilt-only, strict support set.
- **The socket's WHICH bandwidth is real and otherwise unexpressed:** 6–15 bits on buy
  days of pure selection intelligence — the operator's literal question ("WHICH stocks") —
  that the tilt channel only approximates through Δw on already-chosen names. Its sd cost
  (~+30 bp/day while a swapped position diverges, ×2–5 paired-sd inflation) is fatal for
  certifiability but irrelevant if it never touches the verdict arm.
- **Resolution: three registered arms.** (i) VERDICT: incumbent vs incumbent+tilt (E1
  primary, E2 confirm). (ii) SECONDARY (pre-registered context read, honestly wider MDE
  ~8–15 bp/day, never promoted to verdict): incumbent vs hybrid (1c — permutation for
  selection on buy days + tilt for conviction daily, guard per 1a). (iii) ABLATION:
  permutation-only (1a) in the battery for selection-channel attribution. The maximal
  brain thus SPENDS the full bandwidth where it can be measured and keeps the certifiable
  channel clean — C3 and C4 simultaneously, by arm separation rather than compromise.
- `channel_mix` becomes a live gene (B0 = 1.0 = pure tilt) for the hybrid arm only; the
  verdict arm's mix is FIXED 1.0 (no gene may touch the verdict arm's parity geometry).

Organ→expression mapping (extends the Orthogonality §7 contract): M1+M2 directional mu
over support set; M3 scalar gain on T_t; M4 per-name width-shrink/veto; M5 slow sleeve
tilt inside `defensive_fraction` and the β/σ projection (the de-risking-in-disguise rail);
M6 entry-timing/ordering refinement on funded tilts only (zeroed elsewhere, §2). Conviction
c_t per PROPOSAL_EXPRESSION §2 (agreement + dispersion-forecast + organ confidences).

## 4. The executive — earned structure only: a state-conditioned linear gate with a twin

TB-006's lesson: the MLP executive lost to its linear twin on members too correlated to
need arbitration. With six engineered-orthogonal organs the organs WILL disagree and the
trust question becomes real. What structure earns its params on this sample:

**4.1 The regime-label trap, named.** The packet legitimately offers the chassis's own
5-state regime label as a conditioning input. Two measured facts kill it as a TRAINING
input: (i) labels exist only from 2025-08 (inference probs absent earlier) — the F1–F6
fitness era has none, so a label-conditioned gate cannot be trained out-of-fold; (ii) the
live-era label flips 20×/28 days (~74%/day, Role-Finder §3.6) — per-state trust conditioned
on a near-daily-flipping label is conditioning on the picker's gait noise. **Decision:
condition on the label's reconstructible INPUT SPACE, not the label** — observables
computable 2014→present from imported caches: vol tercile (21d book vol), trend sign
(63d SPY), dispersion tercile (M3's own input series), credit-spread z. The trained gate
sees the same information the regime picker digests, without the label's availability gap
or flip noise. The label itself appears in the REPORT as a diagnostic cross-tab (trust
pattern vs live-era label) — read, not trained on.

**4.2 Architecture and arithmetic.** Three candidate structures, decided by params/sample:
- Small attention over 6 organ tokens: ≥300–500 params vs ~300 effective conditioning
  samples (1,507 OOF days ÷ 5) — **REJECTED on arithmetic** (≥1 param/eff for a meta-layer
  whose TB-006 ancestor lost to linear at better ratios).
- Regime-label-conditioned trust matrix (6×5): untrainable per 4.1 — REJECTED.
- **ADOPTED: linear state-conditioned gate.** τ_k(t) = softmax_k( b_k + w_k·state_t ),
  state_t ∈ R^4 (the 4.1 observables, z-scored), params = 6×(1+4) = **30 vs ~300 effective
  → 0.1 params/eff** — the largest executive this sample supports. Trained by gradient on
  the same floored paired-IR surrogate objective the EA uses (fold-pooled, OOF inputs,
  ridge penalty toward the state-independent solution), NOT by the EA (30 params on 6
  fold-scores would be noise-mining; the EA holds only the static prior).
- **The twin control is mandatory (TB-006's lesson, institutionalized):** the gate ships
  only if it beats BOTH (a) the static-trust baseline (EA's organ_trust alone) and (b) a
  frozen-at-init copy of itself, on fold-pooled paired utility, clearing the same
  1×cross-fold-sd adoption gate the EA champion faces. Either failure ⇒ static trust
  ships and the scorecard prints `executive_gate≈0 (measured)`.
- Composition: final trust = softmax( organ_trust_EA + gate_strength · (gate logits) ),
  `gate_strength` ∈ [0,1] an EA gene (B0 = 0) — so the rotation gate (§5) also gets a vote
  on whether conditioning generalizes, and the neutral genome still recovers the incumbent
  bit-for-bit.

## 5. Evolution — PROPOSAL_EVOLUTION_007 adopted; the genome grows by exactly two genes

Adopt in full: floored paired-IR fitness (verdict's own random variable, de-risking removed
from the action space), min over cost×{1.0,1.5}, worst-fold floor, λ_reg shrinkage to B0;
LOFO rotation gate on the SEARCH (≥4/6 unseen folds ΔU ≥ 0 AND pooled mean > 0);
median-of-top-8 deduped champion; B0-in-population; B1 budget-matched random search; both
gates AND; B0 ships on any failure; every variant logged; FITNESS(B0) ≡ 0 anchor.

**Genome for the maximal brain — each addition defended against the generalization gate:**

| Change vs proposal | Genes | Defense |
|---|---|---|
| organ_trust grows to M=6 | +1 (6 total) | same gene class the proposal already sized for M ≤ 6; rotation gate covers it |
| `channel_mix` activates (hybrid arm only) | 0 new (was conditional) | B0 = 1.0 (pure tilt); verdict arm unaffected (FIXED 1.0 there); one float on a bounded blend |
| `gate_strength` (executive coupling, §4.2) | +1 | B0 = 0; lets the rotation gate kill state-conditioning if it doesn't carry to unseen folds — the gene IS a control |
| `m6_timing_gain` ∈ [0,1], B0 = 0 | +1 | M6's only dial (how much entry-timing refinement); zero-cost channel (§2), bounded, shrunk to 0 by λ_reg unless it pays |
| Per-organ-per-tier trust matrices (6×3 = 18 genes) | **REJECTED** | 18 extra genes scored on 6 fold-samples is exactly the §2.1 boundary-pinning machine; tier handling goes to FIXED priors (§7) where it is measured, not evolved |
| Ballast cap | FIXED 0 (verbatim) | NetEdge −8 to −17 bp/bet; not negotiable |

Total: 6 trust + tilt_gain, conviction_temp, dead_zone, cap_core, cap_conditional,
defensive_fraction, event_veto_threshold + event_damp_strength (now bound to M4's
exceedance output, not the retired LLM channel), parity_tol_gross/risk, channel_mix,
gate_strength, m6_timing_gain = **20 genes ≤ 24 hard cap.** Budgets, cadence, controls,
and the 90-min wall-clock cap carried verbatim; rotation runs are never the thing trimmed.

## 6. LLM — the R5 retirement verdict, engaged honestly

The Role-Finder's 46-screen, 0-FDR-survivor decomposition is accepted as binding evidence:
**no llm_* feature enters any member's partition by default, and no LLM pathway is wired
into the live decision path on hope.** The maximal brain keeps exactly three LLM surfaces,
all measured or measuring:

1. **Inside M4, by measurement not fiat (§2 amendment):** the M4-A/M4-B AUC decision is the
   one place the LLM gets to EARN partition space, with the bar (ΔAUC > +0.01, F5–F6
   pooled OOF) pre-registered before training. Expected outcome: M4-A (GDELT-only) ships
   and the LLM-columns zero is printed. This discharges "every type at max value, wherever
   it is" by arithmetic — the Role-Finder's screens tested the LLM features directly; the
   exceedance-target-with-GDELT-context configuration is the one untested corner, and it
   gets one pre-registered look, not a license.
2. **The one admissible NEW role, as a falsifier arm (the packet's named candidate):**
   R-LLM battery arm = verdict-arm brain + disagreement→tilt-width conditioner (Role-Finder
   §5.1 exact spec: unweighted cross-bucket sent std, 252d z, g(z) = clip(1+0.25·tanh z,
   0.75, 1.25) on tilt WIDTH only — C1-safe by construction). **Pre-registered kill: E1
   paired t < +2.0 vs the unconditioned brain ⇒ dead, never touches holdout.** Expected
   effect O(0.1–0.5) bp/day vs E1 MDE — the arm exists to discharge directive 4 by
   measurement; its expected zero is printed in the pre-registration, not discovered later.
3. **Monitoring emission stays** ($0.14–0.20/mo): the record accrues ~22 scored days/mo
   toward the 900-day re-test trigger (verbatim Role-Finder spec); de-chattering (onset
   encoding, geopol demoted to axis, severity ≥ 0.7) applied to the monitoring surface,
   $0 re-scoring.

Scorecard line if both measured looks zero out (expected): `LLM organ: 0 (measured) —
retired from decision path; two pre-registered re-entry tests failed at their bars; $0.14/mo
monitoring + 900-day re-test trigger live.` That sentence IS the value-maximizing role on
this record.

## 7. Universe handling — concentrate where measured, tiered priors, ballast frozen

- **Tilt core (10):** ITA SOXX XRT TLT AGG MUB FXE USO FXI RSP (Universe Selector §4,
  adopted verbatim incl. FXE size-cap and FXI/USO CAST-only flags). `cap_core` applies.
- **Conditional adds (4): VIXY IYR SHY KRE at `cap_conditional` < cap_core.** The maximal
  brain takes all four: the β/σ projection prices VIXY's risk structurally (its huge σ̂
  makes any VIXY leg tiny at matched σ-budget — the Expression Architect's machinery IS
  the risk treatment the Universe Selector asked for). GLD = named first substitute if a
  core name is demoted at member-acceptance.
- **Ballast (the remaining 50): untiltable, cap FIXED 0** — not a gene (§5). The chassis
  holds them as it always has; the brain spends nothing there.
- **Per-symbol skill weighting — YES, as fixed measured priors, tier-grained:** per-organ
  per-name multiplicative masks on mu before combination. Rule pre-registered NOW, numbers
  filled at member acceptance from the NEW members' own F1–F6 OOF per-name stats (the
  Phase-0 table was measured on TB-006 members; M2's new D+5→D+21 skill map is unmeasured
  until retrained): weight 1.0 where that organ's per-name 90% CI > 0 in both halves;
  0.5 where fold-consistent (≥5/6) but CI-spanning; 0 outside core+conditional. Tier
  membership only — never fine rank (Friedman/Kendall evidence: tiers are signal, rank
  3-vs-7 is noise). NOT evolved: evolution sets how hard to lean (caps, trust), never
  re-derives where (carried from the Evolution proposal, extended to the mask).
- 2025 backfill leg: only TLT/USO (+VIXY/SHY conditional) of the tilt set exist there —
  one more reason it is context, never verdict (§8).

## 8. Training plan, surrogate, anchors, window

- **Data:** all TB-006 final-grade assets imported as-is (audit §C); panel target slices
  regenerated: y_rot rank (D+5→D+21), dispersion target, bucket exceedance, y1_z (exists),
  M6 microstructure block from the OHLCV cache. GDELT/LLM features imported, no re-ingest.
- **Folds:** F1–F6 carried, 2026-03-11 holdout firewall untouched. **Embargo amendment:**
  M2's target extends to D+21 → its folds get a 26-td purge/embargo (21+5) instead of the
  blanket 21; all other members keep 21. The C2 gate and all OOF matrices rebuild on the
  new member set; gate runs before any bake-off and after any retrain.
- **Order of operations:** members train → per-name OOF stats → tier masks frozen (§7) →
  C2 gate (+ remediation ladder if needed) → executive gate trained + twin test (§4) →
  EA production run + 6 LOFO rotations + B1 → gates → champion-or-B0 → bake-off replays.
- **Chassis surrogate (fold-era paired fitness):** the Evolution proposal's vectorized
  two-pass walk, adopted; extended to represent both channels (top-N selection makes the
  permutation half representable for the hybrid arm's genome too). Base-arm pass cached
  per fold; only tilt arms run per genome.
- **Anchor checks (pre-registered, before the first EA generation):** (a) neutral-recovery:
  B0 brain arm vs incumbent on the real harness, assert identical trade lists +
  sha256(timeline) match; (b) sign-anchor: champion and B0 through the REAL replay engine
  on the artifact window inside the fitness era (2025-08-04→2026-02-06) — **this requires
  the epoch-aware prices shim, so the shim is built regardless of the verdict-window
  decision** — surrogate-vs-real paired-ΔU sign agreement or stop-and-fix (a finding).
- **Window strategy:** verdict = **live era only** (2026-01-31→present: E1 full-live read
  n≈66 deltas; E2 holdout n≈43), per the Expression Architect's dilution arithmetic (the
  2025 leg can express 2/10 core names — adding n with mean ≈ 0 lowers t). The 2025 leg
  runs ONCE as a labeled context read (the shim exists anyway) with the repaired-timeline
  + backfill caveats printed; never pooled into the verdict.

## 9. Pre-registration sketch (TOURNAMENT_007 frame; exact numbers committed in Phase B)

**Verdict hierarchy, honest to the ceiling arithmetic (expected t at the Grinold ceiling
is 0.6–0.9; the graded ladder is designed so the honest outcomes are informative):**
- **E1 primary (full live window, n≈66 paired deltas, tilt verdict arm):**
  BEATS ⇔ paired t ≥ +2.0 (packet floor). TIES-POSITIVE ⇔ t ∈ (0, 2.0) AND 90% CI
  excludes −2 bp/day (the certifiable "real but small" cell the MDE makes readable).
  TIES ⇔ CI straddles ±2 bp/day. LOSES ⇔ t ≤ −2.0.
- **E2 confirm (holdout n≈43, ONE read, orchestrator-fired):** BEATS additionally requires
  E2 point estimate ≥ 0; an E1 BEATS with E2 < 0 downgrades to TIES-POSITIVE with the
  disagreement printed. E2 never upgrades.
- **Headline line:** the packet's stop-condition sentence, with ΔSharpe + paired t + the
  repaired-timeline caveat.
- **Per-organ battery (the maximal brain's bigger battery, sized against caps):** replay
  arms = incumbent, verdict-tilt, B0-EXPR (hash check), hybrid-secondary, perm-only
  ablation, LOO ×6 (drop_M1..drop_M6, τ renormalized, separate processes), B-disp (M3
  constant control), R-LLM width-conditioner falsifier = **14 arms ≈ 14 × ~67-date replays
  + up to 3 contingent arms if residualization fires (ordered-pair LOO)**. TB-006 ran 19
  battery replays; 14–17 fits the precedent cap. Forecast-altitude falsifiers (per organ,
  §2/§5 of PROPOSAL_ORTHOGONALITY + M6's) run offline on OOF — no replay cost. Three-valued
  organ verdicts; per-arm MDE printed next to every read.
- **Multiplicity ledger:** every screen/read enumerated; BH-FDR(10%) across the per-organ
  forecast battery; the E1/E2 hierarchy holds the family-wise verdict; look ledger appends
  E2 = 1 read, EA = both gates, executive = twin test, M4 = A/B decision, R-LLM = 1 arm.
- **Pre-committed sentences** (Evolution §4 verbatim + executive-twin + M4-A/B + R-LLM
  kill + M6 demotion criteria), all written before Phase C code.

## 10. Cost + build wall-clock

- **Bedrock / LLM spend: $0.00 new** (all LLM surfaces run on existing artifacts; the
  falsifier arm consumes llm_features.parquet; monitoring emission is the existing
  ~$0.14–0.20/mo). Default $3.00 cap untouched — nothing to flag.
- **AWS: $0** (local Mac throughout; no new resources per write surface).
- **Deployed shape if adopted later:** unchanged Lambda + one ~5 KB genome JSON + organ
  inference (M1–M6 are all sub-second CPU models); ≤ $10/mo envelope holds with margin.
- **Wall-clock (TB-006 reuse honest):** shims + lot fix + unit tests ~0.5 day; panel target
  regeneration ~0.5 day; member training (CAST retrain hours-scale, GBM/HAR/net minutes,
  GRU ≤ 1 hr) + OOF + C2 gate ~1 day; executive gate + twin ~0.5 day; EA production +
  rotations + B1 ~20–55 min compute inside a 0.5-day step; 14–17 battery replays at
  TB-006 per-run minutes ~0.5–1 day; evidence assembly ~0.5 day. **Total ≈ 3.5–4.5
  working days**, vs TB-006's from-scratch build — cheaper, as the packet requires. The
  maximal increment over a lean build is ~1–1.5 days (M6 + hybrid arm + executive twin +
  3 extra LOO arms).

## 11. Honest weaknesses — where maximalism loses to a leaner design

1. **Multiplicity burden is real and paid in power.** Six organs × two altitudes + M4-A/B
   + R-LLM + hybrid ≈ a 2–3× bigger look ledger than a two-organ brain; BH-FDR across it
   raises every per-organ bar. The lean design certifies fewer things harder.
2. **Diluted per-organ utility power.** T_max ≤ 10% split across six senses ⇒ each organ's
   marginal contribution is plausibly 0.5–3 bp/day against per-LOO-arm MDEs of ~2–4 bp/day
   — several LOO verdicts will honestly read INDETERMINATE at available power even if the
   organs are real. (Mitigation, not cure: forecast-altitude falsifiers stay powered; the
   trust weights concentrate budget away from weak organs, so dilution is bounded by the
   EA, not uniform.) A 2-organ brain concentrating the full budget on CAST-core names
   would have ~2× the per-organ utility resolution.
3. **Build risk scales with organ count.** Six members + two channels + gate + shim + EA
   rotations is the most moving parts this project has shipped; the mitigations are the
   zero points (everything fails to B0/incumbent) and the strict gate ladder, but
   integration debugging time is the budget line most likely to overrun.
4. **M6 is speculative by design** — the one organ admitted on arithmetic + mechanism
   rather than a measured receipt; prior probability of demotion is material and is the
   declared cost of testing the operator's "find a configuration where each type brings
   value" at full breadth.
5. **The headline ceiling is unchanged.** No organ count fixes n=66; if the committee's
   sole success criterion were E1 BEATS at t ≥ 2, the lean design is equally (un)likely to
   reach it and cheaper. This design's claim is that the certifiable PRODUCT — six organ
   verdicts, a tight headline CI, a measured executive, a measured LLM zero — is what the
   operator actually asked for, and it is bigger here.

---

## Bottom line

Six organs / six model types, every one with a named look, a disjoint partition, honest
sample arithmetic (new M6 GAP-GRU: 35.4k effective vs 2.5k params, junior-most, kill
pre-registered), all behind the C2 gate with the deployed ranker as member zero. Expression
= hybrid by arm separation: tilt-at-matched-gross+β+σ carries the verdict (MDE 1.5–4.6
bp/day), the ranking-socket permutation runs as the registered WHICH-channel secondary
where its sd cost can't touch certifiability. Executive = 30-param state-conditioned linear
gate (regime-label inputs reconstructed as observables; the label itself is untrainable —
absent pre-2025-08 and flipping 74%/day live) behind a mandatory linear/frozen twin test;
attention rejected on arithmetic. EA = PROPOSAL_EVOLUTION_007 verbatim + 2 defended genes
(gate_strength, m6_timing_gain), 20 ≤ 24. LLM = R5 retirement honored; presence reduced to
two pre-registered measurements (M4 ΔAUC, disagreement→width falsifier with E1 kill) and a
$0.14/mo monitoring line. Universe = 10-core + 4-conditional with measured tier masks,
ballast frozen at 0. Verdict = live era, E1-primary t ≥ +2.0 / TIES-POSITIVE cell /
E2-confirm-one-read, 14–17 battery arms inside precedent caps. $0 new Bedrock; ~3.5–4.5
days build. Declared losses to the lean prior: multiplicity, per-organ LOO dilution,
integration risk, and one speculative organ — bought deliberately, because more orthogonal
senses is the only lever that raises expected t at fixed n, and per-organ certifiability is
the operator's actual ask.

— end —
