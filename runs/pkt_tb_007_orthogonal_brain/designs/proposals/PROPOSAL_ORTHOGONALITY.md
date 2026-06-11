# PROPOSAL_ORTHOGONALITY — PKT-TB-007 Phase A, Orthogonality Engineer (role 3)

**Date:** 2026-06-10 · **Inputs:** ANALYST_AUDIT.md, UNIVERSE_EXPLOITABILITY.md (+json),
TB-006 BUILD_SPEC §5, REVIEW_TRAINING_REALIST_PHASE_D §9, ATTACK_TRAINING_REALIST §0
(N1–N4 axioms), store/panel_meta.json.
**Binding constraint served:** operator directive 4 (every model type present ON PURPOSE,
non-redundant) and C2 (pre-registered decorrelation gate ≤ 0.7 before the bake-off).

**TB-006's failure, stated as a design law:** four architectures were pointed at ONE
target family (5d cross-sectional excess) and converged to one book in four costumes
(solo-book corr 0.941). Same target ⇒ same book; architectures do not buy orthogonality,
**questions do**. TR Phase D §9.3 is the prescription this proposal implements: different
targets, different horizons, disjoint feature partitions — measured by a gate, not assumed.

---

## 0. What the regime picker and the deployed ranker already see (the orthogonality datum)

Two chassis senses define the complement we must fill:

1. **The regime picker** emits a per-day, market-level categorical (5 states) consumed as
   buy-threshold + multiplier + PSM. It is constant across the cross-section on any given
   day ⇒ it carries **exactly zero bits about WHICH symbol beats which**. Any
   cross-sectional look is orthogonal to it by construction. Where it is NOT trivially
   orthogonal is market-level second moments (it responds to vol/trend), so the one
   market-level member below (M3) carries an explicit anti-redundancy check against the
   regime multiplier series.
2. **The deployed RankingMLP** (blend 0.35) is a 10-feature price-momentum
   cross-sectional ranker (`return_{1,5,21,63}d, vol_{21,63}d, drawdown_63d, trend_63d,
   rel_strength_{21,63}d` — `training/models/ranking_mlp.py:12`). The chassis already owns
   a generic momentum look. Every member's correlation against THIS score is part of the
   C2 matrix (member zero, §3.2) — a brain member that reproduces the deployed MLP adds
   nothing even if it is orthogonal to its siblings.
3. **Nothing in the production chassis ingests positioning data** (grep: COT/CFTC appears
   nowhere in `src/` or `training/`), event/news data, or any dispersion construct. Axes
   4–5 below are senses the chassis has no organ for at all.

## 1. The look-space decomposition (five axes, five questions)

| # | Axis (the question) | Horizon | Level | Why the regime picker is blind to it |
|---|---|---|---|---|
| A1 | WHO beats whom, fast (5d relative strength) | D→D+5 | cross-sectional, 64 | market-level label has zero cross-sectional bits |
| A2 | WHO beats whom, slow (post-week rotation, macro-conditioned) | **D+5→D+21** | cross-sectional, 64 | same; and the regime label has no macro-interaction structure |
| A3 | WHEN does dispersion pay (tilt-aggressiveness meta) | D→D+5 | market-level scalar | regime sees vol LEVEL; dispersion is the market-removed second moment of the cross-section (corr to VIX partial, ~0.5; member is trained on the increment, gated vs the regime series) |
| A4 | WHICH names are about to decouple (idiosyncratic event risk) | D→D+5 | bucket-level (27), direction-free | regime/ranker ingest no event data; target is a second moment, not a return |
| A5 | WHERE is positioning crowded (contrarian flows) | ~weekly, expressed at 21d | sleeve-level (3 futures complexes) | chassis ingests no COT at all; weekly cadence below the regime picker's clock |

**Pairwise orthogonality is engineered on three independent levers at once** — target
object (return rank / later-window return rank / dispersion scalar / |abnormal move|
exceedance / positioning z), horizon (5d / 5→21d / 5d / 5d / weekly), and feature
partition (§2, disjoint). No pair shares more than one lever.

**The single highest-leverage choice — horizon-disjoint target windows:** A2's target is
the **5-days-later window** `open(D+5)→open(D+21)` cross-sectional excess, NOT
`open(D)→open(D+21)`. The TB-006 failure pair (5d and 21d-from-today) mechanically share
their first 5 days of forward return; D+5→D+21 shares **zero bars** with A1's D→D+5.
Residual correlation can then only come through genuine return autocorrelation — which is
the rotation signal A2 exists to capture — not through target-construction overlap. This
buys decorrelation in the target definition itself, with no leakage machinery to audit.

**Why each is learnable here (sample arithmetic in §6; evidence pointers):**
- A1 is CAST's existing receipt: purged weekly rank-IC **0.105** (ridge twin −0.009),
  per-name CI-significant on 12 names (UNIVERSE_EXPLOITABILITY §3).
- A2's parent (GBM on 5d, macro-conditioned) is TB-006's only E1-positive organ (HAC t
  +2.48, sign-positive in all five defined folds) with per-name skill **disjoint from
  CAST's** (XRT/INDA/XLY/KRE vs ITA/SOXX/bonds/FX/commodities) — symbol-level
  orthogonality already measured on the SAME target; moving it to the later window and a
  slow/macro partition widens it.
- A3: cross-sectional dispersion is strongly autocorrelated (vol-clustering family), the
  canonical HAR fit; predictor set (COR3M implied correlation, VVIX, breadth dispersion)
  exists in T/Z with no new ingestion.
- A4: B-matrix (27×26 GDELT+LLM bucket features, 2015→) exists; exceedance targets are
  denser in signal than signed bucket returns (direction-free ⇒ no sign noise), and this
  is the packet-sanctioned value-maximizing LLM role (item 4: "event-risk conditioning?
  tilt veto?") after TB-006 bounded its direction value at ≤0.
- A5: CFTC TFF cache (ES/UST10Y/VIX, publication-keyed) exists; the look is deliberately
  capacity-zero (§2, M5) because its honest sample is tiny.

Axes considered and NOT proposed: breadth/internals as a sixth member (folded into A3's
predictor set — as a separate member it would be a second market-level scalar competing
for the same ~558 effective samples); vol-level forecasting (RiskNet's job, instrument
not member, carried as-is from TB-006); 63d rotation (N2: h=21 already cuts time samples
to ~140 windows; 63d → ~46, unlearnable cross-sectionally — rejected on arithmetic).

## 2. Model-type assignment ON PURPOSE (each type named to its axis)

| Member | Type (named reason) | Axis | Target (exact) | Feature partition (disjoint by design) |
|---|---|---|---|---|
| **M1 CAST-XS** | **Transformer** — cross-asset attention is the only architecture here that lets symbol s's score condition on the other 63 tokens the same day; that IS the relative-strength question | A1 | `y5_rank`: per-day Gaussian-rank of open(D)→open(D+5) return minus cross-sectional mean (TB-006 target, kept) | X **fast block only**: return_1d, return_5d, return_21d, vol_21d, drawdown_21d, rel_strength_21d, range_pct, volume_z_21d, gap_open_pct, dist_from_52w_high (10 of 14 X cols) + sector/class embeddings. NO 63d stats, NO macro, NO event, NO COT |
| **M2 ROT-GBM** | **GBM** — axis-aligned splits + monotone constraints natively express conditional macro interactions ("small-cap rotation IF credit easing AND breadth thrust"), which attention and linear models express poorly at this sample size | A2 | per-day Gaussian-rank of open(D+5)→open(D+21) cross-sectional excess | **slow per-symbol block**: return_63d, vol_63d, drawdown_63d, rel_strength_63d + T context: ctx_rate_2y/10y, yield_slope, credit_spread, risk_off, spy_ret_21d, spy_vol_21d, fred_nfci, stlfsi4, hy_oas(+d5), icsa_z, t10yie_d21, dfii10(+d63) + br_adv_dec, br_pct_above_50d/200d, br_rsp_spy_21. NO ≤21d per-symbol momentum, NO vol-surface, NO event, NO COT |
| **M3 DISP-HAR** | **HAR (linear time-series)** — dispersion is long-memory; HAR's three-timescale AR structure is the correct ~10-coef inductive bias, and at ~558 effective samples (§6) anything richer is unsupported; the "simple twin" IS the model | A3 | forward 5d realized cross-sectional dispersion: sd over the tilt set of open(D)→open(D+5) returns (demeaned), log-scaled | realized dispersion lags (1d/5d/21d, computed from open_px), **vol-surface block exclusively**: s1_cor3m_z, vvix_z, skew_z, vix_term_slope, vxn_minus_vix, s1_curvature + br_dispersion. NO per-symbol features, NO event, NO COT |
| **M4 EVT-NET** | **Elastic net** — event transmission is sparse and approximately linear in engineered feature space; L1 picks WHICH of 26 bucket features transmit, with honest capacity at the bucket sample size | A4 | per-bucket exceedance: 1{ \|5d market-removed bucket return\| > trailing 80th pct for that bucket } — direction-free decouple risk | **B[27,26] exclusively** (GDELT G1–G5 + LLM sentiment/conf/salience/event one-hots + availability masks, mask-interaction form per TB-006 SK A-K3) + trailing bucket abnormal-vol control. NO price cross-section, NO macro, NO COT |
| **M5 POS-Z** | **Z-score rule (capacity ≈ 0)** — present on purpose BECAUSE the honest sample (~15–25 crowding episodes since 2015, §6) cannot support learning; a pre-registered rule is the only intellectually honest model class for this axis | A5 | none trained — rule: leveraged-net z (156-week window) on ES / UST10Y / VIX TFF; \|z\| > 2 ⇒ contrarian sleeve tilt (equity / duration / vol-adjacent sleeves), decays over 21d | **cot_es_lev_net_z, cot_ust10y_lev_net_z, cot_vix_lev_net_z exclusively** (publication-keyed, m_cot_fresh mask). Nothing else |

Every required type is present with a load-bearing reason: transformer (cross-asset
attention ↔ relative strength), GBM (split-based conditioning ↔ macro interaction),
linear/elastic net (sparse linear transmission ↔ events), HAR sequence structure
(long-memory ↔ dispersion), rules (sample-starved axis ↔ zero capacity). The LLM lives
inside M4 in its packet-item-4 role (event-risk conditioning), not as a tone-direction
proxy (TB-006 bounded that at zero-to-negative).

**Partition disjointness, stated flat:** across M1–M5 the input column sets are pairwise
disjoint (verified against panel_meta T/B/Z/X col lists; the only near-touch is M1's
vol_21d [per-symbol] vs M3's dispersion lags [cross-sectional sd] — different objects,
different level). Phase 0 already showed CAST and GBM hold disjoint per-symbol skill on
OVERLAPPING features; disjoint partitions can only widen that. The deployed-chassis
overlap is owned honestly: M1's 10 fast features share 6 of the deployed MLP's 10 — M1
is the one member whose job is to BEAT an existing chassis sense rather than add a new
one, and its C2 row vs member zero (§3.2) is the receipt either way.

## 3. The C2 gate, operationalized

### 3.1 Statistics (two spaces, both gated — pre-registered in TOURNAMENT_007 before any member trains)

- **G-signal (signal space):** for each directional pair (m,n) ∈ {M1, M2, M4, M5, member
  zero}: per-day cross-sectional Spearman corr of standardized signals s_m[D,·] vs
  s_n[D,·] over the common 64-symbol space, averaged over all OOF validation days of
  F1–F6 (n=1,507). M4 maps bucket→symbol via bucket_map (its symbol-space signal is the
  negated exceedance prob — a width/veto signal); M5 maps sleeve→symbol. Days where M4/M5
  abstain (no event mass / |z|≤2) are excluded from that pair's average and the inclusion
  rate is printed (an organ that is orthogonal only because it is silent gets caught by
  the conditional read).
- **G-book (book space — the apples-to-apples TB-006 statistic):** pairwise Pearson corr
  of daily solo-book returns under the TB-006 fixed solo-book rule (BUILD_SPEC §6,
  unchanged), pooled over F1–F6 OOF days. This is the exact statistic that read **0.941**
  in TB-006; the gate is the same number ≤ **0.7**.
- **G-scalar (the M3 row + chassis-redundancy row):** M3 emits a daily scalar, so its
  cross-sectional corr is degenerate. Its row is: time-series Pearson corr of its
  aggressiveness series vs (a) each member's daily conviction/|tilt| series, (b) the
  regime picker's daily multiplier series, (c) VIX level — all ≤ 0.7, pooled F1–F6. Row
  (b) is also computed for M5 (slow tilts could shadow the regime state).

### 3.2 Ceiling, granularity, cadence

- **Gate:** pooled |ρ| ≤ **0.7** for every pair in every space above, AND no single fold
  > **0.8** (a pooled average must not mask a fold where two members collapse into one).
- **Member zero:** the deployed RankingMLP score is row 0 of the matrix. Its rows are
  **reported, not gated** for M2–M5 (they should be near zero trivially); for **M1 it is
  gated at ≤ 0.8** — softer than sibling pairs because M1 intentionally shares the
  momentum family; if M1 vs member-zero > 0.8, M1 cannot beat 0.35-blend redundancy and
  goes to remediation like any other failure.
- **Cadence:** computed once per member-acceptance round, on OOF predictions only,
  strictly **before** any bake-off replay and before the EA sees any member output. The
  full matrix (all pairs × {pooled, per-fold} × both spaces) is committed to
  `designs/c2_gate_matrix.json` + printed in the report. Re-run after ANY member retrain.
- **No holdout contact:** the gate lives entirely on F1–F6 OOF. Holdout days are never
  touched (E2 discipline unchanged).

### 3.3 Remediation ladder (pre-registered ORDER — no ad-hoc paper-over)

Seniority for contested resources: **M1 > M2 > M4 > M3 > M5** (ranked by measured
standalone evidence: IC 0.105 > E1 +2.48 > new-but-data-rich > new-scalar > capacity-zero).
When a pair fails, apply to the JUNIOR member, one rung at a time, re-running the gate
after each rung:

1. **Re-partition:** remove the junior's columns most correlated with the senior's signal
   (top-3 by |corr|), retrain. One pass only.
2. **Re-target / residualize:** junior retrains on the orthogonalized target (§4).
3. **Horizon shift:** junior's window moves further out (e.g. M2 → D+10→D+31). One pass.
4. **Demote to challenger:** the junior is excluded from the brain arm. It still runs in
   the battery as a non-verdict challenger book and its redundancy is REPORTED in the
   committee report (C2's "never papered over" clause). The brain ships with ≥4 members
   present; if demotions would take it below 4 model types, the report says so and the
   bake-off proceeds with the honest set — non-redundancy outranks the headcount.

## 4. Redundancy-to-value conversion (the orthogonalization option)

When rung 2 fires, the junior member is retrained on **residual targets**, with this
leakage discipline:

- **Construction:** junior target for day D in fold f becomes
  `y_junior_resid[D,s] = y_junior[D,s] − β̂_f · ŷ_senior[D,s]`, where `ŷ_senior[D,s]` is
  the senior's **walk-forward OOF prediction** for D (model trained with fold f and its
  22-td embargo excluded — these already exist as the OOF matrices) and β̂_f is fit on
  training folds only. The junior never consumes a senior prediction whose training data
  included D or its embargo neighborhood. For head-training rows before F1, the senior's
  expanding-window predictions (trained strictly on data before each row) are used; rows
  where no honest senior prediction exists are dropped, and the drop count is printed in
  the manifest.
- **Manifest:** the junior's `walk_forward: true` manifest gains a
  `residualized_against: {member, oof_sha, beta_per_fold}` block; the executive refuses a
  residualized member without it (same build-error pattern as TB-006 §5).
- **Expression consequence (flagged to the Expression Architect):** a residualized member
  predicts the senior's ERROR, so its signal is only meaningful blended WITH the senior;
  the two become an ordered pair in the book composition, and the LOO battery must drop
  them as a pair AND individually (3 arms, not 2) for attribution to stay readable.
- **What is NOT allowed:** training the junior on in-fold senior predictions (circular),
  or "stacking" where the senior is refit after seeing the junior (the boosting loop
  stops at one pass — sample budgets do not support an iterated stack).

Preferred over residualization wherever available: **structural separation** (the §1
horizon-disjoint windows, the §2 disjoint partitions) — already in the base design, which
is why residualization is a remediation rung, not the architecture.

## 5. Per-organ value lines (where value shows up + what kills each member)

| Member | Value should show up at... (from UNIVERSE_EXPLOITABILITY) | Falsifier (pre-registered; all E1-style fold-pool reads + LOO arm) |
|---|---|---|
| M1 CAST-XS | Tilt core: ITA (+.148), FXE (+.134), RSP (+.119), USO (+.116), FXI (+.113), SOXX (+.106), TLT/AGG/MUB; tier membership, not fine rank (Friedman p=.012; fold-pair ρ≈.09) | purged weekly rank-IC < 0.02 ⇒ pre-registered failure prints; forecast-space IC delta vs member-zero ≤ 0 (it must beat the deployed MLP's look, not just exist); LOO arm `drop_M1` |
| M2 ROT-GBM | XRT (+.111), INDA (+.095), XLY (+.086), KRE (6/6 folds) — dispersed non-clone names, macro-conditioned; the **GBM confirmation arm** (TR §9.2) doubles as its prior-evidence check on newly accrued fold F7 | fold-pool rank-IC at the D+5→D+21 target ≤ 0; E1 LOO `drop_M2` CI excludes positive; if the 21d look only re-discovers the 5d look (G-signal vs M1 > 0.7 post-remediation) it demotes |
| M3 DISP-HAR | Timing, not direction: tilt budget spent harder on high-dispersion-forecast days; value concentrated in regimes where COR3M decouples from VIX | OOS forecast R² vs trailing-21d-mean baseline ≤ 0 on F1–F6 ⇒ member replaced by the constant (and that constant arm B-disp is the LOO control); utility read: dispersion-gated tilt vs fixed-gain tilt, fold pool |
| M4 EVT-NET | Event-prone, decouple-heavy names: FXI, EWZ, USO, KRE, XBI/IBB; value = avoided losses / widened uncertainty on decouple days; this is the LLM's one paid role | OOF exceedance AUC ≤ 0.55 (vs trailing-vol-only control net — the LLM/GDELT columns must add AUC over the vol control, else honest zero for the event data, again); LOO `veto_off` arm |
| M5 POS-Z | Positioning extremes only: \|z\|>2 episodes, sleeve-level (TLT/SHY duration sleeve, VIXY-adjacent vol sleeve); ~15–25 episodes ⇒ verdict will likely read **indeterminate at available power** — that is the honest expected outcome and is acceptable for a 0-param organ | per-episode sign tally ≤ 50% on F1–F6 episodes ⇒ rule disabled in the ship config (reported); LOO `drop_M5` |

All five falsifiers run at **forecast altitude AND utility altitude** (TR §9.4: utility-only
attribution under a conservative genome measures the genome, not the organ). Each LOO arm
gets a per-arm MDE printed (TR Phase D §2 nit, fixed by construction this time).

## 6. Sample-budget table (TB-006 honest-arithmetic convention: N1 breadth 12/64
cross-sectional, ÷h overlap, ATTACK_TRAINING_REALIST §0 master conversion)

| Component | Rows (pooled F1–F6 era) | Effective sample | Trained params | Ratio & verdict |
|---|---|---|---|---|
| M1 CAST-XS | ~189k symbol-days | 189k × (12/64) ÷ 5 ≈ **7,100** | ≤16.5k (CAST-Small as shipped, wd 1e-3, rank target, aux head) | ~2.3 params/eff — **MARGINAL, accepted on TB-006 precedent** (shipped at this ratio and produced IC .105 with ridge twin at noise; the anti-memorization spine + heavy shrinkage are the escape; ridge twin re-trained as referee, unchanged) |
| M2 ROT-GBM | ~150k symbol-days with valid D+21 | 150k × (12/64) ÷ 16 (16-day window overlap) ≈ **1,750** | ≤ ~600 leaf values (max_iter 60, max_leaf_nodes 10, max_depth 4, lr 0.05, L2 1.0, binding purged early-stop, monotone constraints kept) | ~0.34 params/eff — **SUPPORTED**; note this is a 4× capacity CUT from TB-006's GBM (2.3k leaves) because the slow horizon divides time samples by 16, not 5 |
| M3 DISP-HAR | 2,790 daily obs (2015-02→2026-02) | 2,790 ÷ 5 ≈ **558** (single market-level series; N2) | ~10 OLS coefs (+ log-link) | ~0.02 — **SUPPORTED** |
| M4 EVT-NET | ~75k bucket-days (2,790 × 27) | 75k × (8/27 effective buckets, N1 sleeve logic) ÷ 5 ≈ **4,500** | ≤702 nominal (26×27), L1-sparsified to ~100–200 active | ≤0.16 nominal — **SUPPORTED** (exceedance targets cluster in time; ÷5 is generous here — printed as a caveat, capacity already 4× under the ⅓ bar) |
| M5 POS-Z | ~580 weekly obs; **~15–25 \|z\|>2 episodes** | episodes ≈ **15–25**; regime-disposition flavor ⇒ N3 floor ~6–8 | **0 trained** (window 156w, threshold 2.0, decay 21d — all pre-registered constants, §16-style provenance) | **SUPPORTED only because capacity = 0**; any request to "tune the threshold" is denied by this row |
| Organ-trust weights (executive/EA genes over 5 members + M3 gain) | 6 fold-scores | **~6–8** (N3) | ≤ ~10 effective (Evolution Engineer's lane; this row is the orthogonality-side constraint) | tiny-or-nothing; TB-006's equal-trust tie was unanswerable at corr .941 — the entire point of the C2 gate is to make THIS row's question answerable for the first time |
| C2 gate itself | 1,507 OOF days × 10 pairs | n/a (measurement, not fitting) | 0 | corr se at n_eff≈300 ≈ 0.06 ⇒ the 0.7 ceiling is measurable with ~±0.06 precision — the gate is powered, unlike E2 |

Totals: ~17.8k trained params system-wide vs ~7.1k on the densest target — same order as
TB-006 (which TR graded executable) with capacity REDISTRIBUTED: the dense 5d look keeps
its budget, the new looks are bought almost free (HAR ~10, net ~200 active, rules 0), and
the slow look pays its honest 16× time-overlap tax up front.

## 7. Composition contract (one paragraph for the Expression Architect — not my lane to fix)

M1+M2 are the directional rank inputs to the chassis socket (in [0,1], per ANALYST_AUDIT
A2 contract; rank-permutation expression preferred for C1 per A5); M3 is a scalar gain on
tilt aggressiveness (never direction); M4 is a per-symbol width/veto (shrinks a name's
tilt toward neutral on high decouple risk); M5 is a slow additive sleeve tilt entering at
weekly cadence. The C1 caveat from UNIVERSE_EXPLOITABILITY §4 binds M5 and any
bond-sleeve tilt: TLT/AGG/MUB/SHY tilts are de-risking in disguise — parity must be
enforced on risk budget, not gross alone.

---

## Bottom line

Five questions, five members, five model types, each present for a named mechanical
reason; orthogonality bought structurally (disjoint targets — including the
horizon-disjoint D+5→D+21 window — disjoint partitions, different levels) and then
MEASURED by a powered, pre-registered gate (pairwise ≤0.7 pooled / ≤0.8 per-fold, signal
AND book space, deployed ranker as member zero) with a fixed remediation ladder ending in
honest demotion, never paper-over. The arithmetic says every member is at or under
TB-006's accepted capacity ratios, and the one sample-starved axis (positioning) is
handled the only honest way: zero trained parameters.

— end —
