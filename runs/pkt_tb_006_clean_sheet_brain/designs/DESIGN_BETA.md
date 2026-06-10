# DESIGN_BETA — "BOOKWRIGHT": a direct-allocation policy brain

**Architect-Beta (panel role 4), PKT-TB-006, 2026-06-10.**
**Prior:** allocation-learner — *skip forecasting, learn the action.*
**Sources:** packet, ASSIGNMENT_BRIEF.md, INFOTROPY_TRANSFER.md, PROPOSAL_{META_EVALUATOR,
EVOLUTION,LLM_SENTIMENT,DATA_EDGE}.md. Designed blind to the incumbent (sealed-incumbent
rule honored; DESIGN_ALPHA/GAMMA not read).

---

## 1. System thesis

At this sample size (~2,959 trading days, ~6 distinct regimes, a 64-ETF cross-section that is
heavily co-moving) the forecast-then-optimize stack pays twice: once to learn per-symbol
expected returns whose signal-to-noise at daily horizon is minuscule, and again when a
separate portfolio constructor amplifies exactly the forecast errors that survived (the
classic error-maximization property of mean-variance on noisy estimates). The
allocation-learner prior collapses the stack: the learnable object is the **book itself** —
a mapping from observable state to target portfolio weights — and the loss is the **realized
forward net utility of holding that book**, computed through the same fill-and-cost mechanics
the bake-off harness uses. This buys three things the forecast stack cannot: (i) costs,
turnover, and risk shaping live *inside* the gradient instead of being bolted on after; (ii)
the model only spends capacity on distinctions that change the action — two states with
identical optimal books are allowed to be identical, however different their forecasts would
be; (iii) credit assignment is exact and one-step (decide at D, utility realized over D..D+h),
which is the only policy learning the brief's §7 sample budget supports. I explicitly do NOT
build long-credit-assignment RL: no bootstrapping, no value-function recursion over episodes,
no online learning. Every policy here is an **offline, batch, short-horizon (h=5 or 21),
dense-reward, heavily regularized differentiable-replay learner** — the variant of "learn the
action" that 2.9k daily steps can actually pay for. Forecast-shaped quantities exist inside
one member (P3's action-value regressor) as internal machinery; the intelligence showcased is
the action.

## 2. System overview (one screen)

```
nightly (Lambda, 03:00 UTC):                       monthly (Mac, launchd):
  ingest: prices, FRED+, CBOE, COT, GDELT 15-min     stage 0  data refresh + GDELT/LLM backfill top-up
  GDELT-rich features (G1..G5) ──┐                   stage 1  policy LIBRARY training (per fold × shaping grid)
  LLM organ → llm_sentiment.json ─┤                  stage 2  executive training (on OOF books)
  state builder → X[64,T,F], z ───┤                  stage 3  EA over genome (library composition + dispositions)
                                  ▼                  stage 4  fine-tune executive calibration on own record
  POLICY ENSEMBLE (frozen weights, loaded from S3 models/):
    P1 cross-asset transformer policy  → book w_P1[64], conf c_P1
    P2 horizon-pair linear policies    → books w_P2s (h=21), w_P2f (h=5)
    P3 GBM action-value policy         → book w_P3
    P4 event/sentiment policy (LLM+GDELT, record-grade gated) → book w_P4
  EXECUTIVE (learned meta-evaluator, ~600 params):
    trust τ[M] over candidate books + deployment f → w_tgt = f · Σ τ_m w_m  (vol-capped)
  intent diff vs w_prev, no-trade band → BUY/SELL/REDUCE intents → harness fill at next open
  audit: meta_decision.json, trust_ledger.parquet, candidate_books.parquet
```

Genome (from evolution) parameterizes: which library variant fills each policy slot, trust
priors/halflife, feature-block gates, allocator risk genes, and the loss-shaping constants the
library grid spans. Everything the EA touches is a disposition; everything gradients touch is
a perception; the measurement instrument is fixed (per PROPOSAL_EVOLUTION §1, adopted).

## 3. State representation (exact)

All from `daily/D/*` (data through D-1 close) plus the brain's own nightly artifacts; in
pretraining, rebuilt from deep-history parquets + refetched OHLCV with the identical
convention (features at D use closes ≤ D-1). Three blocks:

**3.1 Per-symbol panel `X[S=64, T=64, F=14]`** (z-scored per feature, cross-sectionally per day
where marked ×):
returns {1,5,21,63}d (×), vol {21,63}d, drawdown {21,63}d, rel_strength {21,63}d (×),
dollar-volume z, distance-to-52wk-high, breadth-of-own-sector (S0), beta-to-SPY 63d.
(Existing feature family in `asset_features_history.parquet` plus four cheap adds; full OHLCV
refetched from Stooq for opens.)

**3.2 Per-symbol static `s[S, 2 cat]`:** asset_class (6 levels), sector (~30 levels) from
`config/universe.csv` → learned embeddings (8-dim total). **No symbol-identity embedding** —
deliberate memorization defense: the brain cannot learn "SPY always good," only "broad-equity
in this state."

**3.3 Market context `z[~40]`:**
- `context.parquet` columns (rates, slopes, credit proxy, risk-off proxy, vixy returns, vix
  term slope, vvix, skew) — ~14.
- Data-edge adds (§9): CBOE S1 (VIX9D/VIX/VIX3M slope+curvature, COR3M level + 1y z,
  VXN−VIX) ~6; FRED S3 (NFCI z, DFII10 level/Δ63, T10YIE Δ21, HY-OAS level/Δ5, claims z) ~7;
  COT S2 (lev-fund net %OI z for ES/UST10/VIX, publication-lagged) ~3.
- GDELT day-level (§9): G4 theme-novelty JSD + smoothed, doc-count surprise, G5 concentration
  HHIs ~5.
- LLM global axes + status (§8): risk_appetite, rates_pressure, geopol_risk, llm_status_ok,
  llm_available ~5.

**3.4 Bucket features `B[22+, ~8]`** (for P4): per-bucket GDELT G1 mention-share z, G3 tone +
tone-dispersion, G2 country Goldstein/conflict-share (country buckets), LLM sent/conf/sal +
ema3 + d1 (bucket-mapped per PROPOSAL_LLM_SENTIMENT `bucket_map.json`).

**3.5 Portfolio state:** `w_prev[64]` (current book), gross_prev, trailing 21d realized book
vol, drawdown-from-peak of own equity curve, and the executive's trust-ledger stats (§6).

## 4. Action space and harness contract

**Action = the next book.** `a = (w_tgt[64] ∈ [0, w_max]^64 with Σw ≤ 1, long-only; cash =
1 − Σw)`. No leverage, no shorts (universe has no leverage flags; long-only keeps the action
space inside what the harness's BUY/SELL/REDUCE vocabulary expresses cleanly).

**Harness binding (brief §5.1):** decision date D reads `daily/D/*` only; the brain emits
intents; fills at next-session open from `daily/D+1/prices.parquet` through
`_execute_intents`, costs via `apply_transaction_costs` (per-sector half-spread ± 2bps seeded
slippage); marks at D close; harness cluster cap applies downstream (the allocator's own
per-symbol/per-cluster caps are set tighter than the harness rail, so the rail never binds in
normal operation).

```
delta[s] = w_tgt[s] − w_prev[s]
emit BUY where delta > band, SELL/REDUCE where delta < −band; band = no_trade_band gene
(default 100 bps of NAV — wide on purpose; see turnover budget)
sizing: notional = |delta| × NAV at last mark
```

**Turnover is priced three times:** (i) inside every training loss (cost term, same half-spread
table); (ii) by the no-trade band; (iii) by a turnover-smoothness penalty in the executive's
loss. **Pre-registered turnover budget:** expected one-way turnover ≤ 6%/day (≈ blended
3 bps × 6% ≈ 0.18 bp/day ≈ 0.45%/yr cost drag). A design that needs daily full rebalance to
win has lost to its own churn; this one is trained not to want it.

## 5. The policy ensemble — four genuinely different model families (assignment item 1)

"Different model types" under the allocation prior = different *policy families*: different
hypothesis classes about what the state→book mapping looks like, trained on the same utility
objective, disagreeing for structural reasons. All emit a unit-gross candidate book + a
self-confidence scalar.

### P1 — Cross-asset transformer policy (the transformer doing load-bearing work)

The book is a cross-sectional object: how much XLE to hold depends on what bonds, vol, and the
rest of the equity sleeve look like. Per-symbol models cannot see the book; cross-asset
attention is the right inductive bias, not decoration.

```
per symbol s: temporal encoder GRU(F=14 → 32) over X[s, last 64 days, :] → h_s ∈ R^32
token_s = h_s + emb_sector(s) + emb_class(s)              # 8-dim embeds projected to 32
token_0 = MLP(z) ∈ R^32                                   # market token (CLS-like)
encoder: 2 pre-LN transformer layers, d_model=32, 4 heads, FFN=64, dropout 0.2
head:    score a_s = MLP_1(token_s_out);   gross logit from token_0_out
book:    w_P1 ∝ softplus(a_s), per-symbol cap w_max (gene), renormalized to unit gross
conf:    c_P1 = sigmoid(gross logit)        # the transformer's own conviction
```

**Dims/params:** GRU ≈ 4.5k; embeddings ≈ 0.4k; 2 encoder layers ≈ 17k; heads ≈ 2k; z-MLP ≈
1.5k. **Total ≈ 26k parameters**, one weight set shared across all 64 tokens.

**Training:** differentiable replay (§11.2), h=5, loss = −U_h(w_P1(D)) with per-symbol credit
densification: the gradient of U_h decomposes into Σ_s w_s·r_s terms, so although the *loss*
is book-level (~2.9k samples), every symbol-day (~189k, correlated) carries gradient — the
weight-shared encoder is trained by the panel, the attention by the book.

**Memorization defense (the brief §7 demand):** weight sharing (26k params serve 64 tokens ×
2.9k days); no symbol identity; **random symbol-subset dropout** — each training step samples
48 of 64 symbols, so no attention pattern can depend on a specific symbol being present;
feature noise σ=0.1; weight decay 1e-3; 5-seed output averaging; early stop on purged
validation utility. **Attention falsifier (pre-registered):** a "no-attention twin" (identical
GRU + per-symbol MLP head, attention replaced by identity) is trained in the same run. If the
twin matches P1 on validation utility, the transformer is decoration — the dossier says so and
the scorecard's transformer attribution is read against the twin, not against nothing.

### P2 — Horizon-pair convex linear policies (the glass-box family)

Two tiny linear-softplus policies, identical form, different reward horizon — the
"horizon-specialized allocators" arm:

```
P2-slow: w ∝ softplus(W·x_s_slow + b), x_s_slow = {ret21, ret63, vol63, dd63, rs21, rs63, breadth} → trained on U_h, h=21
P2-fast: same form on {ret1, ret5, vol21, dd21, rs21, vixy-beta} → trained on U_h, h=5
≈ 30 params each (shared across symbols). Ridge λ on W. Confidence = cross-sectional
dispersion of its own scores.
```

These are the brain's honest priors: if the deep nets cannot beat two 30-parameter convex
policies, the executive will (measurably, via the trust ledger) listen to the linear ones —
graceful degradation is built into the ensemble rather than bolted on.

### P3 — Gradient-boosted action-value policy (the tabular/nonparametric family)

A fitted one-step action-value: for each symbol-day, regress the realized net 5-day utility of
a unit tilt — `y_{s,D} = Σ_{t=D..D+4} log(1+r_{s,t})/5 − entry_cost_bps(s)/1e4 − cash_rate` —
on `[x_s, B[bucket(s)], z]`, pooled cross-sectionally (~180k rows after embargo).

```
sklearn HistGradientBoostingRegressor: max_depth 3, 300 iters, lr 0.05, min_samples_leaf 200,
L2 1.0, feature_fraction 0.7. Policy: w_P3 ∝ topk(ŷ_s, k=12)·rank-weights, capped, unit gross.
Confidence = mean(ŷ of selected) − median(ŷ all).
```

Different family for real: axis-aligned splits, native handling of the heterogeneous tabular
GDELT/macro/COT features the neural members z-score away, no temporal encoder. Yes, ŷ is a
forecast-shaped internal quantity — it is a *cost-inclusive action value*, and the showcased
output is the selection policy. Memorization defense: depth-3 trees + 200-leaf minimum on 180k
correlated rows is ~thousands of effective parameters against ~600 quasi-independent
cross-sections; monotone-ish regularization via L2; per-fold retrain only.

### P4 — Event/sentiment policy (where LLM + GDELT cash out as actions)

Maps the bucket panel B (§3.4) — **record-grade gated, §10** — to bucket tilts, then to
symbols via bucket_map:

```
MLP: input ≈ 8 feats × salient-bucket pooling + 22-bucket vector ≈ 40 → hidden 16 (tanh) → 26 bucket tilts
w_P4 ∝ softplus(tilt[bucket(s)]) spread equally inside bucket, unit gross. ≈ 1.1k params.
Trained on U_h (h=5) with the llm_available mask as an input (LLM features exist from 2024-01
backfill only; GDELT-rich from 2015-02). Confidence = LLM conf×sal aggregate.
```

P4 is deliberately the only member whose state is dominated by the differentiated data: its
trust line in the executive's ledger is a *running, public measurement of whether the
news organs earn capital* — the assignment's "each organ carries weight" made into a daily
number.

### Ensemble roster

M = 5 candidate books nightly: {P1, P2-slow, P2-fast, P3, P4}. Member gates (EA genome) can
drop any to OFF; the entropy floor in the executive keeps live counterfactual ledgers for all.

## 6. The executive — learned meta-evaluator (assignment item 5)

**Engagement with PROPOSAL_META_EVALUATOR: ADOPT with one structural adaptation.** The
proposal is written in the opinion currency `mu[M,S]` (per-symbol expected-return z-scores)
with a precision-weighted blend. Under the allocation prior the experts already emit **books**,
so I delete the mu/sigma precision blend and replace it with a convex combination of candidate
books — strictly simpler, and it makes the proposal's counterfactual trust ledger *exact*: the
"solo-expert book" it needs for `u_m(D)` is literally the member's own output, no reference
sizing construction required.

Adopted verbatim: the two-head gating architecture (shared-φ trust head + sizing head,
~450–600 params, ≤3k ceiling); the differentiable decision-replay loss (§3.1 of the proposal:
log-growth − one-sided downside − amortized cost − turnover smoothness − entropy-floor term);
sequential w_prev training (§3.2); the counterfactual trust ledger + auxiliary KL
trust-alignment (§3.3); the evolution interface (§3.4 — λ_dn, f_max/σ_cap, ε, band on the
genome); the entropy floor ε=0.05; look-ahead rules incl. the D−h−1 ledger lag (§5.1); the
pretrain/fine-tune split (§5.3 — fine-tune trust logits + sizing head only, on own record
pre-holdout, LR×0.1); `meta_decision.json` + integrated-gradients attribution (§6); all five
kill criteria + LOO baseline (equal trust, fixed f=0.7, same vol cap) (§7); the
simplification ladder (linear gate ships if the MLP can't beat it).

Adapted blend:

```
inputs per proposal §1.1, with mu/sigma replaced by: candidate books w_m[S], confidences c_m,
  agreement g = {pairwise rank-corr of books, mean pairwise L1 distance, gross dispersion}
w_unit = Σ_m τ_m · w_m            # each w_m unit-gross ⇒ w_unit unit-gross, convex
f      = f_max · sigmoid(ψ([z_exec, g, τ-entropy, ledger stats, own-equity drawdown]))
w_tgt  = f · w_unit, then vol-target overlay (trailing-21d predicted book vol ≤ σ_cap gene)
```

`z_exec` = §3.3's z plus LLM global axes and G4 novelty — sentiment conditions trust and
deployment as well as feeding P4 (the proposal's "sentiment enters twice," kept).

**The policy/meta-evaluator line (the packet's care point):** in this design the executive is
itself a policy — a policy *over policies*. The line is held by three structural facts:
(i) **stage separation** — members are trained per-fold and frozen; the executive trains only
on out-of-fold member books (stacking discipline; the walk-forward attestation the proposal
demands is satisfied by construction because I own both sides); (ii) **capacity asymmetry** —
members get 30–26k params for perception; the executive gets ~600 for judgment; it physically
cannot re-derive the members' signals, it can only arbitrate them; (iii) **target
asymmetry** — members optimize their own solo-book utility; the executive optimizes the
*blended* book's utility including the cost of moving between days, i.e. exactly the
decision-grade target the packet demands. Regime labels appear nowhere in any loss path.

## 7. Evolution's role (assignment item 2)

**Engagement with PROPOSAL_EVOLUTION: ADOPT the machinery, ADAPT where it bites.** Adopted
verbatim: the evolved/gradient/fixed division rule; (μ+λ) GA with P=28, G=14, K≤400, seeded
determinism; fold layout F1–F6 with 21-day embargo, fitness data ending ~2026-02-06, holdout
never touched; fitness = min-over-cost-scenarios of (Sharpe − DD penalty), mean − 0.5·std
across folds, parsimony pressure; fold subsampling; B0 default-genome-in-population; the
adoption gate (champion must beat B0 by one cross-fold sd or B0 ships); B1 random-search
baseline; B2 as the E2 attribution read; every-variant logging; $0 AWS cost shape (pure-numpy
walk over cached OOF outputs).

**Adaptation — where evolution bites differently in an allocation-learner.** In a forecast
stack the EA can only tune connective tissue; here the policies' *training losses* contain the
brain's risk dispositions (λ_dn, turnover penalty, horizon), and those are exactly what
gradients must not self-grade and what the proposal assigns to evolution. Retraining members
inside the EA loop is unaffordable, so: **the policy LIBRARY.** Each monthly cycle, stage 1
trains each member family at a small pre-registered grid of loss-shaping constants:

```
P1: λ_dn ∈ {2, 6}                       → 2 variants  (the expensive member gets the coarse grid)
P2s/P2f: λ_dn ∈ {2, 6} × β_to ∈ {lo, hi} → 4 variants each (cheap, seconds)
P3: target horizon h ∈ {5, 10}           → 2 variants
P4: λ_dn ∈ {2, 6}                        → 2 variants
```

The genome's composition genes become **slot-selector genes**: per member slot, an integer
gene picking which library variant occupies it (or OFF). Evolution thereby genuinely evolves
*reward shaping and ensemble composition* — a population of policies in the literal sense —
while genome evaluation stays a cached-matrix walk (each variant's OOF books are precomputed
per fold). Added genes: 5 slot selectors (2–3 bits each as floats) on top of the proposal's
schema; trust priors/halflife, feature-block gates (GDELT theme blocks, LLM block,
vol-structure, macro, COT), allocator risk genes (gross_target, vol_target, max_symbol_weight,
dd_brake, no_trade_band, conviction temp, cash_floor), executive λ_dn/abstain — total L ≈ 36,
inside the ≤40 cap. Gene ranges pre-registered with the schema.

If evolution gates GDELT/LLM blocks to zero, that surfaces in the dossier and the Phase D LOO
still measures the organ independently (proposal §1 note, adopted).

## 8. LLM sentiment organ (assignment item 3)

**Engagement with PROPOSAL_LLM_SENTIMENT: ADOPT wholesale.** Funnel (4 GKG files/night, theme/
actor/country relevance, slug+quote text, cluster dedup, seen-cache, top-120 cap), pinned
Bedrock Haiku with gpt-4o-mini fallback and fail-soft neutral artifact, the bucket map, the
output schema, temperature 0, stored-artifact-only replay (the harness never re-invokes the
LLM), backfill Tier 1 (2026-01-31→present, ≈$0.35, covers replay window + entire holdout) +
Tier 2 (2024-01→, ≈$2.05), the $5 Phase C Bedrock cap, and both falsifier stages (tone-proxy
corr < 0.8 check; LOO retrain-with-neutral-constants on holdout).

**Conflict resolution (the two specialists disagree):** PROPOSAL_DATA_EDGE §1.4 says LLM
features must be inference-only ("never a trained-on feature"); PROPOSAL_LLM_SENTIMENT funds a
2024-01 backfill precisely so they can be trained on. I side with the LLM engineer, with a
guard: **LLM features are trainable inputs only inside P4 and the executive's context, only on
the Tier-2 window (~520 trading days), always accompanied by the `llm_available` mask**; no
deep-history component may condition on them. P4's GDELT loadings train on 11 years; its LLM
loadings train on ~2 years; the mask keeps the regimes separated. The train/serve-skew
argument that killed the V2TONE-proxy option stands.

**Exactly which features enter the state:** per-bucket `llm_sent/conf/sal`, `sent_ema3`,
`sent_d1` → P4 bucket panel (§3.4); global `risk_appetite, rates_pressure, geopol_risk` +
`llm_status_ok, llm_available` → executive context and P3's z (§3.3); `event_flags` → P4 input
and one executive context bit (any-flag-severity max). Event flags do NOT trigger any
deterministic veto — the packet wants learned judgment, so flags influence sizing only through
the learned heads.

## 9. GDELT and the data edge (assignment item 4)

**Engagement with PROPOSAL_DATA_EDGE:**

- **G1–G5 GDELT-rich families: ADOPT in full** (sector theme vectors, country-actor pressure,
  sector-conditioned tone/dispersion, novelty/burst JSD, actor/geo concentration; 4-file/day
  frame; `visible_from` join rule). The **2026-02-05→present top-up backfill is the first
  build task** — without it GDELT has zero holdout coverage and assignment item 4 is
  unevaluable. The 80–110 GB deep backfill runs with the degrade-option (2 samples/day
  2015–2022) if Mac time becomes the binding constraint; density break flagged in the
  manifest.
- **S1 CBOE vol-surface/COR: ADOPT** — deepest history in the system (1990/2006); feeds z and
  P1's market token; COR3M is a direct conditioning input for sizing a 64-ETF book.
- **S3 FRED deep catalog: ADOPT** — pretraining-depth macro context; attribution must beat the
  existing context proxies, as the scout demands.
- **S2 CFTC COT: ADOPT as executive/P3 context only** (weekly, publication-lagged; ~13 holdout
  obs ⇒ it conditions, never drives; the shuffle-placebo falsifier kept).
- **S4 Treasury auctions: REJECT** — episodic, <10 holdout firings, no honest holdout verdict
  possible; feature-count discipline beats a free-but-unverifiable column. (Scout itself
  ranked it first-to-cut.)
- **S0 breadth: ADOPT** (free, internal; in X and z).
- Killed candidates (put/call, ICI, iShares, crypto, AV extras): concur, not used.

GDELT enters decisions through three paths, each separately ablatable: P4's bucket panel
(gated, §10), P3's tabular features, executive/P1 context (G4 novelty + G5 concentration).

## 10. Infotropy mechanism (assignment item 6)

Judging the Liaison's graded transfers from the allocation prior:

- **Attempt B (record-formation label weighting): ADOPT — primary.** It was designed for
  utility targets, and this brain is utility targets all the way down, so it applies to
  *every* learnable component, not just the executive: training-sample weight
  `w_i = base_i · (eps + record_formation_score_i)` with eps=0.25 (floor raised above the
  Liaison's sketch deliberately: an allocation policy must still learn to *stay small* on
  reverting noise days, so noise days keep a real gradient share; at eps→0 the policy never
  sees the states where the right action is inaction). Score computed per the Liaison's
  pseudocode (regime-stat shift × (1 − retracement), h=5/21 matching each learner's horizon).
  **Pre-registered A/B falsifier:** every member + executive trained twice (uniform vs
  record-weighted) in one monthly cycle; ship record-weighted only if purged validation
  utility improves; the dossier reports the delta either way and the scorecard's
  `infotropy=<attr>` is the holdout LOO of this weighting (retrain-uniform arm).
- **Attempt A (R1∧R2∧R3 record-grade ingestion gate): ADOPT, scoped to P4.** Applied as a
  walk-forward feature screen on the GDELT bucket-event features entering P4: a bucket-feature
  family must clear persistence (R1), non-price-echo (R2 — the OOS regression of event
  intensity on the symbol's own lagged returns; this is the leg that kills
  news-chasing-price), and incremental OOS value (R3) on training folds, or it is routed out
  of P4's input. R3-only twin trained as the falsifier per the Liaison. Scoped to P4 because
  that is where the event flood lives; the price-derived features in X need no such gate.
- **Attempts C, D, E: REJECT as infotropy claims; partially absorb as ordinary engineering.**
  The Liaison graded them RESTATEMENT and I take the grade at face value: conviction-weighting
  (C) already exists as the confidence inputs to the trust head; self-record feedback (D)
  already exists as the executive's own-equity drawdown + ledger inputs; breadth gating (E)
  already exists as S0 features. None are labeled infotropy anywhere in the dossier — the
  canon's own overlay-not-rename STOP rule, honored.
- **Attempt F: REJECT** per the Liaison (canon-barred and a rename of vol-squeeze).

## 11. Training plan

### 11.1 Monthly pipeline (local Mac, launchd; wall-clock budget ≤ 90 min)

```
stage 0  data refresh: GDELT top-up + rich parse, CBOE/FRED/COT pulls, LLM backfill delta   ~10 min
stage 1  policy library: per fold f ∈ F1..F6 (expanding head-train windows, 21d embargo):
         P1 ×2 variants ×3 seeds  (~1.5 min/run ⇒ ~54 min worst case; warm-start from prior
            month's weights cuts this to ~20 min steady-state)
         P2 ×8, P3 ×2, P4 ×2 variants (seconds–minutes total)                               ~25–55 min
stage 2  executive: trained on OOF candidate books (record-weighted + uniform twin),
         5 seeds + linear-gate twin + GBM challenger                                        ~5 min
stage 3  EA: cached-matrix fitness walk, K≤400, adoption gate vs B0                          ~8 min
stage 4  fine-tune executive calibration on own live record (pre-holdout ~130 d only)       ~1 min
upload: versioned artifacts + latest.json pointer to s3://…/models/ (existing convention)
```

Shrink order if over budget (packet rule — seeds/epochs before organs): P1 library seeds
3→2, then warm-start-only, then variants 2→1 (EA loses the P1 shaping gene; stated on the
final line if taken).

### 11.2 The shared differentiable-replay loss (members and executive)

```
U_h(D; w) = (1/h)Σ_{t=D..D+h−1} log(1 + r_book_t) − λ_dn·(1/h)Σ min(0, r_book_t)² − cost(D)/h
cost(D)   = Σ_s |w_tgt_s − w_prev_s| · half_spread_bps(sector_s)/1e4      # E[slippage]=0
Loss      = −E_D[ sw(D) · U_h(D) ] + β_to·E_D[‖w(D)−w(D−1)‖₁] + weight decay
            sw(D) = record-formation sample weight (§10)
```

Fills at D's open in pretraining come from refetched OHLCV; the proposal's mandatory
20-pair pretrain-vs-replay fill-price alignment test is adopted (VUG 6:1 handled once, never
twice). Members train teacher-forced then sequential for w_prev, as the executive does.

### 11.3 Walk-forward, purging, seeds, look-ahead

- **Folds:** the EA's F1–F6 layout is the single source of truth for ALL walk-forward
  training (members, executive, EA) — one fold geometry, audited once. Embargo 21 trading
  days both sides of every boundary (> max horizon 21d). 2014–2019 = head-training substrate
  only.
- **Pretrain/fine-tune:** public history → all member + executive structure; own live record
  (~210 d, pre-holdout ~130 d) → executive calibration only (trust logits + sizing head,
  LR×0.1); members are NOT fine-tuned on 130 days (cannot support it, not asked to).
- **Seeds:** master seed sha256("PKT-TB-006-BETA"+window_end); children per stage; 5-seed
  member/executive output averaging; all seeds in manifests.
- **Look-ahead controls:** `visible_from` joins for GDELT/COT/FRED weekly; ledger lag D−h−1;
  LLM artifacts timestamped ≥10.5h pre-open; overlapping-window stats reported HAC-adjusted or
  non-overlapping only.
- **Holdout (2026-03-11→):** touched once, by the one pre-registered configuration (champion
  genome + record-weighting decision + library composition, all frozen on pre-holdout
  validation); looks counted in the run journal. All baseline/ablation arms are defined before
  the first holdout read.

## 12. Sample budget table (every learnable component)

| Component | Params | Training sample | Effective sample (honest) | Memorization defense |
|---|---|---|---|---|
| P1 transformer policy | ~26k | 2.9k book-days; ~189k symbol-day credit terms | ~600 indep. h=5 windows; ~6 regimes | weight sharing ÷64; no symbol IDs; 48-of-64 symbol dropout; feature noise; wd 1e-3; dropout 0.2; 5 seeds; purged early stop; no-attention twin must be beaten |
| P2 ×2 linear policies | ~30 each | same | same | convex + ridge; near-unoverfittable; serve as floor |
| P3 GBM action-value | ~300 trees·d3 (≈ low-k effective) | ~180k pooled rows | ~600 indep. cross-sections | depth 3, min_leaf 200, L2, subsampling; per-fold retrain |
| P4 event MLP | ~1.1k | GDELT cols: ~2.7k d; LLM cols: ~520 d | ~540 / ~100 windows | record-grade gate shrinks input; llm mask; wd; 5 seeds; linear twin |
| Executive gate+sizing | ~450–600 (≤3k) | 2.9k days | ~600 windows | per PROPOSAL_META_EVALUATOR §5.2 battery + simplification ladder |
| EA genome | 36 genes, K≤400 evals | 6 fold-scores/genome | ~6 regime obs | adoption gate (1 cross-fold sd over B0), fold subsampling, B1 random-search control, cost-scenario min |
| Record-formation weights | 0 (derived) | — | — | A/B twin falsifier |

The brain's total *learned* capacity ≈ 30k params, ~85% of it in one weight-shared encoder
whose gradient is fed by the panel. Nothing here trains a per-symbol head; nothing trains on
the holdout; nothing online-learns in Lambda.

## 13. Cost sketch (itemized monthly; target ≤$10, hard fail ~$15)

| Line | Arithmetic | $/mo |
|---|---|---|
| Lambda nightly | existing ~110s + ~60s (LLM funnel+parse 40s, extra ingests 20s) ≈ 170s × 2.94 GB × 22 + morning runs ≈ 11k GB-s | ~6.50 |
| S3 storage+requests | existing ~1.00 + gdelt_rich/cboe/cot parquets ~15 MB + daily artifacts (books, meta_decision, llm json ~60 KB/d) + ledgers | ~1.10 |
| Secrets Manager | unchanged | 1.00 |
| CloudWatch | unchanged | 1.00 |
| ECR | image storage unchanged (no new heavy deps; torch already in) | 0.12 |
| Bedrock (Haiku pinned) | ~$0.0044/night × 22 × 1.5 headroom | 0.15 |
| Data transfer | GDELT/CBOE/FRED pulls inbound (free), outbound minimal | ~0.02 |
| **Total recurring** | | **≈ $9.90** |
| One-time (Phase C) | LLM backfill Tier 1+2 ≈ $2.40; deep GDELT backfill $0 cash (Mac time); Bedrock prototyping under the $5 cap | ≤ $5.00 once |

Headroom note: $9.90 is tight against the $10 target; first reduction lever is moving the LLM
funnel's GKG parse into the same fetch pass as gdelt-rich (shared download, −20s Lambda ≈
−$0.40). Hard-fail margin to $15 is comfortable. Haiku-4.5-class IAM widening (the LLM
engineer's deprecation flag) would move Bedrock to ~$0.55 — still in envelope; flagged, not
assumed.

## 14. Attribution hooks (Phase D leave-one-out; all arms pre-registered before any holdout read)

| Organ | LOO arm | What kills it |
|---|---|---|
| Transformer (P1) | swap P1 → its no-attention twin (primary), and P1 → OFF with trust renormalized (secondary) | holdout paired-daily Δ ≤ 0 vs twin ⇒ attention is decoration |
| Ensemble (any member) | member → OFF, trust renormalized, same genome | ΔSharpe ≤ 0 on holdout ⇒ member ≈ 0 |
| Evolution | champion genome vs B0 default genome, full brain, identical harness (= EA proposal B2) | champion ≤ B0 on holdout, or champion ≤ B1 in fitness (ceremonial-as-optimizer) |
| LLM | retrain brain with all llm_* at neutral constants (primary; per LLM proposal stage 2), feature-neutralized secondary | ΔSharpe ≤ 0 or paired t < 1 on holdout (real LLM output covers entire holdout via Tier 1) |
| GDELT | ablate G1–G5 block everywhere (P4, P3, z) and retrain | |t| < 1 paired daily ⇒ evidence-backed no-signal, reported per item 4 |
| Meta-evaluator | equal-trust + fixed f=0.7 baseline (meta proposal §7) | Δ ≤ 0, static trust (std<0.02), or calibration corr ≤ 0 |
| Infotropy-B weighting | retrain all learners with uniform weights | validation/holdout Δ ≤ 0 ⇒ infotropy attr ≈ 0, scorecard says so |
| Infotropy-A gate | R3-only-screened P4 twin | gated ≤ R3-only ⇒ gate dead, P4 keeps R3 screen |
| Data-edge S1/S3/S2 | column-block ablations vs existing context proxies | per scout's falsifiers (incl. COT shuffle-placebo) |

Plus the standing dumb-twin guard: the full brain must beat two pre-registered non-learned
baselines on pre-holdout validation before it earns its single holdout read — (a) equal-weight
60/40-ish risk-targeted book, (b) P2-slow alone at fixed f. If it cannot, the honest dossier
line is that the learned action stack failed its own floor, and the bake-off proceeds anyway
to report the loss.

## 15. Risks and honest weaknesses (where this prior loses)

1. **The thin spine: ~600 effective book-level observations.** The executive and every
   disposition ultimately rest on ~600 quasi-independent windows over ~6 regimes. The per-
   symbol credit densification feeds the encoders, but the *allocation* signal proper is
   thin. If the true edge lives in per-symbol forecasting precision, Alpha's prior wins.
2. **Closet-baseline risk.** Utility-trained long-only policies gravitate to vol-targeted
   momentum/quality tilts. The deterministic overlays (vol cap, band, cluster caps) plus P2
   may explain most of the performance, leaving the learned parts with ≈0 attribution — the
   showcase fails honestly even if P&L is fine. The dumb-twin guard measures this; it does
   not prevent it.
3. **Overlap inflation.** h=5 windows overlap; anyone reading naive SEs will overclaim. All
   stats HAC/non-overlapping by rule, but the true information content is small and the 62-day
   holdout cannot certify effects below roughly |ΔSharpe| ~0.4–0.5; minimum-detectable-effect
   stated up front in BAKEOFF.md.
4. **Cost-regime baked in.** Action learning internalizes today's fill/cost model. If real
   costs drift, the policy's learned turnover discipline mis-prices; the EA's 1.5× cost
   scenario is the partial hedge.
5. **LLM history asymmetry.** P4's LLM loadings train on ~520 days (one regime-ish); the organ
   may show holdout attribution driven by a single macro episode. The mask + bucket structure
   limits but does not remove this.
6. **Pinned-model fragility.** Haiku-2024-03 deprecation (LLM engineer's flag) is a real
   medium-term operational risk; fallback chain mitigates within the packet.
7. **Library combinatorics vs EA budget.** 36 genes incl. slot selectors searched by ≤400
   evals over 6 fold-scores — evolution may legitimately fail its adoption gate most months
   and ship B0; the design treats that as a feature (honest), but the scorecard's
   `evolution=<attr>` may then be ≈0 by construction of honesty, not by lack of trying.
8. **Where Beta loses to Gamma's prior:** if the differentiated data carries the real edge,
   burying it as one member (P4) + context columns under a 600-param arbiter may under-exploit
   it relative to a design built around the information stream.

## 16. Build order (engineer's checklist)

1. GDELT top-up backfill (2026-02-05→present) + gdelt_rich parser (G1–G5) — unblocks holdout.
2. State builder (X, z, B) with `visible_from` joins + fill-price alignment test (20 pairs).
3. Differentiable replay core (shared loss, costs, teacher-forced/sequential) + purged fold
   splitter pinned to the EA's F1–F6 geometry.
4. P2 pair (floor first — everything else must beat it), then P3, P4, P1 (+ no-attention twin).
5. Record-formation weighter + R1∧R2∧R3 gate (with R3-only twin), A/B harness.
6. LLM organ: 10-day pilot (~$0.05) → Tier 1 → Tier 2 (cap rules per proposal §5).
7. Executive (blend-adapted) + trust ledger + meta_decision.json + LOO baseline runner.
8. Policy library × shaping grid + EA (genome v2 with slot selectors) + B0/B1 baselines.
9. Fine-tune calibration pass; freeze the single pre-registered bake-off configuration;
   register; only then wire the harness Strategy hook and read the holdout once.

— end —
