# BUILD_SPEC — SYN-1, the converged trader's brain (Phase C buildable spec)

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — Synthesis Chair, 2026-06-10.
**Authority:** TOURNAMENT.md §2 (graft/kill tables) and §4 (pre-registration). Every design
decision is made HERE; the builder implements, and does not choose. Any gap discovered at
build time is a finding to surface, not a choice to make silently — log it in
`validation_looks.jsonl` if a choice is forced, with provenance.
**Write surface:** everything under `runs/pkt_tb_006_clean_sheet_brain/prototype/` (+ the
run dir for dossier files). NO production paths (`src/`, `config/`, `training/`,
`optimizer/`, `frontend/`), no deploys, no AWS resource creation beyond reading existing S3
data and the bounded Bedrock calls of §3.

---

## 1. System overview

```
NIGHTLY DECISION (decision date D; all inputs = data through D-1 close):
  data layer:   prices window | context | CBOE S1 | FRED S3 | COT S2 | GDELT G1–G5 | LLM organ
        │            (visible_from joins; masks; trailing-90d z computed from days < d)
        ▼
  feature store: per-symbol panel X[64,63,14] · tabular row T[~55] · bucket panel B[27,~25]
        ▼
  MEMBERS (supervised, frozen at inference; trust-head citizens M=3):
    E1 CAST-Small  16k transformer  → mu1[64], sigma1[64], c1     (price structure only)
    E2 GBM-Cond    ≤2.3k leaves     → mu2[64], sigma2[64], c2     (tabular exogenous world)
    E3 EventHead   675 coefs        → mu3[64], sigma3[64], c3     (events only; abstains)
  INSTRUMENTS (not trust citizens):
    E4 RiskNet+    ridge ~150 coefs → sigma_hat[64], beta_hat[64], book_vol_hat
        ▼
  SOLO-BOOK RULE (fixed, deterministic): mu_m,sigma_m → w_m[64] unit-gross long-only book
        ▼
  EXECUTIVE (~560 params, learned, decision-grade): trust τ[3] + deployment f
    → w_tgt = f · Σ τ_m w_m, vol-capped; consumes trust ledger + context z + info-health
        ▼
  INTENTS: Δw vs w_prev, |Δw| > no_trade_band → BUY/SELL/REDUCE dicts → harness fills next open
  GENOME (EA, monthly): 27 dispositions parameterize everything marked (gene) below
```

Prototype runs entirely from the run dir on the operator's Mac against pinned S3 snapshots
(local disk cache). The nightly Lambda shape is the deployment target described by the cost
worksheet (§14) — NOT built or deployed in this packet.

---

## 2. Data layer

### 2.1 Price data

- **Replay substrate:** `s3://investment-system-data/daily/<D>/prices.parquet`, snapshots
  2026-01-31 → 2026-06-10, pulled once through a **local disk cache wrapped around
  `S3Cache`** (~45 MB total; one-time). `daily/D/` contains data through D-1 close; D's
  OHLC lives in `daily/D+1/`. VUG 6:1 split (effective 2026-04-20) is applied by the
  harness — the feature builder must NOT re-apply it (alignment test §10.4 asserts this).
- **Deep history:** `training/data/asset_features_history.parquet` (2014-08-29→2026-06-05,
  64 symbols, closes + the 12 engineered columns) + full OHLCV refetched from Stooq
  (`https://stooq.com/q/d/l/?s=<SYM>.US&i=d`) for opens (targets need open→open). yfinance
  fallback. Per-symbol inception dates recorded (XLC 2018-06, INDA 2012-02, etc.); symbols
  enter training panels at inception + 70 trading days.
- **Universe:** `config/universe.csv`, 64 symbols, verified; `sector` strings key the cost
  model.

### 2.2 GDELT (build task #0 and #1)

- **Top-up backfill (TASK #0, foreground-launch then background, ≤1.5 h):** GKG + v2
  events, 4 files/day (00/06/12/18 UTC), **2026-02-05 → present** (~2.0–3.5 GB). Without
  it the holdout has zero real GDELT and item 4 is unevaluable.
- **Deep backfill (TASK #1, background, 5–10 h, restartable, $0):** **2 files/day (00/12
  UTC) for 2015-02-18 → 2022-12-31; 4 files/day for 2023-01-01 → 2026-02-04.** Density
  break recorded in the manifest. Fallback rung if wall-clock binds (stated on final line):
  panel start 2019-01-02 at 2/day. Template: `training/scripts/backfill_gdelt.py` frame
  (workers 4–8, politeness delay 0.3 s, per-day extract cache so crashes never re-download).
- **Feature families (one parse pass, emit all; Scout §1.2 constructions verbatim):**
  - **G1** sector theme-pressure: curated dictionary `theme_to_sector.json` (~60–100 GDELT
    theme codes → 13 sleeve buckets); per-bucket daily mention share + trailing-90d z
    (~26 cols).
  - **G2** country-actor pressure: from v2 events CAMEO — per-country mention-weighted mean
    Goldstein + conflict share (QuadClass 3/4) for {US, CN, JP, BR, IN, RU, EU} keyed to
    FXI/EWJ/EWZ/INDA/VGK/EEM/VWO (~14 cols).
  - **G3** tone panel: global avg/std/neg-share + polarity + per-bucket tone (~16 cols).
  - **G4** novelty/burst: per-bucket burst z; theme-novelty = Jensen–Shannon divergence of
    today's top-500 theme distribution vs trailing-90d mean (+21d smoothed); doc-count
    surprise (~4 cols).
  - **G5** concentration: HHI over locations and organizations + US-share (~3 cols).
- **No-look-ahead (binding):** row for UTC day d uses only files stamped ≤ d 18:00 UTC;
  carries `visible_from = d+1`; replay/training join GDELT day d to decision date D only
  where D ≥ d+1. Trailing-90d standardization uses days < d only. Dedicated unit test.
- **Dictionary freeze:** `theme_to_sector.json`, `ACTOR_MAP.json`, `bucket_map.json`,
  `THEMES_FIN` whitelist are authored from the proposals' enumerations, committed, and
  FROZEN **before any validation-fold model selection** (TR G4). The permuted-dictionary
  placebo (TOURNAMENT §4.6.3: 50 seeded permutations, real G1 must beat the 95th
  percentile) runs before any GDELT-fed model ships.
- **Distribution-shift gate** (TOURNAMENT §4.6.2) runs immediately after the top-up,
  before any GDELT-fed training: record-count median ratio ∈ [0.5, 2.0], tone shift ≤ 2 sd,
  bucket-share shifts ≤ 5 sd vs the 2025-09-01→2026-01-31 reference. Result committed to
  `prototype/gdelt_shift_gate.json`.

### 2.3 CBOE / FRED / COT / breadth

- **S1 CBOE** (`cdn.cboe.com/api/global/us_indices/daily_prices/<IDX>_History.csv`): VIX,
  VIX9D, VIX3M, VVIX, SKEW, COR3M, VXN. Features: term slope (VIX9D−VIX3M)/VIX, curvature
  (VIX − (VIX9D+VIX3M)/2), COR3M level + 1y z, VXN−VIX, VVIX z. Drop any row dated ≥ D.
- **S3 FRED** (existing key): NFCI (lag to following Wednesday), STLFSI4 (following
  Thursday), T10YIE Δ21, DFII10 level + Δ63, BAMLH0A0HYM2 level + Δ5, ICSA z (1-week pub
  lag). Daily series lagged 1 day.
- **S2 COT TFF** (`publicreporting.cftc.gov`): leveraged-fund net %OI + 1y z for E-mini
  S&P, UST 10Y, VIX futures; asset-manager net. **Keyed by Friday publication datetime,
  never report date**; `visible_from` = first night run after publication;
  forward-filled. Context-only (enters E2 + executive z; never a standalone driver).
  Shuffle-placebo falsifier kept (validation-fold).
- **S0 breadth** (from held prices, free): 64-ETF advance/decline, %>50d/200d MA,
  cross-sectional return dispersion, RSP−SPY 21d.
- **S4 Treasury auctions: NOT BUILT** (kill K13).

### 2.4 Engineered context

Existing `context.parquet` column family (rates, slopes, credit proxy, risk-off proxy,
vixy returns, vix term slope, vvix, skew) reproduced from deep history for pretraining via
`training/scripts/backfill_historical.py` conventions; live rows read from `daily/D/`.

---

## 3. LLM organ (PROPOSAL_LLM_SENTIMENT adopted; deltas listed)

- **Model:** `anthropic.claude-3-haiku-20240307-v1:0` on Bedrock (the one IAM-allowed ID),
  `temperature: 0`, `max_tokens: 1500`. Fallback chain: Bedrock (2 retries, exp backoff,
  30 s timeout) → 1 schema-repair retry → OpenAI `gpt-4o-mini` (existing key) → neutral
  artifact. `model_used`, prompt SHA-256, raw response stored per artifact.
- **Funnel F0–F4, verbatim from the proposal §1:** 4 GKG files/window; F1 relevance via
  THEMES_FIN ∪ ACTOR_MAP ∪ country-ETF locations; F2 slug-shingle cluster/dedup
  (`n_sources` = distinct domains); F3 5-day seen-cache with ×0.5/day salience decay (a
  story is billed once); F4 rank = `n_sources × (1+|tone|/5) × theme_priority`, **top 120**
  clusters; <10 clusters ⇒ skip call, emit neutral-with-flag.
- **Prompt + schema:** the proposal §2 system prompt and JSON schema VERBATIM (frozen
  strings shipped as `prompts/llm_sentiment_system.txt`, `schemas/llm_sentiment.json`).
  Buckets: the proposal's `bucket_map.json` table (27 keys incl. usd/eur/volatility;
  factor/style ETFs consume us_broad + global-axis loadings). 12 event flags, enum-locked.
- **KILLED (K6):** the event-token annotator extension. The output is the bucket surface +
  3 global axes + event flags ONLY.
- **Fail-soft:** unrecoverable failure ⇒ all-zero artifact with `llm_status ∈ {dark,
  thin_input, fallback_model}`; `llm_status_ok` and `llm_available` are emitted features.
  The organ degrades, never fabricates. Replays read stored artifacts only — the LLM is
  never re-invoked by any replay.
- **Backfill (identical funnel + prompt over archived GKG, chronological seen-cache):**
  1. Pilot: 10 days, ≈$0.05 — validates funnel + schema + truncation rate before bulk.
  2. **Tier 1 (MANDATORY): 2026-01-31 → present** (~89 d, ≈$0.40) — real LLM output over
     the entire replay window and holdout.
  3. Tier 2: 2024-01-02 → 2026-01-30 (~523 d, ≈$2.05) — training depth; runs only if
     cumulative spend < $3.00 at start.
  - **Phase C Bedrock hard cap: $3.10**, logged call-by-call into COST_WORKSHEET.md
    (TOURNAMENT D7). Pre-2024: features 0-filled with `llm_available=0`.
- **Model-cutoff rule** (TOURNAMENT §4.6.1) enforced in the backfill driver as a hard
  assertion: `model_cutoff_date < window_start_date` or the driver errors out and scores
  nothing. Artifacts frozen on model death; `model_used` stored per artifact.
- **Feature emission** (proposal §4 table): `llm_sent/conf/sal_<bucket>`, `_ema3`, `_d1`,
  `llm_risk_appetite`, `llm_rates_pressure`, `llm_geopol_risk`, `llm_event_<flag>` ×12,
  `llm_status_ok`, `llm_available`.
- **Routing (Alpha's table — fixed):** E2 GBM-Cond gets the symbol's-bucket sent/conf/sal
  + d1 + masks; E3 EventHead gets the full bucket surface + event-flag dummies (as
  mask-interaction columns, §5.3); executive z gets the 3 global axes + `llm_status_ok`;
  **CAST gets none** (text-free by design — the LLM arm must be removable without touching
  the transformer).
- **Stage-1 pre-integration kills (proposal §6, run after Tier 1):** degenerate variance;
  corr(`llm_sent_us_broad`, daily V2TONE aggregate) ≥ 0.8 ⇒ expensive tone proxy ⇒ organ
  killed and reported; event flags must fire on ≥3 known scheduled events and <20% of
  ordinary days. Plus: truncation rate (responses hitting max_tokens) must be <2%.

---

## 4. Feature store

One parquet per decision date D (prototype path `prototype/store/<D>/features.parquet` +
`tensors.npz`), built by a single `FeatureStore` class with the no-look-ahead conventions
of §2 baked into the join layer. Contents:

- **X[64, 63, 14]** per-symbol panel (CAST input), z-scored per feature over the trailing
  252 d (stats from days < D): `return_1d, return_5d, return_21d, return_63d, vol_21d,
  vol_63d, drawdown_21d, drawdown_63d, rel_strength_21d, rel_strength_63d, range_pct,
  volume_z_21d, gap_open_pct, dist_from_52w_high`.
- **Current scalars[64, 4]** (CAST token assembly): yesterday's return z, vol z,
  rel-strength z, volume z (cross-sectional).
- **T[64, ~55]** tabular row per symbol (GBM-Cond input): price block (8: ret 1/5/21/63,
  vol 21/63, dd 21, rs 21) + context block (~14) + CBOE S1 (5) + FRED S3 (7) + COT S2 (3)
  + G1 bucket-z + G3 bucket tone for the symbol's bucket (4) + G4 novelty pair + G5 HHI
  (4) + LLM bucket sent/conf/sal + sent_d1 (4) + masks `llm_available, cot_fresh,
  gdelt_available` (3) + S0 breadth (4).
- **B[27, ~25]** bucket event panel (EventHead design matrix): bucket sent/conf/sal +
  sent_d1 (LLM, as mask-interactions) + G1 theme-z + G3 tone + tone-dispersion + G2
  Goldstein/conflict-share (country buckets) + burst z + novelty JSD + HHI + event-flag
  dummies (12, mask-interacted) + `llm_available`.
- **z[~24]** executive context: context block compressed (10: spy_ret_21, spy_vol_21,
  rate_2y, rate_10y, yield_slope, credit_spread, risk_off, vixy_ret_21, vix_term_slope,
  skew z) + CBOE (4: COR3M z, curvature, VXN−VIX, VVIX z) + LLM axes + status (4) +
  **information-health block (6, Gamma graft): `gdelt_available, llm_status_ok,
  n_clusters_z, theme_novelty, field_dispersion (G1 bucket-share entropy), 
  days_since_last_gdelt`**.
- **Portfolio state:** `w_prev[64]`, gross_prev, trailing-21d realized book vol, own-equity
  drawdown-from-peak.

---

## 5. Ensemble members (all supervised; per-fold walk-forward; `walk_forward: true`
manifests mandatory — the executive refuses to train without them, build error)

### 5.1 E1 — CAST-Small (the transformer; ≈16k params)

```
Per-symbol temporal encoder (ONE GRU shared by all 64 symbols):
  GRU(input 14 → hidden 32), over X[s, 63, 14]; last hidden = symbol summary   ≈ 4.5k
Token assembly: token[s] = Linear48→32([GRU 32 | sector_emb 8 (13 sectors)
  | class_emb 4 (6 classes) | current scalars 4])                              ≈ 1.7k
  NO symbol-identity embedding (anti-memorization spine).
Cross-sectional encoder: 1 pre-LN transformer layer, d_model 32, 4 heads,
  FFN 64, dropout 0.20 (attention + token)                                     ≈ 8.5k
Heads (shared): mu 32→16→1; sigma 32→16→1 softplus;
  AUX next-day return-z head 32→16→1 (training only)                           ≈ 1.6k
TOTAL ≈ 16k. Weight decay 1e-3. (CAST-45k and the 2-layer variant: NOT trained — K11.)
```

- **Target:** y[s,D] = open(D)→open(D+5) return minus cross-sectional mean, **per-day
  Gaussian-rank transform** (TR A2; replaces z-clip). Aux target: next-day (D open→D+1
  open) per-symbol return z.
- **Loss:** `Huber(mu, y)·w_rec + 0.25·(1 − soft-rank Spearman surrogate) + GaussNLL(sigma)
  + 0.3·MSE(aux)`; `w_rec` = Infotropy-B sample weight (§9.2).
- **Training:** each day = one batch of 64 tokens; folds per §10.1; early stop on purged-
  fold Spearman (patience 10); OOF seeds {101,102,103}; deploy ensemble seeds
  {11,13,17,19,23}, outputs averaged. c1 = seed-ensemble agreement (mean pairwise rank
  corr of the 5 seeds' mu).
- **Ridge twin (qua-transformer referee):** cross-sectional ridge on the identical
  flattened 63×14=882 inputs + the 4 scalars, per-day Gaussian-rank target, λ ∈
  {1e0…1e4} by fold-internal CV. Trained in the same run, always. CAST ships in the slot
  regardless of the twin result (item-1 presence mandated); attribution is read vs the
  twin (TOURNAMENT §4.3).
- **Functionality bar:** purged-validation weekly rank IC ≥ 0.02 (TOURNAMENT §4.6.4);
  below it the pre-registered conclusion prints.

### 5.2 E2 — GBM-Cond (≈2.3k leaf values)

`sklearn.ensemble.HistGradientBoostingClassifier`: **max_iter 150, max_leaf_nodes 15**,
max_depth 6, learning_rate 0.05, l2_regularization 1.0, **early stopping on purged
validation (binding)**, monotone constraints where economically signed (HY-OAS Δ negative
for credit-sensitive buckets; COR3M z negative for gross-equity P(up)). Input T[~55].
Label: 1{5d excess return > 0}. Output: P → probit z → cross-sectional z = mu2; sigma2 =
rolling per-bucket OOF residual sd; isotonic calibration on OOF folds; c2 = calibrated
mean |P−0.5|. Sample weights = w_rec.

### 5.3 E3 — EventHead (675 coefs)

Per-bucket elastic net over B[27, ~25] (shared design matrix; α by fold-internal CV on
training folds only). **LLM columns enter as mask-interaction pairs** `(x·llm_available)`
plus the mask itself (SK A-K3 fix) so pre-2024 absence ≠ neutral-sentiment. Target:
bucket 5-day abnormal (market-removed) return. mu3 = bucket prediction × `event_weight_cap`
(gene) mapped to symbols via bucket_map; sigma3 = rolling bucket residual sd; c3 = today's
gated event mass (post Infotropy-A runtime gate, §9.1); on quiet nights mu3 → 0 (abstains).
Training span 2015-02-18 →; LLM columns zero+masked before 2024-01.

### 5.4 E4 — RiskNet+ (instrument; ≈150 coefs, ridge)

Three ridge heads on `[vol_21d, vol_63d, vol_5d (HAR daily term), range_pct, VIX level,
S1 slope, S1 curvature, COR3M z, VVIX z, G4 novelty, G5 HHI, llm_geopol_risk]`:
(a) per-symbol forward 5d realized vol `sigma_hat[s]`; (b) per-symbol 63d beta-to-SPY
`beta_hat[s]`; (c) forward 5d realized vol of the equal-weight 64-ETF book
`book_vol_hat` (the executive's vol-cap input). Validation metric QLIKE. The feature set
embeds HAR-RV terms, satisfying TR §3.6's simpler-twin demand by construction (ridge IS
the simple form; no GRU is built — K7). sigma_m for members = member rolling residual sd ×
(sigma_hat / cross-sectional mean sigma_hat) scaling.

### 5.5 Floor baselines (not members; validation-only, zero holdout looks)

Beta's P2 pair as dumb twins: `w ∝ softplus(W·x_s)` linear policies (~30 params each),
slow {ret21, ret63, vol63, dd63, rs21, rs63, breadth} and fast {ret1, ret5, vol21, dd21,
rs21, vixy-beta}, fit by ridge on the utility-proxy (5d forward excess); plus an
equal-weight risk-targeted book. SYN-1 must beat both on pre-holdout validation utility
before its single holdout read; failure is reported and the bake-off proceeds anyway.

---

## 6. Solo-book rule and expert contract

**Fixed deterministic rule (no parameters; same rule = the counterfactual ledger is exact,
Beta's chassis):** for member m on date D:

```
mu_z      = clip(mu_m, ±q95 cross-sectional)            # q95 fixed per day
prec[s]   = 1 / max(sigma_m[s], sigma_floor=0.25)^2
w_raw     = max(mu_z * prec, 0)                         # long-only
w_m       = min(w_raw / sum(w_raw), w_cap=0.10) renormalized to unit gross
```

`expert_opinions.parquet` per D: `member ∈ {cast, gbm_cond, event_head}`, mu[64],
sigma[64], c, manifest {model_version, train_window_end, walk_forward: true, seeds,
code_sha}. `solo_books.parquet`: w_m[64] per member. `risknet.parquet`: sigma_hat,
beta_hat, book_vol_hat. `trust_ledger.parquet` per the meta proposal §4 with **per-fold
standardization of all rolling stats** (SK §0.5): date, member, u_m realized (solo book
through the same half-spread cost table at reference f=0.7 + vol cap), ewma21, ewma63
(both standardized within fold), hit_rate, cf_drawdown, tau_assigned. Ledger lag: rolling
stats at D use only windows fully realized by D−1 ⇒ latest usable decision date D−h−1
(unit-tested).

---

## 7. The executive (≈560 params)

### 7.1 Architecture (PROPOSAL_META_EVALUATOR + Beta blend + Gamma health block)

```
phi: 1 hidden layer width 8, tanh, SHARED across members
  inputs per member: [r_m (4 ledger stats), c_m, agree_m, z (24)]  → 30 dims  ≈ 248 params
trust logit s_m = v·phi_m + b + b_m (per-member bias, 3)                       ≈ 12
tau = softmax(s/T); entropy floor: tau ← 0.95·tau + 0.05/3                     (eps=0.05)
BLEND (convex, book-space — Beta): w_unit = Σ_m tau_m · w_m   (unit gross by construction)
sizing psi: 1 hidden layer width 8 over [z 24, g 3, |mu_blend| stats 3,
  tau entropy 1, ledger means 2, book_vol_hat 1, own drawdown 1] = 35 dims     ≈ 297
f = f_max · sigmoid(psi);  w_tgt = f · w_unit, then vol cap:
  scale down so book_vol_hat(w_tgt) ≤ sigma_cap (gene vol_target_ann)
TOTAL ≈ 560 params (hard ceiling 3k). g = {mean pairwise rank-corr of solo books,
  mean pairwise L1 distance, gross dispersion}.
```

z includes the **information-health block** (§4) — the executive can learn to fade
GDELT/LLM-fed members on dark/thin nights and to shorten effective trust via context.

### 7.2 Loss (differentiable decision replay; proposal §3.1 + SK B-K2 fix)

```
U_h(D) = (1/5)Σ_{t=D..D+4} log(1+r_book_t) − λ_dn·(1/5)Σ min(0,r_book_t)²
         − cost(D)/5
cost(D) = Σ_s |w_tgt_s − w_prev_s| · (half_spread_bps(sector_s) + η_D)/1e4
          η_D ~ U(−2,+2) bps slippage NOISE injected at train time (B-K2)
Loss = −mean_D[ w_rec(D) · U_h(D) ] + β_to·mean_D ‖w_tgt(D)−w_tgt(D−1)‖₁
       + β_tr·KL(softmax(u_realized/T_u) ‖ tau(D))   (aux trust alignment, β_tr=0.1)
       + entropy-floor penalty + weight decay
```

λ_dn = gene `risk_aversion_lambda` (B0 default 2.0); β_to = 0.5 (a-priori, §16); smooth-abs
everywhere; no-trade band straight-through at train, hard at inference. Teacher-forced
first epoch, sequential w_prev thereafter (proposal §3.2). Trains only on OOF member
opinions/books (stacking discipline; attestation enforced).

### 7.3 Training

5 seeds {11,13,17,19,23}, outputs averaged; 200 epochs cap, early stop on purged
validation utility; dropout 0.1 on z; input noise on ledger stats; **linear-gate twin**
(`tau = softmax(A·[r,c,g]+B·z)`, ~100 params) trained in the same run — ships if the MLP
cannot beat it on validation (rung logged); GBM challenger trained at validation only.
**Six LOFO executives** (trained without fold f each) are produced for the EA (§8) — TR S2.

### 7.4 Audit record

`meta_decision.json` per D, schema verbatim from the proposal §6 (trust, logit_attrib via
16-step integrated gradients by input group, sizing_attrib, expert_share per holding,
counterfactuals, expected U_h; realized back-filled at D+5).

### 7.5 Kill criteria (proposal §7, adopted; read per-fold for static trust)

Equal-trust+fixed-f LOO baseline (R08); static trust std(τ)<0.02 per fold; trust collapse
vs positive ledger; calibration corr ≤ 0; challenger parity ⇒ ship simpler rung and say so.

### 7.6 Fine-tune (TR A5/S4 — exact parameter set)

Window: live record 2025-08-04 → **2026-03-03** (= holdout_start − h − 1, by formula).
Frozen: phi, v, blend, psi hidden layer. **Tuned (7 effective params): b_m ×3, softmax
temperature T, psi output gain, psi output bias, f_max sigmoid bias.** LR ×0.1, ≤50
epochs, early stop. Pre/post validation delta reported; de-claimed if ≈0; never narrated
as adapting to "real" fills (same simulator).

---

## 8. Evolution (PROPOSAL_EVOLUTION machinery verbatim; genome below)

- **(μ+λ) GA:** P=28, G=14, elitism 2, K≤400; tournament size 3; uniform crossover
  p_swap=0.5 on 70% of offspring; float mutation σ=0.10 annealed ×0.85/gen, p=0.30;
  binary bit-flip 0.05; early stop after 4 stagnant generations; seed
  `sha256("PKT-TB-006-EA"+window_end)[:8]`; every genome + fitness logged
  (`prototype/ea/generation_<k>.jsonl`).
- **Fitness:** per fold f: fast vectorized walk over **cached OOF matrices using the LOFO
  executive for fold f** (TR S2); `U_f = √252·mean(r)/std(r) − 0.5·MaxDD/0.10`; min over
  cost scenarios c ∈ {1.0, 1.5}; `FITNESS = mean_f − 0.5·std_f − 0.02·(#active gates)`;
  4-of-6 fold subsampling per generation, elites + champion rescored on all 6. Fitness
  data ends 2026-02-06 (21 td before holdout); the EA never sees the holdout.
- **Baselines (pre-committed):** B0 DEFAULT_GENOME in the population every generation;
  **adoption gate: champion ships only if FITNESS(champ) − FITNESS(B0) > 1.0 × cross-fold
  sd, else B0 ships and the dossier says so**; B1 budget-matched random search (K=366,
  same folds/seeds); B2 = replay R09 (TOURNAMENT §4.3). The proposal's three ceremonial-
  verdict sentences apply verbatim.

### 8.1 Genome (27 genes; decoded `genome.json`; ranges are part of this pre-registration)

| Gene | Type/range | B0 default |
|---|---|---|
| trust_prior[3] (cast, gbm, event) | logit [−2, +2] | 0.0 each |
| trust_halflife_days | log [5, 60] | 21 |
| member_gate[3] | {0,1} | all 1 |
| feature_gate[8]: {G1-themes, G2-country, G3-tone, G4G5-novelty-conc, LLM-block, CBOE-S1, FRED-S3, COT-S2} | {0,1} | all 1 |
| gross_target | [0.30, 1.00] | 0.60 |
| vol_target_ann (σ_cap) | log [0.06, 0.18] | 0.10 |
| max_symbol_weight | [0.02, 0.15] | 0.08 |
| dd_brake_threshold | [0.05, 0.20] | 0.10 |
| dd_brake_strength | [0.0, 1.0] | 0.50 |
| no_trade_band (weight space) | [0.000, 0.030] | 0.010 |
| conviction_temp | log [0.25, 4.0] | 1.0 |
| cash_floor | [0.00, 0.30] | 0.10 |
| risk_aversion_lambda (λ_dn) | log [0.5, 8.0] | 2.0 |
| abstain_threshold | [0.0, 0.5] | 0.10 |
| **record_weight_eps** | **[0.20, 0.50]** (D4 adjudication) | 0.25 |
| **event_weight_cap** | [0.5, 2.0] | 1.0 |

(`p_min` removed — K10/K18. Hard rails NOT genes: gross ≤ 1.0, harness cluster cap,
universe, fold/embargo scheme, fitness constants, cost model.)
Cadence: EA monthly on cached matrices (3–8 min, cap 25); nightly Lambda reads the static
~5 KB genome; daily trust adaptation = the genome-parameterized deterministic decay rule
(no online learning).

---

## 9. Infotropy mechanisms (the two TRANSFERS, Alpha's scoping)

### 9.1 Transfer A — record-grade ingestion screen (train-time, family-level)

Applied to candidate event feature FAMILIES entering EventHead/GBM (G1 buckets, G2
country, G3 tone, burst, novelty, HHI, LLM event flags) — **never per-event at inference**
(K5): per training fold, family F passes iff conjunctively:
- **R1 persistence:** mean over family-active days of 1{|regime_stat shift over [t+1,t+h]|
  > k·sd_base}, k=1, h=10, regime_stat = bucket 21d realized vol and 21d drift (both must
  pass at the 0.5 level on average).
- **R2 non-self-encoding:** 1 − OOS R² of family intensity ~ lagged bucket returns
  [t−5,t−1] ≥ 0.5 (walk-forward regression; kills price-echo families).
- **R3 downstream reuse:** OOS AUC/IC lift of the fold model with vs without F > 0
  (per-family ablation — ~10–20 fits/cycle, FA B.4).
Failing families are routed out of the design matrix for that fold. **Inference-time:
R2-only down-weight** — event-day rows whose intensity is well-explained by the symbol's
own trailing returns (rolling R2 test, past data only) are down-weighted ×0.25 in
EventHead's design matrix (leak-free: R2 needs only past returns). **Falsifier:** R3-only-
screened twin trained per fold; if the conjunctive gate adds no walk-forward lift, the
gate is dead and R3 screening remains; verdict prints in the scorecard parentheses.

### 9.2 Transfer B — record-formation label weighting (anti-HILL)

Per training sample (move at t, symbol/bucket s, horizon h matching the learner):
`score = clip(|regime_stat(s,[t+1,t+h]) − regime_stat(s,[t−h,t−1])|/sd_base ×
(1 − fraction_retraced_within(s,t,h)), 0, 1)`;
`w_rec = clip(eps + score, eps, 1)` with eps = `record_weight_eps` (gene, default 0.25,
floor 0.20 — D4). Applied to CAST loss, GBM sample weights, executive utility terms.
**Mandatory A/B falsifier:** every weighted learner has a uniform-weight twin trained in
the same cycle; record-weighting ships only on a purged-validation win; replay arm R12 is
the scorecard read.
(C, D, E content exists under standard names — confidence inputs, ledger/drawdown inputs,
breadth features — and is never labeled infotropy. F not built.)

---

## 10. Training procedure

### 10.1 Fold geometry (single source of truth for members, executive, EA)

```
F1 2020-02→2021-02  F2 2021-02→2022-02  F3 2022-02→2023-02
F4 2023-02→2024-02  F5 2024-02→2025-02  F6 2025-02→2026-02   (~250 td each)
Head-training for fold f: expanding window 2014-08-29 (CAST/RiskNet) or 2015-02-18
(GBM/EventHead GDELT-rich panel) → fold_f.start − 22 trading days.
EMBARGO: 22 trading days (h_max+1; TR S1) at every boundary.
Fitness/selection data ends 2026-02-06. HOLDOUT 2026-03-11→ : never touched by any
training, tuning, ladder decision, or EA pass; read only per TOURNAMENT §4.
2014–2019: head-training substrate only, never a fitness fold.
```

### 10.2 Target construction (no-look-ahead, audited)

Features for D use closes ≤ D−1 (the `daily/D/` convention); y uses open(D)→open(D+5)
from refetched Stooq OHLCV; GDELT joins via `visible_from`; COT by publication date; FRED
weekly by documented lags; LLM artifacts from files stamped before D 03:00 UTC (≥10.5 h
pre-open). Overlapping-window statistics reported HAC (Newey–West, 10 lags) or
non-overlapping only.

### 10.3 Seeds & determinism

Master/component/deploy/OOF seeds per TOURNAMENT §4.4 (fixed list). All seeds, code SHA,
params hash, snapshot range, cost-model version, wall-clock, and exact command in every
manifest. Replays pin to S3 snapshots via the disk cache; no live re-fetch mid-experiment.
Mac venv torch version recorded as found (2.10.0 per FA) — pinned in the manifest, not
changed.

### 10.4 Mandatory alignment tests (run before any member trains)

(a) 20 random (symbol, date) pairs in the overlap window: pretrain fill price ==
replay-engine fill price (VUG 6:1 applied once, never twice); (b) ledger D−h−1 lag unit
test; (c) GDELT `visible_from` join test; (d) COT publication-key test; (e) feature-store
row for a live date == rebuilt-from-deep-history row (tolerance 1e-6 on shared columns).

### 10.5 Monthly-cycle budget (one full training cycle, Mac CPU)

```
data refresh + GDELT incremental + CBOE/FRED/COT pulls          ~10 min
CAST-Small: 6 folds × 3 OOF seeds + 5-seed deploy + ridge twin  ~45–70 min
GBM-Cond 6 folds + EventHead + RiskNet+ (+ uniform twins)        ~15 min
trust-ledger rebuild (vectorized)                                 ~3 min
executive: 5 seeds + linear twin + 6 LOFO executives              ~8 min
EA: P=28 × G=14 cached-matrix walk + B1                           ~8 min (cap 25)
TOTAL                                                             ~1.5–2.2 h
Shrink order (packet rule): OOF seeds 3→2 → deploy 5→3 → CAST early-stop patience →
EA G 14→8 → GDELT panel-start 2019. Stated on the final line if taken.
```

---

## 11. Intent-emission contract (replay-harness integration)

For decision date D the brain reads `daily/D/*` (+ its own stored artifacts for D) and
emits:

```
delta[s] = w_tgt[s] − w_prev[s]
intents  = [{"symbol": s, "action": "BUY" if delta>0 else ("SELL" if w_tgt[s]≈0 else
             "REDUCE"), "sizing": abs(delta[s]) × NAV_at_last_mark}
            for s where abs(delta[s]) > no_trade_band]
```

written to the run-dir equivalent of `daily/D/trade_intents.json`. Harness side:
`run_variant(cache, variant, strategy=SYN1Strategy, trading_dates, universe_df)` with a
`Strategy` adapter calling the brain's nightly function; fills at next-session open via
`_execute_intents` + `apply_transaction_costs` (seeded rng per TOURNAMENT §4.4); harness
cluster cap applies downstream (the brain's own per-symbol cap 0.10 and cluster-aware
gross genes sit tighter than the rail in normal operation); marks at D close. Replays
never re-call the LLM and never re-fetch data. Expected one-way turnover ≤ 6%/day
(pre-registered budget; printed in the dossier).

---

## 12. Artifacts and manifests

- **Per-day (prototype mirror of `daily/<D>/`):** `llm_sentiment.json`,
  `gdelt_rich_row.parquet`, `expert_opinions.parquet`, `solo_books.parquet`,
  `risknet.parquet`, `meta_decision.json`, `trade_intents.json`.
- **Per-cycle:** versioned model artifacts + `latest.json` pointer (existing `models/`
  convention, written to the run dir, NOT to S3 models/), `genome_<date>.json`,
  `ea_manifest_<date>.json`, `ea/generation_*.jsonl`, `trust_ledger.parquet`,
  fold OOF matrices (`oof/fold_<f>_<member>.npz`).
- **Per-comparison (EVIDENCE_PROTOCOL):** one JSON + one MD with full-period AND
  holdout-only deltas, paired daily stats, run manifest. All N variants logged.
- **Ledgers:** `holdout_looks.jsonl` (init 0, budget ≤20), `validation_looks.jsonl`
  (init 0), `bedrock_spend.jsonl` (call-by-call, cap $3.10).

---

## 13. Attribution battery implementation

Exactly the 14 planned runs + ≤6 contingency of TOURNAMENT §4.4. Implementation notes:
- Drop arms (R04–R07): zero the member's book, renormalize τ over survivors (entropy floor
  redistributed); same genome; NO retrain (K17).
- Retrain arms (R03, R10–R12): full per-fold retrain of affected components + executive;
  E1 read = pooled F1–F6 OOF daily utility difference (with vs without), HAC t; E2 read =
  the replay pair vs R01 on identical dates, slippage seed 4242.
- Every arm computes BOTH the E1 primary statistic and the E2 sign confirmation before any
  verdict is written; verdict assignment is mechanical per TOURNAMENT §4.3.
- One local disk cache serves the whole battery; replays ~1–3 min each.

---

## 14. Cost worksheet (deployed shape — for the dossier's mandatory itemization)

| Line | Arithmetic (FA audited, list price + conservative carry) | Marginal $/mo | Absolute |
|---|---|---|---|
| Lambda night | 110→~175 s (+25 s GDELT 8 zips, +40 s LLM funnel/call, +5 s inference, +5 s CSVs) × 22 × 2.94 GB ≈ 11.3k GB-s | +$0.06 | $6.20 |
| Lambda morning | unchanged | +$0.00 | incl. |
| Bedrock Haiku | (10k in + 1.5k out) × 22 × 1.5 headroom | +$0.15 | $0.15 |
| S3 storage + requests | +~20–25 MB steady + ~70 req/day | +$0.01 | $1.06 |
| ECR | no new heavy deps (torch/sklearn in image) | +$0.00 | $0.12 |
| Secrets / CloudWatch / SNS | unchanged | +$0.00 | $2.00 |
| Data transfer | inbound free | +$0.00 | $0.01 |
| **TOTAL** | | **≈ +$0.25/mo marginal** | **≈ $9.5/mo** (vs $10 target; ≈$3.6 at honest list price) |
| One-time (Phase C) | Bedrock pilot+T1+T2 ≤ $3.10 cap; GDELT backfills $0 cash (Mac time); training $0 | ≤$3.10 once |

Lever (Beta, free): merge the LLM funnel's GKG parse into the gdelt_rich fetch pass
(−20 s Lambda). Contingency: Haiku retirement ⇒ artifacts frozen (§3); live organ falls to
gpt-4o-mini at comparable cents; IAM widening to a Haiku-4.5-class ID is flagged as a
follow-on (≈$0.55/mo Bedrock — in envelope), never assumed.

---

## 15. Build order (with wall-clock budgets; foreground target ≈5–7 h)

```
0. LAUNCH GDELT top-up backfill 2026-02-05→present (background; ≤1.5 h)      [first action]
1. LAUNCH deep GDELT backfill (2/day 2015-02→2022-12, 4/day 2023→; background 5–10 h)
2. Disk cache over S3Cache; pull replay-window snapshots (~45 MB)                 ~15 min
3. Feature store + §10.4 alignment tests + CBOE/FRED/COT pulls                    ~1–1.5 h eng
4. Dictionaries authored from the proposals' enumerations; FREEZE COMMIT;
   permuted-dictionary placebo harness                                            ~30 min
5. GDELT shift gate (needs top-up done) → gdelt_shift_gate.json                   ~10 min
6. LLM organ: funnel+prompt+schema; pilot 10 d ($0.05) → Tier 1 ($0.40) →
   Tier 2 ($2.05, conditional <$3 cumulative); Stage-1 kills checked              ~1–2 h (API-bound;
                                                                                   overlaps step 3)
7. Members: RiskNet+ → EventHead (+Infotropy-A screen) → GBM-Cond → ridge twin
   → CAST-Small (6 folds × 3 seeds + deploy 5) → uniform-weight twins             ~1.5–2 h
8. Solo books + trust ledger + executive (5 seeds + linear twin + 6 LOFO) +
   fine-tune (cutoff 2026-03-03, 7 params)                                        ~30 min
9. EA + B0/B1 + adoption gate → champion-or-B0 genome                             ~15 min
10. Floor-baseline checks (validation); diversity statistic; break-even IC line;
    validation-look ledger flushed                                                ~15 min
11. FREEZE the single SYN-1 configuration; registration commit
12. Wire Strategy adapter; battery R01–R14 (+contingency as triggered); BAKEOFF.md,
    ATTRIBUTION.md, COST_WORKSHEET.md                                             ~3–4 h
```

Shrink ladder if the session breaks: §10.5 order; never drop an organ, never drop the
bake-off; every reduction on the final line.

---

## 16. Registered constants (provenance — SK B-K4; everything below is a-priori unless noted)

| Constant | Value | Provenance |
|---|---|---|
| h (member/executive horizon) | 5 td | meta proposal §3.1, a-priori |
| Embargo | 22 td | TR S1 (h_max+1) |
| Holdout start / fine-tune cutoff | 2026-03-11 / 2026-03-03 | optimizer config / TR S4 formula |
| q95 clip, sigma_floor, w_cap | per-day 95th pct / 0.25 / 0.10 | solo-book rule, a-priori |
| Entropy floor eps_τ | 0.05 | meta proposal |
| β_to / β_tr / aux-head weight | 0.5 / 0.1 / 0.3 | a-priori (meta proposal scale; TR §2.1) |
| Gaussian-rank target; wd 1e-3; dropout 0.2 | — | TR A2/A1 |
| GBM caps 150 iters / 15 leaves | — | TR A3 |
| EventHead 27×25; α by fold-CV | — | Alpha E3 |
| Infotropy-A: k=1, h=10, R2 bar 0.5, runtime ×0.25 | — | Liaison A + Alpha scoping, a-priori |
| record_weight_eps default 0.25, floor 0.20 | — | D4 adjudication |
| Reference sizing f=0.7 + vol cap (ledger + LOO baseline) | — | meta proposal §3.3/§7 |
| EA P/G/K, fitness constants, cost scenarios {1.0,1.5} | — | EVOLUTION proposal (fixed, not genes) |
| LLM: top-120 clusters, max_tokens 1500, temp 0, 4 files/day | — | LLM proposal |
| Turnover budget ≤6%/day one-way | — | Beta §4 (the bound, pre-registered) |
| Battery caps 20 replays / 9 retrains; slippage seed 4242 | — | FA B.3 / TOURNAMENT §4.4 |
| Diversity floor 0.90; tone-proxy bar 0.8; truncation bar 2% | — | SK B-K1 / LLM proposal / G-K2 fix |
| Functionality bar: weekly rank IC ≥ 0.02 | — | SK A-K4 derivation (TOURNAMENT §4.6.4) |

Anything not in this table that turns out to be decision-relevant is appended here with
provenance and counted in the validation-look ledger.

---

*End of BUILD_SPEC. The builder's first action is step 0 of §15. No holdout replay before
TOURNAMENT.md §5's three preconditions are met.*
