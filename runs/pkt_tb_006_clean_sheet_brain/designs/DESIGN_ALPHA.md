# DESIGN_ALPHA — "FORECAST FIRST" — the prediction-stack trader's brain

**Panel role 3 of 12 — Architect-Alpha (prediction-stack prior)**
**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — 2026-06-10
**Posture:** designed blind to the incumbent (sealed-incumbent rule honored; sources: packet,
ASSIGNMENT_BRIEF.md, EVIDENCE_PROTOCOL.md, the Liaison's INFOTROPY_TRANSFER.md, and the four
blind specialist proposals). Universe = the verified 64 ETFs of `config/universe.csv`.

---

## 1. System thesis

The edge in a forecast-first brain does not come from predicting *the market* — ~2,900 daily
observations spanning ~6 regimes cannot support that and the brief says so. It comes from
predicting the **cross-section**: which of 64 highly-related ETFs will do relatively better
over the next five sessions, a target that is market-neutral by construction (z-scored per
day), refreshes 64 correlated-but-distinct readings every day, and is where daily-bar
predictability actually lives (relative momentum/reversal structure, vol-risk transfer,
news-to-sector transmission). So the brain is a stack: **a small set of specialized
forecasters, each owning one predictable component** (relative price structure; exogenous
tabular conditioning; event/news shocks; risk magnitude), standardized into one currency
(expected 5-day excess return, cross-sectionally z-scored, with uncertainty), then a
**meta-labeling layer** that decides *whether acting on the blended forecast clears costs*,
and a **learned executive** that decides *how much to listen to which forecaster today and
how much capital to deploy* — trained end-to-end on realized forward utility of its own
choices, never on labels about the world. Forecasting is the perception; meta-labeling is the
precision trigger; the executive is the judgment. Every learnable layer is sized to the
sample that actually exists.

---

## 2. The ensemble — four forecasters + one trigger, five genuinely different jobs

One shared currency (the executive's contract, §5): each μ-expert emits, nightly for decision
date D from data through D-1 close, `mu[s]` = expected 5-day open(D)→open(D+5) excess return,
z-scored cross-sectionally, plus per-symbol `sigma[s]` and a scalar self-confidence `c`.
M = 3 μ-experts (E1–E3). E4 and E5 are non-μ organs with their own contracts.

### E1 — CAST: Cross-Asset Sequence Transformer (the flagship; assignment item 1's transformer)

**Job:** relative-value ranking of the 64-name cross-section from price/vol structure alone.
It answers "which symbols are rich/cheap vs their peers this week," nothing else.

**Architecture (exact):**

```
Per-symbol temporal encoder (ONE network shared by all 64 symbols):
  input:  63 trading days × 14 features per symbol
          [return_1d, return_5d, return_21d, return_63d, vol_21d, vol_63d,
           drawdown_21d, drawdown_63d, rel_strength_21d, rel_strength_63d,
           range_pct (hi-lo)/close, volume_z_21d, gap_open_pct, dist_from_52w_high]
  GRU: input 14 → hidden 32, last hidden state = 32-dim symbol summary       ≈ 4.5k params

Token assembly (per symbol s):
  token[s] = [GRU summary (32) | sector embedding (8, 13 sectors from universe.csv)
              | asset-class embedding (4, 6 classes) | 4 current scalars
              (yesterday's return z, vol z, rel-strength z, volume z)]  → d_model = 48
  NO per-symbol identity embedding — symbols are distinguishable only by behavior and
  sector/class. This is the anti-memorization spine.                        ≈ 0.13k params

Cross-sectional transformer (attention across the 64 tokens of ONE day):
  2 layers, d_model 48, 4 heads, FFN 96, pre-LN, dropout 0.20
  context length = 64 tokens (the cross-section). Time is handled by the GRU;
  attention is spent where the target lives: symbol-vs-symbol relative structure.
                                                                              ≈ 38k params
Output heads (shared across symbols):
  mu head 48→16→1; heteroscedastic sigma head 48→16→1 (softplus)             ≈ 1.6k params

TOTAL ≈ 45k parameters (fallback variant d_model=32/FFN 64 ≈ 22k, see ladder below).
```

**Targets and loss:** y[s,D] = open(D)→open(D+5) return minus the cross-sectional mean,
z-scored per day (clipped at ±3). Loss = Huber(mu, y)·w_rec + 0.25·(1 − daily Spearman
surrogate via soft-rank) + Gaussian NLL term for sigma, where `w_rec` is the infotropy
record-formation sample weight (§8, Transfer B). Trained with each day as one batch of 64.

**Why it won't memorize 2.9k days (brief §7, binding — the Training Realist's audit line):**
(a) the target is **cross-sectional and market-neutral** — the component with only ~6 regimes
of variation (the market) is subtracted out by the per-day z-score; what remains is relative
structure with ~64 readings/day → ~186k symbol-day targets, of which a conservative effective
count (correlation clusters of ~8 truly distinct behaviors/day × ~580 non-overlapping 5-day
windows) is **~20–35k effective samples vs 45k weights** — aggressive but defensible only
because of (b)–(f); (b) zero symbol-identity capacity: one shared GRU, one shared
transformer, sector-level embeddings only (a symbol cannot be looked up, only described);
(c) dropout 0.20, weight decay 1e-4, early stop on purged-fold Spearman; (d) **5-seed
ensemble averaged at inference** (seeds fixed, recorded); (e) ranking-flavored loss — rank
targets are far less memorizable than levels; (f) the **simplification ladder** (mandatory,
reported): CAST-45k vs CAST-22k vs cross-sectional ridge on the identical 14×63 feature set.
If ridge matches CAST on pre-holdout validation Spearman, the dossier says the transformer
does not do real work — that is the honest assignment-item-1 answer, pre-registered.

### E2 — XGB-Cond: gradient-boosted conditional-direction model

**Job:** per-symbol *absolute* 5-day direction probability conditioned on everything tabular
and exogenous — macro (FRED deep catalog), vol surface (CBOE), positioning (COT), GDELT-rich
aggregates, LLM features where they exist. The inductive bias CAST lacks: axis-aligned
thresholds, interactions, native missing-value handling (LLM/COT masks).

- `sklearn.ensemble.HistGradientBoostingClassifier` (already in requirements-training):
  ≤300 iterations, 31 leaves, depth ≤6, lr 0.05, L2 1.0 → **≤9.3k leaf values** vs ~150k
  pooled symbol-day rows per training fold. Monotone constraints where economically signed
  (e.g. HY-OAS Δ negative for credit-sensitive buckets).
- Features (~55): per-symbol price block (8) + context.parquet block + CBOE S1 block
  (term slope/curvature, COR3M z, VXN−VIX) + FRED S3 block (NFCI z, DFII10 level/Δ63,
  T10YIE Δ, HY-OAS level/Δ) + COT S2 (3 nets % OI, z) + GDELT G1 bucket-z + G3 bucket tone
  + LLM sent/conf/sal for the symbol's bucket + masks (`llm_available`, `cot_fresh`).
- Output adapted to currency: P(excess>0) → probit z → cross-sectional z-score; sigma = its
  rolling per-bucket residual sd; calibration by isotonic fit on OOF folds.

### E3 — EventHead: news-shock transmission model

**Job:** the event specialist. Given tonight's *event structure* (GDELT G1–G5 + LLM bucket
sentiment + LLM event flags), forecast the abnormal (market-removed) 5-day return of the 27
news buckets, mapped to symbols via `bucket_map.json`. It is deliberately tiny and only
speaks when events do.

- Per-bucket elastic-net linear model, shared design matrix (~25 event features:
  bucket sent/conf/sal + sent_d1 + bucket theme-z + bucket tone + burst z + novelty JS +
  concentration HHI + relevant event-flag dummies): 27 × 25 = **675 coefficients**, α tuned
  by fold-internal CV only.
- **Infotropy Transfer A (record-grade gate) lives here** (§8): candidate event features pass
  the R1∧R2∧R3 conjunctive screen in walk-forward training; at inference, events failing the
  trace test (R2-style price-echo check) are down-weighted ×0.25 before entering the design
  matrix. EventHead emits c (self-confidence) = today's gated event mass; on quiet nights its
  mu → 0 and the executive's trust ledger sees it abstaining rather than guessing.
- Training span: 2015-02-18 → (GDELT-rich availability); LLM features carry the
  `llm_available` mask and are zero before 2024-01 (Tier 2 backfill boundary).

### E4 — RiskNet: forward risk magnitude (not a μ-expert)

**Job:** direction-free risk perception: per-symbol forward 5-day realized vol
`sigma_hat[s]` and market beta `beta_hat[s]`, feeding (i) every expert's sigma scaling,
(ii) the executive's vol-target overlay (predicted book vol), (iii) the meta-labeler's cost-
vs-vol feature. Two shared ridge regressions (≈ 2×45 = **90 coefficients**) on
[vol_21d, vol_63d, range_pct, VIX level/slope/curvature, COR3M z, G4 novelty, G5 HHI,
llm_geopol_risk]; upgraded to a 20→16→2 MLP (~**370 params**) only if ridge loses on
validation QLIKE. Forecast vol is one of the few things daily data genuinely supports.

### E5 — MetaTrigger: the meta-labeling layer (the López-de-Prado piece of this prior)

**Job:** precision filter. The primary stack says *what looks good*; MetaTrigger says
*whether the bet clears costs* — the canonical secondary model of meta-labeling.

- Label (walk-forward, per symbol-day where the equal-trust blended signal exceeds a fixed
  salience floor |z|>0.3): `1 if sign(mu_blend_eq[s,D]) · y_raw[s,D] > 2·half_spread[s]`
  (the trade would have been profitable after round-trip spread), else 0.
- Model: HistGradientBoostingClassifier, ≤100 iterations, 15 leaves (**≤1.5k leaves**) on
  ~80k qualifying symbol-day events. Features: |mu_blend|, expert sign-agreement, dispersion
  of expert mu's, sigma_hat, cost_bps[s], spread-vs-expected-edge ratio, vol regime z, LLM
  conf for the symbol's bucket, days-since-event.
- Output `p_win[s] ∈ [0,1]`; position scaling `min(1, max(0, (2·p_win − 1)/(2·p_min − 1)))`
  with threshold gene `p_min` owned by evolution (§6). Trained on the **equal-trust** blend
  (stationary across months) — the executive's τ then tilts the blend modestly; this
  approximation is stated and its error monitored (corr of gated books, dossier plot).

**Why this is five jobs, not five voters:** E1 sees only relative price structure; E2 sees
the exogenous tabular world; E3 speaks only at events; E4 forecasts magnitude with no
direction; E5 forecasts *the profitability of acting*, conditioned on the others. Remove any
one and a distinct input→decision pathway goes dark — which is exactly what the attribution
arms in §11 test.

---

## 3. Forecast → action pipeline (replay-harness compatible)

Nightly (3:00 UTC, Lambda, decision date D — all inputs from `daily/D/*` = data through D-1
close; fills happen at D's open via `daily/D+1/prices.parquet` per the harness contract;
costs via `apply_transaction_costs` per-sector half-spread + seeded slippage):

```
1. Ingest: prices window, context, GDELT-rich top-up (8 files), CBOE CSVs, FRED, COT(weekly),
   LLM organ call (one batched Bedrock invocation, §7) → feature store rows for D.
2. Experts: CAST (5-seed mean), XGB-Cond, EventHead → mu[3,64], sigma[3,64], c[3].
   RiskNet → sigma_hat[64], beta_hat[64], book-vol predictor.
3. Executive (§4): trust ledger stats r[3,K] → tau[3]; blend:
     mu_blend = Σ_m tau[m]·mu[m]      prec = Σ_m tau[m]/sigma[m]^2
     w_raw    = clip(mu_blend, ±q95)·prec ;  w_dir = max(w_raw, 0)   (long-only book)
4. MetaTrigger: w_dir[s] *= meta_label_scale(p_win[s]; p_min)        (precision gate)
5. Normalize → w_unit; sizing head f ∈ [0, f_max] (executive); vol-target overlay using
   RiskNet's predicted book vol vs sigma_cap → w_tgt = f · w_unit (scaled).
6. Intents: delta = w_tgt − w_prev; emit {symbol, action BUY/SELL/REDUCE, sizing} only where
   |delta| > no_trade_band (gene). Harness cluster cap applies downstream; fills at next open
   with the half-spread table — turnover is priced in the executive's training loss (§4), in
   the EA's fitness (§6), and braked by the no-trade band, so it is honest, not assumed away.
7. Write artifacts: llm_sentiment.json, expert_opinions.parquet, meta_decision.json,
   trade_intents.json (+ trust-ledger update, expected-vs-realized backfill).
```

Replays never re-call the LLM — stored per-day artifacts only (the LLM engineer's rule,
adopted). Expected turnover shape: 5-day horizon + no-trade band + turnover penalty in two
losses → ~10–25% of book value per week; at 1–8 bps half-spread this is ~1–4 bps/week of
cost, visible in every utility number reported.

---

## 4. The executive — meta-evaluator (assignment item 5)

**Verdict on PROPOSAL_META_EVALUATOR: ADOPT, with three adaptations.** The proposal is the
strongest organ on the table and is exactly shaped for this prior: tiny gating network
(≈450–600 params, ceiling 3k), shared-φ trust head over per-expert rolling counterfactual
P&L, deterministic blend, sizing head, **differentiable decision replay** loss = realized
forward utility U_h (h=5) of its own choices through the same half-spread cost table, purged
walk-forward, entropy-floored τ, full audit JSON with integrated-gradients attribution,
5-seed ensemble, linear-gate simplification ladder, fine-tune-on-live-record (~130
pre-holdout days, trust/sizing layers only). All adopted as written, including its §7 kill
criteria and the `walk_forward: true` attestation it demands from every expert's historical
opinions — my training plan (§9) is built to satisfy that attestation natively.

Adaptations (stated per the contract's own graft points):
1. **MetaTrigger insertion** — the meta-label scale is applied between blend and
   normalization (pipeline step 4). During executive training the gate is differentiable
   (sigmoid relaxation of the p_min threshold, straight-through at inference), so the
   executive learns sizing *given* the trigger's behavior rather than fighting it.
2. **M=3 μ-experts** (CAST, XGB-Cond, EventHead). RiskNet and MetaTrigger are not trust-head
   citizens — they are instruments, attributed by ablation (§11), not by τ. This keeps the
   trust problem at its smallest honest dimensionality.
3. **Context vector z (≈18 dims):** the proposal's market block + LLM global axes
   (risk_appetite, rates_pressure, geopol) + G4 novelty + G5 concentration + `llm_status_ok`
   — so the executive can learn to discount dark-LLM nights and to shorten trust half-lives
   when news novelty spikes.

Rejected within the proposal's own option set: attention-over-experts (their rejection
adopted verbatim — the transformer requirement is E1's job, not the executive's); RL
(brief §7 kills it); the GBM challenger is **kept** as the mandated challenger ablation.

Training targets are decision-grade throughout: the loss IS realized U_h of the would-have
book (growth − one-sided downside − amortized entry cost − turnover smoothness), λ_dn and
friends pre-registered constants surfaced to evolution. **No regime labels exist anywhere in
this system** — there is no regime classifier organ at all; regime awareness enters only as
continuous context features. The showcased intelligence is allocation, by construction.

---

## 5. Expert contract (the exact interface the executive and EA consume)

```
expert_opinions.parquet (per decision date D):
  expert ∈ {cast, xgb_cond, event_head}
  mu[s]      float  — 5d excess-return z (cross-sectional, clipped ±3)
  sigma[s]   float  — per-symbol uncertainty (rolling residual sd × sigma_hat scaling)
  c          float  — scalar self-confidence (CAST: seed-ensemble agreement;
                       XGB: mean |p−0.5| calibrated; EventHead: gated event mass)
  manifest: {model_version, train_window_end, walk_forward: true, seeds, code_sha}
trust_ledger.parquet — per the meta-evaluator proposal §4, unchanged.
risknet.parquet      — sigma_hat[s], beta_hat[s], book_vol_hat.
meta_trigger.parquet — p_win[s].
genome.json          — EA champion (or B0), §6.
```

---

## 6. Evolution's role (assignment item 2)

**Verdict on PROPOSAL_EVOLUTION: ADOPT with genome extensions.** The division of labor is
exactly right for a prediction stack: gradients own perception (E1–E5 weights — dense,
differentiable, per-sample losses), evolution owns disposition (the non-differentiable,
risk-shaped connective tissue scored on cost-adjusted walk-forward path utility), contracts
and rails stay fixed. Adopted as written: (μ+λ) GA P=28/G=14/K≤400, 6 regime folds tiled
2020→2026-02 with 21-day embargo, fitness = min-over-cost-scenarios (Sharpe − DD penalty),
mean − 0.5·std across folds, parsimony pressure, fold subsampling, DEFAULT_GENOME-in-
population, the adoption gate (champion must beat B0 by one cross-fold sd or **B0 ships**),
B1 budget-matched random-search control, every-variant logging, $0 AWS.

Genome for this design (32 genes ≤ 40 cap; the proposal's schema + ranges, M=3, B=8 blocks):
- trust_prior[3], trust_halflife — initial listening posture + adaptation speed.
- member_gate[3] — ensemble composition (the EA may bench an expert; reported, and §11
  attribution independently measures it — adopted stance on gates-vs-scorecard).
- feature_gate[8]: {GDELT-themes G1, GDELT-tone G3, GDELT-novelty/conc G4+G5, GDELT-country
  G2, LLM-sentiment block, CBOE-vol-surface, FRED-macro, COT-positioning}.
- allocator risk genes (8): gross_target, vol_target, max_symbol_weight, dd_brake
  threshold/strength, no_trade_band, conviction_temp, cash_floor.
- executive shaping (2): risk_aversion λ_dn, abstain_threshold.
- **NEW (3, declared ranges for pre-registration):** `p_min ∈ [0.50, 0.70]` (MetaTrigger
  threshold — where the precision/recall trade-off sits is a disposition, not a perception);
  `event_weight_cap ∈ [0.5, 2.0]` (multiplier on EventHead's mu before blending — how loud
  the event channel may be); `record_weight_eps ∈ [0.05, 0.5]` (floor of the infotropy-B
  label-weighting at retrain time, §8 — evolution decides how hard the anti-HILL prior bites).
- Where the EA bites hardest in THIS system: τ-priors × p_min × gross/vol genes jointly set
  the aggression frontier — exactly the surface backprop cannot reach because it lives
  behind discrete intents, the cost table, and max-drawdown.

Cadence adopted: monthly evolution on the Mac (3–8 min over cached OOF matrices — the OOF
matrices are a free by-product of §9's fold training); nightly Lambda reads a static ~5 KB
genome; daily trust adaptation is the genome-parameterized deterministic decay rule.

---

## 7. LLM sentiment organ (item 3) and GDELT as load-bearing data (item 4)

**Verdict on PROPOSAL_LLM_SENTIMENT: ADOPT in full.** Haiku-on-Bedrock (the one IAM-allowed
model, verified alive), ≤120 deduped GKG-derived headline-equivalents/night (URL slugs +
quotations + themes/actors — no new hosts), one batched call, validated JSON over 27 buckets
+ 3 global axes + 12 event flags, fail-soft-never-fabricate with `llm_status` as a feature,
temperature 0, stored artifacts (replay never re-invokes), **backfill Option A** (identical
funnel+prompt over archived GKG): Tier 1 2026-01-31→present (covers replay window + entire
holdout, ≈$0.35), Tier 2 2024-01→ (≈$2.05), $5 packet hard cap. Model-deprecation risk
flagged as they flag it; gpt-4o-mini fallback via the existing key.

Adjudication of the one inter-proposal conflict: the Data Edge Scout says "LLM features must
never be trained on" (history too shallow); the LLM engineer's Option A creates 2024→
training coverage with masks. **Ruling for this design:** LLM features ARE training inputs
to E2 and E3 but only ever behind the `llm_available` mask, with both models required to
remain valid when the mask is 0 (pre-2024 history trains the non-LLM pathways); the
executive consumes LLM aggregates live regardless. This gets train/serve consistency where
real output exists and honest absence where it doesn't — and the holdout contains genuine
LLM output everywhere it is read, so the item-3 attribution is a real measurement.

**Exactly which LLM features enter which models:**
- E2 XGB-Cond: `llm_sent/conf/sal_<bucket>` (symbol's bucket), `llm_sent_<bucket>_d1`,
  masks.
- E3 EventHead: full bucket surface + event-flag dummies + severities (its primary food).
- Executive z: `llm_risk_appetite, llm_rates_pressure, llm_geopol_risk, llm_status_ok`.
- MetaTrigger: `llm_conf_<bucket>` (news confidence as a precision feature).
- CAST: **none** (kept text-free deliberately — its attribution must isolate price-structure
  skill, and the LLM block must be removable without touching the transformer).

**Verdict on PROPOSAL_DATA_EDGE: ADOPT G1–G5 + S1 + S3; ADOPT S2 as context-only; CUT S4.**
- **First action of Phase C (adopted as the Scout's "most urgent task"):** the GDELT top-up
  backfill 2026-02-05→present — without it the holdout has no real GDELT and item 4 cannot
  be evaluated at all. Then the full G1–G5 re-pull 2015-02-18→ (80–110 GB, local, free,
  restartable; degrade to 2/day sampling pre-2023 if wall-clock demands — flagged in the
  manifest as they specify).
- **GDELT features → models:** G1 sector theme-z → E2 + E3; G2 country-actor pressure
  (Goldstein, conflict share) → E3 (country buckets: FXI EWJ EWZ INDA VGK EM) + E2; G3
  per-bucket tone/dispersion → E2 + E3; G4 novelty (JS divergence) + doc-surprise →
  executive z + RiskNet + MetaTrigger; G5 actor/geo concentration → executive z + RiskNet.
  This is GDELT doing **structural** work (who/where/how-concentrated), not a mood number —
  the differentiated use the packet demands.
- S1 CBOE (COR3M, term curvature, VXN−VIX; depth to 1990/2006) → E2 + RiskNet + executive z.
  Adopted at rank 1 for the reason they give: implied correlation is the market's price of
  "everything moves together," which is the central risk of a 64-ETF long-only book.
- S3 FRED deep catalog (NFCI, STLFSI4, DFII10, T10YIE, HY-OAS, ICSA with publication-lag
  rules) → E2 + executive z. Attribution must beat the existing context proxies, as they
  warn.
- S2 COT TFF: adopted as **conditioning only** (weekly, ~13 holdout obs — their own
  skepticism): 3 nets-%-OI z features → E2 + executive z; publication-date keying (the
  classic leak) implemented exactly as specified; their shuffle-placebo falsifier kept.
- S4 Treasury auctions: **CUT.** Episodic, no holdout verdict arithmetically possible, and
  this design needs feature-count discipline more than a tail conditioner. (Their own "first
  candidate to cut.")
- S0 breadth freebies (A/D, %>200d, dispersion, RSP−SPY): folded into CAST's current-day
  scalars and executive z; claimed as nothing.

---

## 8. Infotropy mechanism (item 6)

Engaging the Liaison's graded transfers:
- **Transfer B (record-formation label weighting, anti-HILL) — ADOPT, widest deployment.**
  Sample weights `w_rec = clip(shift·(1−reversion), eps, 1)` applied to (i) CAST's Huber/rank
  loss, (ii) XGB-Cond's sample weights, (iii) the executive's per-date utility terms. The
  brain is paid to forecast moves that lay down durable structure, not round-trips. `eps` is
  the EA gene `record_weight_eps` (§6). A/B falsifier (uniform vs record weights) is a
  pre-registered training-time ablation; if OOS Spearman/utility doesn't improve, weights
  revert to uniform and the scorecard reads infotropy = no-attr for this arm.
- **Transfer A (R1∧R2∧R3 record-grade event gate) — ADOPT, scoped to EventHead.** Implemented
  as specified: R1 persistence of the bucket regime stat, R2 non-self-encoding (event
  intensity not reconstructible from lagged returns — the reverse-causality screen), R3
  walk-forward AUC lift; conjunctive product gates feature admission, runtime trace-bucket
  down-weight ×0.25. Falsifier kept: gated vs R3-only event sets, walk-forward; no lift ⇒
  the gate dies and R3-screening remains.
- **C (B-parameter trust), D (self-record feedback), E (bottleneck-width gate) — REJECT as
  infotropy claims, per the Liaison's own RESTATEMENT grading.** Their useful content
  already exists in this design under standard names (C: the trust head conditions on
  counterfactual P&L and confidence; D: the trust ledger IS self-record feedback; E: breadth
  enters as ordinary context features) and will not be sold as an infotropy edge.
- **F (possibility-space/exergy) — REJECT**, on the Liaison's NO-TRANSFER verdict and the
  canon's own speculative-unregistered bar.
- Scorecard consequence: `infotropy=<attr>` is measured on Transfers A+B specifically (§11);
  an honest zero is reportable without touching the rest of the brain.

---

## 9. Training plan

**Common walk-forward frame (shared with the EA — one fold layout for the whole brain):**
the Evolution Engineer's six ~250-day folds F1–F6 (2020-02→2026-02), expanding head-training
windows from 2014-08-29, **21-trading-day purge/embargo** between train end and fold start
(> the longest label horizon), holdout 2026-03-11→ never touched by any training, tuning, or
EA pass; read once per pre-registered configuration at bake-off, looks counted.

**Target construction (no look-ahead, audited):** features for date D use closes ≤ D-1
(matching the `daily/D/` convention); y uses open(D)→open(D+5) from refetched Stooq OHLCV;
the meta-evaluator proposal's mandatory alignment test adopted (20 random symbol-date fill-
price asserts vs replay, VUG 6:1 split handled once, never twice); GDELT rows join via
`visible_from` (Scout's rule); COT keyed by publication date; weekly FRED series lagged by
documented publication delays; LLM day-D artifact built only from GKG files stamped before
D's 03:00 UTC run.

**Pretrain / fine-tune split:**
- Pretrain on public history: CAST + RiskNet from 2014-08; E2/E3 from 2015-02 (GDELT-rich
  boundary), LLM columns masked before 2024-01. Per-fold OOF opinions generated with
  `walk_forward: true` manifests — the executive's attestation requirement satisfied at the
  source.
- Fine-tune on the brain's own ~210-day live record: **executive only** (trust logits +
  sizing head, LR×0.1, ≤50 epochs, pre-holdout ~130 days), per the meta proposal. Forecaster
  heads are NOT fine-tuned on 130 days — they cannot learn structure from it and we do not
  pretend otherwise.

**Monthly Mac run (launchd, 1st of month 02:00; budget ≤ 2h, measured-order estimates):**

```
data refresh (GDELT 8 files/day backlog, CBOE/FRED/COT pulls)            ~10 min
CAST: 6 folds × 3 seeds (warm-start from prior fold) + deploy 5 seeds    ~45–60 min
XGB-Cond: 6 folds + deploy                                               ~12 min
EventHead + RiskNet (linear/ridge)                                        ~2 min
MetaTrigger: 6 folds + deploy                                             ~5 min
counterfactual trust-ledger rebuild                                       ~3 min
executive: 5 seeds × 200 epochs (600 params)                              ~5 min
EA: P=28 × G=14 over cached OOF matrices                                  ~8 min (cap 25)
artifact upload (models/ + latest.json pointer convention)                ~2 min
TOTAL                                                                     ~95–110 min
Shrink lever if over (packet rule — shrink seeds/epochs before organs):
OOF seeds 3→2, CAST epochs cap, then d_model 48→32.
```

**Seeds & determinism:** master seed per component = sha256(component + train_window_end);
all child seeds recorded in manifests; 5-seed inference ensembles for CAST and executive;
EA determinism per its proposal; cost-model RNG seeded per EVIDENCE_PROTOCOL.

---

## 10. Sample budget table (every learnable component)

| Component | Params/capacity | Nominal sample | Effective sample (honest) | Memorization defense |
|---|---|---|---|---|
| CAST (E1) | ~45k (fallback 22k) | 186k symbol-days | ~20–35k (cross-sectional ranks, clusters ~8/day, 5d overlap) | no symbol identity; shared GRU+attention; market-neutral rank target; dropout .2; wd; early stop; 5 seeds; ridge/22k ladder |
| XGB-Cond (E2) | ≤9.3k leaves | ~150k rows/fold | ~25k (same clustering) | depth/leaf caps, L2, monotone constraints, OOF isotonic calibration |
| EventHead (E3) | 675 coefs | ~2,790 event-days × 27 buckets | ~8–15k bucket-days w/ events | elastic net; R1∧R2∧R3 feature gate; abstains on quiet nights |
| RiskNet (E4) | 90 (MLP fallback 370) | 186k symbol-days | vol is persistent — ample | ridge default; QLIKE validation; vol targets are the easy problem |
| MetaTrigger (E5) | ≤1.5k leaves | ~80k bet events | ~12–20k | small trees; single threshold consumed (p_min, evolved not fit) |
| Executive | ~450–600 (cap 3k) | 2,959 days | ~600 (5d windows) | adopted battery: shared φ, decay, dropout, noise, 5 seeds, linear-gate ladder |
| EA genome | 32 genes, K≤400 evals | 6 fold-scores/genome | ~6 regime obs | adoption gate (>1 cross-fold sd over B0), B1 random-search control, fold subsampling, cost-scenario min |
| LLM organ | 0 trained params | — | — | zero-shot; schema-clamped; fail-soft |

System-wide: ~57k trained parameters total, dominated by CAST — and CAST's target is the one
place 186k symbol-days genuinely exist. Nothing per-symbol, nothing per-regime, no RL.

---

## 11. Attribution hooks (leave-one-out arms; all pre-registered in TOURNAMENT.md before build)

Every arm: identical replay harness, identical snapshots/seeds/cost model, holdout-only
verdict (2026-03-11→, ~62 days — minimum detectable effects stated up front), paired
daily-difference stats per EVIDENCE_PROTOCOL. "Kill" = scorecard records ≈0 attribution.

| Organ | LOO arm | Kills it |
|---|---|---|
| CAST / transformer | drop expert from set; executive retrained on M=2 | holdout paired Δμ ≤ 0 or ΔSharpe ≤ 0 vs full brain |
| transformer *qua* transformer | CAST vs cross-sectional ridge on identical features, same slot | ridge matches CAST validation Spearman ⇒ "transformer not doing real work," reported |
| XGB-Cond | drop expert, M=2 | same form |
| EventHead | drop expert, M=2 | same form |
| RiskNet | replace sigma_hat/book-vol with trailing 21d realized | no holdout degradation |
| MetaTrigger | identity gate (scale=1 everywhere) | ungated brain ≥ gated on holdout (after costs — the gate's whole claim is cost-aware precision) |
| Executive | B0: τ=1/M, f=0.7 fixed + vol cap (meta proposal §7) | paired Δ ≤ 0; or static-trust (std τ < .02); or calibration corr ≤ 0 |
| Evolution | champion vs DEFAULT_GENOME (B2), + B1 random-search control | adoption-gate fail / loses to B0 on holdout / ≤ B1 ⇒ ceremonial, reported per its three pre-committed verdicts |
| LLM organ | all `llm_*` → neutral constants, brain retrained + neutralized variants | ΔSharpe ≤ 0, paired t < 1, or removal improves return (LLM proposal stage-2 verbatim) |
| GDELT | G1–G5 block ablated (and G1-dictionary sub-falsifier) | |t| < 1 paired ⇒ evidence-backed no-signal per item 4, with the "62-day window" honesty clause |
| Infotropy A | R1∧R2∧R3 gate vs R3-only event screen | no walk-forward lift |
| Infotropy B | record-weighted vs uniform-weighted retrain | no OOS Spearman/utility lift |

Holdout-look ledger: each organ contributes exactly the arms above, counted; no arm is run
until pre-registration is committed.

---

## 12. Cost sketch (itemized monthly, deployed shape)

| Line | Arithmetic | $ /mo |
|---|---|---|
| Lambda compute | night ~110s→~175s (+GDELT 8 zips ~25s, +LLM call ~40s incl. retries, +5 expert/executive inference <5s, +CSV pulls ~5s); 22×175s×2.94GB + 22 morning×15s×2.94GB ≈ 12.3k GB-s; carried at the cost-of-record shape | ~6.20 |
| Bedrock (Haiku, pinned) | ~10k in + 1.5k out tokens/night × 22 × 1.5 headroom (LLM proposal worksheet) | 0.15 |
| S3 storage + requests | existing ~1.0 + new parquets/artifacts ~15–25 MB steady (gdelt_rich, opinions, ledger, meta_decision, llm json) + ~30 PUT/GET/day | 1.06 |
| ECR | 1.2 GB image, unchanged (no new heavy deps — torch/sklearn already in) | 0.12 |
| Secrets Manager | unchanged | 1.00 |
| CloudWatch + SNS | unchanged | 1.00 |
| Data transfer | inbound free; outbound artifacts trivial | ~0.01 |
| **TOTAL** | | **≈ $9.5/mo** — under the $10 target; ~$5.5 headroom to hard-fail |
| One-time (Phase C) | LLM backfill Tier1+Tier2 ≈ $2.40 + $0.05 pilot (cap $5, logged per call); GDELT 80–110 GB to local Mac $0; Mac training time $0 | ≤ $5 once |

If the Haiku ID is retired mid-program: gpt-4o-mini fallback at comparable cents, or the
flagged IAM widening (≈ +$0.40/mo at Haiku-4.5-class pricing) — both inside the envelope.

---

## 13. Risks & honest weaknesses (where this prior loses)

1. **The prior's core bet can simply be false at this cadence.** Daily-bar 5-day
   cross-sectional predictability in 64 liquid, deeply-arbitraged ETFs may be ~0 after 1–8
   bps spreads. If mu's OOS Spearman is ~0.02, the whole stack is an elaborate way to hold a
   vol-targeted diversified book — and an allocation-learner prior (Beta) that never claims
   forecast skill would beat me on parsimony while matching me on P&L.
2. **Error stacking.** Forecast → blend → trigger → sizing is three approximation layers
   before capital moves; a policy trained directly on utility has one. My mitigation (the
   executive's utility-trained gate) is also my admission that the pure prediction stack
   needed an allocation-learner organ at its head.
3. **CAST is the most likely showcase casualty.** 45k params vs ~25k effective samples is
   the most aggressive ratio in the design; a real chance exists that ridge matches it and
   the headline transformer claim dies on the ladder. The honest fallback (report it, ship
   ridge in the slot) satisfies the evidence rules but weakens the showcase — Gamma's prior,
   where the transformer digests genuinely high-dimensional event data, may give a
   transformer more honest work than price bars can.
4. **Event/news organs face a short real-data runway.** LLM features exist from 2024,
   genuine holdout GDELT only after the top-up; ~62 holdout days with wide error bars means
   item-3/item-4 attributions may be statistically indistinguishable from zero even when
   real — reported as "underpowered," which is honest but unsatisfying.
5. **Long-only single-book structure.** Forecast z's want a long/short expression; clipping
   to long-only throws away half the cross-sectional information. If the harness/portfolio
   contract admits short or cash-tilt expression cleanly, the stack improves; I designed
   conservatively (w_dir = max(w_raw,0), cash floor) rather than assume shorting.
6. **MetaTrigger trains on the equal-trust blend** while serving a τ-tilted one — a stated
   approximation that could miscalibrate the gate precisely when the executive deviates most
   (i.e., when it matters).
7. **62-day holdout arithmetic** caps every claim in this document: the bake-off verdict will
   lean on E1 cross-fold evidence + one E2 read, and the pre-registered minimum-detectable-
   effect line must temper the final-line scorecard.

---

## 14. What was adopted / adapted / rejected (one-table register)

| Source | Verdict |
|---|---|
| PROPOSAL_META_EVALUATOR | ADOPT + 3 adaptations (MetaTrigger insertion, M=3, z extensions) — §4 |
| PROPOSAL_EVOLUTION | ADOPT + 3 new genes (p_min, event_weight_cap, record_weight_eps) — §6 |
| PROPOSAL_LLM_SENTIMENT | ADOPT in full incl. backfill Option A and kill criteria — §7 |
| PROPOSAL_DATA_EDGE | ADOPT G1–G5, S1, S3; S2 context-only; **CUT S4**; conflict with LLM-training ruled via masks — §7 |
| INFOTROPY_TRANSFER A | ADOPT (scoped to EventHead ingestion) — §8 |
| INFOTROPY_TRANSFER B | ADOPT (CAST/XGB/executive label weights; eps on genome) — §8 |
| INFOTROPY_TRANSFER C, D, E | REJECT as infotropy claims (Liaison's own RESTATEMENT grade); content present under standard names — §8 |
| INFOTROPY_TRANSFER F | REJECT (Liaison's NO-TRANSFER) — §8 |

*End of DESIGN_ALPHA. Build order: GDELT top-up → feature store + alignment tests → E1–E5
walk-forward training → executive + ledger → EA → attribution arms wired BEFORE any holdout
read, per pre-registration.*
