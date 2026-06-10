# DESIGN_GAMMA — The Information-Funnel Brain

**Architect-Gamma (panel role 5 of 12) — PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN**
**Prior: information-edge.** The system exists to digest *differentiated* information — GDELT
event structure, LLM-read text, cross-asset entropy/flow measures — into decisions others
cannot make. Price models are a commodity; the differentiated *input stream* is the edge.
Written blind to the incumbent (sealed-incumbent rule honored — no forbidden file opened).
Designed from the packet, ASSIGNMENT_BRIEF.md, INFOTROPY_TRANSFER.md, and the four mechanism
proposals only.

---

## 1. System thesis

The 64 instruments are the most efficiently priced objects on earth, and the brain decides at
the previous day's close to fill at the next open — a full session of lag. So I concede the
thing a price-prediction prior cannot: **at daily cadence, directional alpha on SPY's *level*
is ~0 after one-day lag.** That is not where an information edge lives, and a design that
pretends otherwise is lying. The edge survives in three places the efficient-market argument
does *not* close: (1) **cross-sectional attention allocation** — given a structured,
machine-read information field aligned to exactly these 64 sleeves, *which* sleeve deserves
capital today is a far less efficient question than *whether* SPY goes up, because it depends
on a dispersed, slowly-diffusing news structure no retail aggregate reconstructs; (2) **the
covariance/volatility regime** — the *price of diversification* (realized correlation, sleeve
vol) is forecastable from the information field's dispersion/novelty/burst structure faster
and differently than from price-only statistics, and it is a *sizing* edge, not a direction
edge, so it is not arbitraged away by the level being efficient; (3) **second-order
positioning** — who is crowded (COT), how concentrated the news mass is (HHI), how unlike
recent news today is (Jensen–Shannon novelty) — these condition *how much to bet*, which the
efficient-level argument never touches. The brain therefore does not try to out-predict the
tape on direction. It builds the differentiated information field nobody else builds, and
spends it on *allocation and sizing* — the questions that stay open at daily cadence. The
funnel is the moat; the models exploit it; the executive turns it into bets.

---

## 2. The information funnel (the centerpiece)

The funnel is the system. Everything downstream is a consumer of it. It runs once per night
inside the existing 3008 MB / 900 s Lambda (incremental cost in §11), parse-once-emit-many.

### 2.1 Raw differentiated sources (all free, all on the current allowlist)

| Stream | Raw object | First usable | Why retail bots lack it |
|---|---|---|---|
| **GDELT GKG v2 15-min** | themes (col 8), actors/orgs/persons (12/14), V2TONE (15), quotations (22), locations, source URL slug (4) | 2015-02-18 | needs a theme→sector dictionary + 15-min parsing nobody ships free |
| **GDELT v2 events** | CAMEO actor/event, Goldstein, QuadClass, NumMentions | 2015-02-18 | CAMEO→country-ETF plumbing |
| **LLM-read text** | URL-slug headline-equivalents + quotations (no page fetch) | backfill-bounded (~90d real; Tier-2 to 2024) | a model that *reads* the field, not a keyword count |
| **CBOE vol surface (S1)** | VIX9D/VIX/VIX3M term structure, COR3M implied correlation, VVIX, SKEW, VXN | 1990/2006/2011 | COR3M + curvature are practitioner objects; retail stops at VIX level |
| **CFTC COT TFF (S2)** | leveraged-fund / asset-mgr net positions, OI | 2006-06-13 weekly | positioning extremes, publication-lagged |
| **FRED deep (S3)** | NFCI, STLFSI4, T10YIE, DFII10, BAMLH0A0HYM2, ICSA | 1967–2003 by series | real-rate/breakeven decomposition pricing TIP/GLD/TLT |
| **Treasury auctions (S4)** | bid-to-cover by tenor | 1979 episodic | duration-demand surprise |
| **Universe-internal breadth (S0)** | A/D, %>50/200d MA, RSP−SPY, dispersion | from prices | free; bottleneck-width proxy (Infotropy E) |
| Prices OHLCV | 64 ETFs | 2014-08-29 | commodity — present only to anchor the targets |

I **adopt PROPOSAL_DATA_EDGE in full** (G1–G5 GDELT-rich families; S1 CBOE; S2 COT; S3 FRED;
S4 Treasury; S0 breadth) and the killed list (no CBOE P/C, no ICI, no iShares, no crypto). The
DATA_EDGE depth-audit table is binding and reproduced in §10. The urgent prerequisite it names
— **the 2026-02-05→present GDELT top-up backfill** — is build-task #0; without it GDELT is
dark across the entire holdout and item 4 cannot be evaluated at all.

### 2.2 Pipeline stages (exact constructions, no-look-ahead discipline)

```
F0  ACQUIRE   4 GKG + 4 events files/day at 00/06/12/18 UTC (all precede 21:00 UTC US close).
              Weekend gap: the Tue 03:00 run sweeps Sat–Mon (72h window).
F1  PARSE     one pass over GKG -> all G1..G5 + LLM text package; one pass over events -> G2.
F2  MAP       theme/actor/location codes -> the 64-ETF sleeve buckets via curated dictionaries
              (theme_to_sector.json ~60-100 codes -> 13 sleeves; ACTOR_MAP; country->country-ETF).
F3  STANDARDIZE  every count/tone/Goldstein feature -> trailing-90d z (mean,sd from days < d).
F4  NOVELTY   per-bucket burst-z (today vs trailing-90d); theme-novelty = JS divergence of
              today's top-~500 theme distribution vs trailing-90d mean (+21d smoothed);
              doc-count surprise; actor/geo HHI concentration; breadth/dispersion (S0).
F5  LLM ORGAN reads top-120 record-graded clusters -> per-bucket {sent,conf,sal} + 3 global
              axes + event flags (§6), AND emits structured event tokens (§6.3, my extension).
F6  ASSEMBLE  model-ready tensors with explicit missing-masks + visible_from join keys (§2.3).
```

**No-look-ahead (binding, every stream):** a feature row for UTC day `d` is built only from
files/series stamped at or before `d`'s defined cutoff (GDELT: `d` 18:00 UTC; daily series:
`d` close; weekly series: documented publication lag — COT keyed to Friday publication not
Tuesday report-date, NFCI to following Wednesday, STLFSI4 to following Thursday). Every row
carries a `visible_from` column. The replay join attaches GDELT/feature day `d` to decision
date `D` **only where `D ≥ d+1`** (the 03:00 UTC night run for D reads streams through `D−1`),
filling at D+1 open per the harness. Backfilled rows use `visible_from`, never the raw date —
this is the single easiest leak in the whole design and it gets a dedicated unit test.

### 2.3 The model-ready feature object

Per decision date D, the funnel emits two objects:

- **`field_state[D]`** — a ~140-dim day-level vector: macro+vol-surface+COT+breadth (~40), G1
  sector theme-pressure z (~26), G3 sector tone panel (~16), G2 country pressure (~14), G4
  novelty/burst/HHI (~10), LLM global axes + event flags (~16), and the **information-health
  block** (~6): `gdelt_available`, `llm_status_ok`, `n_clusters`, `theme_novelty`,
  `field_dispersion`, `days_since_last_gdelt`. The health block is my prior's specific gift to
  the executive (§4): the brain must *know when its differentiated input is thin or dark*.
- **`event_tokens[D]`** — a variable-length set (≤K=64) of record-graded event tokens (§6.3),
  each a structured record `{theme_code, actor_bucket, sleeve_target, goldstein, tone,
  novelty_z, n_sources, llm_sent, llm_conf, record_grade}`. This is the transformer's food.

Both carry `field_mask` so any dark stream degrades gracefully (§9), never fabricates.

---

## 3. Ensemble of genuinely different model types (item 1)

Four members with genuinely different architectures and *different jobs*. Two
forecast cross-sectional *direction* over different representations; one forecasts the *sizing*
regime; one is the LLM organ. None votes on the same label.

### 3.1 Member 1 — **Event-Theme Transformer (ETT)** — the transformer doing real work

Attention over the **event/theme field**, not over a price sequence. This is the design's
centerpiece and the literal answer to "at least one transformer doing real work."

- **Input:** `event_tokens[D]` (≤64 tokens) + a 5-day temporal context of `field_state` summaries.
  Each event token is embedded from its structured fields: theme-code embedding (shared
  table, ~120 codes → 16-dim), actor-bucket embedding (~24 → 8-dim), sleeve-target embedding
  (13 → 8-dim), plus scalar features (goldstein, tone, novelty_z, n_sources, llm_sent×conf,
  record_grade) projected to the residual width. `d_model = 32`.
- **Architecture:** 2 encoder layers, 2 heads, FFN width 64, learned [CLS]-per-sleeve query
  tokens (13 sleeve queries attend over the event set → per-sleeve pooled representation).
  Self-attention lets "China export-curb" + "semis" + "high novelty" events *interact* —
  that interaction is the thing a bag-of-features model cannot represent and the reason a
  transformer earns its slot here.
- **Output:** per-sleeve expected 5-day excess return `mu_ETT[13]` + uncertainty `sigma_ETT[13]`
  (heteroscedastic head), expanded to 64 symbols via the sleeve map.
- **Parameter count:** embeddings 120·16 + 24·8 + 13·8 ≈ 2,216; projections+2 layers
  (≈ 2·(4·32·32 + 2·32·64) ) ≈ 24k; sleeve queries 13·32 ≈ 416; heads ≈ 1k. **≈ 28k params.**
- **Distinct job:** "which sleeves does *today's structured news interaction* favor."
- **Memorization defense (§7):** d_model=32 and shared embeddings cap capacity; pretrained on
  the GDELT-rich panel 2015-02→ (~2,790 days) with heavy dropout (0.2 attention, 0.1 token),
  walk-forward folds, 5-seed averaging, record-formation label weighting (Infotropy B, §7),
  and a **simplification rung**: if a bag-of-events MLP (mean-pooled tokens, ~3k params)
  matches it on purged validation, ship the MLP and report the transformer added nothing — an
  honest finding about whether event *interaction* carries weight.

### 3.2 Member 2 — **Information-State Regime Encoder (ISRE, GRU)** — the sizing forecaster

- **Input:** 21-day sequence of `field_state[D−20..D]` (the ~140-dim day vector).
- **Architecture:** single-layer GRU, hidden 24 → two heads: (a) per-sleeve 5-day realized-vol
  forecast `vhat[64]`, (b) market-wide realized-correlation forecast `chat` (the COR3M target).
  **≈ (140·24·3 + 24·24·3 + heads) ≈ 12–14k params.**
- **Output:** the *sizing-relevant* targets — `vhat`, `chat`, regime embedding `z_reg[8]`. It
  does **not** forecast direction; it forecasts the covariance regime, which §1 argues is the
  forecastable, non-arbitraged part. Its outputs feed the executive's sizing head and the
  precision-weighting in the blend.
- **Distinct job:** "how violently and how *together* will things move, and what regime are we
  in." Different target (variance/correlation), different architecture (recurrent), different
  inductive bias (temporal smoothing) from the ETT.
- **Memorization defense:** vol/correlation targets are dense and high-SNR (vol is the most
  forecastable thing in markets), so 14k params over ~2,800 day-sequences is defensible;
  weight decay, 5-seed, purged walk-forward.

### 3.3 Member 3 — **Cross-Sectional GBM Ranker (LightGBM)** — the tabular workhorse

- **Input:** the full per-symbol funnel feature row (sleeve-mapped funnel features broadcast to
  the symbol + symbol-specific price/breadth features + missing-masks), ~80 features × 64
  symbols × days.
- **Architecture:** gradient-boosted trees, rank objective (lambdarank on forward 5-day excess
  return), ≤ 400 trees, max_depth 4, min_child 200, heavy L1/L2 — strangled to avoid the
  lookup-table failure §7 warns of.
- **Output:** per-symbol cross-sectional score `mu_GBM[64]` (z-scored per day).
- **Distinct job:** robust nonlinear tabular cross-sectional ranking with a *completely
  different inductive bias* (axis-aligned splits, no gradient) from the two nets — the genuine
  "different model type," and the member most able to exploit sparse/masked funnel features
  without overfitting their interactions.
- **Memorization defense:** depth-4 trees + min_child 200 + feature/bagging fraction 0.6 +
  early stop on purged validation; sample-size honest because trees handle the sparse-mask
  regime gracefully.

### 3.4 Member 4 — **LLM Sentiment Organ** — as an ensemble member

The LLM organ (§6) emits per-bucket `sent×conf` which, mapped to symbols, is a fourth opinion
vector `mu_LLM[64]` with `sigma_LLM` from `1−conf`. It is a member (its mu row enters the
blend) **and** a context source (its global axes enter `field_state`) — exactly the dual role
PROPOSAL_META_EVALUATOR §1.1 describes. Zero learnable parameters of its own at inference (the
"learning" is the frozen Haiku weights); its historical opinions are backfill-bounded (§9).

### 3.5 (Optional) Member 5 — Positioning Contrarian — small linear head

A ~30-param logistic over COT extremes + breadth + crowding (Infotropy D self-record feedback)
emitting a contrarian risk-appetite tilt. Kept behind an evolution gate (§5); if it earns no
attribution it is gated off and reported, not silently dropped.

**Common currency:** every member is adapted to the meta-evaluator's `mu[M,S]` unit
(z-scored per-symbol expected excess return) by its own organ, plus `sigma[M,S]` and a
self-confidence scalar `c[M]` — per the meta-evaluator contract.

---

## 4. The meta-evaluator executive (item 5)

**I adopt PROPOSAL_META_EVALUATOR almost wholesale** — it is architect-agnostic, decision-grade
by construction (differentiable decision replay, loss = realized forward utility through the
*same* fill/cost mechanics, never a regime label), and its parameter economy (≤~600 params,
shared trust `phi`, deterministic blend, measurement-driven trust inputs) is exactly right for
the sample budget. Its trust ledger, entropy floor, simplification ladder, integrated-gradient
audit record, and leave-one-out baseline (equal trust + fixed f) are taken verbatim. I do not
"upgrade" it to attention-over-experts or RL — the proposal's rejection of those is correct.

**My single adaptation (the information-edge contribution to the executive):** the context
vector `z` is extended with the funnel's **information-health block** (§2.3): `gdelt_available,
llm_status_ok, n_clusters, theme_novelty, field_dispersion, days_since_last_gdelt`. Rationale:
in *my* prior the executive's most important learned behavior is **knowing when its
differentiated input is trustworthy** — to lean on the GDELT/LLM-fed experts on information-rich
nights and fall back toward the price/vol experts when the field is thin or dark. This is the
mechanism by which "how much to listen to which model today" becomes genuinely information-
conditioned rather than purely performance-conditioned. It costs ~6 inputs (phi width
unchanged-ish, total still < 1k params) and is the difference between an executive that
*has* a differentiated input organ and one that *understands* it.

Decision-grade targets, outputs (trust simplex `tau[M]`, deployment fraction `f`, target
weights `w_tgt[64]`), the `meta_decision.json` audit record, and the §7 falsifiers are taken
from the proposal unchanged.

---

## 5. Evolution's role (item 2)

**I adopt PROPOSAL_EVOLUTION's division of labor** (gradients own perception; evolution owns
disposition; fixed = the measurement instrument) and its (μ+λ) GA, fold layout, fitness =
min-over-cost-scenario (Sharpe − DD penalty) meaned−0.5·std across six regime folds, the
DEFAULT_GENOME-in-population, the B0/B1/B2 baselines, the adoption gate (champion must beat the
hand-set default by >1 cross-fold σ or the default ships), and the $0 AWS cost line — all
verbatim.

**Where evolution bites hardest in an information-edge brain — the extension:** in this design
evolution's most valuable genes are the **funnel/feature-stream gates**, not just the allocator
risk knobs. The genome (still ≤40 genes) carries, in addition to the proposal's trust/allocator
genes:

```
  funnel_gate[b]      b=1..B   {0,1}   feature blocks: {GDELT-G1-themes, GDELT-G2-country,
                                       GDELT-G3-tone, GDELT-G4-novelty, GDELT-G5-HHI,
                                       LLM-sentiment, CBOE-vol-surface, COT-positioning, FRED-deep}
  record_grade_tau    float [0.0, 0.6]  ingestion gate threshold (Infotropy A, §7)
  novelty_floor       float [0.0, 2.0]  min novelty-z for an event to become a transformer token
  member_gate[m]      m=1..M   {0,1}    ensemble composition (incl. Member 5 contrarian)
```

This is the honest place evolution earns its keep in my prior: it **decides which
differentiated streams pay their complexity cost**, with the parsimony term (−0.02 per active
gate) forcing each stream to justify itself and the leave-one-out attribution (§12)
independently confirming. If evolution gates GDELT-G2 (country pressure) to zero across all
folds, that is the evidence-backed no-signal finding the packet explicitly permits — surfaced,
never hidden. The `record_grade_tau` and `novelty_floor` genes mean **the Infotropy mechanisms
are themselves evolution-tuned**, which is the cleanest way to test whether they carry weight:
if evolution drives `record_grade_tau`→0 (gate off), Infotropy-A added nothing.

---

## 6. LLM sentiment organ & GDELT load-bearing (items 3, 4 — home turf)

### 6.1 LLM organ — adopt PROPOSAL_LLM_SENTIMENT, then push it

I adopt the proposal's core: Haiku-on-Bedrock (`anthropic.claude-3-haiku-20240307-v1:0`,
verified alive), GKG-slug + quotation text path (no new hosts, no page fetch), the F0–F4
nightly selection funnel, 22 buckets + 3 global axes + event flags, the validated JSON schema,
fail-soft `llm_status` as an emitted feature, `temperature:0` + stored artifacts + replay reads
artifacts (never re-invokes), and **Option A backfill** (LLM-score archived GKG so the holdout
contains genuine LLM output and attribution is real, not a V2TONE proxy). The pre-committed
kill criteria — including the **<0.8 correlation with raw V2TONE** non-proxy test — are taken
verbatim; if the LLM is just an expensive tone reimplementation, it dies and the GDELT organ
keeps the free tone feature.

### 6.2 GDELT load-bearing — adopt all of G1–G5

GDELT is not a sentiment garnish here; it is the **structural backbone** of the funnel. G1
sector theme-pressure vectors are the cross-sectional attention signal §1 calls the edge; G2
country pressure keys directly to FXI/EWJ/EWZ/INDA/VGK; G3 sector-conditioned tone dispersion
is a vol precursor; G4 novelty/burst is the Infotropy-flavored information-theoretic feature;
G5 actor/geo HHI is a crisis-localization gauge. The theme→sector dictionary is the single most
differentiated artifact in the system — and (honestly, §13) the single biggest curation-overfit
risk. The DATA_EDGE no-look-ahead `visible_from` rule and the 4-files/day sampling frame are
binding.

### 6.3 The push beyond the proposals: **the LLM as event-token annotator**

The peers would use the LLM only to score buckets. I push it one level further, into the
information-edge sweet spot: **the LLM also annotates the top record-graded clusters into
structured event tokens** the Event-Theme Transformer consumes. The same nightly call that
returns the bucket scores returns, per top cluster, a compact `{sleeve_target, direction,
magnitude∈[0,1], novelty∈[0,1], event_type}` tuple (an extra ~15 tokens/cluster in the same
JSON, negligible cost). This makes the LLM a **reader that tokenizes the event field for the
transformer**, not just a mood-meter — the LLM's language understanding does real structural
work that the keyword theme-map cannot (resolving "Powell signals patience" → rates_duration
bullish, novelty-low; "undersea cable severed near Taiwan" → semis/china bearish, novelty-high).
This is the strongest honest version of "an LLM doing real sentiment analysis whose removal
measurably hurts," because removing it removes both the bucket scores *and* the transformer's
richest token features.

### 6.4 Engaging the Infotropy record-grade gate concretely (item 6, Attempt A)

The Liaison's **Attempt A (record-grade conjunctive R1∧R2∧R3 ingestion gate) was designed for
exactly my prior**, and I judge it concretely useful and **adopt it as the event-ingestion
gate** between the funnel and the transformer/LLM-token stream. Each candidate event/cluster
`e` on sleeve `s` at day `t` is scored:

```
R1 persistence       held = mean_{d=1..h}[ |regime_stat(s,[t+1,t+d]) - base| > k·sd_base ]   # does the move STAY?
R2 non-self-encoding R2 = clip(1 - OOS_R2(event_intensity(e) ~ lagged_returns(s,[t-5,t-1])),0,1)  # not a price echo
R3 downstream reuse  R3 = max(0, AUC_oos(model|with e) - AUC_oos(model|without e))            # adds OOS info
record_grade(e) = R1 * R2 * R3        # conjunctive — zero if any leg fails
# only events with record_grade > record_grade_tau (evolution-tuned gene, §5) become tokens.
```

This is genuinely non-standard: R2 specifically kills **reverse-causality events** (news echoing
a move that already happened — the dominant false-signal class in GDELT/sentiment), which a
naive R3-predictive screen lets through. It is cheap (rolling stats + one OOS regression + one
ablation on data already ingested) and falsifiable (R3-only screened set vs R1∧R2∧R3 gated set;
if the gate adds no OOS lift, evolution sets `record_grade_tau`→0 and we report it dead). I
adopt **B (record-formation label weighting, anti-HILL)** too — see §7. I treat **C/D/E** as
the Liaison graded them: C's anti-circularity rule informs the trust ledger (measure
selectivity at the output distribution, never back from P&L) but is not sold as edge; **D
(self-record feedback)** I adopt as the Member-5 contrarian + an executive context feature;
**E (bottleneck-width / breadth rate-of-change)** I adopt as a plain S0/G4 funnel feature.
**F is rejected** (collapses to vol-squeeze trading; canon's own STOP rule forbids the rename).

---

## 7. Infotropy mechanism — concrete adoption (item 6)

| Liaison grade | Mechanism | Gamma verdict | Where it lives |
|---|---|---|---|
| TRANSFERS | A — record-grade R1∧R2∧R3 ingestion gate | **ADOPT** | event→token gate (§6.4), threshold = evolution gene |
| TRANSFERS | B — record-formation label weighting (anti-HILL) | **ADOPT** | ETT + GBM + meta-evaluator loss |
| RESTATEMENT | C — B-parameter trust structure | **ADAPT (rigor only)** | trust ledger anti-circularity; not sold as edge |
| RESTATEMENT | D — self-record feedback | **ADOPT (as plain factor)** | Member-5 + executive context |
| RESTATEMENT | E — bottleneck-width regime gate | **ADOPT (as plain feature)** | S0 breadth / G4 in `field_state` |
| NO-TRANSFER | F — possibility-space/exergy | **REJECT** | — |

**B — anti-HILL label weighting (the load-bearing adoption).** Every supervised target (ETT
direction, GBM rank, executive utility) is sample-weighted by record-formation score:

```
record_formation_score(move@t,s,h) = clip( (|regime_stat(s,[t+1,t+h]) - regime_stat(s,[t-h,t-1])| / sd_base)
                                            * (1 - fraction_retraced_within(s,t,h)), 0, 1 )
sample_weight[i] = base_weight[i] * (eps + record_formation_score[i])     # eps = 0.05
```

A move that **persisted** (laid down a durable regime record) gets near-full weight; a move that
fully **round-tripped** (transient, HILL-like, near-efficient/unforecastable surprise) is
down-weighted toward `eps`. This concentrates the ensemble's scarce capacity on the
forecastable part of the target and stops the transformer overfitting one-day noise — directly
serving §1's thesis that the *durable regime* (not the transient level) is the exploitable
part. Falsifier: A/B train each model with uniform vs record-formation weights; if the latter
doesn't improve purged-validation Sharpe/calibration, set eps→1 (effectively off) and report it.

Honest grading: A and B are the two genuinely non-standard, canon-motivated, cheaply-falsifiable
mechanisms. The rest are useful but standard quant in canon vocabulary, and I do not sell them
as an infotropy edge. F does not survive contact with daily bars. The final-line `infotropy`
attribution will be whatever the leave-one-out arm (§12) measures — and if A and B both ablate
to zero, the honest verdict is `infotropy=no-transfer`, per packet.

---

## 8. Action pipeline (replay-harness contract)

```
NIGHT RUN (03:00 UTC Tue–Sat, decision date D, data through D-1 close):
  1. funnel: acquire GDELT/CBOE/COT/FRED/Treasury -> field_state[D], event_tokens[D]   (§2)
  2. LLM organ: read top-120 clusters -> llm_sentiment.json + event-token annotations    (§6)
  3. experts: ETT, ISRE, GBM, LLM, (contrarian) -> mu[M,64], sigma[M,64], c[M]           (§3)
  4. executive: gating MLP -> tau[M], f, w_tgt[64]; deterministic blend                  (§4)
  5. intents: delta = w_tgt - w_prev; emit BUY/SELL/REDUCE where |delta|>no_trade_band   (§4)
                                       -> daily/D/trade_intents.json + meta_decision.json
MORNING RUN (14:45 UTC Mon–Fri): harness fills intents at D+1 OPEN with the per-sector
  half-spread + ±2bps seeded slippage cost model; marks to D close; VUG 6:1 split handled once.
```

This satisfies the replay-engine contract exactly: a decision at D from `daily/D/*` (through
D−1 close) producing trade-intent dicts filled at next open through `src/utils/
transaction_costs.py`, with the harness cluster cap clamping correlated BUY intents downstream.
Turnover is priced honestly in the executive's training loss (the `cost(D)` term uses the *same*
per-sector spread table) and the no-trade band (evolution gene) is the turnover brake — a
daily-rebalance design pays 3–10 bps/side in the niche names and the executive is trained
knowing it. Replays read stored artifacts (LLM never re-invoked), pin to S3 snapshots, fixed
seeds, cost-model version in the manifest — per EVIDENCE_PROTOCOL.

---

## 9. Training plan — the uneven-history problem (my hardest problem)

History depth is wildly uneven across streams; this is the design's central difficulty and the
missing-mask discipline is its answer.

```
Depth ladder (binding, from DATA_EDGE §3):
  prices + CBOE vol-surface + FRED .......... 2014-08 / 1990 / 1967  (deepest)
  GDELT-rich G1-G5 .......................... 2015-02-18  (the binding common-panel constraint)
  COT positioning ........................... 2006 weekly
  LLM-derived features ...................... ~90 real days (Tier-1) + Tier-2 to 2024-01; ZERO before
  brain's own live record ................... ~210 days (fine-tune/calibrate only)
```

**Three-phase training, each model degrading gracefully when a stream is dark:**

1. **Pretrain on the deep price+vol panel (2014-08 → 2026-02).** ISRE and a price-only ETT
   variant pretrain here on the always-present streams; GDELT/LLM token inputs are
   **zero-filled with `field_mask=0`** before 2015-02 and before the LLM backfill window. Every
   model **consumes the mask as a real input** — the ETT learns to attend to nothing when
   `event_tokens` is empty; the GBM has the masks as features; the ISRE has the health block.
   A model is *never* trained to assume a stream is present.
2. **Joint train on the common panel (2015-02 →) walk-forward.** Six expanding-window folds
   (DATA_EDGE/EVOLUTION layout), **21-trading-day purge/embargo** on each split boundary
   (longer than the 21d max label horizon → no label leakage). LLM features carry their
   `llm_available` mask: real where Tier-1/Tier-2 backfill exists (≥2024-01), zero+masked
   before. **The LLM is never a trained-on historical feature where it doesn't exist** — its
   pre-2024 contribution to gradient training is exactly zero by mask, and the bake-off/holdout
   (2026-03-11→, fully inside Tier-1) is where its attribution is honestly read.
3. **Fine-tune/calibrate on the own-record (~130 pre-holdout days).** Per the meta-evaluator
   proposal: freeze `phi` and the blend; fine-tune only the trust-logit layer and sizing head
   at LR×0.1 to calibrate to the real morning-fill/cost regime. 130 days cannot support new
   structure and we do not ask it to. The last ~62 days (holdout) are never trained on.

**Walk-forward attestation (the single most important leakage guard):** each expert hands the
executive **walk-forward-generated** historical opinions (the expert trained only on data before
the dates it opines on) with a `walk_forward: true` manifest field; the executive refuses to
train without it (build error). In-sample expert history would teach the executive to over-trust
the most overfit expert — the design's most likely silent failure.

**Seeds & budget:** 5 seeds everywhere, recorded; deterministic GA seed from
`sha256("PKT-TB-006-EA"+window_end)`. Monthly local-Mac budget (1–2 h): head training is the
cost driver. ETT ~28k params + ISRE ~14k + GBM over the panel, 6 folds × 5 seeds — torch 2.1.2
CPU. If the 1–2 h window is breached, **shrink seeds then epochs then GA generations — never
drop an organ or the bake-off** (packet stop-condition). Estimated 35–70 min; comfortable.

---

## 10. Sample budget table (every learnable component)

| Component | Params | Effective sample | Memorization defense |
|---|---|---|---|
| **ETT** (event transformer) | **~28k** | ~2,790 days × interacting events; effective ~600 indep 5-day windows | d_model=32, shared embeddings, dropout 0.2/0.1, 5-seed, anti-HILL weights, bag-of-events fallback rung |
| **ISRE** (GRU regime) | **~14k** | ~2,800 day-sequences; vol/corr targets dense & high-SNR | hidden 24, weight decay, 5-seed, purged WF; vol is the most forecastable target so ratio defensible |
| **GBM ranker** | trees (≤400×depth-4) | 189k symbol-days but cross-correlated → ~600 indep day-cross-sections | depth 4, min_child 200, feat/bag frac 0.6, early stop, L1/L2 |
| **LLM organ** | 0 learnable (frozen Haiku) | n/a (inference organ) | not trained; backfill-bounded; non-proxy & schema falsifiers |
| **Member-5 contrarian** | ~30 | weekly COT ~1,040 obs | linear, evolution-gated, drop if no attribution |
| **Meta-evaluator** | **~450–600** (ceiling 3k) | ~2,959 days → ~600 indep h=5 windows | shared phi, deterministic blend, measurement-driven trust, linear-gate fallback rung |
| **Evolution genome** | ≤40 genes, ≤400 evals | 6 regime folds | adoption gate >1σ over default, B1 random-search baseline, parsimony −0.02/gate, fold subsampling |

The binding fact (§7 brief): ~11.7y spans only ~6–8 distinct regimes, ETFs are cross-correlated,
so *effective* sample is hundreds of windows, not 189k. Every learnable component is sized to
that reality with an explicit simpler-fallback rung; the transformer is the most data-hungry
and is the first organ the Skeptic should attack (§13).

---

## 11. Cost sketch (itemized monthly vs ≤$10 target / $15 hard fail)

| Line | Arithmetic | Monthly |
|---|---|---|
| Base Lambda (night ~110s + morning ~15s, 3008 MB) | existing | ~$6.00 |
| S3 storage + requests (incl. gdelt_rich ~10 MB, feature parquets ~5 MB, daily artifacts) | existing + ~$0.06 | ~$1.06 |
| Secrets Manager | existing | ~$1.00 |
| CloudWatch | existing | ~$1.00 |
| ECR (1.2 GB image, pure-numpy/torch already in it) | existing | ~$0.12 |
| **LLM organ (Bedrock Haiku)** nightly ~$0.0044×22 + Lambda parse + S3 | PROPOSAL_LLM_SENTIMENT | ~$0.20 |
| **Data-edge funnel** (+8 GDELT zips, ~10 CSV/API calls, +20–40s Lambda, S3) | PROPOSAL_DATA_EDGE | ~$0.05 |
| **Executive + evolution** (local training, nightly forward pass <1ms, tiny JSON) | proposals | ~$0.01 |
| **TOTAL MONTHLY** | | **≈ $9.4/month** |
| One-time backfill: LLM Tier-1+Tier-2 Bedrock ($2.40) + GDELT-rich GKG/events (~80–110 GB, $0 cash, local Mac 10–20 h) | | **~$2.40 once** |
| Phase-C live Bedrock hard cap (logged call-by-call) | | **$5.00 total** |

**≈ $9.4/month — inside the $10 target, well clear of the $15 hard fail.** The information-edge
prior is the most data-hungry, and the proof it fits is that **the differentiated streams are
free-over-HTTP and the only marginal AWS cost is GDELT bandwidth (~$0.05) and Haiku tokens
(~$0.20)** — the funnel's bytes are paid in local Mac time, not AWS dollars. If a reduction is
forced, the order is: drop LLM Tier-2 backfill depth → reduce GDELT to 2 files/day pre-2022 →
shrink seeds — never drop an organ.

---

## 12. Attribution hooks (leave-one-out per organ — Phase D, pre-registered)

Each arm retrains the full brain with one organ neutralized, replays on the **identical
harness, holdout-only (2026-03-11→), paired daily-difference stats** per EVIDENCE_PROTOCOL. An
organ "carries weight" iff holdout ΔSharpe > 0 with positive paired-daily mean and t respecting
protocol form; a measured zero is reported as zero (a real finding, per packet).

| Organ | Ablation | What kills it |
|---|---|---|
| **Transformer (ETT)** | replace with bag-of-events MLP | event *interaction* adds no OOS lift → transformer attribution ≈ 0 |
| **Ensemble (each member)** | drop member m, renormalize trust | member's marginal holdout Sharpe ≤ 0 |
| **Evolution** | champion genome vs DEFAULT_GENOME (B0) + random-search (B1) | champion ≤ B0 on holdout, or ≤ B1 fitness → ceremonial |
| **LLM** | replace all `llm_*` + event-token annotations with neutral constants | ΔSharpe ≤ 0 or paired t < 1 or corr(sent, V2TONE) ≥ 0.8 (tone proxy) |
| **GDELT-rich** | ablate G1–G5 block (sub-arm: G1 theme-map alone) | \|t\| < 1 → evidence-backed no-signal (item 4) |
| **Meta-evaluator** | replace with equal-trust + fixed-f baseline | full brain ≤ baseline brain on holdout, or static trust (std(tau)<0.02) |
| **Infotropy A+B** | set `record_grade_tau`→0 and anti-HILL `eps`→1 | no OOS lift → `infotropy=no-transfer` honestly |

The ~62-day holdout has wide error bars; every arm reports its minimum-detectable-effect up
front and leans on the E1 full-period + cross-fold evidence where the holdout is too short
(COT, Treasury especially). Looks against the holdout are counted in the run journal; one
pre-registered config per arm.

---

## 13. Risks & honest weaknesses (where my prior loses)

1. **Daily lag may close the direction edge entirely.** §1 concedes direction on the level is
   ~efficient; if cross-sectional *attention* alpha is also arbitraged at daily cadence, the
   whole funnel only buys sizing/regime value and the brain may not beat a price-only incumbent
   on return — only (maybe) on risk-adjusted return. This is the most likely way I lose.
2. **GDELT is dark across the entire holdout right now.** Item 4 cannot be evaluated until the
   2026-02-05→present top-up backfill runs. If that backfill reveals the 15-min files are
   thinner/different than the historical parquet assumed, GDELT attribution could be a
   measurement artifact, not signal. Build-task #0; failure here is fatal to the showcase's
   marquee organ.
3. **The theme→sector dictionary is the edge — and a curation-overfit trap.** The most
   differentiated artifact (G1) is hand-built; a dictionary tuned (even implicitly) to make the
   backtest work is exactly the data-mined coincidence the Skeptic will hunt. Defense: the
   dictionary is frozen and committed *before* any holdout read, and the record-grade gate +
   anti-HILL weighting regularize against spurious theme→return links — but this is the
   weakest seam.
4. **The LLM organ has no deep history.** It can never be a trained-on feature pre-2024; the
   ensemble must work with it masked-off for most of pretraining and only "switched on" for the
   recent window. If its real attribution is dominated by the ~62 holdout days, the effect is
   under-powered and the honest verdict may be "promising but unproven on this sample."
5. **The transformer is the most data-hungry component on the thinnest effective sample.** 28k
   params over ~600 effective windows is defensible only with aggressive sharing/dropout and
   the anti-HILL weighting; the bag-of-events fallback exists precisely because event
   *interaction* may not be learnable at this sample size, in which case the marquee
   "transformer doing real work" degrades to a pooled-MLP and I must report it.
6. **Novelty/burst features may be noise.** JS-divergence theme-novelty is information-
   theoretically pretty but may not predict returns at all; it is gated and ablated, and if it
   dies it dies on the record.
7. **Short holdout + many organs = multiple-looks risk.** Seven attribution arms on ~62 days is
   a multiple-comparisons hazard; the dossier must state looks consumed and treat a single
   +0.1-Sharpe arm among seven as noise, per EVIDENCE_PROTOCOL.

---

*Design Gamma complete. The funnel is the moat; four genuinely different models exploit it
(event-theme transformer, regime GRU, GBM ranker, LLM); the adopted meta-evaluator turns it
into bets while learning when its differentiated input is trustworthy; evolution decides which
streams pay their way; two Infotropy mechanisms (record-grade event gate, anti-HILL label
weighting) carry the canon's only non-standard transfers; ≈$9.4/month.*
