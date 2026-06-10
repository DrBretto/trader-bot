# PROPOSAL — LLM Sentiment Organ (panel role 9)

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — **Author:** LLM Sentiment Engineer — 2026-06-10
**Serves:** all three architects blindly. Designed from the assignment brief only (sealed-incumbent respected).
**One-line:** Haiku-on-Bedrock reads ~120 deduplicated headline-equivalents per night extracted from GDELT GKG
15-minute files (URL slugs + quotations + themes/actors — zero new network hosts), emits a validated JSON
sentiment surface over 22 asset buckets + 3 global axes, costs ≈ $0.004/night live and ≈ $2.40 one-time to
backfill to 2024-01, and pre-commits its own kill criteria.

**Live evidence gathered today (2026-06-10), within allowlist + bounded-Bedrock allowance:**
- Bedrock probe: `anthropic.claude-3-haiku-20240307-v1:0` invoked successfully (us-east-1, profile
  `personal`; response `msg_bdrk_01Q5dgXJ4j8J2GgsJiRBAFvS`, 9 in / 4 out tokens). The pinned model is ALIVE.
- GKG probe: `gdeltv2/20260609030000.gkg.csv.zip` = 3.0 MB zip / 9.3 MB raw, **720 records, 27 tab-separated
  fields**. V2TONE (idx 15) populated 720/720; Quotations (idx 22, `offset|len|verb|quote` packets) populated
  132/720 (~18%); **565/720 (~78%) of source URLs carry a wordy slug (≥4 alpha tokens)** — i.e., real
  headline-equivalent text is genuinely extractable without fetching a single article page.

**Model-risk flag (must appear in the dossier):** `claude-3-haiku-20240307` is deprecated upstream
(first-party API retirement was listed for Apr 2026). It is verified alive on Bedrock today, but Bedrock EOL
is a real medium-term risk. Design runs on the pinned ID (no IAM change needed for this packet); fallback is
`gpt-4o-mini` via the existing Secrets Manager key; productionization should flag an IAM widening to a
Haiku-4.5-class Bedrock ID as a follow-on (cost impact at $1/$5 per MTok: ≈ $0.55/mo — still in envelope).

---

## 1. Text acquisition — what the LLM actually reads

**Source: GDELT GKG v2 15-minute English files only** (`data.gdeltproject.org` — already allowlisted, verified
alive in brief §3.4). **No new hosts.** Fetching the articles behind GKG's source URLs would require
allowlisting arbitrary news domains — rejected; flagged here per packet instructions, not assumed.

The densest real-text package per token, per GKG record (field indices verified live):
1. **URL slug → headline-equivalent** (idx 4, `V2DOCUMENTIDENTIFIER`): last path segment, split on `-`/`_`,
   strip extension/date/ID tokens. ~78% of records yield ≥4 words ("fed-holds-rates-steady-as-inflation-cools"
   → "fed holds rates steady as inflation cools"). Records with non-wordy slugs are dropped from the text set
   (they still count toward salience via clustering).
2. **Quotations** (idx 22): actual quoted sentences from the article ("optional premium add-on service" style
   packets). Attach the first quotation, truncated to 25 words, for the subset that has one (~18%).
3. **Structured context**: V2THEMES (idx 8, with char offsets — take the 4 earliest = headline-proximate
   themes), V2ORGANIZATIONS/V2PERSONS (idx 14/12, first 3), V2TONE tone+polarity (idx 15), source domain
   (idx 3).

### Nightly selection funnel (concrete rules, in order)

Run inside the existing nightly Lambda step, before inference:

| Step | Rule | Expected volume |
|---|---|---|
| F0 fetch | 4 sampled English GKG files per 24h window: timestamps `06:00, 12:00, 18:00` UTC of D-1 and `00:00` UTC of D (all published ≥3h before the 03:00 UTC run). Window widens to "since last successful run" across the weekend gap (the Tue 03:00 run sweeps Sat–Mon, 72h). | ~12–50 MB zip; 3k–10k records |
| F1 relevance | Keep records where (a) ≥1 theme in the curated whitelist `THEMES_FIN` (ECON_STOCKMARKET, ECON_INTEREST_RATE/CENTRALBANK, ECON_INFLATION, ECON_OILPRICE/ECON_GASPRICE, ECON_BANKING/BANKRUPTCY, ECON_CURRENCY_EXCHANGE_RATE, ECON_TRADE_DISPUTE/TARIFF, ECON_HOUSING_PRICES, ECON_EARNINGS, ECON_DEBT, plus geopolitical escalation themes ARMEDCONFLICT/BLOCKADE/SANCTIONS), **or** (b) ≥1 actor/org in the curated `ACTOR_MAP` (Federal Reserve/FOMC/named central banks, OPEC, US Treasury, megacap names for sector mapping — NVIDIA/TSMC/Intel→semis, JPMorgan/regional-bank names→financials, Pfizer/Moderna→healthcare, Boeing/Lockheed→defense, Exxon/Chevron→energy…), **or** (c) location field names a country with a country ETF (China→FXI, Brazil→EWZ, India→INDA, Japan→EWJ, Eurozone→VGK/FXE). | ~400–1,500 records |
| F2 cluster/dedup | Normalize slug words (lowercase, stopword-strip, sort) → 6-word shingle hash = `cluster_id`. Syndicated copies collapse; `n_sources` = count of distinct source domains in cluster. | ~150–400 clusters |
| F3 seen-cache | Drop clusters whose `cluster_id` appears in the rolling seen-store (S3 JSON, 5-calendar-day window). Their previously assigned bucket scores decay into today's salience priors (×0.5/day) so persistent stories register without re-billing. **This is the caching/dedup answer: a story is billed once.** | ~100–300 novel clusters |
| F4 rank & cap | Score = `n_sources × (1 + |tone|/5) × theme_priority`; take **top 120** clusters. Floor: if <10 clusters survive (GDELT outage / dead files), skip the call and emit neutral-with-flag (§2). | ≤120 story lines |

### Bucket map (the LLM scores buckets, not symbols)

22 buckets + 3 global axes; static table shipped with the organ (`bucket_map.json`):

| bucket | symbols | bucket | symbols |
|---|---|---|---|
| us_broad | SPY QQQ IWM DIA RSP VTI VOO IVV VT | healthcare_biotech | XLV XBI IBB |
| tech | XLK XLC ARKK | industrials_defense | XLI ITA IYT |
| semis | SOXX SMH | consumer_disc | XLY XRT |
| financials_banks | XLF KRE | consumer_staples | XLP |
| energy_oil | XLE USO | utilities | XLU |
| natgas | UNG | materials | XLB |
| real_estate | VNQ IYR | rates_duration | TLT IEF SHY TIP AGG BND MUB |
| intl_dev | VEA EFA | credit | LQD HYG |
| europe | VGK | gold_pm | GLD SLV |
| japan | EWJ | commodity_broad | DBC |
| em_broad | VWO EEM | usd / eur | UUP / FXE |
| china | FXI; india: INDA; brazil: EWZ | volatility | VIXY |

Factor/style ETFs (MTUM QUAL VLUE USMV VIG SCHD VUG VTV) have no news identity of their own: they consume
`us_broad` sentiment plus the global axes (`risk_appetite`, `rates_pressure`) — defensive factors load
negatively on risk_appetite, growth styles positively. Global axes: `risk_appetite` ∈ [-1,1],
`rates_pressure` ∈ [-1,1] (+1 = hawkish/higher-rates news), `geopolitical_risk` ∈ [0,1].

---

## 2. The nightly call

**Model:** `anthropic.claude-3-haiku-20240307-v1:0` on Bedrock (the only IAM-allowed ID — verified alive).
**Fallback chain:** Bedrock (2 retries, exp. backoff, 30s timeout) → 1 schema-repair retry (validator errors
appended) → OpenAI `gpt-4o-mini` (existing key, same prompt/schema; artifact records `model_used`) → neutral
artifact. **Determinism:** `temperature: 0` (supported on this model generation), `max_tokens: 1500`; prompt
SHA-256, model ID, and raw response stored in the artifact. The LLM is still not bit-deterministic — so **the
replay harness never re-invokes the LLM**: replays read the stored per-day artifacts. One call per night
(single batch of ≤120 story lines).

### Prompt template (verbatim)

System (frozen string, never interpolated):

```
You are the sentiment organ of a daily ETF allocation system. You read financial news
headlines (with themes, actors, tone metadata, and occasional quotes) collected since the
last trading session, and you emit a single JSON object scoring market-relevant sentiment.

Rules:
- Score each BUCKET only from stories actually relevant to it. No stories => sent 0, conf 0.
- "sent" is directional for the bucket's assets over the next 1-5 trading days: +1 strongly
  bullish, -1 strongly bearish, 0 neutral/mixed. For rates_duration, +1 means bullish BOND
  PRICES (dovish/falling yields). For volatility, +1 means rising volatility expected.
- "conf" reflects agreement and source breadth, not your general market view.
- "sal" reflects how much of tonight's news mass concerns the bucket (0 = none, 1 = dominant).
- event_risk flags are for scheduled or breaking events that could gap prices: use only
  flags from the allowed list, with the affected buckets.
- Use ONLY the provided stories. Do not use memorized knowledge of what markets did.
- Output ONLY the JSON object. No prose, no markdown fences.
```

User message (built nightly; `{...}` are filled slots):

```
DATE: {decision_date}  WINDOW: {window_start_utc} -> {window_end_utc}
BUCKETS: us_broad, tech, semis, financials_banks, energy_oil, natgas, real_estate, intl_dev,
europe, japan, em_broad, china, india, brazil, healthcare_biotech, industrials_defense,
consumer_disc, consumer_staples, utilities, materials, rates_duration, credit, gold_pm,
commodity_broad, usd, eur, volatility
ALLOWED EVENT FLAGS: cb_decision, cb_speech, inflation_print, jobs_print, earnings_megacap,
geopolitical_escalation, sanctions_tariffs, credit_event, energy_supply_shock, election_political,
regulatory_action, natural_disaster

STORIES ({n} clusters, fields: id | headline | themes | actors | tone | n_sources | quote?):
1 | fed holds rates steady as inflation cools | ECON_INTEREST_RATE,ECON_INFLATION | Federal
Reserve;Jerome Powell | -0.7 | 23 | "we need greater confidence inflation is moving down"
2 | china export curbs hit chipmakers | ECON_TRADE_DISPUTE,ECON_STOCKMARKET | NVIDIA;TSMC | -3.1 | 11
... ({up to 120 lines})

Respond with the JSON object now.
```

### Output schema (validated with jsonschema; clamped on ingest)

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["as_of", "buckets", "global", "event_flags"],
  "properties": {
    "as_of": {"type": "string"},
    "buckets": {"type": "object",
      "patternProperties": {"^[a-z_]+$": {
        "type": "object", "additionalProperties": false,
        "required": ["sent", "conf", "sal"],
        "properties": {
          "sent": {"type": "number", "minimum": -1, "maximum": 1},
          "conf": {"type": "number", "minimum": 0, "maximum": 1},
          "sal":  {"type": "number", "minimum": 0, "maximum": 1},
          "drivers": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}}}}},
    "global": {"type": "object", "additionalProperties": false,
      "required": ["risk_appetite", "rates_pressure", "geopolitical_risk"],
      "properties": {
        "risk_appetite":    {"type": "number", "minimum": -1, "maximum": 1},
        "rates_pressure":   {"type": "number", "minimum": -1, "maximum": 1},
        "geopolitical_risk":{"type": "number", "minimum": 0,  "maximum": 1}}},
    "event_flags": {"type": "array", "items": {"type": "object",
      "required": ["flag", "buckets"], "additionalProperties": false,
      "properties": {
        "flag": {"type": "string", "enum": ["cb_decision","cb_speech","inflation_print",
          "jobs_print","earnings_megacap","geopolitical_escalation","sanctions_tariffs",
          "credit_event","energy_supply_shock","election_political","regulatory_action",
          "natural_disaster"]},
        "buckets": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "number", "minimum": 0, "maximum": 1}}}}
  }
}
```

**Fail-soft (organ never invents):** any unrecoverable failure (Bedrock + fallback dark, schema invalid after
repair, <10 stories) emits `{"llm_status": "dark"|"thin_input"|"fallback_model", buckets: all sent=0 conf=0
sal=0, global: zeros, event_flags: []}` plus SNS alert. `llm_status` is itself an emitted feature so the
meta-evaluator can learn to discount dark nights — the organ degrades, it never fabricates.

---

## 3. Historical backfill for training — decision: LLM-score archived GKG (Option A)

**Options weighed honestly:**
- **(B) V2TONE as historical proxy, LLM live-only.** Cheap, but creates train/serve skew (the brain trains
  on tone-shaped features then receives differently-distributed LLM output live), and the bake-off
  "LLM organ" attribution would mostly be measuring V2TONE — which is the GDELT organ's feature, not mine.
  Rejected as a primary; V2TONE aggregates remain available to the GDELT organ as *separate, distinctly named*
  features so attribution stays clean.
- **(A) Batch-score archived GKG with the identical funnel + prompt.** Chosen. The archive (GKG 2.0 from
  2015-02-18) lets the exact nightly pipeline be replayed offline; cost is trivial (below).

**Backfill plan (runs on the operator's Mac, not Lambda):**
- **Tier 1 (mandatory): 2026-01-31 → present** (~89 trading days) — covers the replay window and the entire
  E2 holdout (2026-03-11 →). Real LLM output exists everywhere the bake-off reads. ≈ **$0.35**.
- **Tier 2 (planned): 2024-01-02 → 2026-01-30** (~523 trading days) — training depth. ≈ **$2.05**.
- **Pre-2024:** LLM features are emitted as missing (0-filled + `llm_available=0` mask feature). Trainers
  must treat the mask as a real input; the Training Realist should verify no component implicitly assumes
  full-history coverage. Going deeper later is linear: full 2015→present ≈ $11 one-time, deferred.
- **Sampling grid is identical live and backfill** (4 files/day at the same UTC stamps) — no train/serve
  window mismatch. Download: ~940 calendar days × 4 files × ~3 MB ≈ **11 GB** to local disk (free, overnight);
  raw files cached locally so reruns are free.
- **No look-ahead:** each backfilled day D uses only files timestamped inside D's live window (§4); weekends
  fold exactly as the live cadence folds them. Seen-cache is replayed forward chronologically so dedup
  behavior matches live.
- **Attribution consequence, stated:** with Option A the bake-off window AND holdout contain genuine LLM
  output, so leave-one-out attribution for this organ is a real measurement, not a proxy measurement.

## 4. Signal definition — what the brain consumes

Artifact: `daily/<D>/llm_sentiment.json` shape per §2 (prototype: run-dir equivalent). Tabular features
emitted to the feature store, nightly cadence, one row per decision date D:

| feature | range | notes |
|---|---|---|
| `llm_sent_<bucket>` | [-1, 1] | 27 buckets incl. usd/eur/volatility |
| `llm_conf_<bucket>` | [0, 1] | |
| `llm_sal_<bucket>` | [0, 1] | |
| `llm_sent_<bucket>_ema3` | [-1, 1] | 3-day EMA, smoothed view |
| `llm_sent_<bucket>_d1` | [-2, 2] | day-over-day delta (news momentum) |
| `llm_risk_appetite`, `llm_rates_pressure` | [-1, 1] | global axes |
| `llm_geopol_risk` | [0, 1] | |
| `llm_event_<flag>` | {0,1} ×12 | global; per-bucket severity available in the JSON |
| `llm_status_ok`, `llm_available` | {0,1} | fail-soft + backfill-coverage masks |

Per-symbol view = bucket value via `bucket_map.json` (factor/style symbols = us_broad + axis loadings).
**Timestamp discipline:** the value keyed to decision date D is computed at the 03:00 UTC run of day D from
GKG files published strictly before the run (latest stamp D 00:00 UTC) — i.e., ≥10.5h before D's 14:45 UTC
morning execution. Conforms to the `daily/D/` = "knowable before D's open" convention (brief §3.1/§5.1);
replay reads stored artifacts only, never re-calls the model.

## 5. Cost worksheet line (Haiku Bedrock: $0.25/MTok in, $1.25/MTok out — packet figures; nothing newer assumed; pinned 2024-03-07 model verified live)

| item | arithmetic | cost |
|---|---|---|
| Nightly input | 1.6k prompt/instructions + 120 stories × ~50 tok ≈ 7.6k → budget 10k | $0.0025 |
| Nightly output | ~1.0–1.5k JSON → budget 1.5k | $0.0019 |
| Per night ≈ $0.0044 × 22 runs/mo × 1.5 retry/fold headroom | | **≈ $0.15/mo** |
| Lambda increment | +~40s/night parse @3008MB = 120 GB-s × 22 | ≈ $0.05/mo |
| S3 | ~10 KB/day artifact + seen-store | < $0.01/mo |
| **Live organ total** | | **≈ $0.20/mo** |
| Backfill one-time | Tier 1 89d × $0.0044 ≈ $0.35; Tier 2 523d ≈ $2.05 | **≈ $2.40 once** |
| **Phase C hard cap** | **$5.00 total Bedrock/OpenAI spend for this packet**, logged call-by-call in COST_WORKSHEET.md. Sampling: 10-day pilot (~$0.05) to validate funnel+schema → Tier 1 → Tier 2 only if cumulative spend < $3. | |

(At a Haiku-4.5-class price of $1/$5 per MTok the live line becomes ≈ $0.55/mo — envelope-safe either way.)

## 6. Falsifier — pre-committed kill criteria

**Stage 1 — pre-integration sanity (cheap kills before any bake-off):**
1. Non-degenerate: backfilled `llm_sent_us_broad` daily variance > 0 and not >95% one sign.
2. **Not a tone proxy:** corr(`llm_sent_us_broad`, daily-aggregate V2TONE) < 0.8 over the backfill window.
   If ≥ 0.8, the organ is an expensive reimplementation of a free field — report and kill (the GDELT organ
   keeps the tone feature).
3. Event flags fire on ≥3 known scheduled events in-window (FOMC decision days, CPI prints) and on <20% of
   ordinary days.

**Stage 2 — bake-off leave-one-out (the packet's attribution bar):** primary ablation = retrain the full
brain identically with all `llm_*` features replaced by their neutral constants (sent/conf/sal/axes = 0,
status = dark); secondary = neutralize features in the already-trained brain. Both replayed on the identical
harness, holdout-only read (2026-03-11 →, where real LLM output exists), paired daily-difference stats per
EVIDENCE_PROTOCOL. **The organ dies if:** holdout ΔSharpe(with − without) ≤ 0, or the paired daily t-stat
< 1, or removal *improves* holdout return. Any of these is reported as zero-attribution in the assignment
scorecard, with the honest implication for the showcase — per packet, a measured zero beats a silent garnish.
