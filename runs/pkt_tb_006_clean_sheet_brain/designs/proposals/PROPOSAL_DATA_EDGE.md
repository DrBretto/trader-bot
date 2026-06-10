# PROPOSAL — Data Edge Scout (panel role 10)

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — Phase A mechanism proposal, served blind to all three architects.
**Author:** Data Edge Scout, 2026-06-10.
**Charter:** differentiated data beyond GDELT + the historical-depth problem.
**Verification discipline:** every reachability/depth claim below marked **[verified]** was checked live on 2026-06-10 from this sandbox with small HTTP probes (HEAD, range GETs, 1-row API queries). No bulk downloads were performed. All sources are free, no signup, on the existing network allowlist unless flagged.

---

## 1. GDELT depth plan (co-owned with the LLM Sentiment Engineer)

### 1.0 Ground truth, verified

- GDELT v2 15-min **GKG**, **events (export)**, and **mentions** files all return HTTP 200, from 2015-02-18 through present **[verified: 20150218230000 and 20260608120000 for all three families]**.
- File sizes **[verified Content-Length]**: GKG ≈ 2.1–7.9 MB/file (2015 and 2020 samples ~7.9 MB; 2026 samples 2.1–4.4 MB); v2 events ≈ 49–125 KB/file. GDELT 1.0 daily events (one file/day, back to 1979) ≈ 6–12 MB/day, still being published **[verified 20260608]**.
- The brief's critical fact stands: the live pipeline's `gkgcounts` endpoint is dead and `historical_gdelt.parquet` **ends 2026-02-04**. Consequence (see §3): **the replay window and the entire holdout currently have NO real GDELT data.** Any GDELT organ requires a top-up backfill 2026-02-05 → present before it can even be evaluated. This is the single most urgent data task in the packet.

### 1.1 Sampling frame (shared by all features below)

Keep the proven `training/scripts/backfill_gdelt.py` frame: **4 samples/day at 00/06/12/18 UTC**, for BOTH GKG and v2 events files, for BOTH backfill and live ingestion. Why this exact frame:

- All four timestamps precede the 21:00 UTC US close, so a feature row for UTC day *d* is fully knowable by *d*'s close — no intraday leakage and **no train/serve mismatch** (live nightly pulls the same four files the backfill did).
- **No-look-ahead timestamp rule (binding, all GDELT features):** the row for UTC day *d* is built only from files stamped `d` 00/06/12/18 UTC; it becomes visible to the brain at decision date **D ≥ d+1** (the 03:00 UTC night run for decision date D may read GDELT days ≤ D−1). In replay, GDELT day *d* joins to `daily/D/` only where D ≥ d+1. Backfilled rows carry a `visible_from` column; the replay join uses it, never the raw date.

One pass over the GKG files extracts ALL features below (parse once, emit many columns) — the marginal cost of each extra feature family over the existing tone-only backfill is parsing, not bandwidth.

### 1.2 Feature families to backfill (2015-02-18 → present)

| # | Family | Construction (exact) | Why retail bots don't have it |
|---|--------|----------------------|-------------------------------|
| G1 | **Sector theme-frequency vectors** | From GKG `V2Themes` (col 8): curated dictionary mapping ~60–100 GDELT theme codes (ECON_INTEREST_RATE, ECON_BANKING, ECON_OILPRICE, ECON_INFLATION, ENV_OIL/GAS, ARMEDCONFLICT, CYBER_ATTACK, HEALTH_PANDEMIC, TAX_FNCACT_*, WB_* finance codes…) onto the **13 sector/asset buckets of the 64-ETF universe** (tech/semis, financials/banks, energy, health/biotech, defense, transports, retail/consumer, REIT, utilities, materials/gold, rates/credit, USD/EUR, vol/crisis). Daily features: per-bucket mention share, and per-bucket z vs trailing 90-day mean/sd. ~26–30 columns. | Retail sentiment = one market-wide mood number. A **sector-resolved news-pressure vector aligned to the tradable universe** requires building the theme→sector map; nobody ships it free. |
| G2 | **Country-actor pressure for the country-ETF sleeve** | From v2 events files (CAMEO Actor1/2CountryCode + QuadClass + Goldstein + NumMentions): per-country daily (i) mention-weighted mean **Goldstein score**, (ii) **conflict share** = mentions in QuadClass 3/4 ÷ all mentions, for {US, CN, JP, BR, IN, RU, EU-aggregate} — keyed directly to FXI, EWJ, EWZ, INDA, VGK, EEM/VWO. ~14 columns. | Maps geopolitical event structure to specific tickers the brain trades. Free, but requires CAMEO plumbing retail bots never do. |
| G3 | **Tone dispersion & polarity, sector-conditioned** | From GKG `V2Tone` (col 15, fields tone/positive/negative/polarity): keep global avg/std/neg-share (continuity with existing parquet), add **polarity** (intensity of charged language) and **per-bucket tone** for the G1 buckets (tone of energy news vs tech news…). ~16 columns. | Dispersion/disagreement (high tone_std within a sector) is a different signal from level — disagreement precedes vol. Sector conditioning multiplies the existing single-number tone into a panel. |
| G4 | **Event-novelty / burst measures** | (i) Per-bucket **burst z**: today's G1 bucket count vs trailing 90-day distribution (already implied by G1 z). (ii) **Theme-novelty**: Jensen–Shannon divergence between today's full theme-frequency distribution (top ~500 themes) and the trailing-90-day mean distribution — "how unlike recent news is today's news"; one scalar + a 21-day smoothed version. (iii) **doc-count surprise** vs trailing 90d. ~4 columns. | A trailing-distribution novelty scalar is an information-theoretic feature (natural infotropy hook for role 6) that no free feed publishes. |
| G5 | **Actor/geo concentration** | Herfindahl–Hirschman index over (i) `V2Locations` country codes and (ii) `V2Organizations` from GKG, per day: news concentrated on one place/org = localized crisis; flat = diffuse calm. Plus US-share of locations. ~3 columns. | Concentration is orthogonal to tone and count; trivially computable in the same parse, absent from every retail stack. |

Total: ~60–65 new daily columns, ~4,140 rows (2015-02-18 → present), one parquet ≈ 5–10 MB.

### 1.3 Backfill bytes & compute (one-time, local Mac, $0 cash)

| Job | Files | Transfer | Wall clock (est.) |
|---|---|---|---|
| GKG re-pull, 4/day × ~4,140 days | ~16,560 zips × ~2–8 MB | **~80–110 GB** | ~10–20 h with 4–8 workers (restartable; cache per-day extracts so a crash never re-downloads) |
| v2 events, 4/day × ~4,140 days | ~16,560 zips × ~50–125 KB | **~1.5–2 GB** | ~1–2 h |
| Top-up (2026-02-05 → present, both) | ~500 files | ~1.5 GB | <1 h — **do this first**; it unblocks holdout evaluation |

Bandwidth is free; the only spend is Mac time. Politeness delay 0.3 s/worker as in the existing script. If 10–20 h is unacceptable, degrade GKG to 2 samples/day (00/12) for 2015–2022 and 4/day from 2023 — halves transfer, keeps the holdout-era data at full density; flag the density break in the manifest.

### 1.4 Handoff to the LLM engineer

GKG rows carry source-article URLs and `SourceCommonName`. The cheap text path for the LLM organ: rank each day's finance-bucket GKG records, hand the top-N **headlines/URL slugs** (not fetched pages — most article hosts are off-allowlist) to the LLM. Depth warning the LLM engineer must respect (see §3): LLM-derived features can be backfilled at bounded token cost over the **replay window only (~90 trading days)** — never over 11 years. The LLM organ therefore must be zero/few-shot at inference time, not a source of trained-on historical features.

---

## 2. Differentiated free sources beyond GDELT — verified shortlist

Ranked by (signal plausibility × history depth × implementation cost). All on already-allowlisted hosts.

### S1 — CBOE volatility-surface & implied-correlation indices (`cdn.cboe.com`) — KEEP, rank 1

- **What/verified:** `https://cdn.cboe.com/api/global/us_indices/daily_prices/<IDX>_History.csv` — HTTP 200, clean CSV, **current through 2026-06-09** [verified]. Depths [verified first rows]: VIX 1990, SKEW 1990, VVIX 2006, COR3M (3-month implied correlation) 2006, VIX9D 2011, VIX3M 2011 (file starts 2011 here), VXN 2009.
- **Cadence/lag:** updated end-of-day; the 03:00 UTC night run reads through D−1 — fits exactly. **No-look-ahead rule:** drop any row dated ≥ D.
- **Features:** (a) full **term-structure slope and curvature** VIX9D/VIX/VIX3M (the live feed has only one slope, from stooq, since ~2025); (b) **COR3M level + 1y z** — implied correlation is the market's price of "everything moves together," a direct input for sizing a 64-ETF cross-sectional book; (c) VXN−VIX spread (tech-specific fear vs broad, keyed to QQQ/XLK/SOXX).
- **Differentiated:** COR1M/COR3M and term-structure *curvature* are practitioner objects; retail bots stop at VIX level. Depth to 1990/2006 supports real pretraining.
- **Skepticism:** the VIX complex is the most-mined data in finance; expect the marginal edge over the existing `vix_term_slope/vvix/skew` columns to be modest. The honest case rests on COR3M + curvature + 35y depth, not on "VIX but again."
- **Cost:** ~7 small CSVs nightly (~2 MB total), seconds of Lambda time.

### S2 — CFTC Commitments of Traders, TFF report (`publicreporting.cftc.gov` Socrata) — KEEP, rank 2

- **What/verified:** JSON API live; earliest TFF report **2006-06-13** [verified]; markets include **E-MINI S&P 500 / S&P 500 Consolidated, UST 10Y NOTE, VIX FUTURES** [verified]; fields include `lev_money_positions_long/short`, `asset_mgr_positions_long/short`, `open_interest_all` [verified live row 2026-06-02].
- **Cadence/lag:** weekly; Tuesday positions published **Friday ~15:30 ET**. The Friday 22:00 ET night run can use it same-day. **No-look-ahead rule:** key every row by *publication* datetime (Friday), never by `report_date` (Tuesday) — the classic COT leak. `visible_from` = first night run after publication.
- **Features:** leveraged-fund net position as % of OI, and its 1y z, for ES (→ equity sleeve), UST 10Y (→ TLT/IEF), VIX futures (→ VIXY, and as a crowding gauge); asset-manager net as the slow-money counterpart. ~8 columns, weekly, forward-filled with the publication-lag rule.
- **Differentiated:** positioning *extremes* are a contrarian conditioning signal for the meta-evaluator's risk appetite — structurally different information (who is positioned how) from anything price-derived.
- **Skepticism:** weekly + 3-day lag caps it at regime-tempo value; only ~1,040 obs since 2006 (~13 in the holdout) — it can condition, never drive. Well-known to professionals, so "differentiated" means vs retail bots only.
- **Cost:** 3 API calls nightly (or weekly), KBs.

### S3 — FRED deep catalog (`api.stlouisfed.org`, key already in Secrets Manager) — KEEP, rank 3

- **What/verified:** API host reachable (keyless probe returns the expected 400 key-required JSON [verified]; key exists per brief §4). Target series and depths: **NFCI** (weekly financial conditions, 1971→), **STLFSI4** (weekly stress, 1993→ [verified via public fredgraph CSV]), **T10YIE** 10y breakeven (daily, 2003→), **DFII10** 10y real rate (daily, 2003→), **BAMLH0A0HYM2** HY OAS (daily, 1996→ [verified reachable]), **ICSA** initial claims (weekly, 1967→).
- **Cadence/lag:** daily series next-day on FRED; weekly indices published with up to ~5 business-day delays. **No-look-ahead rule:** lag each weekly series by its documented publication delay (NFCI: following Wednesday; STLFSI4: following Thursday); daily series by 1 day. If the Training Realist wants rigor, ALFRED vintages exist on the same API.
- **Features:** financial-conditions z (NFCI), real-rate level/63d momentum (DFII10), breakeven momentum (T10YIE), HY OAS level/Δ (BAMLH0A0HYM2 — a *daily* credit spread vs the live proxy), claims surprise. Direct drivers for the bond/commodity/defensive third of the universe (TLT, IEF, TIP, GLD, HYG, LQD, XLU, USMV).
- **Differentiated:** not exotic — the differentiation is **depth** (30–55 years for pretraining macro context) and that real-rate/breakeven decomposition specifically prices TIP/GLD/TLT, which generic equity bots ignore.
- **Skepticism:** macro features are slow; most of their value may be absorbed by existing context columns (rate_2y/10y, credit_spread_proxy). Attribution must beat that incumbent-of-its-own.
- **Cost:** 6 extra series on the existing FRED step; negligible.

### S4 — Treasury auction results (`api.fiscaldata.treasury.gov`) — KEEP (small), rank 4

- **What/verified:** `auctions_query` endpoint live, **back to 1979-10-31** [verified], fields incl. `bid_to_cover_ratio`, `high_yield`, `security_term` [verified live row]. Sandbox note: TLS chain failed through the sandbox proxy (probe needed `-k`); data confirmed real. Flag a one-time CA-bundle check in the Lambda/local env before relying on it.
- **Features:** bid-to-cover z by tenor bucket and auction-day dummy — auction-demand surprise for the duration sleeve (TLT/IEF).
- **Skepticism:** sparse/episodic (a few rows/week); realistically a tail-risk conditioner. Keep only because it costs ~nothing and is genuinely unusual in retail stacks. First candidate to cut if the feature count needs trimming.

### S0 — (freebie, not a new source) Universe-internal breadth

From data already held: daily advance/decline across the 64 ETFs, % above 50d/200d MA, cross-sectional return dispersion, equal-weight-vs-cap-weight spread (RSP−SPY). Zero new bytes, zero new hosts; listed so the architects don't overlook that "breadth" needs no external feed. Not claimed as data-edge.

### Killed candidates (explicit)

| Candidate | Verdict | Reason |
|---|---|---|
| **CBOE put/call ratios** (`totalpc.csv` etc.) | KILL | Archive reachable [verified] but **frozen at 2019-10-04** [verified tail]; current P/C endpoints on cdn.cboe.com return 403 [verified]. Train-on-history/serve-on-nothing mismatch. |
| **ICI weekly fund flows** | KILL | Host 301s to an HTML/XLS stats page; parsing fragile, ~1-week lag, weekly; cost > plausible signal. |
| **iShares holdings/flows** | KILL (for this packet) | Holdings CSV endpoint live [verified HTTP 200 text/csv] but **point-in-time only — zero history**, so no training or holdout read is possible. Forward-collection-only; out of scope. |
| **CryptoCompare BTC (risk-appetite proxy)** | KILL | API now requires a key: 401 "API key required" [verified] — violates no-signup. CoinGecko free tier caps history at 365d; insufficient. |
| **Stooq extra indices/futures/FX** | NOT PURSUED | All sandbox probes (incl. the documented-working `?s=spy.us&i=d` form) returned stooq's 404 page [verified] — almost certainly sandbox-egress/quota, since the production pipeline uses stooq daily; but S1 (CBOE) supersedes it for vol indices with deeper, cleaner history. No new stooq dependency. |
| **Alpha Vantage extras** | KILL as a family | 25 req/day free cap; redundant with existing price paths; nothing differentiated. Stays as the existing fallback only. |
| **Anything off-allowlist** (Reddit, SEC EDGAR full-text, Finnhub, NewsAPI, …) | NOT PROPOSED | New-host operator action required; per charter these carry lower priority and the shortlist above already covers positioning, vol-surface, macro, fiscal, and news-structure axes. None flagged as worth the ask right now. |

---

## 3. Historical-depth audit

Replay window = 2026-01-31 → present; holdout = **2026-03-11 → present (~62 trading days)**.

| Data family | Honest first-usable date | Covers replay window? | Covers holdout? | Training implication |
|---|---|---|---|---|
| Prices OHLCV (64 ETFs) | 2014-08-29 (per-symbol inceptions vary; XLC 2018) | YES | YES | Primary pretraining substrate, ~2,960 d |
| Engineered context (rates, vol, credit proxy) | 2014-12-10 | YES | YES | Pretraining-grade |
| GDELT tone (existing parquet) | 2015-02-18 | **NO — ends 2026-02-04** | **NO** | **Dead for evaluation until top-up backfill runs** |
| GDELT-rich G1–G5 (proposed) | 2015-02-18 (after backfill) | YES (after backfill) | YES (after backfill) | ~2,790 trading days — pretraining-grade for small models |
| LLM sentiment (over GDELT text) | backfill-bounded: replay window only (~90 d) | YES (bounded tokens) | YES (bounded tokens) | **Never a trained-on feature; zero/few-shot inference organ only** |
| CBOE vol-surface/COR (S1) | 1990 / 2006 / 2011 by index | YES | YES | Deepest family in the system; full pretraining |
| CFTC COT TFF (S2) | 2006-06-13, weekly | YES | YES (~13 obs) | Conditioning feature only; never per-day attribution alone |
| FRED deep catalog (S3) | 1967–2003 by series | YES | YES | Pretraining-grade macro context |
| Treasury auctions (S4) | 1979, episodic | YES | YES (sparse) | Dummy/z feature; no standalone model |
| Brain's own live record | 2025-08-04 (~210 d decision-grade) | partial | partial | Fine-tune/calibrate only — never primary training |

**Headlines:** (1) Every learnable component can train on a common panel from **2015-02-18** (GDELT-rich is the binding constraint; prices/CBOE/FRED extend deeper for price-only pretraining, e.g. a transformer pretrained 2014→ on prices+vol-surface, with GDELT-rich joining from 2015-02). (2) **GDELT is currently absent from the entire holdout** — the 2026-02-05→present top-up backfill is a prerequisite for evaluating assignment item 4 at all, and it is cheap (<1 h). (3) Nothing LLM-derived has history: the LLM must be an inference-time organ.

---

## 4. Storage / cost line (itemized)

| Item | One-time | Monthly |
|---|---|---|
| GKG backfill transfer (~80–110 GB) + events (~2 GB) | $0 (local Mac bandwidth/time, 10–20 h restartable) | — |
| `gdelt_rich.parquet` in S3 (~10 MB) + cboe/cot/fred/treasury parquets (~5 MB total) | — | ~$0.0004 (15 MB × $0.023/GB) |
| Daily incremental artifacts (~50–100 KB/day × 30 d, before 365-d lifecycle deletion) | — | ~$0.002 cumulative-steady-state |
| Nightly fetch compute: +8 GDELT zips (~20 MB) + ~10 small CSV/API calls, est. +20–40 s Lambda at 3008 MB | — | ~40 s × 22 d × 2.94 GB ≈ 2,600 GB-s ≈ **$0.04** |
| S3 PUT/GET requests (~20/day) | — | ~$0.003 |
| **Total new** | **$0 cash** | **≈ $0.05/month** |

No new hosts, no keys beyond those already in Secrets Manager, no paid signups, lifecycle rules unchanged, S3 versioning stays off.

## 5. Falsifiers (pre-committed; leave-one-out on the holdout per EVIDENCE_PROTOCOL)

- **GDELT-rich (G1–G5):** ablate the whole block → if holdout paired daily-difference t-stat of the full brain vs ablated brain is |t| < 1, GDELT-rich is dead; report per packet item 4 as evidence-backed no-signal (and distinguish "no signal" from "holdout too short" honestly — 62 days is wide-error-bar territory). Sub-falsifier: if G1 sector z-vectors show zero feature-importance in the meta-evaluator across all retrains, kill the theme dictionary specifically before killing GDELT wholesale.
- **S1 CBOE:** ablate COR3M + curvature columns (keep the legacy slope) → no holdout degradation ⇒ the family added nothing over the existing vol columns; drop and say so.
- **S2 COT:** shuffle-publication-date placebo: if real COT does not beat a 1-week-shifted COT on validation folds, it is forward-fill artifact, kill. Holdout has only ~13 obs — verdict must lean on the full-period E1 read plus the placebo, stated openly.
- **S3 FRED extras:** ablation vs the *existing* context columns (not vs nothing) → if ΔSharpe ≤ 0 on validation and holdout, the deep catalog duplicated the proxies; keep only the proxies.
- **S4 Treasury:** if auction-day features fire < 10 times in the holdout (likely), no holdout verdict is possible — grade it E1-only and either carry it explicitly as unproven or cut it; never claim holdout support that arithmetic can't deliver.

— end —
