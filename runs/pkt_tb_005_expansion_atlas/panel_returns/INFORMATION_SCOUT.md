# Information Scout — panel return (verbatim subagent output)

Probes run 2026-06-09/10 from repo root. Probe-environment notes embedded in candidate blocks where load-bearing. All FRED probes used the system's existing key (`investment-system/fred-key`, Secrets Manager) against `api.stlouisfed.org` — already on allowlist, already free.

### [D-01] Equity vol term structure (VIX9D / VIX / VIX3M ratios)
- hypothesis: Term-structure slope (9d/30d, 30d/3m) distinguishes "event premium" spikes from persistent stress regimes far better than VIX level or SKEW percentile alone; backwardation onset is a classic daily-cadence risk-off trigger.
- mechanism: Vol curve inversion reflects dealers pricing imminent realized vol; it mean-reverts over days-weeks, exactly the system's cadence, and the data is free index closes.
- source: `https://query1.finance.yahoo.com/v8/finance/chart/^VIX9D` and `^VIX3M` (`range=10y&interval=1d`, browser UA); cost free; host ON allowlist; probe: TESTED-OK (both 2513 daily bars 2016-06→2026-06-09, OHLC fields; `range=max` is broken for ^VIX3M — returns 1 bar — use explicit range/period1).
- history depth: 10y daily verified via chart API; ^VIX9D exists from 2011, ^VIX3M from 2006 (use period1 epoch for deeper pulls).
- integration: new expert signal (or direct upgrade of the existing vol-complexity expert from percentile-of-levels to curve-shape).
- lambda/training fit: 2 extra Yahoo calls/night, ~2 floats/day stored alongside ^VVIX/^SKEW; trivial.
- evidence path to E2: backfill 10y, add slope features to vol-complexity expert, replay 2025-08→2026-06 with holdout 2026-03-11+; kill if regime-flip timing unchanged.
- suggested verdict: pilot-now

### [D-02] MOVE index (treasury implied vol)
- hypothesis: The system allocates heavily across treasuries/credit but has zero bond-vol input; MOVE spikes lead duration drawdowns and credit-spread widening at daily cadence.
- mechanism: Equity VIX misses rates-driven stress (2022-style regimes); MOVE is the direct free observable for that orthogonal risk axis.
- source: `https://query1.finance.yahoo.com/v8/finance/chart/^MOVE?range=10y&interval=1d` (browser UA); cost free; host ON allowlist; probe: TESTED-OK (2513 daily bars 2016-06→2026-06-09, OHLC; symbol exists from 2002 via period1).
- history depth: 2002→present daily (10y verified at daily granularity).
- integration: new expert signal input (macro/credit expert gets a vol dimension) + sizing input for treasury sleeve.
- lambda/training fit: 1 call/night, 1 float/day.
- evidence path to E2: add MOVE percentile + 5d change to macro/credit expert; replay; kill if treasury-sleeve sizing decisions unchanged on holdout.
- suggested verdict: pilot-now

### [D-03] Asset-class vol surfaces: OVX (oil), GVZ (gold), VXN (Nasdaq)
- hypothesis: Per-sleeve implied vol (gold vol for GLD/SLV sizing, oil vol for energy/commodity sleeve, VXN for tech-factor tilt) lets sizing be risk-aware per asset class instead of SPY-vol-for-everything.
- mechanism: Same structural edge as D-02 applied per sleeve; honest caveat — incremental over realized vol the system can already compute, so the edge may be small.
- source: Yahoo chart `^OVX`, `^GVZ` (TESTED-OK, 2513 daily bars each, 10y, OHLC); FRED `VXNCLS` probe: TESTED-OK (6613 obs, daily since 2001, latest 2026-06-08); cost free; hosts ON allowlist.
- history depth: OVX 2007+, GVZ 2008+, VXN 2001+ daily.
- integration: candidate-scoring feature + sizing input per sleeve.
- lambda/training fit: 2–3 calls/night, 3 floats/day.
- evidence path to E2: bolt onto layered sizing as per-sleeve vol scalar; replay holdout; kill if Sharpe/drawdown delta indistinguishable from realized-vol-only control.
- suggested verdict: pilot-now (behind D-01/D-02; ship together as one "vol surface" ingest)

### [D-04] Real credit spreads: ICE BofA HY & IG OAS (replace HYG/IEF price proxy)
- hypothesis: The credit expert currently proxies spreads with an ETF price ratio that confounds duration moves with credit moves; actual option-adjusted spreads are the clean signal.
- mechanism: OAS is the thing the proxy is trying to estimate; using the real series removes duration contamination for free.
- source: FRED `BAMLH0A0HYM2` (HY OAS) + `BAMLC0A0CM` (IG OAS) via existing keyed API; cost free; host ON allowlist; probe: TESTED-OK (HY latest 2026-06-08 = 2.75; IG = 0.75; **but API count=793 obs ≈ only ~3.1 years served** — FRED appears to have truncated ICE BofA history; verify before training on it).
- history depth: ~mid-2023→present daily as served today (covers full replay window 2025-08→2026-06; NOT enough for decade-scale model training — proxy ratio still needed for deep backfill).
- integration: new expert signal input (direct upgrade of macro/credit expert).
- lambda/training fit: 2 extra FRED series in the existing FRED step; near-zero.
- evidence path to E2: swap proxy→OAS in credit expert, replay 235 stored dates, compare fusion decisions on holdout; kill if decision diff is zero.
- suggested verdict: pilot-now

### [D-05] Real yields & inflation breakevens (DFII10, T10YIE)
- hypothesis: Gold/silver and TLT allocation is driven by real rates and inflation expectations, neither of which the system observes; nominal DGS10 conflates the two.
- mechanism: Real-rate direction is the textbook daily driver of gold; the decomposition is free and already inside the existing FRED pipeline.
- source: FRED `DFII10`, `T10YIE` via existing key; cost free; ON allowlist; probe: TESTED-OK (DFII10 6113 obs, T10YIE 6114 obs, both daily since 2003, latest 2026-06-08/09).
- history depth: 2003→present daily.
- integration: regime-fusion gate context + candidate-scoring feature for metals/duration sleeve.
- lambda/training fit: 2 series added to FRED step; near-zero.
- evidence path to E2: add real-yield 21d change to gold/TLT candidate scoring; replay holdout; kill if metals-sleeve picks unchanged.
- suggested verdict: pilot-now

### [D-06] Fed net-liquidity composite (WALCL − RRP − TGA)
- hypothesis: Slow-moving liquidity tide (balance sheet minus reverse-repo minus Treasury cash) gates risk-on regimes at weekly cadence; popular as a "don't fight the drain" filter.
- mechanism: Mechanical reserve arithmetic does move risk assets at multi-week horizon, but the honest sentence is: it's a crowded narrative with weak out-of-sample evidence at daily granularity.
- source: FRED `WALCL` (TESTED-OK, weekly, 1225 obs since 2002) + `RRPONTSYD` (TESTED-OK, daily, 6088 obs, latest 2026-06-09) — both free, ON allowlist; TGA daily via Treasury DTS `api.fiscaldata.treasury.gov` probe: TESTED-OK (record_date 2026-06-08, opening balance fields, no key) but host NOT on allowlist → that leg needs operator sign-off.
- history depth: WALCL 2002+, RRP 2003+ (nonzero from 2013+), TGA DTS 2005+.
- integration: regime-fusion gate (slow risk-on/off tilt).
- lambda/training fit: 2 FRED series now, 1 parked host later; weekly-change features, tiny.
- evidence path to E2: two-leg version (WALCL−RRP) pilotable immediately; replay holdout as a fusion tilt; revive TGA leg only if two-leg shows signal.
- suggested verdict: parked-promising (revival: two-leg pilot shows holdout delta, or operator signs off api.fiscaldata.treasury.gov)

### [D-07] Broad trade-weighted dollar (DTWEXBGS) replacing DEXUSEU
- hypothesis: International ETF and commodity sleeve performance keys off the broad dollar, not the single EUR cross currently ingested.
- mechanism: Strict upgrade of an existing input — same pipeline, strictly more representative index, free.
- source: FRED `DTWEXBGS` via existing key; cost free; ON allowlist; probe: TESTED-OK (5330 obs, daily since 2006, latest 2026-06-05 — note ~2-business-day publication lag).
- history depth: 2006→present daily (lagged 2 days).
- integration: candidate-scoring feature for EFA/EEM/commodity sleeves; context feature.
- lambda/training fit: 1 series swap/add in FRED step.
- evidence path to E2: add broad-dollar 21d momentum to international sleeve scoring; replay holdout.
- suggested verdict: pilot-now (lowest-effort entry on this list)

### [D-08] Weekly financial-conditions indices (NFCI, STLFSI4)
- hypothesis: A vetted composite of 100+ stress indicators as a slow regime prior, cross-checking the home-built fragility expert.
- mechanism: Fed-curated breadth the system can't replicate for free; honest caveat — weekly + revised + lagged, so it can only be a slow gate, never a trigger.
- source: FRED `NFCI` (2891 weekly obs since 1971), `STLFSI4` (1692 weekly obs) via existing key; cost free; ON allowlist; probe: TESTED-OK (latest 2026-05-29 for both).
- history depth: NFCI 1971+, STLFSI 1993+ weekly.
- integration: regime-fusion gate (prior/confidence scaler on fragility expert).
- lambda/training fit: 2 series, weekly; trivial.
- evidence path to E2: use NFCI z-score as fragility-expert confidence multiplier; replay holdout; kill if fusion outputs identical.
- suggested verdict: pilot-now (bundled with other FRED adds)

### [D-09] Daily Economic Policy Uncertainty index (USEPUINDXD)
- hypothesis: News-based policy-uncertainty level adds event-risk context that GDELT tone aggregates (which measure sentiment, not uncertainty) don't carry.
- mechanism: EPU is constructed from newspaper term counts and updates daily with 40 years of history — a free, pre-built "richer news structure" series; honest caveat — noisy day-to-day, use smoothed.
- source: FRED `USEPUINDXD` via existing key; cost free; ON allowlist; probe: TESTED-OK (15134 obs, daily since 1985, latest 2026-06-08 = 362.06).
- history depth: 1985→present daily — deepest history of any candidate here; full training-scale.
- integration: candidate-scoring feature + input to entropy-shift/news expert.
- lambda/training fit: 1 FRED series; trivial.
- evidence path to E2: add 21d EPU percentile to context features, retrain-lite or replay-with-feature; holdout comparison.
- suggested verdict: pilot-now

### [D-10] GDELT: fix the broken fetch, then add theme/event structure
- hypothesis: Event-class counts (PROTEST, MILITARY, ECON_BANKRUPTCY themes; CAMEO conflict event rates) carry regime information beyond average tone.
- mechanism: Before any "richer use": **the production fetch path appears broken** — `src/steps/ingest_gdelt.py` requests `data.gdeltproject.org/gdeltv2/{YYYYMMDD}.gkgcounts.csv.zip` which probe-returns **404** (daily gkgcounts files live under GKG v1.0: `/gkg/{YYYYMMDD}.gkgcounts.csv.zip`), so gdelt features are likely riding their `gdelt_available=False` defaults in production; fixing the path is free signal recovery.
- source: `http://data.gdeltproject.org/gkg/20260608.gkgcounts.csv.zip` probe: TESTED-OK (200, 2.7MB; 2015 file also 200, 2.2MB); v2 15-min GKG `20260609120000.gkg.csv.zip` TESTED-OK (200, 5.4MB per 15-min — too heavy for thin Lambda full-day aggregation, 96 files/day); cost free; host ON allowlist.
- history depth: v1 daily gkgcounts 2013-04→present; v2 15-min 2015-02→present.
- integration: new expert signal (event-risk) + repair of existing context features.
- lambda/training fit: v1 daily file = 1 fetch + 1 parse/night, fine; full v2 GKG aggregation belongs in monthly local training, not Lambda.
- evidence path to E2: step 1 fix path and confirm `gdelt_available=True` flips in nightly run; step 2 add 3–5 theme-count features, replay holdout.
- suggested verdict: pilot-now (the path fix is a bug repair, not even an expansion)

### [D-11] CFTC Commitments of Traders positioning (weekly)
- hypothesis: Non-commercial net positioning extremes in VIX, gold, silver, crude, 10Y futures are contrarian/confirming inputs for the corresponding ETF sleeves at weekly cadence.
- mechanism: Crowded-positioning unwind risk is one of the few free positioning datasets in existence; honest caveat — weekly with 3-day lag (Friday release of Tuesday data), so it's a slow tilt only.
- source: Socrata API `https://publicreporting.cftc.gov/resource/6dca-aqww.json` (legacy futures-only; disaggregated = `72hh-3qpy`), no key, `$select`/`$order` filtering; probe: TESTED-OK (latest report_date 2026-06-02, long/short fields by contract market) — **probe required sandbox-disable; host NOT on production allowlist → operator sign-off needed**; cost free.
- history depth: API serves 1986→present weekly.
- integration: candidate-scoring feature for gold/silver/oil/treasury/VIXY sleeves.
- lambda/training fit: 1 JSON call/week, ~10 floats/week; trivial once host approved.
- evidence path to E2: backfill 2024→present, add net-positioning percentile to metals/energy scoring, replay holdout.
- suggested verdict: parked-promising (revival condition: operator allowlists `publicreporting.cftc.gov`; data verified working and free)

### [D-12] VIX futures term structure / roll yield (CBOE settlement files)
- hypothesis: VX front/second-month contango-backwardation is the direct driver of VIXY P&L (roll drag) and a sharper risk-regime gauge than spot VIX; the system trades VIXY blind to it.
- mechanism: VIXY's expected daily bleed is literally computable from the curve — this is mechanics, not alpha speculation; the catch is purely the host.
- source: `https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{expiry-date}.csv` probe: TESTED-OK (per-contract daily Trade Date/Settle/Volume/OI rows, e.g. Jun-2026 contract history from 2025-09); the delayed-quotes term-structure JSON endpoint returned 403 (use per-contract CSVs + expiry calendar instead); **host `cdn.cboe.com` NOT on allowlist → sign-off needed**; cost free.
- history depth: CBOE historical VX CSVs cover ~2013→present per contract.
- integration: new expert signal (vol-curve) + direct VIXY sizing input.
- lambda/training fit: ~8 small CSV fetches/night (active contracts) + expiry-calendar logic; modest but real parsing code.
- evidence path to E2: reconstruct front/second continuous series, gate VIXY entries on backwardation, replay holdout where VIXY trades occurred.
- suggested verdict: parked-promising (revival: operator allowlists `cdn.cboe.com`; meanwhile D-01's VIX9D/VIX3M slope is the on-allowlist approximation)

### [D-13] Equity put/call ratio
- hypothesis: Aggregate put/call extremes as daily contrarian sentiment.
- mechanism: Honest sentence: the free historical sources have rotted — Yahoo `^CPC` is dead and CBOE's archive moved behind shifting endpoints — so the acquisition cost exceeds the marginal signal over D-01's curve features.
- source: Yahoo `^CPC` probe: TESTED-FAIL ("No data found, symbol may be delisted"); CBOE delayed-quotes JSON: 403 AccessDenied; cost free-in-theory; allowlist: Yahoo on, CBOE off.
- history depth: n/a (no working free source found).
- integration: would have been candidate-scoring sentiment feature.
- lambda/training fit: n/a.
- evidence path to E2: none without a source.
- suggested verdict: killed (no reliable free historical source; SKEW+VVIX+term-structure cover the options-sentiment axis)

### [D-14] Internal market breadth from the existing 65-symbol universe
- hypothesis: % of universe above 50d/200d MA, count at 21d/63d highs-lows, and cross-sectional return dispersion are classic breadth/regime confirmations — computable from prices already stored, zero new ingestion.
- mechanism: Breadth divergence (index up, breadth down) is a well-documented multi-day regime-top pattern, and here it is literally free — the parquets already exist back to 2014.
- source: existing S3/local price parquets; cost zero; allowlist n/a; probe: NOT-PROBED (no fetch — data already in `prices` parquets and local context parquets 2014-12→2026-02).
- history depth: 2014→present from local context parquets; 235 replay dates ready.
- integration: regime-fusion gate (breadth confirmation) + fragility-expert sibling.
- lambda/training fit: pure pandas on data already loaded each night; zero network.
- evidence path to E2: compute breadth series over stored parquets TODAY, correlate with regime labels, then replay holdout with breadth as fusion gate — the cheapest full E2 on this list.
- suggested verdict: pilot-now

### [D-15] Overnight foreign-index closes for international sleeve (lead-lag)
- hypothesis: Nikkei/Stoxx/FTSE/HSI same-day closes are known hours before US open and inform EFA/EEM/per-country ETF scoring beyond what the US-listed ETF's own (stale) close encodes.
- mechanism: Honest sentence: most of the overnight information is already in the ETF's opening price by the time morning execution fills, so the residual daily-cadence edge is probably thin — the cleaner use is as regime context (global risk breadth), not per-trade alpha.
- source: Yahoo chart `^N225`, `^STOXX50E`, `^FTSE`, `^HSI`, `000001.SS`; cost free; ON allowlist; probe: TESTED-OK (all 5: ~2425–2526 daily bars, 2016-06→2026-06-09). Note: Stooq could not be probed as alternate — it served a JS proof-of-work anti-bot challenge to curl and 404'd python-requests from this network during testing (consistent with its existing demotion to fallback in `ingest_prices.py`).
- history depth: 10y+ daily via Yahoo.
- integration: candidate-scoring feature for international sleeve + global-breadth context.
- lambda/training fit: 5 Yahoo calls/night; small.
- evidence path to E2: add foreign-momentum features to intl-sleeve scoring; replay holdout; kill if intl picks unchanged.
- suggested verdict: parked-promising (revival: D-14 breadth pilot succeeds, making global-breadth the natural extension)

### [D-16] Crypto as risk-appetite / weekend-information bridge
- hypothesis: BTC trades through the weekend, so Monday-morning decisions could read 48h of risk-appetite the equity feeds can't see; BTC drawdown/vol also correlates with speculative-factor regimes.
- mechanism: Honest sentence: BTC-equity correlation is regime-dependent and the weekend-gap edge for a daily ETF allocator is plausible but unproven — this is the most speculative live candidate here.
- source: CoinGecko `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily` probe: TESTED-OK (daily prices array; free tier caps history at 365d); Binance probe: TESTED-FAIL (US geo-block: "Service unavailable from a restricted location"); CryptoCompare probe: TESTED-FAIL (now requires API key post-Coindesk); Kraken `OHLC?interval=1440` probe: TESTED-OK but only ~720 daily candles served; all hosts ON allowlist.
- history depth: effective free daily depth ~1–2y (CoinGecko 365d + Kraken 720d) — fine for replay window, thin for training.
- integration: regime-fusion context feature (weekend risk gauge for Monday runs).
- lambda/training fit: 1 call on Sunday-night/Monday runs; trivial.
- evidence path to E2: replay Mondays only in holdout with BTC-weekend-return feature; kill if Monday decisions unchanged.
- suggested verdict: parked-promising (revival: Monday-only replay slice shows any signal; cheap to test but low prior)

### [D-17] Seasonality / calendar-event features (zero-fetch)
- hypothesis: Turn-of-month flows, FOMC-day/pre-FOMC drift, OpEx week, and month-end rebalancing are documented daily-cadence effects the model currently cannot represent (it has no calendar awareness).
- mechanism: These are among the most replicated calendar anomalies in the literature, and the cost is a static table — FOMC dates are published years ahead, so a hardcoded list needs updating once a year.
- source: static config table (FOMC schedule, OpEx = 3rd Friday, month-end), no network; cost zero; allowlist n/a; probe: NOT-PROBED (nothing to probe).
- history depth: unlimited (deterministic backfill to any date).
- integration: candidate-scoring feature + LLM-risk-veto context ("tomorrow is FOMC").
- lambda/training fit: zero network, ~5 boolean/int features.
- evidence path to E2: add calendar flags to stored feature parquets retroactively (deterministic), replay holdout; also check whether pre-FOMC days explain any existing replay losses.
- suggested verdict: pilot-now

### [D-18] Macro nowcasts (GDPNow via FRED)
- hypothesis: Growth-nowcast direction shifts the equity-vs-duration regime prior between official data releases.
- mechanism: Honest sentence: FRED's GDPNOW series stores one value per target quarter (count=60) and overwrites vintages, so the daily-revision signal that would actually carry information is not what FRED serves — the usable content at daily cadence is near zero.
- source: FRED `GDPNOW` via existing key; probe: TESTED-OK (count=60 quarterly obs, latest 2026-04-01 = 3.29); Atlanta Fed's own vintage file lives off-allowlist; cost free.
- history depth: 2011→present, quarterly granularity as served.
- integration: would be regime-fusion prior.
- lambda/training fit: trivial but pointless at served granularity.
- evidence path to E2: none worth running at quarterly granularity.
- suggested verdict: killed (FRED-served granularity destroys the nowcast's value; revisit only if atlantafed.org vintage CSV gets allowlisted AND a growth-regime gap is demonstrated)

### [D-19] ETF fund-flow proxy via shares outstanding
- hypothesis: Daily SO changes = creation/redemption flow, a real positioning signal per ETF.
- mechanism: Honest sentence: the only free path (Yahoo quoteSummary) is crumb-gated and serves point-in-time, not historical, so building a flow history means months of fragile daily scraping before the first backtest is even possible.
- source: `query2.finance.yahoo.com/v10/finance/quoteSummary/SPY?modules=defaultKeyStatistics` probe: TESTED-FAIL (401 "Invalid Crumb"; the cookie+crumb dance via fc.yahoo.com that yfinance does could work but is the system's most breakage-prone dependency already); ICI weekly flows: host unreachable from sandbox and NOT on allowlist; cost free-in-theory.
- history depth: none retrievable historically — accumulate-forward only.
- integration: would be candidate-scoring feature.
- lambda/training fit: fragile scraping, no backfill, slow evidence accumulation.
- evidence path to E2: ≥6 months of accumulation before any test — fails the replay-harness test cheaply criterion.
- suggested verdict: killed (no backfillable free source; evidence loop too slow)

### [D-20] Short interest / squeeze indicators (FINRA)
- hypothesis: Crowded shorts in sector ETFs flag squeeze-driven rallies.
- mechanism: Honest sentence: bi-monthly publication with a multi-day lag on a 65-ETF (not single-stock) universe makes this nearly information-free at daily cadence.
- source: FINRA equity short-interest files (finra.org; NOT on allowlist); probe: NOT-PROBED (cadence kills it regardless of reachability); cost free.
- history depth: years available, bi-monthly.
- integration: n/a.
- lambda/training fit: n/a.
- evidence path to E2: not worth a pilot.
- suggested verdict: killed (cadence/lag mismatch; ETF-level short interest is mostly arbitrage plumbing, not sentiment)

### [D-21] Full yield-curve shape features (DGS5, DGS30 → level/slope/curvature)
- hypothesis: The macro/credit expert sees only DGS10−DGS3MO; adding 5y and 30y enables curvature/butterfly and long-end steepening features that differentiate "bull steepener" (recession-onset) from "bear steepener" (term-premium/supply) regimes — which demand opposite treasury-sleeve positioning.
- mechanism: The 2s10s vs 5s30s distinction is standard rates-desk regime reading, free, and slots into the existing FRED step unchanged.
- source: FRED `DGS5` (16810 obs, 1962+), `DGS30` (12865 obs, 1977+) via existing key; probe: TESTED-OK (latest 2026-06-08: DGS5 4.29, DGS30 5.03); cost free; ON allowlist.
- history depth: 1962+/1977+ daily — full training scale.
- integration: new features in macro/credit expert + treasury-sleeve scoring.
- lambda/training fit: 2 series; trivial.
- evidence path to E2: add slope/curvature deltas, replay holdout, inspect treasury-sleeve decision diffs.
- suggested verdict: pilot-now (bundle with D-04/D-05 as one "FRED expansion" PR)

### [D-22] Funding-stress monitor (SOFR-based spreads)
- hypothesis: Money-market stress (SOFR spikes vs policy rate) precedes broader risk-off and is invisible to all current inputs.
- mechanism: Honest sentence: true funding blowups are rare enough that this feature would be flat for years then matter once — high tail value, near-zero replay-window testability.
- source: FRED `SOFR` probe: TESTED-OK (2135 obs, daily since 2018, latest 2026-06-08 = 3.63; pair with `IORB` for spread); `TEDRATE` confirmed discontinued 2022-01; cost free; ON allowlist.
- history depth: 2018→present daily.
- integration: LLM-risk-veto context + emergency fusion gate.
- lambda/training fit: 1–2 series; trivial.
- evidence path to E2: replay window contains no funding event, so E2 cannot validate it — that is the kill argument despite zero cost.
- suggested verdict: parked-promising (revival: bundle opportunistically with another FRED PR as an untested safety tripwire, explicitly flagged unvalidatable)

### [D-23] GDELT event-level structure for geo/sector shock typing (beyond D-10's counts)
- hypothesis: Typed events (CAMEO codes: sanctions, armed conflict, central-bank statements) with actor-country mapping could feed sector/country ETF scoring — e.g., conflict-event spikes → energy/gold tilt.
- mechanism: Honest sentence: event-to-price mapping is the graveyard of news-trading projects; per-15-min v2 files are 1–5MB each (96/day), so doing this properly busts the thin-Lambda constraint and the daily v1 counts in D-10 capture most of the recoverable signal.
- source: `data.gdeltproject.org/gdeltv2/*.export.CSV.zip` probe: TESTED-OK (lastupdate.txt lists current export/mentions/gkg files); cost free; ON allowlist.
- history depth: 2015→present (v2), 1979→2013 historical archive.
- integration: would be new expert signal.
- lambda/training fit: fails thin-Lambda at full granularity; training-side-only aggregation possible monthly.
- evidence path to E2: only after D-10's daily counts show signal; build typed aggregates offline and replay.
- suggested verdict: parked-promising (revival condition: D-10 theme counts show holdout signal)

### [D-24] Trend/attention data (Google Trends, Wikipedia pageviews)
- hypothesis: Retail attention spikes ("recession", "stock market crash" searches) as sentiment extremes.
- mechanism: Honest sentence: hosts are off-allowlist, Google Trends is aggressively rate-limited/normalized (values are relative, re-scaled per request — unstable for replay), and the literature edge decayed post-2015.
- source: trends.google.com / wikimedia.org REST; NOT on allowlist; probe: NOT-PROBED (allowlist + known instability); cost free.
- history depth: 2004+/2015+ but non-stationary scaling.
- integration: n/a.
- lambda/training fit: poor (rate limits vs Lambda retries).
- evidence path to E2: not worth sign-off request.
- suggested verdict: killed (unstable scaling makes replay non-reproducible; D-09 EPU covers the attention axis with a stable daily series)

## SCOUT'S TOP PICKS
1. **D-10 GDELT path fix** — production fetch 404s today (v2 daily gkgcounts path doesn't exist; v1 path TESTED-OK); this is recovering an input the system already believes it has, before any expansion.
2. **D-01+D-02 vol term structure + MOVE (one Yahoo ingest PR)** — both TESTED-OK with 10y daily history on an allowlisted host; directly upgrades the vol-complexity expert and gives the treasury-heavy book its missing bond-vol axis.
3. **D-14 internal breadth** — zero new ingestion, computable on stored parquets back to 2014, full E2 runnable on the replay harness this week; cheapest possible evidence-per-effort on the list.
Honorable mention: the FRED bundle (D-04/D-05/D-07/D-21) is one small PR on an existing keyed pipeline — all probes TESTED-OK — and should ride along with whichever pick ships first.
