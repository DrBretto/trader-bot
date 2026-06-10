# Feasibility Auditor — panel return (verbatim subagent output)

## ENVELOPE GRADES

Graded: every `pilot-now` candidate (12 D, 10 M, 7 S = 29), plus envelope spot-checks on parked candidates (see OVERRIDES — none flipped to pilot, one re-laned). Grounding read: `src/handler.py` (12-step night phase; vol indices step 3 already Stooq-fragile, prices Yahoo-primary since 2026-05-12 with Alpha Vantage/yfinance fallbacks), `src/steps/ingest_prices.py`, `src/steps/ingest_gdelt.py`, `config/data_sources.json` (6 FRED series today), `config/optimizer.committee_20260606.json` (`max_runtime_minutes: 25`), `automation/run_training.sh` + monthly launchd plist, `requirements-lambda.txt` (thin: boto3/requests/openai/yfinance; pandas via layer).

### Information Scout

**D-01 — FITS** — 2 Yahoo chart calls/night on the already-primary host; same fragility class as existing price ingest, must ship with graceful-default-on-fetch-fail (gdelt_available pattern). — Pilot: ~4-6h (backfill script + feature add + replay), local compute minutes. — Ship: +2 nightly HTTP calls (~2-4s), +2 floats/day in signals.parquet, $0, maintenance = Yahoo chart-endpoint drift (known, shared with existing ingest; note Scout's `range=max` bug for ^VIX3M must be encoded as explicit period1).

**D-02 — FITS** — 1 Yahoo call, 1 float/day. — Pilot: ~2h riding D-01's harness. — Ship: negligible; same Yahoo fragility pool. Bundle with D-01 as one ingest module.

**D-03 — FITS-WITH-CARE** — care: mixed-source (2 Yahoo + 1 FRED) for one feature family; ship inside the D-01/D-02 module and the FRED bundle respectively, NOT as a third ingest path. Marginal-signal honesty already flagged by Scout. — Pilot: ~3h marginal after D-01. — Ship: +2-3 calls/night, 3 floats/day, $0.

**D-04 — FITS-WITH-CARE** — care: FRED serves only ~3.1y of ICE BofA OAS — fine for the 235-date replay, NOT for decade-scale training; keep the HYG/IEF proxy for deep backfill and document the dual-source split or training silently trains on the proxy while inference reads OAS (train/serve skew). — Pilot: ~3-4h. — Ship: +2 series in existing FRED step, ~0s, $0, low fragility (keyed FRED is the system's most reliable source).

**D-05 — FITS** — 2 FRED series, existing pipeline, 2003+ history (full training scale). — Pilot: ~2-3h. — Ship: near-zero; bundle in the FRED PR.

**D-07 — FITS-WITH-CARE** — care: 2-business-day publication lag (Scout verified latest = 2026-06-05 on a 06-09 probe); feature must be lag-aware in replay or it leaks. — Pilot: ~2h. — Ship: 1 series, near-zero.

**D-08 — FITS-WITH-CARE** — care: weekly + revised series; replay must pin as-of values, and FRED serves current-vintage only — a revised NFCI is a small but real replay-determinism leak; freeze the backfill parquet once and never re-pull history. — Pilot: ~2-3h. — Ship: 2 weekly series, near-zero.

**D-09 — FITS** — 1 FRED series, daily since 1985, deepest history on the list. — Pilot: ~2h. — Ship: near-zero.

**D-10 — FITS (bug-fix lane, not expansion)** — verified independently, see GDELT section below. — Pilot: ~1-2h for the path fix + 1 nightly run to confirm `gdelt_available=True`; theme-count expansion is a separate ~4-6h pilot. — Ship (fix): zero delta — same 1 fetch/night, ~2-3MB zip, well inside runtime. Ship (theme expansion): still 1 file/night; v2 15-min aggregation stays banned from Lambda per Scout — agreed.

**D-14 — FITS (cleanest envelope on the board)** — zero network, computes on parquets already loaded nightly; pure pandas on existing layer. — Pilot: ~4-6h, all local, full E2 runnable immediately. — Ship: +O(seconds) nightly compute, 0 new failure surface, $0, zero source fragility.

**D-17 — FITS-WITH-CARE** — care: hardcoded FOMC table is a once-a-year manual maintenance item; put it in config with an expiry assertion (fail loudly when the table runs out of future dates rather than silently emitting `fomc=False` forever). — Pilot: ~3-4h (retroactive deterministic backfill is the cheap part). — Ship: zero network, ~5 features, $0.

**D-21 — FITS** — 2 FRED series, 1962+/1977+ history. — Pilot: ~2-3h. — Ship: near-zero; bundle in the FRED PR.

### Model Architect

**M-01 — FITS** — replay-only, zero new data; if rules win, the Lambda gets *simpler* (torch regime path exits). — Pilot: ~6-8h (the `regime_source` harness extension is the cost; it's shared by M-02/M-03/M-12/M-13), compute = 235-date replay × variants × 5 seeds on the MacBook, well under an hour. — Ship: negative cost potential (smaller image, fewer model loads).

**M-02 — FITS** — same harness, +1 hyperparameter. — Pilot: ~1h marginal. — Ship: a few lines of EMA state; no torch.

**M-03 — FITS-WITH-CARE** — care: `hmmlearn` is a new dependency — training-side requirements only; production inference must ship as the pure-numpy forward step the Architect describes, with fitted params exported to a small artifact. Do NOT let hmmlearn into `requirements-lambda.txt` / the container. — Pilot: ~4-6h after M-01 harness; training = seconds. — Ship: numpy-only inference, tiny artifact in S3, monthly refit = seconds inside the existing launchd window.

**M-05 — FITS** — static-weight sweep is zero new code (`ensemble_overrides` exists); adaptive variant ~20 lines. — Pilot: ~2-4h. — Ship: a rolling scalar; or, on the expected flat result, *minus* one model from the Lambda load.

**M-06 — FITS** — temperature scalar fit offline on stored probs; apply-time is arithmetic in the handler. — Pilot: ~3-4h. — Ship: one config scalar, $0.

**M-09 — FITS** — first version is arithmetic derived from existing labeler inputs; replay-only. — Pilot: ~6-8h (the dial→multiplier map needs care to match regime centroids). — Ship: arithmetic; learned-dial version later re-enters as needs-retraining and must be re-audited then.

**M-12 — FITS-WITH-CARE** — care: `lightgbm` is training-side only; production inference must be exported trees (pure python/numpy) — same rule as M-03: no new heavy deps in the Lambda image. Walk-forward monthly refit fits the launchd window (minutes). — Pilot: ~6-8h, local minutes. — Ship: small tree artifact in S3; image unchanged if export discipline holds.

**M-13 — FITS** — replay-only twin of M-01, shares the harness; rules already in `src/models/baseline_health.py`. Same possible-simplification upside (AE exits the image). — Pilot: ~2-3h marginal. — Ship: zero-to-negative.

**M-14 — FITS (cheapest on the M list)** — replay wiring already exists (`optimizer/replay.py:307-309`); config-only sweep. — Pilot: ~2-3h including rank-IC sanity. — Ship: zero — model already budgeted and deployed inert.

**M-16 — FITS** — offline correlation study + a supported config override. — Pilot: ~2-3h. — Ship: zero, possible knob deletion (simplification).

### Strategist

**S-01 — FITS** — ~40 lines engine logic + replay on stored artifacts; SHY data already stored. — Pilot: ~4-6h. — Ship: zero new data/infra; +2 params into optimizer space (see aggregate note on the 25-min optimizer window).

**S-02 — FITS** — sleeve logic, all data stored. — Pilot: ~6-8h (core carve-out touches holdings count, cluster caps, sell triggers — more surface than S-01). — Ship: zero new data; param bundle grows.

**S-03 — FITS** — engine logic ported from an existing overlay; replay-now. — Pilot: ~4-6h. — Ship: zero new data; actually *reduces* maintenance surface (one decision path instead of engine+overlay).

**S-05 — FITS (cheapest on the S list)** — one universe-config cell, two replay runs. — Pilot: ~1-2h. — Ship: zero; arguably negative (removes a toxic state).

**S-08 — FITS-WITH-CARE** — care: trailing-60d correlation matrix computed in the nightly engine — fine on the existing pandas layer (65×65 on 60 rows is trivial), but the cluster-map-ON control arm MUST run first exactly as the Strategist says, or this ships complexity the existing knob already covers. — Pilot: ~6-8h. — Ship: +O(seconds) nightly compute, +2 params.

**S-09 — FITS** — buy-gate flag, replay-now; reduces turnover (transaction-cost savings are real modeled dollars). — Pilot: ~3-4h. — Ship: zero new data, +1 param.

**S-10 — FITS** — universe config edit + one pre-check query against stored fills. — Pilot: ~1-2h (bundle with S-05 as one replay run, per Strategist). — Ship: negative cost — ~8 fewer symbols fetched nightly.

## AGGREGATE CREEP ASSESSMENT

Sum of all pilot-now DATA adds if everything ships: **+6 Yahoo chart calls** (D-01 ×2, D-02 ×1, D-03 ×2; plus D-15 ×5 if ever revived), **+10 FRED series** (D-04 ×2, D-05 ×2, D-07 ×1, D-08 ×2, D-09 ×1, D-21 ×2 — config grows 6→16 series), **+0 GDELT** (D-10 is the same single fetch, fixed), **+0 network** for D-14/D-17.

- **Nightly runtime:** ~15-40s added worst-case with retries. Against a minutes-scale budget this fits — runtime is NOT the binding constraint.
- **Failure surface IS the binding constraint.** The night phase currently has 3 external sources (Yahoo, FRED, GDELT) with two known-fragile (Yahoo throttling/endpoint drift, Stooq already demoted to fallback after anti-bot challenges). Six more Yahoo symbols means six more ways step 3 degrades, on the host the system is *most* dependent on. Every new series must ship with the `gdelt_available` pattern: per-series graceful default + availability flag, never a step failure. A new index symbol must never be able to kill the run.
- **data_sources.json burden:** make the FRED series list and the Yahoo index list config-driven (FRED already is — extend the array; Yahoo indices currently hardcoded in handler step 3 — move to config in the same PR). Ten series as config entries is cheap; ten series as bespoke code paths is not.
- **Recommended discipline:** (1) **Two ingest-PR bundles maximum for this whole atlas**: one "vol surface" Yahoo module (D-01+D-02+D-03's Yahoo legs) and one "FRED expansion" PR (D-04/D-05/D-07/D-08/D-09/D-21 — all six, one PR, one backfill parquet). (2) Thereafter cap at **~1 new external-source bundle per quarter**, and a new HOST (vs new series on an existing host) always counts as a full bundle by itself. (3) Every series entry carries: source, first-date, lag-days, fallback-default — enforced by the validate_data step.
- **Second-order creep — the optimizer window:** the S-candidates collectively add ~8-10 new params to the search space against a hard `max_runtime_minutes: 25` weekly window. That window, not Lambda runtime, is where strategy-shape creep bites. Recommend each shipped sleeve/gate retires or freezes at least one existing knob (M-16's disagreement-throttle deletion and the M-01/M-13 possible model retirements are the natural offsets).

## GDELT D-10 VERIFICATION

**Confirmed — the Scout's finding is correct.** Independently re-probed 2026-06-10:
- `http://data.gdeltproject.org/gdeltv2/20260608.gkgcounts.csv.zip` → **404** (this is the exact URL built at `src/steps/ingest_gdelt.py:35`)
- `http://data.gdeltproject.org/gkg/20260608.gkgcounts.csv.zip` → **200**

`fetch_gdelt_daily_aggregate` catches the non-200, returns `gdelt_available=False` defaults, retries yesterday (same broken path), and the handler proceeds — so production has been silently running on GDELT placeholder zeros. Note `config/data_sources.json` also carries `"gdelt": {"base_url": "http://data.gdeltproject.org/gdeltv2/"}` but the ingest step hardcodes the URL — fix both, and prefer reading from config so the next path migration is a config change.

**This is a bug-fix lane item, not an expansion candidate.** It should not consume an atlas pilot slot, should not wait on triage, and should not count against any adds-per-quarter budget. It is recovering an input the system already believes it has. The *theme-count expansion* half of D-10 remains a legitimate expansion candidate, sequenced after the fix confirms live data flows.

## STORAGE PATTERN RECOMMENDATION

Cheapest pattern that preserves replay determinism, for all new daily scalars (D-01..D-09, D-21, D-14 breadth, D-17 flags):

1. **Forward path:** new floats ride as columns in the existing `daily/<date>/signals.parquet` (alongside hy_spread_proxy etc.). No new artifact files, no new S3 prefixes, no schema service. Kilobytes/day; storage cost rounds to zero against the ~$9/month total. Lifecycle policies unchanged.
2. **Never rewrite historical `daily/<date>/` artifacts.** Backfill rewrites would silently change what a past replay reads — that is the determinism breach to avoid.
3. **Backfill path:** one immutable parquet per bundle (e.g., `reference/backfill/fred_expansion_v1.parquet`, `reference/backfill/vol_surface_v1.parquet`), written once, versioned by filename suffix (NOT S3 versioning — stays off per hard rule). The replay harness and monthly training merge backfill values by date with a strict rule: stored daily artifact value wins where present, backfill fills earlier dates only. Re-pulls that would change history require a new `_v2` file, never an overwrite.
4. **Missing-value contract:** every new column has a documented default + `<name>_available` flag (the existing GDELT pattern), so old dates without the column replay identically before and after the add.
5. **D-17 calendar features need no storage at all** — deterministic from date + config table; compute at read time in both Lambda and replay.

No new launchd jobs, no new Lambda functions, no new AWS services are required by ANY pilot-now candidate. The only items on the whole 56-candidate board that would require new infrastructure or hosts are: D-06's TGA leg (api.fiscaldata.treasury.gov — off allowlist), D-11 (publicreporting.cftc.gov — off allowlist), D-12 (cdn.cboe.com — off allowlist), D-23 at full v2 granularity (training-side only, correctly contained by the Scout) — all already parked, all correctly. Loudly: none of these may ship without operator host sign-off, and none justify a new Lambda or paid service even then.

## OVERRIDES

No fatal envelope breaches found among the 29 pilot-now candidates — no BREAKS grades, no kill overrides. The enumerators' envelope hygiene was good (off-allowlist hosts were self-parked; heavy-granularity GDELT was self-contained to training side). Three re-lanings/conditions, not kills:

1. **D-10 (path fix) — re-laned out of the atlas entirely**: bug-fix lane, immediate, not an expansion pilot. The theme-count half stays as a normal expansion candidate behind the fix.
2. **M-03 / M-12 — conditional grade**: FITS only under the export discipline (hmmlearn/lightgbm never enter the Lambda image; inference ships as numpy/pure-python artifacts). If a pilot result comes back requiring the library at inference time, that flips to BREAKS (thin-handler line: no heavy DS deps in request path) and the candidate dies in that form.
3. **Parked-status verifications (no flips):** D-06, D-11, D-12 are correctly parked on allowlist grounds — confirmed each requires operator sign-off and nothing else blocks them. D-16's hosts are genuinely on-allowlist (CoinGecko/Kraken), so its park is on signal grounds, not envelope — correctly labeled. S-11/S-12's "start the artifact clock now" config commit is envelope-clean (~7-10 more symbols in the existing nightly fetch, trivial) — the park is on evidence grounds only, also correct.
