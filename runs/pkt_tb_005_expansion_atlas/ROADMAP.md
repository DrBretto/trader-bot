# PKT-TB-005 — Expansion Roadmap

Sequenced follow-on list for the surviving `parked-promising` candidates, each
with its evidence path pre-designed so a future packet can execute without
redesign. Ordering = information value per unit work, respecting unlock
dependencies. Verdicts and revival conditions are canonical in `ATLAS.md`;
this file adds sequencing and the ready-to-run evidence designs.

## Wave 0 — now, zero holdout looks, zero evidence claims

These ship as riders or run as free offline checks. None may claim holdout
evidence; their job is to set up the NEXT window's reads.

| # | Item | What to do | Pre-designed evidence path (for later) |
|---|------|------------|----------------------------------------|
| 0.1 | GDELT path fix (re-laned D-10) | Bug-fix lane: point `ingest_gdelt.py` at the v1 `/gkg/` daily path, read base_url from `config/data_sources.json`; confirm `gdelt_available=True` flips in the next nightly | After ≥1 quarter of REAL tone data accrues: replay tone-feature on/off on the new window |
| 0.2 | Vol-surface Yahoo module (D-01+D-02+D-03 Yahoo legs) | One ingest PR: ^VIX9D, ^VIX3M (explicit period1 — range=max broken), ^MOVE, ^OVX, ^GVZ; per-series graceful default + `_available` flag; new columns in signals.parquet; immutable backfill parquet `reference/backfill/vol_surface_v1.parquet` | Next-window replay: vol-curve slope feature in vol-complexity expert ON/OFF; D-02 needs a rates-vol event in window to be readable |
| 0.3 | FRED expansion PR (D-04/D-05/D-07/D-21 + optional D-22 tripwire) | Extend the FRED series array: BAMLH0A0HYM2, BAMLC0A0CM, DFII10, T10YIE, DTWEXBGS (lag-aware: 2-business-day), DGS5, DGS30, optional SOFR+IORB; same storage pattern | Next-window replay: OAS-instead-of-proxy in macro/credit expert ON/OFF; real-yield feature in metals scoring ON/OFF |
| 0.4 | S-11/S-12 artifact clock start | One universe.csv commit: leveraged sleeve symbols (leverage_flag=1) + DBMF/JEPI/BTAL; they begin accruing daily features/health/inference artifacts | After 60+ accrued trading days: pipeline-faithful replay with the sleeve symbols eligible vs not |
| 0.5 | S-03 engineering port | Port `topup_on_psm_rise` semantics into the decision engine behind a param; claim consistency-with-overlay ONLY (the holdout is mined for top-up claims) | Forward shadow (E3-style logging) confirms value; never a fresh read on this holdout |
| 0.6 | D-14 offline breadth check | Free: compute breadth series (% above 50d/200d MA, dispersion) on stored parquets 2014→now; correlate vs spy_vol_21d and regime labels | Revive only if breadth adds information beyond spy_vol_21d; then next-window replay as a fusion gate |
| 0.7 | M-08 precondition query | Free: share of holdout buys that are net losers after costs, from stored fills | If ≥30%: meta-labeling pilot enters the next window's slate |

## Wave 1 — next accrued window (≈ one quarter of clean artifacts)

Gate for the whole wave: the next holdout window must be pre-registered BEFORE
reads, and its look budget set the same way (Bonferroni-adjusted t threshold).

| # | Item | Unlock | Pre-designed evidence path |
|---|------|--------|----------------------------|
| 1.1 | **Ranking-model clean read** (M-14 completion — highest priority) | Next monthly retrain pins a training cutoff; ≥21 trading days accrue after it | Pre-register: canon-object (three_line_replay run_variant) AND optimizer-object replays of blend 0 vs active blend, read only on post-cutoff dates; plus 21d rank-IC on post-cutoff dates. This run's evidence (in-window IC +0.20 vs out-of-window ~0.00; optimizer-object holdout -5.9% with blend vs +1.6% without) says treat blend 0.35 as UNPROVEN and possibly harmful |
| 1.2 | Baseline-block second read (M-01/M-02/M-13 confirmation) | One more quarter of genuine deep-era inference accrues | Same harness (pilots/pilot_baseline_check.py), same arms, new window; the t≥3.0 standard applies to the SECOND read independently |
| 1.3 | M-06 confidence calibration | ≥60 genuine deep-era pre-window prob days | Fit temperature on accrued probs; ECE before/after + one pre-registered window read |
| 1.4 | M-05/M-16 powered re-audits | Same accrual | Re-run pilots/offline_audits.py at n≥120; this run's direction: disagreement predicts label flips (ρ=+0.37, p=0.003) — keep the throttle pending the powered read |
| 1.5 | M-03 HMM regime baseline | M-01 verdict says regime path matters at all (this run: label-path deltas were noise — weak motivation; do not run before 1.2 confirms) | Fit on pre-window context (hmmlearn training-side only; numpy export); filtered probs swap via the same harness |
| 1.6 | S-09 split cadence | PKT-TB-003 lands its cost model | Replay buy-cadence {1,3,5,regime-change-only} under the NEW cost model |
| 1.7 | S-08 correlation-aware selection | PKT-TB-002 runs the cluster-map-ON arm first | Only if a residual pathology remains after cluster-ON |
| 1.8 | D-16 / D-17 (Mondays / event cells) | An accrued window that actually contains Mondays and ≥10 events per class — requires fixing the Monday artifact gap first (see pipeline-reliability item below) | Monday-only paired slice; per-event-class paired slices |
| 1.9 | D-01/D-02/D-03 evaluation | Wave-0.2 ingested + a window containing at least one vol/rates event | Expert-signal ON/OFF replays per the Scout's designs |

## Wave 2 — operator-gated (host sign-offs or harness builds)

| # | Item | Gate | Pre-designed evidence path |
|---|------|------|----------------------------|
| 2.1 | D-12 VIX futures curve | Operator allowlists `cdn.cboe.com` (the Skeptic ranks this the most-real data mechanism on the board — lead any sign-off request with it) | Reconstruct front/second continuous series from per-contract CSVs; gate VIXY entries on backwardation; replay where VIXY traded (note: this run found VIXY traded 3 times in 194 days — the question may be moot after S-05 ships) |
| 2.2 | D-11 CFTC COT | Operator allowlists `publicreporting.cftc.gov` | Positioning-percentile feature in metals/energy scoring; next-window replay |
| 2.3 | D-06 net liquidity (full) | Two-leg (WALCL−RRP) shows next-window delta, or operator allowlists `api.fiscaldata.treasury.gov` for the TGA leg | Fusion-tilt replay |
| 2.4 | S-02 core-satellite | A pipeline-faithful long-history replay harness incl. a bear window (does not exist; building it is its own packet) | Core {0,20,40}% arms with the bear window carrying the verdict |
| 2.5 | S-04 hedge sleeve / S-13 vol targeting | PKT-TB-004 attribution of the existing five throttles + a window with a real drawdown | Budget-keyed sleeve arms / brake-replacement arms per ATLAS |
| 2.6 | M-11 / M-12 learned fusion & GBM regime | Signal-series backfill 2014→now (itself a Wave-0-able build) + M-01 second read | Walk-forward refit, probs swap via the same harness |

## Run-level repairs that gate the roadmap (route to their lanes; not expansion)

1. **Monday artifact gap + May hole (ops lane, HIGH).** The holdout window has
   ZERO Monday artifacts and a 13-day hole (2026-05-08→05-21); ~30% of trading
   days have no stored decision artifacts. Until fixed, every window-level
   metric the system reports is biased and D-16/D-17-class candidates are
   untestable. Diagnose the night-pipeline schedule/persistence path.
2. **paper_trader partial-SELL landmine (bug lane, HIGH).** `execute_trade`
   SELL credits cash for the sold shares but pops the ENTIRE holding — any
   future partial-sell caller silently burns the remainder (this run's sleeve
   pilot v1 lost 97% to it; engine sells are full-position today, which is the
   only reason production is safe). Fix before any scale-in/partial-exit work
   (S-15 depends on it).
3. **Stored-inference fallback era (training/ops lane).** 128/194 stored days
   are rule-fallback one-hots; add a manifest flag so every future analysis can
   condition on genuine-model eras mechanically; pin the labeler version in
   training manifests (teacher-version skew: deep-era pre-holdout agreement
   with current-code rules is 28.6% vs 77.8% on holdout).
4. **Ranking blend live-risk review (operator decision, URGENT).** Blend 0.35
   is live in production on a model trained 2026-04-29 with no clean
   out-of-sample read (see 1.1). The operator should decide whether to keep it
   live pending the clean read; this packet takes no production action per its
   hard constraints.
