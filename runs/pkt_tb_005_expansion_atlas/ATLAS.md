# PKT-TB-005 — Expansion Atlas

> Synthesized by the Analyst seat from the five panel returns under the
> committee resolution rules (Skeptic demotions and Feasibility re-lanings
> stand; stricter verdict wins on conflict). Source returns: `panel_returns/`.
> Pilot numbers live in `PILOT_EVIDENCE.md`; follow-on sequencing in `ROADMAP.md`.

## Scope

This atlas is the complete triage of 56 expansion candidates for the trader-bot nightly system, enumerated across three angle families: **D — new information/data sources** (D-01..D-24, Information Scout), **M — model architecture and inference** (M-01..M-16, Model Architect), and **S — portfolio strategy shapes** (S-01..S-16, Strategist). Every candidate carries a final verdict — `pilot-now`, `parked-promising`, or `killed` — reflecting the post-Skeptic, post-Feasibility resolution: Skeptic demotions and Feasibility re-lanings stand, and where they conflict the stricter verdict wins. The evidence bar for this run is deliberately hard: the holdout window is 45 perforated days (2026-03-11→2026-06-09) of a rally-dominated market, the holdout-look budget is 12 pre-registered arm-reads, and "shows something" means a paired daily-delta t ≥ 3.0. Candidates whose data should flow now but whose claims cannot be honestly evaluated on this window are encoded as `parked-promising (ship-ingest rider)`: ingest under the Feasibility bundling discipline, claim **no** holdout evidence, evaluate on the next accrued quarter.

## Summary

| Family | pilot-now | parked-promising | killed | Total |
|---|---|---|---|---|
| D — data | 0 | 19 | 5 | 24 |
| M — model | 4 | 11 | 1 | 16 |
| S — strategy | 3 | 10 | 3 | 16 |
| **Total** | **7** | **40** | **9** | **56** |

Of the 40 parked: 6 are ship-ingest riders (D-01, D-02, D-04, D-05, D-07, D-21; D-22 may ride as an optional unvalidated tripwire), 2 are offline-audit riders (M-05, M-16), and 1 is a port-as-engineering rider (S-03). Riders do work now but claim zero holdout evidence this run.

## Triage Ledger

### D family — information / data sources

| ID | Candidate | Verdict | One-line reason | Revival condition | Envelope |
|---|---|---|---|---|---|
| D-01 | VIX term-structure slope | parked-promising (ship-ingest rider) | Most picked-over free vol signal; marginal effect over VIX/VVIX/SKEW is below MDE on 45 perforated days — ingest, claim nothing | Next quarter's accrued window | FITS — graceful-default per series; encode explicit period1 (^VIX3M range=max bug) |
| D-02 | MOVE treasury implied vol | parked-promising (ship-ingest rider) | Holdout contains no rates-vol event (max DD -2.5%), so its claimed value cannot show — ingest and accrue | Next quarter's accrued window | FITS — bundle with D-01 module |
| D-03 | Per-sleeve vol (OVX/GVZ/VXN) | parked-promising | Implied ≈ realized vol + premium — the scout's own "incremental over realized vol" caveat is the kill argument for a look; Yahoo legs folded into D-01 vol-surface module | Accrued window shows per-sleeve implied vol beats realized-vol control | FITS-WITH-CARE — mixed-source; no third ingest path |
| D-04 | Real HY/IG OAS spreads | parked-promising (ship-ingest rider) | Data-quality fix, not an edge — no look charged; ~3.1y served history must be resolved before training (keep proxy for deep backfill) | Next quarter's accrued window | FITS-WITH-CARE — train/serve skew if proxy/OAS split undocumented |
| D-05 | Real yields + breakevens | parked-promising (ship-ingest rider) | "Real rates drive gold" is priced in everywhere — fine as context, indefensible as a look-consuming alpha pilot | Next quarter's accrued window | FITS |
| D-06 | Fed net-liquidity composite | parked-promising | Crowded narrative, weak daily-cadence evidence; TGA leg needs an off-allowlist host | Two-leg (WALCL−RRP) pilot shows delta, or operator allowlists api.fiscaldata.treasury.gov | FITS-WITH-CARE — TGA leg needs host sign-off |
| D-07 | Broad dollar (DTWEXBGS) | parked-promising (ship-ingest rider) | Strict input upgrade, near-zero cost; 2-business-day publication lag must be modeled or the backfill leaks | Next quarter's accrued window | FITS-WITH-CARE — lag-aware replay |
| D-08 | NFCI / STLFSI conditions | parked-promising | Revised series, FRED serves current vintage only — vintage-honest evaluation is impossible from this source | Live-accrued as-of values (forward accrual sidesteps the vintage problem) | FITS-WITH-CARE — freeze backfill parquet once, never re-pull |
| D-09 | Daily EPU index | parked-promising | Noisy, retro-revised, 40-year-old published index every quant has mined; deep history doesn't rescue the mechanism | Demonstrated news-uncertainty gap, read on an accrued window | FITS |
| D-10 | GDELT theme/event counts | parked-promising | News-trading graveyard; stays parked behind the path fix (see re-laned note) and behind evidence the counts carry anything | Path fix confirms `gdelt_available=True` live AND theme counts show offline signal | FITS — 1 daily file; v2 15-min aggregation banned from Lambda |
| D-11 | CFTC COT positioning | parked-promising | Working, free, verified — but host off allowlist; weekly+lagged so slow tilt only | Operator allowlists publicreporting.cftc.gov | BREAKS — off-allowlist host |
| D-12 | VIX futures curve (CBOE) | parked-promising | Most real mechanism on the data list (VIXY roll math is mechanics, not alpha) — blocked purely by host | Operator allowlists cdn.cboe.com (should lead any future sign-off request) | BREAKS — off-allowlist host |
| D-13 | Equity put/call ratio | killed | No working free source — Yahoo ^CPC dead, CBOE archive 403; acquisition cost exceeds marginal signal | — | — |
| D-14 | Internal universe breadth | parked-promising | Breadth over 65 heavily duplicated ETFs is mostly re-measured index vol (n_effective ≪ 65) | Zero-cost offline check shows information beyond spy_vol_21d | FITS — cleanest envelope on the board (zero network) |
| D-15 | Overnight foreign closes | parked-promising | Residual edge thin (info already in ETF opens); cleaner as global-breadth context | D-14 breadth check succeeds, making global breadth the natural extension | FITS |
| D-16 | Crypto weekend bridge | parked-promising | Untestable this run: holdout contains **zero Mondays** — literally no evaluation dates for a weekend-gap feature | Accrued window with Monday artifacts; Monday-only slice shows any signal | FITS — hosts on allowlist; park is on signal grounds |
| D-17 | Calendar/seasonality flags | parked-promising | Post-2015 attenuation plus n≈3 per event class in holdout — unreadable; free deterministic retro-build may proceed, no evidence claim | Accrued window with enough event cells per class | FITS-WITH-CARE — FOMC table needs expiry assertion |
| D-18 | GDPNow nowcast | killed | FRED-served granularity (one value/quarter, vintages overwritten) destroys the nowcast's value | — | — |
| D-19 | ETF flow via shares outstanding | killed | No backfillable free source (crumb-gated, point-in-time only); ≥6 months accumulation before any test | — | — |
| D-20 | FINRA short interest | killed | Bi-monthly + lagged on an ETF universe — nearly information-free at daily cadence | — | — |
| D-21 | Full curve shape (DGS5/DGS30) | parked-promising (ship-ingest rider) | Curve curvature is universal rates-desk furniture — rides the FRED bundle, no separate look | Next quarter's accrued window | FITS |
| D-22 | SOFR funding-stress monitor | parked-promising | Zero funding events in holdout — confirmed untestable; may ride the FRED bundle as an explicitly unvalidated tripwire | A funding event in an accrued window | FITS |
| D-23 | GDELT typed-event structure | parked-promising | Event-to-price mapping is the news-trading graveyard; full v2 granularity busts thin-Lambda anyway | D-10 theme counts show holdout signal | FITS-WITH-CARE — training-side aggregation only |
| D-24 | Google Trends / Wikipedia attention | killed | Per-request re-scaled values make replay non-reproducible; hosts off allowlist; edge decayed post-2015 | — | — |

**Re-laned out of the atlas:** D-10's path-fix half. Production `ingest_gdelt.py` builds a v2 URL that 404s (verified twice independently); the v1 path works. This is a bug repair recovering an input the system already believes it has — bug-fix lane, immediate, consumes no pilot slot, no holdout look, no adds-per-quarter budget. Fix the hardcoded URL and the stale `config/data_sources.json` base_url together; prefer config-driven. Only the theme-count half remains in the ledger above.

### M family — model architecture / inference

| ID | Candidate | Verdict | One-line reason | Revival condition | Envelope |
|---|---|---|---|---|---|
| M-01 | Rule-labeler direct swap | pilot-now | Mandatory baseline: does the deep pair beat its own teacher — with Skeptic repairs (deep-era conditioning, variant (ii) headline, teacher-version check, paired-daily t≥3.0) | — | FITS — possible negative cost (torch regime path exits Lambda) |
| M-02 | Smoothed/hysteresis rule baseline | pilot-now | Rides M-01 at near-zero marginal cost; tests whether the deep pair is an expensive EMA | — | FITS |
| M-03 | Gaussian HMM regime baseline | parked-promising | State-to-label hand-mapping is researcher degrees-of-freedom dressed as unsupervision; sequenced behind the bar M-01 sets | M-01 shows the regime path matters at all | FITS-WITH-CARE — hmmlearn training-side only; numpy-export inference |
| M-04 | Changepoint transition flag | parked-promising | ~1-2 genuine breaks in window, all at the left edge — replay evidence would be anecdote | M-01 shows regime path matters AND offline event study shows ≥3-day lead | FITS |
| M-05 | Calibration-weighted ensemble | parked-promising (offline-audit rider) | Only ~66 days of genuine member probs exist (128/194 are fallback one-hots) — the redundancy finding is a correlation on 66 days, not a replay | Audit result + next accrued quarter of deep-era probs | FITS |
| M-06 | Confidence calibration layer | parked-promising | Only ~21 genuine pre-holdout prob days to fit on — "a calibration layer fit on 21 observations is noise wearing a lab coat" | Another quarter of deep-era probs accrues | FITS |
| M-07 | Conformal regime sets | parked-promising | Coverage would be coverage-of-the-teacher (pseudo-label problem); sequenced behind M-06 and M-01 | M-06 shows miscalibration AND M-01 shows deep probs carry signal | FITS |
| M-08 | Meta-labeling trade filter | parked-promising | Trained on synthetic replay trades — imports the simulator's assumptions into the labels | Stored-fills query shows ≥30% of holdout buys are net losers after costs | FITS — GBM/tiny MLP, no torch in handler |
| M-09 | Continuous risk dial | parked-promising | Many hand-chosen maps judged on a rally window where "fewer whipsaws" wins mechanically; inherits M-01's verdict for free if the regime path is inert | M-01 shows the regime path is not inert | FITS |
| M-10 | Learned position sizing | killed | ~235 days of decision data = an overfitting machine; the evolutionary param search already occupies this niche at sane capacity | — | — |
| M-11 | Learned fusion over expert signals | parked-promising | ~170 trainable pre-holdout days will memorize the one drawdown it saw; the production-only design survives re-examination on sample-size grounds | Signal-series backfill completed (itself a flagged data candidate) | FITS |
| M-12 | GBM regime on real forward outcomes | parked-promising | Forward-regime prediction on ~2,800 rows is the precise graveyard genre; trailing description may be all the engine needs; third-wave = not this run | M-01/M-03 establish the bar on a future run | FITS-WITH-CARE — lightgbm training-side only; exported-tree inference |
| M-13 | Health model honesty check | pilot-now | M-01's twin on a larger decision surface (buy thresholds, HEALTH_COLLAPSE/DROP); shares the harness and the fallback-era conditioning fix | — | FITS — AE may exit the image |
| M-14 | Ranking blend sweep | pilot-now (rank-IC only) | Only real-label model in the stack, wiring exists — but RankingMLP trained 2026-04-29 > holdout start 2026-03-11: holdout replay is contaminated, so standalone rank-IC is the only admissible read; **0 holdout looks** | — | FITS — cheapest on the M list |
| M-15 | Vol-targeting sizing baseline | parked-promising | Worth an afternoon — once someone else builds the per-date multiplier hook | The hook gets built for any other candidate, after M-01 lands | FITS |
| M-16 | Disagreement-signal audit | parked-promising (offline-audit rider) | The claimed 235 disagreement values are really n≈66 (rest are fallback zeros) — se(ρ)≈0.13 makes the audit directional only; flip the throttle only alongside the M-01 verdict | Audit direction + M-01 verdict; next accrued quarter for power | FITS — possible knob deletion |

### S family — strategy shapes

| ID | Candidate | Verdict | One-line reason | Revival condition | Envelope |
|---|---|---|---|---|---|
| S-01 | Cash sleeve (SHY) | pilot-now | Not an edge — T-bill carry on structurally idle 10-40% reserve is arithmetic; replay is a **harm check** (clean liquidation into buys), carry never claimed as a replay-measured win | — | FITS — ~40 lines, all data stored |
| S-02 | Core-satellite beta floor | parked-promising | A 30-40% permanent SPY core on a +11% rally holdout wins mechanically — the canonical adopt-beta-call-it-alpha trap; no bear window exists in stored artifacts to price the cost | Pipeline-faithful long-history replay including a bear window | FITS |
| S-03 | Engine-native top-up | parked-promising (port-as-engineering rider) | Overlay trigger was tuned against this same holdout — the window is mined; port proceeds for one-decision-path hygiene, claims only consistency-with-overlay, zero fresh evidence credit | Forward data confirms value post-port | FITS — reduces maintenance surface (engine+overlay → engine) |
| S-04 | Hedge sleeve with regime budget | parked-promising | Holdout max DD -2.5% — far too benign to price crisis convexity; only the carry-cost side is observable | PKT-TB-004 shows the throttle stack lacks downside control AND a window with a real drawdown exists | FITS |
| S-05 | VIXY ejection from scored universe | pilot-now | Negative-carry decay asset in a momentum scorer that structurally buys post-spike is a category error; tie favors removal; run the free fills query first | — | FITS — cheapest on the S list; arguably negative cost |
| S-06 | Regime-conditional universe masks | killed | Zero expressiveness over existing `regime_compatibility` (already expresses 0.0); re-tuning is PKT-TB-002's lane | — | — |
| S-07 | Conviction-weighted sizing | parked-promising | Value hinges on score *magnitude* (not rank) carrying information — unestablished; single-window wins presumptively overfit | PKT-TB-002/ranking evidence that score magnitude is calibrated | FITS |
| S-08 | Correlation-aware selection | parked-promising | Trailing-60d correlations among duplicated index ETFs are ~0.8+ — the penalty mostly re-derives the static cluster map with noise; the cluster-map-ON control arm IS the experiment and belongs to PKT-TB-002 | Cluster-map-ON arm (PKT-TB-002) leaves a residual pathology | FITS-WITH-CARE — cluster-ON control must run first |
| S-09 | Split cadence (daily sells, slow buys) | parked-promising | Turnover-cost verdict is hostage to the fill/cost model PKT-TB-003 is about to change — a read now has a known expiry date | PKT-TB-003 lands; re-run is cheap then | FITS |
| S-10 | Universe dedup (7 duplicate sets) | pilot-now | Slots that mean something at zero information cost; tie wins on legibility; bundles with S-05 in one replay | — | FITS — negative cost (~8 fewer nightly fetches) |
| S-11 | Leveraged ETF sleeve | parked-promising | Needs-new-data is dispositive: no stored artifacts exist for unlisted symbols; pipeline-faithful replay impossible until they accrue | 60+ days of accrued live artifacts (start the clock now — config commit is envelope-clean) | FITS — clock-start trivial |
| S-12 | Diversifier ETFs (DBMF/JEPI/BTAL) | parked-promising | Same artifact-accrual block as S-11; precondition check (can the scorer like low-momentum assets?) is free | Same artifact clock as S-11, started in the same commit | FITS |
| S-13 | Book-level vol targeting | parked-promising | A sixth exposure brake before PKT-TB-004 attributes the existing five is how the five happened | PKT-TB-004 finds the throttle stack incoherent — then it's a *replacement*, not an addition | FITS |
| S-14 | Minimum-deployment floor | killed | Worse failure mode stands alone: in a low-score environment it forces the book into its own weakest ideas; survives only as a comparison arm inside (parked) S-02 | — | — |
| S-15 | Scale-in entries | parked-promising | Net sign genuinely unclear and it requires S-03's top-up plumbing, which hasn't landed | S-03 port lands and pilots successfully | FITS |
| S-16 | Reserve as SHY/IEF ladder | killed (as standalone) | Folded into S-01 as the ladder arm — live in this run's slate; the marginal IEF claim is already expressible through the scorer (compat 1.1-1.2) | — | FITS — rides S-01 |

Analyst note: the Skeptic's slate budgeted 2 looks for the S-05+S-10 bundle; the locked slate charges 3. The 10-of-12 total below uses the locked arithmetic.

## Final pilot slate (locked, 10 of 12 holdout arm-reads)

1. **Baseline honesty block — M-01 + M-02 + M-13** (mandatory; 5 looks) with the Skeptic's four repairs: deep-era conditioning (66-day comparison, not 235), variant (ii) as the only admissible label-path headline, teacher-version pinning/reproduction, and the paired-daily t≥3.0 noise model. A statistical tie kills the torch regime path under the removal asymmetry.
2. **M-14 ranking sweep — rank-IC only** (0 looks): training-date contamination (trained 2026-04-29, after holdout start 2026-03-11) makes any holdout replay read inadmissible.
3. **S-01 cash sleeve + S-16 ladder arm** (2 looks): harm-check framing — carry is established by arithmetic, the replay only verifies clean liquidation into buys.
4. **S-05 + S-10 universe-config bundle** (3 looks): tie-favors-removal standard; settles the standing VIXY question jointly with PKT-TB-004.

## Run-level findings surfaced during triage

- **(a) The stored inference series is mostly the rule labeler.** 128 of 194 stored inference days (2025-08-04→2026-02-05) are rule-fallback one-hots (conf=1.0, zero embedding, psm=1.0); genuine deep-ensemble output exists only from ~2026-01-31. Every prior analysis that treated the stored series as "the deep ensemble's track record" — including champion-selection evidence — was looking at the rule labeler for two-thirds of it. This finding outranks any single candidate.
- **(b) Production GDELT fetch 404s.** The v2 daily-gkgcounts path built in `src/steps/ingest_gdelt.py` does not exist (v1 path works, verified twice); production has been silently riding `gdelt_available=False` placeholder zeros. Re-laned to the bug-fix lane.
- **(c) The holdout window is perforated and rally-dominated.** 45 of ~65 trading days stored (~70% coverage), **zero Mondays**, a 13-day artifact hole (2026-05-08→05-21); SPY +11.2% over the window with max drawdown only -2.5%. The artifact gaps are a pipeline-reliability finding that routes to operations — they silently bias every window-level metric the system reports, not just this packet's.
- **(d) The ranking model has no clean out-of-sample read.** RankingMLP (trained 2026-04-29) sits in production at blend 0.35 with zero uncontaminated holdout days available; only forward accrual produces one.

## Holdout honesty rule adopted for this run

- **Look budget:** ≤ 12 pre-registered holdout arm-reads total across all pilots; every arm declared in the run dir before its replay executes; K stated in the run report.
- **Evidence standard:** "shows something" = paired daily-delta **t ≥ 3.0** (Bonferroni for ~12 looks) AND mean delta ≥ +0.05%/day. t ∈ [2.0, 3.0) = suggestive — park, requires a second independent read on the next accrued quarter. t < 2.0 = "consistent with noise," stated verbatim.
- **Primary metric:** the paired daily-delta series. Endpoint/fold_score deltas and holdout Sharpe deltas are never primary (SE of a 45-day annualized Sharpe ≈ 2.4 — inadmissible at any plausible magnitude). Seed-std over 5 slippage seeds measures fill noise only, never the verdict's uncertainty — the dominant noise term is the single 45-day window.
- **Removal asymmetry:** for removals/simplifications (S-05, S-10, torch-path exit), a noise result IS bounded evidence of harmlessness — report the MDE bound and let the simpler arm win ties.
- **Mechanics-not-alpha:** for arithmetic effects (S-01 carry), the replay is a harm check only; the report must not claim the effect as a replay-measured win.
- All window-level numbers carry a "45/65 coverage" footnote.

## Ingest riders shipped without evidence claims

Feasibility bundling discipline — **two ingest PRs maximum for this whole atlas**, then at most **1 new source-bundle per quarter**, and a new HOST always counts as a full bundle by itself:

1. **Vol-surface Yahoo module:** D-01 + D-02 + D-03's Yahoo legs — one module, per-series graceful default + availability flag (a new index symbol must never kill the run); move the hardcoded Yahoo index list into config in the same PR.
2. **FRED expansion PR:** surviving bundle members **D-04 / D-05 / D-07 / D-21** (D-08 and D-09 were demoted to parked by the Skeptic and do not ship), plus **D-22 as an optional, explicitly-unvalidated funding-stress tripwire**; config grows by extending the existing FRED series array; D-07 ships lag-aware (2-business-day publication lag).
3. **Storage pattern (all new daily scalars):** forward path = new columns in the existing `daily/<date>/signals.parquet` — no new artifacts, prefixes, or services; **never rewrite historical daily artifacts**; backfill = one immutable versioned parquet per bundle (`reference/backfill/<bundle>_v1.parquet`, re-pulls require `_v2`, never overwrite; no S3 versioning per hard rule); stored daily value wins over backfill where both exist; every new column carries a documented default + `<name>_available` flag plus source/first-date/lag-days metadata enforced by validate_data. D-17's calendar features, if ever built, need no storage (deterministic from date + config).
4. **Other zero-look riders this run:** D-10 GDELT path fix (bug-fix lane), M-05/M-16 offline audits on the 66 genuine deep-era days, S-03 engineering port, S-11/S-12 artifact-clock-start config commit.

No new launchd jobs, Lambda functions, AWS services, or hosts are required by anything shipping this run. The only candidates anywhere on the board needing new hosts (D-06 TGA leg, D-11, D-12, D-23 at full granularity) are parked and may not ship without operator sign-off.

## Post-pilot verdict updates (see PILOT_EVIDENCE.md for numbers)

- **S-01 cash sleeve: piloted → parked-promising.** The harm check was
  confounded by a production landmine the pilot itself discovered
  (`paper_trader.execute_trade` partial-SELL pops the whole holding); the
  pilot's forced full-churn workaround tripled costs (suggestive harm,
  t=-2.32, attributable to wiring). Revival condition: partial-SELL fix lands,
  then re-run the v3 design with clean partial liquidation. The carry argument
  stands on arithmetic.
- **S-05 + S-10: piloted → proven harmless, removals win** under the
  pre-registered tie-favors-removal standard (deltas ≈ 0, |t| ≤ 0.2; VIXY
  traded 3× in 194 days, duplicates co-held 2 days). Production adoption is a
  future packet per the evidence-only constraint.
- **M-01/M-02 baseline check: statistical tie** (no |t| ≥ 2 in either harness
  variant; delta sign flips with the ranking layer). **Scope correction
  (operator-prompted):** every evaluated deep-era day ran the
  short-corpus-MISTRAINED models (pre-2026-06-06 training-data fallback bug;
  GRU at 9% OOS before the fix); the corrected models have exactly 1 stored
  day and are untested. The tie verdict scopes to the ensemble as-it-ran; the
  next-quarter re-read (ROADMAP 1.2) is the first genuine test of the
  corrected models and carries the keep/kill decision.
- **M-13 health check: noise-with-direction** — rule health beat the trained
  AE by +14.5pp on the holdout endpoint at a quarter of the drawdown (t=+1.18,
  below the bar). Same scope correction: the losing AE was the short-corpus
  one; the corrected AE's first test is the next-window read.
- **M-14 ranking: no clean out-of-sample evidence exists**; the active layer
  (blend 0.35) showed -7.5pp holdout endpoint and ~3× max drawdown vs blend 0
  on this harness (t=-0.65, noise at daily level; contamination favors the
  layer). Routed to the operator as the run's most urgent decision input.
