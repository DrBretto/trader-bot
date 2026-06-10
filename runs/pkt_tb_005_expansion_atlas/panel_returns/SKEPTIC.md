# Skeptic — panel return (verbatim subagent output)

## ATTACKS

**Data candidates (Information Scout)**

- D-01 — DEMOTED pilot-now → ship-ingest-only, no holdout look this run — vol-curve shape is the single most picked-over free vol signal in existence; as marginal input over VIX/VVIX/SKEW already ingested, its detectable effect on 45 perforated holdout days is far below MDE (see honesty rule), so ingest now, evaluate on the next window.
- D-02 — DEMOTED pilot-now → ship-ingest-only — the holdout contains no rates-vol event (within-window SPY drawdown is only -2.5%), so a MOVE feature cannot show its claimed value on this window; ingest and accrue.
- D-03 — DEMOTED pilot-now → parked — implied vol ≈ realized vol + premium is the textbook dead retail signal; the scout's own caveat ("incremental over realized vol the system can already compute") is the kill argument for spending a holdout look.
- D-04 — DEMOTED pilot-now → ship-ingest-only — this is a data-quality fix not an edge, so it shouldn't be charged a holdout look; the ~3.1y served history flag must be resolved before anything trains on it.
- D-05 — DEMOTED pilot-now → ship-ingest-only — "real rates drive gold" is priced into every macro desk on earth; fine as regime context, indefensible as a holdout-look-consuming alpha pilot.
- D-06 — upheld (parked) — correctly parked; net-liquidity is a crowded narrative with weak daily-cadence evidence.
- D-07 — DEMOTED pilot-now → ship-ingest-only — strict input upgrade, near-zero cost, but the 2-business-day publication lag must be modeled in replay or the backfilled feature leaks two days of future dollar information.
- D-08 — DEMOTED pilot-now → parked — NFCI/STLFSI are *revised* series and FRED serves current vintage only, so any backfilled replay hands the strategy data production could never have had; vintage-honest evaluation is impossible from this source.
- D-09 — DEMOTED pilot-now → parked — daily EPU is noisy, retro-revised, and news-uncertainty→daily-returns is graveyard; the "deepest history" virtue is irrelevant when the mechanism is a 40-year-old published index every quant has already mined.
- D-10 — UPHELD, reclassified bug-fix not expansion — ships regardless of committee verdict and consumes zero holdout looks; the "then add theme structure" half is classic news-trading graveyard and stays parked behind evidence the counts carry anything.
- D-11/D-12 — upheld (parked) — correctly parked on allowlist grounds; D-12's mechanism (VIXY roll math) is the most real on the data list and should lead any future sign-off request.
- D-13 — kill upheld (fair) — no working free source is dispositive.
- D-14 — DEMOTED pilot-now → parked-pending-free-offline-check — "breadth" computed over 65 heavily duplicated ETFs (SPY/VOO/IVV/VTI count as four) is mostly a re-measurement of index vol with n_effective far below 65; run the zero-cost offline correlation-vs-regime-labels check first and revive only if it adds information beyond spy_vol_21d.
- D-15/D-16 — upheld (parked), with D-16 hardened to untestable: the holdout artifact set contains **zero Mondays**, so a Monday-morning weekend-gap feature has literally no evaluation dates this run.
- D-17 — DEMOTED pilot-now → parked — post-publication attenuation hit turn-of-month/FOMC drift hard after 2015, and the holdout contains ~3 of each event class (n≈3 per cell is unreadable); the deterministic retro-feature build is free and may proceed, but no holdout evidence claim is possible.
- D-18/D-19/D-20 — kills upheld (fair) — granularity destruction, no backfill, and cadence mismatch respectively are all dispositive.
- D-21 — DEMOTED pilot-now → ship-ingest-only — curve curvature is universal rates-desk furniture; ride the FRED bundle, no separate look.
- D-22 — upheld (parked) — the scout's own honesty ("replay window contains no funding event") is exactly right; confirmed, the window has none.
- D-23/D-24 — kill/park upheld (fair).

**Model candidates (Model Architect)**

- M-01 — UPHELD (mandatory slot) — not an edge claim but a demolition tool; immune to the graveyard prior; design repairs required (see audit).
- M-02 — UPHELD (rides M-01) — near-zero marginal cost in the same harness.
- M-03 — DEMOTED pilot-now → parked behind M-01 — 5-state Gaussian HMM with hand-mapping of states-to-labels by fitted means is researcher-degrees-of-freedom dressed as unsupervision; sequence it only after M-01 establishes whether the regime path matters at all.
- M-04 — upheld (parked) — correctly parked; ~1-2 genuine breaks in window = anecdote.
- M-05 — DEMOTED pilot-now → offline-audit-only, no holdout look — discovered fact: stored inference is rule-fallback one-hot for 128/194 days, so only ~66 days of genuine member probs exist; the member-redundancy finding the architect actually wants is a correlation computation on those 66 days, not a replay.
- M-06 — DEMOTED pilot-now → parked — the temperature would be fit on pre-holdout stored probs, of which only **~21 days** are genuine deep-ensemble output (the rest are conf=1.0 fallback); a calibration layer fit on 21 observations is noise wearing a lab coat; revive after another quarter of deep-era probs accrues.
- M-07 — upheld (parked) — correctly sequenced behind M-06/M-01; coverage-of-the-teacher caveat is honest.
- M-08 — upheld (parked) — correct; synthetic-label import is real and the precondition query (share of losing holdout buys) is the right gate.
- M-09 — DEMOTED pilot-now → parked behind M-01 — a smooth-dial rewrite of every regime consumer is many hand-chosen maps evaluated on a rally window where "fewer whipsaw transitions" wins mechanically; if M-01 shows the regime path is inert, dials inherit the verdict for free.
- M-10 — kill upheld (fair) — the evolutionary search already occupies the niche at sane capacity.
- M-11 — upheld (parked) — correctly parked on sample size; the flagged signal-backfill is the right data candidate.
- M-12 — DEMOTED pilot-now → parked behind M-01 — forward-regime prediction on ~2,800 rows is the precise genre the graveyard is full of; the architect's own caveat (trailing description may be all the engine needs) is the attack; third-wave means not-this-run.
- M-13 — UPHELD (rides M-01) — higher decision surface than regime; same harness; same fallback-era conditioning required.
- M-14 — UPHELD with precondition — the only real-label model, wiring exists, one pre-registered holdout confirm; precondition: verify the RankingMLP training cutoff predates 2026-03-11, else the holdout read is contaminated and only the standalone rank-IC is admissible.
- M-15 — upheld (parked) — correct revival condition (hook gets built anyway).
- M-16 — DEMOTED pilot-now → offline-audit-only — the claimed "235 stored disagreement values" is false: 128 are fallback zeros, leaving n≈66, where se(ρ)≈0.13 means |ρ|<0.1 is indistinguishable from |ρ|=0.2; run the underpowered audit for direction, flip the threshold only alongside the M-01 verdict, no separate holdout look.

**Strategy candidates (Strategist)**

- S-01 — UPHELD — survives the graveyard prior because it is not an edge: T-bill carry on structurally idle cash is arithmetic, not alpha; the replay's job is a harm check (clean liquidation into buys), not effect detection — its ~0.4-2pp/yr carry is below holdout MDE and must not be sold as a measured replay win.
- S-02 — DEMOTED pilot-now → parked — a 30-40% permanent SPY core on a +11% rally holdout wins mechanically; this is the canonical adopt-beta-call-it-alpha trap, and the evidence that would price the core's cost (a bear window) does not exist in stored artifacts; park until pipeline-faithful long-history replay exists.
- S-03 — DEMOTED pilot-now → port-as-engineering, no fresh evidence credit — the overlay's +11.85% line was found by tuning the trigger against this same holdout, so the window is already mined for top-up claims; the engine-native port is justified for one-decision-path hygiene and must only claim consistency-with-overlay, with value confirmation deferred to forward data.
- S-04 — upheld (parked) — correct, and confirmed harder: within-holdout SPY drawdown is -2.5%, far too benign to price crisis convexity.
- S-05 — UPHELD — removing a negative-carry decay asset from a momentum scorer that structurally buys it post-spike is mechanism-sound; tie favors removal; run the free fills query first (may be a non-event).
- S-06 — kill upheld (fair) — zero added expressiveness over compat=0.0.
- S-07 — upheld (parked) — correct; score-magnitude calibration is unestablished.
- S-08 — DEMOTED pilot-now → parked behind the cluster-map-ON arm — trailing-60d correlations among duplicated index ETFs are uniformly ~0.8+, so the penalty mostly re-derives the static cluster map with estimation noise; the strategist's own "run cluster-ON first" arm is the whole experiment and belongs to PKT-TB-002's tuning lane.
- S-09 — DEMOTED pilot-now → parked — a turnover-cost claim whose verdict is hostage to the fill/cost model that PKT-TB-003 is about to change; per the evidence protocol that re-run is cheap later, so don't spend a holdout look on a verdict with a known expiry date.
- S-10 — UPHELD — near-zero cost, tie-wins-on-legibility, bundles with S-05 in one replay.
- S-11/S-12 — upheld (parked) — needs-new-data is dispositive this run; starting the artifact clock in one config commit is cheap and correct.
- S-13 — upheld (parked) — correct; sixth brake before attribution of the first five is how the five happened.
- S-14 — kill upheld, reason amended — stands on worse-failure-mode grounds alone (forces the book into its weakest ideas) even with S-02 now parked rather than piloted.
- S-15/S-16 — upheld (parked / folded) — S-16's SHY+IEF ladder survives as an arm inside the S-01 pilot.

**Unfair-kill review: no candidate was killed unfairly.** All seven kills (D-13, D-18, D-19, D-20, D-24, M-10, S-06, S-14) rest on source-availability, granularity, cadence, capacity, or expressiveness facts, not on taste.

## HOLDOUT HONESTY RULE

Computed from the actual cache: the holdout is **45 stored decision days** spanning 2026-03-11→2026-06-09, but it is perforated — **zero Mondays**, a 13-calendar-day artifact hole (2026-05-08→2026-05-21, ~8 trading days), and 2026-03-30/31 missing: 45 of ~65 trading days (~70% coverage).

Minimum detectable effect, paired daily deltas, n=45:

| daily-delta sd | t=2.0 | t=3.0 |
|---|---|---|
| 0.10% | 0.030%/day = 1.3pp/window ≈ 7.5pp/yr | 0.045%/day = 2.0pp/window ≈ 11pp/yr |
| 0.20% | 0.060%/day = 2.7pp/window ≈ 15pp/yr | 0.089%/day = 4.0pp/window ≈ 23pp/yr |
| 0.30% | 0.089%/day = 4.0pp/window ≈ 23pp/yr | 0.134%/day = 6.0pp/window ≈ 34pp/yr |

Sharpe is worse: SE of an annualized Sharpe estimated on 45 days is ≈ √(252/45) ≈ 2.4, so **holdout Sharpe deltas are inadmissible as primary evidence at any plausible magnitude**.

**Binding rule for this run:**
1. **Holdout look budget: ≤ 12 pre-registered arm-reads total** across all pilots (the recommended slate uses ~10-11). Every arm is declared in the run dir before its holdout replay executes; K is stated in the run report per EVIDENCE_PROTOCOL.
2. **"Shows something" = paired daily-delta t ≥ 3.0** (Bonferroni for ~12 looks at α≈0.05) **AND mean daily delta ≥ +0.05%/day.** t ∈ [2.0, 3.0) = "suggestive — park, requires a second independent read on the next accrued quarter before adoption." t < 2.0 = "consistent with noise," stated as such verbatim.
3. **Endpoint/fold_score deltas and holdout Sharpe deltas are never primary**; the paired daily series is (protocol §2). Seed-std over 5 slippage seeds measures fill noise only and must not be presented as the uncertainty of the verdict — the dominant noise term is the single 45-day window (n=1 window).
4. **Asymmetry for removals/simplifications** (S-05, S-10, torch-path removal): a noise result IS bounded evidence of harmlessness — report the MDE bound ("any harm exceeds X pp/yr with prob ≤5%") and let the simpler arm win ties.
5. **Mechanics-not-alpha candidates** (S-01 carry): the effect is established by arithmetic; the holdout replay is a harm check only, and the report must not claim the carry as a replay-measured win.
6. Cumulative/endpoint metrics over this holdout understate/mis-state reality because of the 13-day hole; all window-level numbers carry a "45/65 coverage" footnote.

## HOLDOUT REPRESENTATIVENESS

What the window actually contains (measured from `data_cache.pkl`):
- **Rally-dominated:** cumulative SPY over the 45 stored days = **+11.2%** (≈ +60%/yr pace); within-window max drawdown only **-2.5%**; max single-day move -1.79%/+2.54%; 3 days with |move|>1.5%.
- **A trailing-window stress echo, not a stress event:** deep-ensemble labels = risk_on 28, panic 9, choppy 5, risk_off 2, calm 1; recomputed rule labels = risk_on 27, panic 12, risk_off 4, choppy 2. The panic labels (2026-03-13→04-03 region) reflect 21-day trailing windows reaching back into the pre-holdout late-Feb/early-Mar decline; the holdout itself never falls more than 2.5%.
- **Structure:** zero Mondays; 13-day artifact hole in mid-May; 10 deep-vs-rule label disagreement days (78% agreement), concentrated at transitions — so M-01 is substantive, not a foregone tie.
- **Stored-inference regime break:** conf=1.0 one-hot fallback (zero embedding, psm=1.0) for every day 2025-08-04→2026-02-05 (128/194 days); genuine deep-ensemble output exists only from 2026-01-31 (~66 days, 45 of them holdout).

Candidate classes that CANNOT be honestly evaluated on this window:
- **Crisis/hedge convexity** (S-04, D-22, VIXY-as-hedge half of S-05): no funding event, no crash day, -2.5% max drawdown — only the carry cost side is observable.
- **Bear/regime-cost of permanent beta** (S-02, S-14): a rally window can only flatter them.
- **Monday/weekend machinery** (D-16, any Monday calendar flag): zero Monday artifacts exist.
- **Event-class calendar claims** (D-17): ~3 FOMC, ~3 OpEx, ~3 month-ends — n≈3 per cell.
- **Changepoint lead-time** (M-04): ~1-2 genuine breaks, all at the window's left edge.
- **Rates-vol / bond-stress inputs** (D-02, D-21's bear-steepener case): no rates event in window.
- **Anything needing new artifacts** (S-11, S-12): undefined on this window by construction.

## BASELINE CHECK AUDIT

1. **[CRITICAL — bounds the verdict, discovered in data] The "stored deep-ensemble arm" is the rule labeler for 66% of the window.** 128/194 stored inference days (2025-08-04→2026-02-05) are one-hot fallback: conf=1.0, zero embedding, position_size_multiplier=1.0. The full-period E1 "deep vs rules" comparison is rules-vs-rules for most of its length; genuine deep output = ~21 pre-holdout days + 45 holdout days. Not invalidating if conditioned, fatal if not. **Fix (cheap):** condition every deep-vs-rule statistic on deep-era dates (conf<1.0 era); report the fallback share in the manifest; state the verdict as a 66-day comparison, not 235.
2. **[CONFIRMED — would invalidate variant (i) as headline] conf=1.0 hands the rules arm a mechanical sizing win.** Holdout deep psm: mean 0.798, min 0.5; one-hot rules pin psm at the 1.0 clip → ~25% more gross exposure into a +11% rally window. The architect anticipated this with variant (ii) (copy stored conf/disagreement). **Fix:** variant (ii) is the only admissible label-path verdict; variant (i) minus (ii) is reported as the sizing-path decomposition, never as "rules beat deep"; report average gross exposure per arm (protocol already requires it).
3. **[Bounds only] Monthly retraining on growing data.** The deep arm on date t used the most recent pre-t retrain — production-faithful, no lookahead; but the verdict evaluates the production *process*, not a fixed model. **Fix:** record retrain dates in the manifest; no design change needed.
4. **[Real, needs one check] Teacher-version skew.** Recomputed current-code rule labels agree with deep labels 78% on holdout but only **26% on the 19 deep-era pre-holdout days** — either the labeler thresholds/calibration changed since the deployed students were distilled, or context_df reconstruction differs from production context. If the teacher changed, M-01 is still a valid "which regime source decides better" test but the "student vs own teacher / distillation ceiling" narrative is wrong. **Fix (cheap):** pin the labeler version in the deployed models' training manifest and reproduce rule labels from the production context parquet, not ad-hoc recomputation; report both agreement numbers.
5. **[No leak found] Context features.** Labeler inputs are trailing 21-day windows ending at t; fills execute at next_date prices from the snapshot; credit_spread_proxy comes from signal_row computed at t. One watch-item: cache-rebuilt features that didn't exist in production at the time (e.g., vix_term_slope is NaN pre-holdout) must not enter any arm.
6. **[Would overclaim significance] "Exceeds by more than one seed-std" is the wrong noise model.** Five slippage seeds measure fill noise, not window noise; the decision rule must be the paired-daily t≥3.0 standard above. A sub-threshold deep arm does not "lose" statistically — but per the removal asymmetry, a tie licenses killing the torch regime path on cost/complexity grounds with the MDE bound stated.
7. **[Inherited] M-13 health twin** carries flaws 1, 3, 4 identically (AE health in the fallback era is presumably rule-fallback too); same conditioning fix, same manifest disclosures.

Net: the design is sound after repairs 1, 2, 4, 6; none of the flaws requires new infrastructure.

## RECOMMENDED PILOT SLATE

1. **Baseline honesty block: M-01 + M-02 + M-13** (mandatory slot) — with the four repairs above; sets the bar every future regime/health candidate must clear, and a null result removes torch from the Lambda path. (~5-6 holdout looks.)
2. **M-14 ranking-blend sweep** — the only real-label model in the stack, wiring already exists, sweep pre-holdout + one pre-registered holdout confirm; precondition: verify RankingMLP training cutoff < 2026-03-11 or downgrade to rank-IC-only. (1 look.)
3. **S-01 cash sleeve (SHY, with S-16 SHY+IEF ladder arm)** — deterministic carry, hardest candidate to lose with; replay is a harm check, carry claimed by arithmetic, not by the window. (2 looks.)
4. **S-05 + S-10 universe-config bundle** — two one-line config changes, one replay run, tie-favors-removal standard; settles the standing VIXY question jointly with PKT-TB-004. (2 looks.)

Zero-holdout-look riders (ship without evidence claims): D-10 GDELT path fix (bug repair); FRED/Yahoo ingest bundle D-01/D-02/D-04/D-05/D-07/D-21 (accrue for next window, model D-07's 2-day lag); M-05/M-16 offline audits on the 66 genuine deep-era days; S-11/S-12 universe-add artifact clock start.

Total looks: ~10-11 of the 12-look budget.

## DISSENT-WORTHY POSITIONS

1. **The stored-inference fallback discovery (128/194 one-hot days) is a run-level finding that outranks any single candidate** and must appear in the committee report regardless of pilot outcomes: every prior analysis that treated the stored 235-day inference series as "the deep ensemble's track record" — including any champion-selection evidence — was looking at the rule labeler for two-thirds of it.
2. **The holdout has already been partially mined.** The S-03 overlay's trigger value was tuned against this same holdout in prior champion work; I dissent from any framing of a fresh S-03 holdout read as confirmation, and more broadly from treating 2026-03-11 as a virgin boundary — this run's reads inherit an unknown number of prior looks, which strengthens, not weakens, the t≥3.0 standard.
3. **Data-ingest candidates should never be charged holdout looks, and conversely must never claim holdout evidence.** I demoted eight "pilot-now" data candidates to ship-ingest-only on this principle; if the committee instead replays them this run, I dissent: with 45 perforated days the reads are pre-destined to be noise and will burn the window's look budget on unanswerable questions.
4. **The holdout artifact gaps (zero Mondays, 13-day May hole) are a pipeline reliability finding** that should route to operations: ~30% of holdout trading days have no stored decision artifact, which silently biases every window-level metric the system reports, not just this packet's.
5. **If the baseline block ends in a statistical tie, I will read it as a kill of the torch regime path,** per the removal asymmetry (tie + complexity cost ⇒ simpler arm wins). Committee members who require affirmative proof of harm to remove a layer should register the opposing position now, before results exist.
