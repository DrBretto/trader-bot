# PKT-TB-005 — Pilot Evidence

Per `committee/EVIDENCE_PROTOCOL.md`. Harness: `pilots/common.py` — a verbatim
copy of `optimizer/replay.py:run_replay_for_dates` (production decision engine,
production fill/cost model, unmodified) with recording + pilot hooks; validated
**byte-identical** to the production function on the holdout segment before any
pilot ran. Data: pinned local cache (`pilots/data_cache.pkl`) of S3 daily
artifacts, 194 aligned snapshots 2025-08-04→2026-06-09. Seeds fixed
(20260217+). Manifests per pilot in `manifests/`.

**Segments.** E1 = full cached window (194 decision days; in-sample for the
active bundle, which was tuned on parts of it). E2 = holdout, decision dates ≥
2026-03-11 (45 stored days). **Holdout coverage caveat (applies to every
number below): 45 of ~65 trading days (~70%), zero Mondays, a 13-day artifact
hole 2026-05-08→05-21; the window is rally-dominated (SPY +11.2%, max
intra-window drawdown -2.5%).**

**Evidence standard (Skeptic rule, binding).** Primary = paired daily-return
deltas on identical dates. t ≥ 3.0 AND mean ≥ +0.05%/day = shows something;
t ∈ [2,3) = suggestive; t < 2 = consistent with noise. Endpoint/Sharpe deltas
are never primary. Removal asymmetry: for simplifications, a noise result is
bounded evidence of harmlessness; the simpler arm wins ties. Holdout arm-reads
consumed this run: **11 of the 12-look budget** (battery 5, sleeve 2, universe
3, ranking contrast 1; rank-IC 0).

**Production-faithfulness note.** The active bundle
(`beat-champion-participation-2026-06-06-v1`) carries `ranking_blend: 0.35`;
the canon control replays WITH it (battery v2 onward). The v1 no-ranking
battery is preserved as a logged variant
(`pilots/results_baseline_check_NORANKING_v1.json`) and turned out to be
load-bearing — see Pilot 4. This harness measures the pipeline-faithful
optimizer object, NOT the dashboard line (`three_line_replay` + champion
overlays + real 3/11 portfolio seed); absolute returns differ between the two
objects by construction (see `docs/BEAT_CHAMPIN_20260606.md` §3 — sic, the
beat-champion doc); all verdicts here are arm-vs-arm within one object.

---

## Pilot 1 — Mandatory baseline check (M-01 + M-02), E1+E2

**Question.** Does the GRU+Transformer regime ensemble beat its own teacher —
the 5-rule threshold labeler it was distilled from?

**Design** (Model Architect; Skeptic repairs R1–R4 applied): arms swap the
per-date `inference['regime']` with no lookahead (labeler inputs are
production-computed trailing context features from each date's stored
context.parquet). `rules_storedsizing` (rule labels + stored
confidence/disagreement/psm) is the **only admissible label-path verdict**;
`rules_raw` (one-hot, conf=1) measures label+sizing jointly; `rules_ema`
halflife was selected on PRE-HOLDOUT only (hl=3 in v2). EMA sweep variants all
logged in the results file.

**Run-level discovery (R1):** 128/194 stored inference days are rule-fallback
one-hots (conf=1.0) — the stored "deep ensemble" series IS the rule labeler
until ~2026-01-31. All 45 holdout days are genuine deep output; only 21
pre-holdout deep days exist. Deep-vs-rule stats are reported conditioned on
deep-era dates. **Teacher-version skew (R3):** current-code rule labels agree
with stored deep labels 77.8% on holdout but only 28.6% on the 21 deep-era
pre-holdout days — the deployed students' teacher was likely a different
labeler calibration; the pilot therefore answers "which regime source decides
better," not strictly "student vs its own teacher."

**Results (v2, production-faithful, seed 20260217; full battery + paired stats
in `pilots/results_baseline_check.json`):**

| arm | E1 full ret | E1 maxDD | E2 holdout ret | E2 maxDD | paired-vs-control holdout t | deep-era-only full t |
|---|---|---|---|---|---|---|
| ens_stored (control) | +4.58% | -12.29% | -5.87% | -12.21% | — | — |
| rules_storedsizing (admissible) | -1.85% | -10.45% | -7.73% | -10.23% | **-0.68** | -1.71 |
| rules_raw | -7.02% | -14.86% | -9.60% | -14.25% | -1.05 | -2.57 |
| rules_ema (hl3) | +6.85% | -4.57% | -0.26% | -3.82% | +0.47 | +0.13 |
| ens_nodis (no disagreement penalty) | +6.01% | -12.10% | -4.61% | -12.03% | +0.58 | +0.60 |

v1 contrast (no ranking layer; logged variant): control holdout +1.63%,
rules_raw +4.21% (paired t **+0.91**), rules_ema(hl5) -0.46% (its pre-holdout
selection did not transfer — honest selection-trap datapoint).

**Verdict (per the binding standard): statistical tie.** No arm reaches |t| ≥
2.0 on the holdout in either harness variant, and the SIGN of the rules-vs-deep
delta flips between the no-ranking (+0.91 favoring rules) and with-ranking
(-0.68 favoring deep) variants. The ensemble **does not beat** the dumb
baseline by any measurable margin; neither does the baseline measurably beat
the ensemble. MDE bound: with sd(daily delta) ≈ 0.49% (storedsizing arm), any
true difference larger than ~±0.15%/day (≈±37pp/yr) is excluded at t=2; smaller
real differences are invisible on this window. Per the Skeptic's pre-registered
removal asymmetry, the tie is kill-license for the torch regime path on
complexity grounds; the opposing position (require affirmative harm before
removal) is recorded in COMMITTEE_REPORT Dissent.

Multiple-candidates note (protocol): 5 battery arms read the holdout once each;
the EMA arm additionally consumed a 3-value pre-holdout selection sweep
(variants logged); no arm was selected on the holdout.

## Pilot 2 — Health-model honesty check (M-13), E1+E2

Same harness, arm `health_rules`: per-date `asset_health` replaced by the
rule-based composite (`src/models/baseline_health.py`, cross-sectional ranks of
momentum/vol/drawdown/relative-strength from each date's stored features).

| arm | E1 full ret | E1 maxDD | E2 holdout ret | E2 maxDD | paired holdout t |
|---|---|---|---|---|---|
| ens_stored (trained AE health) | +4.58% | -12.29% | -5.87% | -12.21% | — |
| health_rules | **+13.43%** | **-3.56%** | **+8.63%** | -2.26% | **+1.18** |

**Verdict: consistent with noise (t=+1.18 < 2), direction strongly favors the
rule health.** The rule-health arm beat the trained autoencoder by +8.8pp E1 /
+14.5pp E2 endpoint with a quarter of the drawdown — but the daily-level test
does not clear even the suggestive bar on 45 days, so per the standard this is
a direction, not a result. Because swapping AE→rules is a simplification
(torch artifact exits the image), the removal asymmetry applies: the burden now
sits on the AE to show it beats the free rules on the next window. Flagged as
the single most promising follow-on read in ROADMAP (rides the baseline-block
second read, 1.2).

## Pilot 3 — Ranking model (M-14): rank-IC only + paired layer contrast

**Precondition failure (Skeptic):** the active RankingMLP
(`models/ranking_expanded_unconditioned`) was trained **2026-04-29** — inside
the holdout window. A holdout replay read of blend-on-vs-off is contaminated
*in favor of* the ranking arm; the planned sweep was downgraded.

**Rank-IC (`pilots/results_ranking_ic.json`, 0 looks):** Spearman IC of model
scores vs realized forward 21d returns: **in-training window IC = +0.20
(t=9.3, n=149 dates)**; on dates whose forward window crosses the training
boundary IC = **-0.00 (t=-0.04, n=24)**; **zero** clean post-training dates
exist at the 21d horizon (forward windows + the May artifact hole consume them
all). Exploratory short horizons (target mismatch, labeled as such):
post-training 5d IC +0.14 (t=2.0, n=12), 10d IC +0.14 (t=1.8, n=7) —
suggestive that some cross-sectional signal survives, far from proof.

**Paired layer contrast (`pilots/results_ranking_contrast.json`, 1 look,
blend 0.35 vs 0, same bundle/seed/data):**

| arm | E1 full ret | E1 Sharpe | E1 maxDD | E2 holdout ret | E2 maxDD |
|---|---|---|---|---|---|
| blend 0.35 (production) | +4.58% | 0.45 | **-12.29%** | -5.87% | -12.21% |
| blend 0 | +9.08% | 1.32 | **-4.02%** | +1.64% | -1.55% |

Paired daily t: full **-0.30**, holdout **-0.65** — consistent with noise at
the daily level (sd ≈ 0.85-1.6%/day; the damage concentrates in a handful of
days). Endpoint and path: the ranking layer cost -4.5pp full-period, -7.5pp
holdout, and **tripled max drawdown** on this harness — while the contamination
asymmetry runs in its favor.

**Verdict:** the only real-label model in the stack has **no clean
out-of-sample evidence**, in-sample IC that collapses at its training boundary,
and a markedly worse drawdown path when active on the pipeline-faithful
harness. Not statistically condemned (t < 2) — but it is LIVE in production at
blend 0.35 with nothing supporting it. Routed to the operator as the most
urgent decision input of this run (ROADMAP item 1.1 pre-designs the clean
read; the production-keep/disable decision is the operator's, per the packet's
no-production-action constraint).

## Pilot 4 — Cash sleeve S-01 (+S-16 ladder arm), E1+E2 harm check

**Framing (binding):** the carry is arithmetic (T-bill yield on the 10-40%
structurally idle reserve); the replay is a **harm check only**.

**Wiring:** engine sees sleeve as cash (portfolio view; sleeve excluded from
scoring/holdings count); intent hook sells sleeve before engine buys, sweeps
idle cash above a $500 buffer into SHY (or SHY/IEF 50/50).

**Two wiring bugs found and preserved as logged variants:**
- v1 (`results_cash_sleeve_PARTIALSELL_BUG_v1.json`): -97.5% holdout —
  exposed a **production landmine**: `paper_trader.execute_trade` SELL credits
  cash for the sold shares but **pops the entire holding**, so any partial
  SELL silently destroys the remainder. Production is safe only because engine
  sells are always full-position today.
- v2 (`results_cash_sleeve_FRAGMENT_BUG_v2.json`): -83% — holding-fragment
  mismatch (each sweep appends a new holding; SELLs must be per-fragment).
- v3 (final, `results_cash_sleeve.json`): per-fragment full liquidation.

**Results (v3):**

| arm | E1 full ret | E1 costs | E2 holdout ret | paired full t | paired holdout t |
|---|---|---|---|---|---|
| control | +4.58% | $518 | -5.87% | — | — |
| sleeve_shy | +2.44% | **$1,429** | -6.93% | **-2.32** | -1.78 |
| sleeve_ladder | +2.43% | $1,462 | -7.39% | -1.35 | -1.32 |

**Verdict: harm detected, attributable to the pilot wiring, not the concept.**
The forced full-liquidate-and-resweep churn (the only safe pattern given the
partial-SELL landmine) tripled transaction costs and produced a suggestive-harm
read (t=-2.32 full period). A ship-shaped sleeve (clean partial sells) would
not churn this way; its cost profile cannot be measured until the
`paper_trader` partial-SELL bug is fixed. **S-01 verdict updated:
parked-promising — revival condition: partial-SELL fix lands (bug lane), then
re-run this pilot's v3 design with partial liquidation.** The carry argument
stands on arithmetic; this window could never have measured it anyway
(Skeptic MDE).

## Pilot 5 — Universe config bundle: S-05 VIXY ejection + S-10 dedup, E1+E2

**Pre-check (control fills):** VIXY traded **3 times** in 194 days; duplicate
pairs co-held on **2 days** (SMH+SOXX only). Both questions are near-non-events
in practice — which is itself the finding.

**Results (`pilots/results_universe_config.json`):**

| arm | E1 full ret | E2 holdout ret | paired full t | paired holdout t |
|---|---|---|---|---|
| control | +4.58% | -5.87% | — | — |
| vixy_off | +4.61% | -5.85% | +0.04 | +0.04 |
| dedup (9 tickers off) | +4.68% | -5.87% | +0.19 | 0.00 |
| both | +4.70% | -5.85% | +0.16 | +0.04 |

**Verdict: clean tie → removals win** (pre-registered tie-favors-removal
standard). Harmlessness bound: any true cost of removal exceeding ~0.06%/day
(≈15pp/yr) is excluded at t=2; observed deltas are ≈0 and weakly positive.
S-05 and S-10 are **proven harmless and structurally simplifying** (VIXY's
toxic post-spike buy path closed; 9 duplicate tickers freed; ~8 fewer nightly
fetches). The VIXY *hedge-value* half of the question remains untestable on
this window (no crash day) and is coordinated with PKT-TB-004 per the atlas.

## Offline audits (0 looks, `pilots/results_offline_audits.json`)

- **M-16 (disagreement validity), n=66 deep-era days:** disagreement vs forward
  5d vol: ρ = -0.08 (nothing); disagreement vs regime-label flip within 5d:
  **ρ = +0.37, p = 0.003** — disagreement DOES flag imminent label instability.
  Direction: **keep the throttle** (contra the deletion hypothesis); battery's
  ens_nodis arm agrees (removing the penalty: t=+0.58, noise). Underpowered;
  powered re-read next quarter (ROADMAP 1.4).
- **M-05 (member redundancy), n=66:** gru-vs-transformer prob-vector
  correlation: median 0.87 but mean 0.46, p10 = -0.31 — the members genuinely
  diverge on a meaningful fraction of days; member-retirement is NOT supported.

---

## Addendum pilot 6 — Corrected-models counterfactual (operator-requested), E1+E2

**Question.** How would the stretch have gone had the models been trained
right? (All deep-era stored days ran short-corpus-mistrained models; the
corrected 2026-06-06 retrain — regime GRU 9%→73% OOS — has one stored day.)

**Design.** Per-date inference regenerated with the CURRENT production models
loaded via the production `ModelLoader` (regime `ensemble_v20260606`; health
resolved to the `health_v20260601.pkl` artifact — only regime was retrained on
2026-06-06, per BEAT_CHAMPION "health model unchanged"), from each date's
stored context/features (as-of-date inputs only), swapped via the harness
hook; production-faithful in every respect including the discovery below.
Ranking layer (blend 0.35) identical in both arms. In-sample caveat: corrected
models trained on the 11-year corpus through ~2026-02-03 (assumption; pkl does
not record the range) → days ≤ 2026-02-03 are in-sample for them; the holdout
is entirely post-training = the clean read. 12th and final holdout arm-read.

**Results (`pilots/results_corrected_models.json`):**

| arm | E1 full ret | E1 maxDD | E2 holdout ret | E2 maxDD | paired t full | paired t holdout | paired t post-training-only (n=66) |
|---|---|---|---|---|---|---|---|
| stored record (mistrained) | +4.58% | -12.29% | -5.87% | -12.21% | — | — | — |
| corrected 2026-06-06 models | +0.11% | -9.03% | -5.33% | -8.08% | -0.26 | **+0.009** | -0.34 |

Label agreement: corrected-vs-stored 80.9%, corrected-vs-rules 56.7%; mean
disagreement 0.354 (throttle active ~daily), mean confidence 0.77.

**Verdict: the corrected models replay the same stretch statistically
identically to the mistrained record** (holdout paired t = 0.009 — as close to
a coin-flip as the harness can produce; drawdown path modestly better, endpoint
modestly worse, all within noise). The 9%→73% label-accuracy improvement does
not cash out in portfolio terms on this window — consistent with the
baseline-check tie: the regime label path is not where this system's P&L is
made or lost, at least under the current bundle and its ranking layer. The
operator's hypothetical is answered: **the stretch would have gone essentially
the same.**

**New run-level finding #7 (production inference tiling).**
`ModelLoader.predict_regime` builds the model input by TILING the single
current day's context row 21× (`np.tile(features, (seq_length, 1))`) — the
GRU/Transformer, trained on real 21-day sequences, have never been fed a real
temporal sequence at production inference. Every sequence-modeling capacity
argument for the deep pair is moot under this serving path; fixing it (feed
the actual trailing 21-day context window, which the pipeline has) is a bug-fix
lane candidate that would need its own E1/E2 read.

## Final line

All six pilots carry E1+E2 manifests (`manifests/`); every variant including
three bugged/superseded runs is logged; 12/12 pre-registered holdout arm-reads
consumed (budget exhausted — no further holdout reads this window); no pilot result exceeds the t≥3.0 evidence bar — every directional
claim above is graded suggestive-or-noise and parked accordingly; the S-01
sleeve pilot reports E1+E2 numbers whose harm component is attributed to a
pilot-wiring constraint (production partial-SELL landmine), so its
concept-level verdict is evidence-incomplete pending that fix.
