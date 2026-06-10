# Portfolio Strategist — panel return (verbatim subagent output)

Substrate review done: read `src/steps/decision_engine.py` (1083 lines), `optimizer/replay.py`, `src/steps/paper_trader.py`, `config/universe.csv`, `config/decision_params.active.json`, and the three_line_replay overlay stack. Key grounding facts used below: buy loop hard-excludes already-held symbols (engine cannot add to winners — top-up exists only as a `three_line_replay` overlay, not in the live engine); all 65 universe symbols have `leverage_flag=0` so the leveraged param stack is dead code; idle cash earns nothing while `min_cash_reserve_by_regime` forces 10–40% idle; replay takes an arbitrary `decision_dates` list (cadence experiments are cheap); `regime_compatibility` multipliers can already express 0.0 (hard exclusion).

### [S-01] Cash sleeve: park idle cash in SHY/BIL
- hypothesis: the book can hold "cash that earns the risk-free rate" instead of literal zero-yield cash — today the 10–40% regime reserve plus any undeployed residual is pure drag, worst exactly in risk_off/panic when the reserve is 40%.
- mechanism: SHY returned ~4-5% annualized over the replay window with near-zero drawdown; sweeping the post-reserve residual AND the reserve itself into SHY (counted as cash-equivalent by the reserve gate, sold first in the sizing cascade) converts a structural 10–40% dead allocation into T-bill carry with almost no added risk.
- testability: replay-now — SHY is in the universe with full stored price/feature artifacts; wiring = new sleeve logic in `decision_engine.run` (post-buy-loop sweep + sell-SHY-first when buys need cash) run through `optimizer/replay.py` unchanged.
- integration: decision-engine logic change (small, ~40 lines) + one param (`cash_sleeve_symbol`, `cash_sleeve_enabled`).
- evidence path to E2: replay full 235 dates with sleeve on vs active champion bundle; win = holdout (2026-03-11→2026-06-10) return ≥ champion + ~0.3pp with maxdd within 0.2pp and no panic-window degradation (sleeve must liquidate cleanly into buys).
- suggested verdict: pilot-now

### [S-02] Core-satellite: permanent beta core + active 8-position satellite
- hypothesis: a baseline market participation the scoring stack cannot take away — today the engine can sit in cash through a rally if nothing clears the buy bar (the documented 2026-05 cash-during-rally failure, currently patched only by threshold tuning).
- mechanism: a fixed core (e.g., 30–40% SPY or SPY/IEF) held outside the 8-slot/sell-trigger machinery guarantees rally participation while the satellite expresses regime alpha; this is a shape (exposure floor) that no amount of threshold re-tuning reproduces because thresholds still gate to zero when scores are weak.
- testability: replay-now — SPY/IEF stored; wiring = sleeve logic (core bought day 1, excluded from holdings count, cluster caps, and sell triggers except panic) piloted in a run dir against the replay harness.
- integration: new sleeve logic in decision engine + param bundle (core weights, panic-exempt flag).
- evidence path to E2: replay with core 0%/20%/40%; win = holdout return beats champion with maxdd penalty < half the return gain, AND the 2026-03-25..04-10 down-window maxdd stays within ~1pp (the core will hurt here — that's the honest cost being measured).
- suggested verdict: pilot-now

### [S-03] Engine-native top-up of held winners
- hypothesis: adding to an existing position on strength — `filter_buy_candidates` hard-filters `~isin(current_holdings)`, so the live engine literally cannot increase a winner; the only top-up is the `topup_on_psm_rise` overlay in `src/utils/three_line_replay/extender.py`, which the champion's own evidence says is load-bearing (trigger 1.2→1.1 was part of the beat-champion win) but which doesn't exist in the production decision path.
- mechanism: drifted-down winners with rising scores re-earn capital instead of the book carrying stale half-size positions; the overlay evidence already suggests positive value, this ports the shape into the canonical engine where the optimizer can tune it.
- testability: replay-now — pure engine logic change replayed on stored artifacts; the overlay version provides a prior to beat.
- integration: decision-engine logic change (allow held symbols through the buy filter when current weight < target × topup_trigger; clamp by headroom) + 1–2 params.
- evidence path to E2: replay engine-native top-up vs champion-with-overlay; win = matches or beats the overlay's holdout +11.85% line while removing the overlay (one decision path, optimizer-tunable).
- suggested verdict: pilot-now

### [S-04] Explicit hedge sleeve with a regime/fragility-keyed budget
- hypothesis: holding a hedge BEFORE it scores well — today TLT/GLD/UUP/VIXY enter only by clearing the same momentum/health buy bar as everything else, so the book buys protection after vol has already spiked (hedge-as-trend-follower, which is backwards for crisis convexity).
- mechanism: a budget (e.g., 0% calm → 10–15% risk_off/fragile) allocated to a fixed hedge basket regardless of score threshold gives the book negative-correlation exposure during transitions, the one thing score-gated buying structurally cannot express; honest caveat — in the mostly-benign replay window this will likely cost carry, so the test is really about the down-window.
- testability: replay-now — all hedge candidates have stored artifacts; wiring = sleeve logic piloted in a run dir.
- integration: new sleeve logic + param bundle (basket, budget-by-regime, fragility trigger).
- evidence path to E2: replay budget {0, 5, 10, 15}% keyed to regime and to `fragility_score`; win = down-window (2026-03-25..04-10) maxdd improves ≥0.5pp at total holdout return cost ≤0.5pp; if the window is too benign to discriminate, park with that finding.
- suggested verdict: parked-promising (revival: PKT-TB-004 attribution shows the existing throttle stack does NOT already provide cheap downside control — don't build a second brake before knowing what the first one does)

### [S-05] VIXY: remove from the scored universe (hedge-sleeve-only or out)
- hypothesis: nothing — this REMOVES a mis-expressed shape; VIXY's contango decay makes it a structurally negative-carry asset that the health/momentum scorer can only ever buy after a spike (selling low buying high), and panic compat 1.3 actively boosts it at the worst entry point.
- mechanism: a decay asset scored by the same machinery as SPY is a category error; either it lives in an explicit hedge sleeve (S-04) with its own entry/exit or it should be ineligible — leaving it scored is silent tail risk in every panic regime.
- testability: replay-now — flip `eligible` 0 in universe config (or compat→0.0), replay both ways; also query stored fills to see if VIXY ever actually traded (it may be a non-event, which is itself the answer).
- integration: universe config (one cell).
- evidence path to E2: replay eligible-on vs eligible-off over full window + holdout; win = off ≥ on (any tie favors removal — fewer toxic states); coordinate verdict with PKT-TB-004's attribution of VIXY's historical contribution.
- suggested verdict: pilot-now (cheapest test in this entire list)

### [S-06] Regime-conditional universe eligibility masks
- hypothesis: hard per-regime universes (e.g., risk_off universe excludes ARKK/XBI/KRE entirely).
- mechanism: weak — `regime_compatibility` multipliers already express this continuously and can be set to 0.0 for hard exclusion, and the panic asset-class filter already exists in `filter_buy_candidates`; a binary mask adds zero expressiveness over existing knobs, and tuning those knobs is PKT-TB-002's lane.
- testability: replay-now (it's just compat values), but that's the point — it's not a new shape.
- integration: param bundle (already).
- evidence path to E2: n/a — any pilot here is an optimizer sweep, not a strategy-shape pilot.
- suggested verdict: killed (fully expressible via existing regime_compatibility multipliers; re-tuning owned by PKT-TB-002)

### [S-07] Conviction-weighted sizing (score-proportional position sizes)
- hypothesis: a book where weight reflects relative conviction — today `compute_position_size` starts every buy at `max_position_weight` × shared multipliers; score determines ONLY queue order, so the #1 and #8 candidates get identical size.
- mechanism: scaling base weight by score rank or score-above-threshold concentrates capital in the strongest signals; honest caveat — value depends entirely on whether `final_score` magnitude (vs rank) carries information, which the ranking-model blend may or may not provide, so this could easily be noise-fitting.
- testability: replay-now — engine logic change (one line in sizing: `base_dollars *= f(score)`), replayed on stored artifacts.
- integration: decision-engine logic change + param (sizing curve).
- evidence path to E2: replay flat vs linear-in-score vs rank-step sizing; win = holdout return improvement that survives BOTH the pre-holdout and holdout windows (single-window wins here are presumptively overfit).
- suggested verdict: parked-promising (revival: evidence from PKT-TB-002/ranking work that score magnitude is calibrated, not just ordinal)

### [S-08] Correlation-aware marginal selection (diversification bonus at buy time)
- hypothesis: "is this candidate a NEW bet or the same bet again?" — the greedy buy loop takes top scores and only the static cluster map (`DEFAULT_SECTOR_CLUSTERS`, currently opt-in and OFF in the active bundle: `sector_clusters` not set) restrains correlated stacking; the 2026-06-05 growth book (four labels, ~65% one correlated bet) is the documented pathology.
- mechanism: penalizing a candidate's score by its trailing-60d correlation to the current book is a dynamic version of the cluster cap that adapts to actual correlations instead of a hand-written label map, and unlike the cap it shapes WHICH symbol fills a slot rather than just clamping dollars.
- testability: replay-now — correlations computable from stored daily prices inside the engine; pilot code in a run dir (needs a price-history window passed in, which `features_df` already carries).
- integration: decision-engine logic change (scoring adjustment) + params (lookback, penalty scale).
- evidence path to E2: replay penalty {0, 0.1, 0.2} vs champion AND vs simply turning the existing cluster map ON (the cheap incumbent it must beat); win = better holdout return/maxdd than both.
- suggested verdict: pilot-now (but run the cluster-map-ON arm first — if that captures most of the value, this gets killed as over-engineering)

### [S-09] Split cadence: daily sells, slower buys
- hypothesis: asymmetric reaction speed — today buys and sells both fire daily, so choppy regimes churn entries that trailing stops then whipsaw out (paying spread+slippage both ways); the engine cannot express "stay defensive intraday-week but only commit new capital weekly."
- mechanism: risk-off actions (stops, health collapse, panic) need daily latency, but entry edge at daily granularity is plausibly noise — gating BUY generation to N-day intervals (or to regime-change days) cuts turnover where transaction costs are measurably modeled in `paper_trader.apply_transaction_costs`.
- testability: replay-now — `run_replay_for_dates` already accepts arbitrary date lists, but the clean version is a pilot flag in the engine (suppress buys unless date % N == 0 or regime changed), replayed on stored artifacts.
- integration: decision-engine logic change (buy-gate) + param (buy_cadence_days / on_regime_change).
- evidence path to E2: replay buy-cadence {1 (control), 3, 5, regime-change-only}; win = holdout return within 0.3pp of daily while cumulative transaction costs drop ≥30%, or return improves outright in choppy-dominated sub-windows.
- suggested verdict: pilot-now

### [S-10] Universe dedup: disable near-duplicate tickers
- hypothesis: position slots that mean something — SPY/VOO/IVV/VTI, AGG/BND, VNQ/IYR, SOXX/SMH, XBI/IBB, EFA/VEA, EEM/VWO are 7 duplicate pairs/quads; with only 8 slots, two slots on SPY+VOO is one bet wearing two hats, defeating position-count diversification even where cluster dollar-caps clamp size.
- mechanism: marking duplicates `eligible=0` (keep the most liquid of each set) frees slots for genuinely distinct exposures at zero information cost; weak-but-honest note — duplicates only burn slots if the scorer actually buys both, so check fills first.
- testability: replay-now — universe config edit (eligible flags), replay; pre-check with a one-liner against stored fills for duplicate co-holding frequency.
- integration: universe config.
- evidence path to E2: replay deduped (65→~57 effective) vs control; win = holdout ≥ control (a tie still wins on book legibility); if duplicates never co-held, record null result and close.
- suggested verdict: pilot-now (near-zero cost)

### [S-11] Universe expansion: leveraged ETFs (SSO/QLD/UPRO/TQQQ)
- hypothesis: convex risk-on expression in calm_uptrend — the engine carries a complete, currently-DEAD leveraged stack (`leverage_flag` column, `trailing_stop_leveraged` 6%, 10-day hold cap, `leveraged_constraints.max_weight` 0.10) with zero `leverage_flag=1` symbols in universe.csv to use it.
- mechanism: a 10%-capped, 10-day-capped 2x sleeve in confirmed uptrends is the exact shape the dead params were built for, and the regime classifier is the gating signal that makes it defensible; real value claim is honest but unverifiable retroactively.
- testability: needs-new-data — stored daily artifacts (prices/features/health/inference) do not exist for symbols before they're added; pipeline-faithful replay evidence is impossible until the symbols accrue live artifacts (free daily prices are fetchable, but health/inference artifacts depend on the model pipeline, so backfill = building a feature/inference backfill harness, which is a project not a pilot).
- integration: universe config + (only then) param bundle.
- evidence path to E2: add symbols to universe now, let artifacts accrue, evaluate after ~60 live trading days of forward sim; interim proxy (non-pipeline-faithful) replay using raw price history is possible but must be labeled as such.
- suggested verdict: parked-promising (revival: 60+ days of accrued live artifacts; cheap to start the clock now since universe adds cost nothing)

### [S-12] Universe expansion: diversifier ETFs (DBMF managed futures, JEPI covered-call, BTAL anti-beta)
- hypothesis: return streams structurally uncorrelated to everything in the current 65 (which is equity+rates+commodity beta wall-to-wall) — a genuinely new axis, unlike S-11 which is more of an existing axis.
- mechanism: managed-futures and anti-beta ETFs historically pay off in exactly the regimes where this book's only defense is idle cash; honest caveat — daily-cadence health scoring may chronically under-score these low-momentum assets, so they may never clear the buy bar without compat support.
- testability: needs-new-data — same artifact-accrual problem as S-11, per candidate symbol.
- integration: universe config + regime_compatibility entries for the new sectors.
- evidence path to E2: same as S-11 (add now, accrue artifacts, forward-evaluate); precondition check is free — confirm the scorer CAN like such assets by inspecting SHY/GLD score histories in stored artifacts.
- suggested verdict: parked-promising (revival: same artifact clock as S-11; start it in the same config commit)

### [S-13] Book-level realized-vol targeting
- hypothesis: sizing the whole book off realized portfolio vol (target e.g. 10% annualized) instead of regime-label proxies.
- mechanism: weak in this stack — the regime multipliers, ensemble multiplier, fragility/entropy modifiers, risk throttle, AND the optional gross-exposure cap are already five overlapping exposure brakes; adding a sixth before PKT-TB-004 attributes what the existing five do is exactly how this system accumulated five.
- testability: replay-now technically (realized vol computable from the replay portfolio path; engine change to feed it back), but attribution must come first.
- integration: decision-engine logic change (feedback loop — new state).
- evidence path to E2: only after PKT-TB-004: if attribution shows the throttle stack is redundant/incoherent, replace (not augment) with vol-targeting and replay; win = same maxdd profile with fewer knobs.
- suggested verdict: parked-promising (revival: PKT-TB-004 finds the existing throttle stack incoherent; this is then a simplification candidate, not an addition)

### [S-14] Minimum-deployment floor by regime (forced participation)
- hypothesis: "never less than X% invested in calm_uptrend" as a hard floor, buying best-available even below the score threshold.
- mechanism: addresses the same cash-during-rally failure as S-02 but by overriding the scorer's veto rather than carving out a core — which means in a true low-score environment it forces the book into its own weakest ideas; S-02 achieves the participation floor without ever buying a below-threshold name.
- testability: replay-now — engine logic change (post-buy-loop: if deployed < floor, take next candidates ignoring threshold).
- integration: decision-engine logic change + param (floor-by-regime).
- evidence path to E2: only as a comparison arm inside the S-02 pilot (core vs floor vs control), not its own pilot.
- suggested verdict: killed (dominated by S-02 — same expressiveness gain, worse failure mode; keep only as an S-02 comparison arm)

### [S-15] Scale-in entries (half position now, half on N-day confirmation)
- hypothesis: path-dependent entries — today every buy is all-at-once at full computed size; the book cannot express "probe then commit."
- mechanism: halves the cost of whipsaw entries in choppy regimes (enter, stop out 3 days later is a recurring fill pattern worth confirming in stored fills) at the cost of half-missing fast winners; net sign is genuinely unclear, which is what the replay is for.
- testability: replay-now — engine logic change (buy at 0.5× size, tag holding, top-up to full if health/score holds after N days — note this REQUIRES the S-03 top-up plumbing, so it sequences behind it).
- integration: decision-engine logic change + params (probe fraction, confirm days).
- evidence path to E2: after S-03 lands: replay probe {0.5, 1.0(control)} × confirm {2,3,5}; win = holdout return ≥ control with stop-out count down ≥20%.
- suggested verdict: parked-promising (revival: S-03 pilots successfully — this is a one-param extension of its plumbing)

### [S-16] Reserve-as-ladder: regime cash reserve held as SHY/IEF duration ladder
- hypothesis: the 40% risk_off/panic reserve as a defensive ALLOCATION (short+intermediate treasuries) rather than dead cash.
- mechanism: mostly subsumed by S-01 (SHY sleeve) — the marginal claim is that IEF adds rate-rally convexity in risk-off, but IEF is already buyable by the scorer in those regimes with compat 1.1–1.2, so only the SHY leg is structurally inexpressible today.
- testability: replay-now (same wiring as S-01 with a two-asset sleeve).
- integration: sleeve logic (S-01 variant).
- evidence path to E2: an arm inside the S-01 pilot (SHY-only vs SHY+IEF ladder), not its own pilot.
- suggested verdict: killed as standalone (folded into S-01 as a pilot arm; IEF exposure is already expressible through the scorer)

STRATEGIST'S TOP PICKS
1. **S-01 cash sleeve (SHY)** — pure structural drag removal, ~40 lines, all data stored, hardest candidate in the list to lose with; run S-16's ladder as an arm.
2. **S-03 engine-native top-up** — the live engine provably lacks a shape the champion's own evidence says is load-bearing (the topup overlay); porting it into the canonical path is the highest-evidence expressiveness gap on the board and unlocks S-15.
3. **S-05 VIXY ejection + S-10 dedup (bundled config pilot)** — two one-line universe-config changes sharing one replay run; cheapest possible E2 evidence, and S-05 settles the standing VIXY question jointly with PKT-TB-004.
