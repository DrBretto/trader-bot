# PROPOSAL_EXPRESSION — PKT-TB-007 Phase A, Expression Architect (role 4)

**Charter:** C1 (exposure parity STRUCTURAL, binding the RISK BUDGET not just gross) and
C3 (the brain must be able to SPEND its signal). Design the expression mechanism(s) by
which the orthogonal brain's outputs become decisions through the incumbent chassis.

**Substrate verified directly (not taken on faith):** `score_candidates` blend
(`src/steps/decision_engine.py:241-254`), `filter_buy_candidates` threshold/health/vol/
LLM-veto/panic filters (`:294-323`), `compute_position_size` takes NO score argument
(`:767-890` — score is selection-only, confirming the Analyst), buy loop ordering +
slots/cash/cluster clamps (`:1263-1343`), replay injection point `_compute_ranking_scores`
(`replay_engine.py:174-187`) and its module-global model cache hazard (`:145-149`),
`run_variant` post_decision hook (`:656-658`), open-fill execution + `position_map`
last-lot-wins (`:298-299`, `:407-477`), TB-006 adapter w_prev/held_shares overwrite bug
(`strategy_adapter.py:320,:335-339`), sanctioned monkeypatch pattern
(`run_replay.py:258-269`).

---

## 1. The expression-channel menu

### 1a. Rank permutation through the ranking socket (Analyst's structural candidate)

**Mechanism.** Brain arm monkeypatches `RE._compute_ranking_scores`. The patched function
(i) calls the original to get the incumbent's deployed-MLP scores, (ii) locally replays the
incumbent's own candidate pipeline (blend at 0.35 → regime multiplier → clip → threshold /
health / vol-bucket / LLM-veto / panic filters — all inputs are in the day's artifacts, so
this counterfactual is EXACT, not estimated), (iii) chooses a permutation of the
incumbent's `final_score` values over the **feasible set** (candidates passing every
score-independent filter), and (iv) solves back for the injected ranking vector:
`r'_i = (final_target_i / mult_i − 0.65·h_i) / 0.35`. Solved `r'` may leave [0,1] — the
engine does not clamp `ranking_scores` values (verified `:243-251`; only `final_score`
clips at `:254`), so the solve is mechanically sound in replay; disclosed as out-of-contract
vs the production-scale convention. Symbols outside the feasible set keep their incumbent
scores untouched; the full 64-symbol dict is always supplied (the 0.5 default at `:245`
never fires).

**What is preserved exactly (by construction, under feasible-set permutation):**
- the multiset of `final_score` values → **count above `buy_score_threshold_by_regime`
  is preserved exactly** (the marginal-buy gross channel A3.1 is closed);
- the health filter, vol-bucket filter, LLM veto, panic filter (score-independent for
  non-high-vol names; see guard below for the 0.80 gate);
- sizing FORMULA inputs that don't depend on symbol identity: PV, regime_adj, PSM,
  throttle (verified: `compute_position_size` has no score argument).

**What still varies (the residual exposure-drift channels, quantified):**
1. **vol_adj composition** (`:807-812`): swapping a low-vol for a high-vol name moves that
   position ±15% (1.10 vs 0.80 around med 1.0) — at deployed `max_position_weight 0.30`,
   up to ±4.5% NAV per buy.
2. **llm_confidence_adj** (`:824`): per-symbol, up to 0.5 → up to −50% on one position
   (live era only; 2025 has no LLM artifacts).
3. **0.80 high-vol gate** (`:307`): score can ADMIT a high-vol-bucket name in
   calm_uptrend — a risk-appetite channel, not an intelligence channel.
4. **Cluster-cap clamps** (`:1309-1315`) bind differently by symbol → second-order gross
   differences.
5. **Path divergence**: different holdings compound through sells/trims/cash forever.

**Intelligence vs risk appetite.** (5) and the WHICH part of (1)-(4) are the point —
picking different symbols IS the brain. What is NOT acceptable is a *systematic* tilt of
the score mass toward higher-beta / higher-vol names: that raises book beta and re-imports
the TB-006 confound in reverse (the Universe Selector's mirror warning: tilting INTO
TLT/AGG/MUB/SHY at matched gross is de-risking in disguise; tilting into VIXY/high-vol is
re-risking in disguise). Either direction makes the verdict measure risk appetite.

**The parity guard (C1, enforced IN THE BRAIN, not the harness).** Because the brain can
compute the incumbent's counterfactual buy set exactly (step ii above), the guard is exact:

- Predict each arm's would-be buy list (symbol, dollars via the full
  `compute_position_size` formula — every input is known pre-trade) for the unpermuted
  and permuted assignments.
- Compute predicted marginal book risk: `Δβ$ = Σ dollars_i·β̂_i` and
  `Δσ$ = Σ dollars_i·σ̂_i`, with `β̂` = trailing 126d OLS beta to SPY and `σ̂` = trailing
  21d realized vol, both point-in-time from the imported OHLCV cache (no leakage).
- **Constraint:** permuted-vs-unpermuted |Δβ$| ≤ 0.10 × unpermuted β$-mass of the buy
  list (and ≥ floor 0.02% NAV·β to avoid zero-division on tiny days), same ±10% band on
  σ$; predicted gross of the buy list within ±1% NAV.
- **Repair:** greedy — revert the single swap contributing most to the violated budget;
  iterate; if no within-band non-identity permutation exists, fall back to identity
  (neutral) and log `guard_infeasible`.
- High-vol gate: the permutation support EXCLUDES high-vol-bucket symbols whose
  feasibility depends on `final_score > 0.80` (channel 3 closed structurally), unless the
  unpermuted assignment already admitted one (then it may be swapped only for another
  already-admitted high-vol name).
- Ex-post diagnostic (pre-registered print, not a knob): realized daily gross gap and
  rolling 21d realized-beta gap between arms; disclosure threshold |Δβ| > 0.05 sustained
  5 days.

**Bandwidth.** Feasible non-held candidates k ≈ 10–25 (live era); decision-relevant
content = which symbols occupy the s executed buy slots (s = min(passers, slots, cash),
typically 0–3): ~log2(C(k,s)·s!) ≈ 6–15 bits on a buy day. **On no-buy days this channel
expresses NOTHING** — a structural C3 ceiling: the brain can only speak when the chassis
happens to be buying (TB-006 incumbent averaged ~1 action/decision-date including sells;
buy days are a minority — measure in Phase C smoke).

**Failure modes:** guard infeasibility on thin candidate days (logged, neutral fallback);
chaos amplification — one swapped marginal buy held for weeks dominates the paired diff
(a single 25%-of-NAV divergent position adds ~30 bp/day to paired sd while diverged, vs
the 6–15 bp tilt regime); the solve produces extreme `r'` when `mult_i` is small
(cap |r'| ≤ 3 and re-repair); model-cache hazard `:147-149` (irrelevant if arms run in
separate processes — REQUIRED, below).

**Replay shape:** brain arm = separate process, `RE._compute_ranking_scores` patched at
runtime (try/finally, the `START_PORTFOLIO_DATE` pattern); incumbent arm = unpatched
process; same seeds, windows, cost overlay (4242).

### 1b. Post-decision intent tilt at matched risk (TB-006 adapter lineage, lot-fixed)

**Mechanism.** A `Strategy.post_decision` that does NOT replace the incumbent's intents
(the TB-006 mistake-context) but **edits them at the margin**: starting from the chassis's
own intents for this arm, reallocate dollars among `support = (chassis buy set ∪ held
names)` — optionally ∪ tilt-core names (the "b-extended" design knob, committee's call):

1. Build lot-aggregated `w_prev`, `held_shares`, NAV from guarded marks (§3 fix).
2. Brain emits a desired long-short tilt `Δw` over the support set with conviction-scaled
   budget `Σ|Δw|/2 = T_t = c_t · T_max` (§2).
3. **Parity projection (by construction, not statistical):** project Δw onto
   `{Σ Δw_i = 0}` (cash/gross unchanged) ∩ `{Σ Δw_i·β̂_i = 0}` (beta-matched) ∩
   `{|Σ Δw_i·σ̂_i| ≤ ε_σ}` (vol-budget-matched) — a 2-equality/1-box projection on ≤ ~15
   names, closed-form or 10-line iterative. Long legs may not exceed per-name cap 5% NAV;
   short legs floored at −w_i (no shorting); no tilt on names the chassis is already
   selling fully.
4. Convert to intent edits: scale chassis BUY dollars up/down; add SELL trims of held
   names; add BUYs (b-extended) — min_order respected, residual rounding imbalance
   (≤ one min_order) absorbed by shrinking the largest buy leg so cash never drifts.
5. Write the expression log (§5) and return the edited intents. The cluster cap
   (`:663-668`) then applies to both arms' intents identically.

This is the bond-tilt warning answered structurally: TLT/AGG/MUB/SHY tilts are forced to
be funded beta-for-beta (a TLT long must pair against low-beta sources or be β-offset by
an equity-side add), so "tilt into bonds" cannot lower book beta vs the incumbent arm.

**Bandwidth.** Δw over ~10–15 support names, quantized at min_order/NAV ≈ 0.25%, budget
T=8%: ≈ 30–60 bits/day, **expressible EVERY day** (trims/topups of held names need no
chassis buy activity). Conviction is continuous. Strictly more expressive than (a) per
day and unconditionally available — this is the C3 carrier.

**Failure modes:** path divergence still exists (the chassis runs tomorrow on the tilted
book — a tilted-in position can trip a stop the incumbent never owns); mitigated because
the tilt engine re-targets daily relative to its OWN arm's chassis decisions and the 5d
horizon naturally unwinds; book-overlap printed daily as a diagnostic. Execution-level
parity is intent-level (fills at next open with gap guard, cash clipping `:430-432`);
realized gross gap printed ex post. min_order quantization eats small-conviction days
(below ~1.5% NAV total tilt at $100k NAV nothing trades — honest dead zone, logged).

**Replay shape:** pure `Strategy` object via the existing harness hook — no engine
monkeypatch needed for the channel itself (the lot fix patches `_execute_intents`, §3,
applied identically to BOTH arms' processes). Separate processes per arm regardless
(uniform discipline + the ranking-model global cache).

### 1c. Hybrid: socket permutation for selection + bounded tilt for conviction

Channel (a) decides WHICH names enter on buy days; channel (b) expresses HOW MUCH
conviction daily at matched risk. Composable because (a) acts pre-decision (scores) and
(b) post-decision (intents): one process, both patches. Inherits (a)'s path-divergence
sd inflation, so the hybrid arm is NOT the certifiability carrier.

### Channel recommendation

- **Verdict pair (pre-registered primary): incumbent vs incumbent+tilt (1b, strict
  support).** Parity is by construction (gross AND β AND σ budgets), neutral recovery is
  trivial, paired sd is minimal → carries certifiability (C4).
- **Registered secondary arm: hybrid (1c)** — answers the operator's WHICH question
  (selection intelligence) with the guard of (1a); read as context with its honestly
  larger sd, never as the verdict.
- **(1a) alone:** keep as a battery ablation (selection-only attribution), not a verdict.

## 2. The conviction path (C3) and neutral recovery

**Organ → expression mapping.** Each organ k emits per-symbol `mu_k` (cross-sectional,
tilt-core + support names) + a scalar self-confidence `q_k`. The combiner (genes owned by
the Evolution Engineer; semantics fixed here):

- direction: `s = Σ_k τ_k · rank_z(mu_k)` over the support set, τ = trust weights
  (EA-trained, verdict-aligned fitness per C3);
- conviction: `c_t = σ( a·agree_t + b·disp_t + Σ_k d_k·q_k − θ )` ∈ [0,1], where
  `agree_t` = mean pairwise rank-corr of the active organs' mu vectors (organs agree →
  spend), `disp_t` = the dispersion-forecast organ's predicted cross-sectional spread
  (nothing to pick when dispersion is low) — when organs disagree or dispersion is
  forecast flat, c_t → 0 and the arm IS the incumbent;
- expression: `T_t = c_t·T_max` (hard cap T_max ≤ 10% NAV one-sided, pre-registered, not
  a gene above the cap) into the §1b projection; for the hybrid's permutation half, the
  allowed Kendall-tau distance from identity scales with c_t.

**Neutral-recovery property (load-bearing; proof per channel):**
- (1b): `c_t = 0 ⇒ T_t = 0 ⇒ Δw = 0 ⇒ post_decision returns the incumbent's intents
  list UNCHANGED (the same objects)`; execution, cost overlay, and valuation are
  deterministic given intents → **bit-for-bit identical daily series.**
- (1a): neutral ⇒ the patched `_compute_ranking_scores` returns exactly the original
  function's output (it calls the original and passes it through; identity permutation is
  pass-through, not a re-solve) → identical `ranking_scores` dict → identical decisions.
- Hybrid: conjunction of both.
- **Mechanical check (zero-attribution baseline arm B0-EXPR, pre-registered):** run the
  brain arm with conviction forced 0 over the smoke window; assert
  `sha256(timeline.json) == incumbent arm's` (manifests differ, series must not). This
  arm doubles as the attribution zero point for the LOO battery.

## 3. The lot fix (C5)

**Defect (verified):** two sites, one bug class — per-symbol structures built by
overwrite instead of aggregation over multi-lot positions.
1. Adapter: `held_shares = {p.symbol: float(p.shares) for p in positions}`
   (`strategy_adapter.py:320`) and `w_prev[idx] = p.shares·mk/nav` (`:335-339`) — last
   lot wins (the TB-006 SCHD 2×738-share seed → adapter saw one lot → unintended ~22%
   SCHD book). NAV accumulation (`:321-328`) was already correct.
2. Harness: `position_map()` is `{p.symbol: p}` (`replay_engine.py:298-299`) and the SELL
   path clamps `shares_to_sell = min(requested, that_one_lot.shares)` (`:455-456`) — a
   full exit of a multi-lot symbol silently sells only the last lot. This goes LIVE for
   TB-007: the tilt adapter's BUY intents append new `Position`s (`:436-446`), so the
   brain arm creates multi-lot symbols routinely; the live-era seed state may also carry
   multi-lot holdings (check at build; TB-006's 2026-03-11 seed did).

**Fix spec:**
- Adapter (TB-007 copy of `strategy_adapter`-style code): build both maps by
  aggregation — `held_shares[sym] += shares`; `w_prev[idx] += shares·mk/nav`.
- Harness: runtime monkeypatch in the TB-007 runner (sanctioned try/finally pattern;
  production `src/` untouched per write surface), applied **identically to both arms'
  processes**: in `_execute_intents`, BUY into an already-held symbol increments that
  `Position` (shares-weighted entry_price, `peak_price = max`) instead of appending; SELL
  iterates all lots of the symbol. Incumbent-arm behavior is unchanged whenever its book
  is single-lot (the chassis never buys held names — `filter_buy_candidates:286` — so
  incumbent multi-lots can only come from the seed; the patch is a no-op there or fixes
  the same defect for both arms identically).
- **Unit test (`test_lot_aggregation.py`):** portfolio with SCHD in two 738-share lots,
  mark $25, cash $10k → assert (i) adapter `w_prev[SCHD] = 1476·25/nav` and
  `held_shares[SCHD] = 1476.0`; (ii) a full-exit SELL intent of 1476 shares leaves 0
  SCHD lots and credits 1476·open; (iii) BUY of a held symbol yields ONE lot with summed
  shares and weighted entry; (iv) regression: single-lot behavior byte-identical
  pre/post patch on a 3-day synthetic replay.

## 4. Window strategy recommendation

**Recommend: PRIMARY verdict window = live era, 2026-01-31 → present** (67 decision
dates, 66 paired daily deltas; holdout ≥ 2026-03-11 = 44 dates/43 deltas for the E2
read). **Do not pay the 2025 epoch shim for the verdict; run the 2025 leg only if another
role needs it, and then strictly as context.** Grounds:

1. **The brain cannot speak in 2025.** The backfill era trades a 25-name universe; of the
   Universe Selector's 10-name tilt core only **TLT and USO exist** (2/10; conditional
   adds: VIXY, SHY = 2/5). A verdict leg where the brain's exploitable names mostly don't
   exist measures nothing — it adds n with expected mean ≈ 0, which DILUTES the paired t
   rather than helping it.
2. The 2025 leg also has: no LLM artifacts (one organ structurally dark), the single-date
   prices convention requiring the epoch shim (as-coded it replays ZERO days), all four
   repair epochs, and a Mon–Fri cadence unlike the live record — every one a disclosed
   asymmetry the committee would carry for negative expected information.
3. **The paired-sd arithmetic (tilt construction carries certifiability):**
   - TB-006 baseline: paired sd **59.5 bp/day**, n=44 → realized MDE ≈ 21 bp/day. The two
     books were essentially unrelated (26% vs ~full gross).
   - Tilt arm: daily delta ≈ Σ Δw_i·r_i. Pure-tilt sd ≈ T·σ_LS with σ_LS (daily vol of
     the unit long-short basket, tilt-core vs book, ETF pairs) ≈ 100–130 bp. T = 5–8% →
     5–10.4 bp; path-divergence inflation ×1.2–1.5 (calibrate on TB-006 R-runs) →
     **≈ 6–15 bp/day paired sd** — a 4–10× MDE shrink, consistent with ~90% book overlap.
   - MDE at paired t = 2.0: full live read n=66 → 2.0·sd/√66 = **1.5–3.7 bp/day**;
     holdout n=43 → **1.8–4.6 bp/day** (vs TB-006's 21).
4. **Honest ceiling (print this in the pre-registration, Chair's adjudication):** because
   effect and sd both scale with T, the t-stat is T-invariant; what t measures is the
   tilt overlay's information ratio. t ≥ 2.0 needs daily IR ≥ 2/√n → annualized overlay
   Sharpe ≈ **3.9 (n=66)** / **4.8 (n=43)**. Grinold ceiling for the tilt set (blend-IC
   ≈ 0.08, ~10 names, 5d horizon, transfer ≈ 0.7, costs ~6–12 bp RT) ≈ ann IR 1.2–1.8 →
   expected t(66) ≈ 0.6–0.9. Perfect parity makes the verdict MEASURE intelligence; it
   does not manufacture enough days for a realistic edge to clear t ≥ 2. The likely
   honest outcome is TIES-with-positive-point-estimate and a tight CI — itself the
   certifiable product. Adding the 2025 leg does not fix this (point 1: its expected
   contribution to the numerator is ~0). If the committee wants more certifiable days,
   the answer is forward accumulation, not the backfill era.

## 5. Per-organ expression observability (audit schema + LOO arms)

**Per-decision log** `expression_log/<D>.json` (written by the expression layer in both
non-neutral channels; JSONL mirror for batch reads):

```
{ "date": D, "channel": "tilt|perm|hybrid", "neutral": false,
  "conviction": {"c": 0.62, "agree": 0.41, "disp_forecast": 1.8e-3,
                  "organ_conf": {"cast": .7, "gbm": .5, "event": .2, ...}},
  "trust": {"cast": .38, "gbm": .29, ...},                    # τ after gates
  "organ_mu": {"cast": {"ITA": +1.2, ...}, ...},              # rank_z, support set
  "combined": {"ITA": +0.9, ...},
  "guard": {"beta_pre": ..., "beta_post": ..., "sigma_pre": ..., "sigma_post": ...,
             "gross_buy_pred_incumbent": ..., "gross_buy_pred_brain": ...,
             "repairs": ["swap_reverted:VIXY<->XLP"], "infeasible": false},
  "tilt": {"ITA": +0.031, "SPY": -0.018, ...},                # final Δw
  "organ_attribution": {"cast": {"ITA": +0.020, ...}, ...},   # see below
  "perm": {"applied": [["SOXX",0.71],["XLK",0.66]], "kendall_d": 3},  # channel a/c
  "intents_edits": {"scaled": [...], "added": [...], "suppressed": [...]},
  "book_overlap_vs_chassis": 0.93 }
```

- **organ_attribution:** the combiner is linear in organ z-scores pre-projection, so per
  decision we recompute Δw with organ k zeroed (τ renormalized) and log the difference —
  "which organ moved which name how far," exact w.r.t. the projection actually used,
  cheap (≤ 6 extra 15-name projections/day).
- **LOO arms (battery):** one replay arm per organ with that organ's `member_gate` off,
  τ renormalized, all else identical — **separate process per arm** (model-cache hazard
  + uniform discipline), own out dir + nightly-audit copy, per the TB-006 `run_replay.py`
  pattern. Plus B0-EXPR (conviction ≡ 0; must hash-match incumbent, §2) as the zero
  point. Per-organ verdict line = LOO arm vs full-brain arm, paired, same dates — the
  packet's three-valued organ verdicts attach here.
- Switchability is config, not code: organ gates live in the genome/arm manifest, so the
  battery enumerates arms without touching the expression layer.

---

## Bottom line

Tilt channel (1b) at structurally matched gross+β+σ is the verdict carrier: parity by
construction (the projection), neutral = bit-for-bit incumbent (B0-EXPR hash check),
paired sd ≈ 6–15 bp/day → MDE 1.5–4.6 bp/day (4–10× better than TB-006). The ranking
socket (1a) is real but selection-mostly; with the feasible-set permutation + exact
counterfactual beta/vol guard it becomes a clean WHICH channel — run it inside the hybrid
secondary arm, not the verdict. Lot bug is two sites (adapter aggregation + harness
last-lot-wins sell), both specified with the unit test. Verdict window: live era only;
the 2025 leg can express 2 of 10 tilt-core names and would dilute, not strengthen, the t.
The committee should pre-register knowing the honest ceiling: at n=66/43 paired days,
t ≥ 2.0 demands an overlay Sharpe ≈ 3.9–4.8 vs a Grinold ceiling ≈ 1.2–1.8 — parity makes
the measurement clean, not the window long.
