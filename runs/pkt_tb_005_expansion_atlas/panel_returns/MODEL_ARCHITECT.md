# Model Architect — panel return (verbatim subagent output)

### [M-01] Rule-Labeler Direct Swap (the honest dumb baseline)
- hypothesis: Replacing the GRU+Transformer ensemble's regime probs with the one-hot output of the rule labeler (`compute_regime_labels_from_baseline` / `baseline_regime_model` — identical 5 if/elif rules) changes nothing material in decisions, exposing that the deep pair adds no information over its own teacher.
- mechanism: The deep pair is a distillation of these exact rules — its ceiling is the teacher plus whatever smoothing/anticipation the 21-day sequence gives it; the only honest first question is whether that delta cashes out in portfolio terms. Regime enters decisions only via the label (compat multipliers, thresholds, panic gates) and confidence/disagreement (sizing multiplier), so a per-date prob swap isolates the model's entire contribution.
- testability: replay-only. Labeler inputs (spy_return_21d, spy_vol_21d from SPY in prices.parquet; vixy_return_21d from VIXY; credit_spread_proxy from signal_row's hy_spread_proxy) are reconstructable per date from stored artifacts with no lookahead (all trailing windows).
- integration: regime swap.
- lambda/training fit: zero — if baseline wins, Lambda gets SIMPLER (rules already exist in `src/models/baseline_regime.py`, no torch path needed).
- evidence path to E2: replay champion bundle over holdout 2026-03-11→2026-06-10 with inference['regime'] swapped per date; 5 seeds (slippage rng); compare fold_score/annualized return/maxDD/Sharpe vs unmodified replay. "Win" for the deep pair = it beats the rules by more than seed noise; anything else is a kill argument for the torch regime path.
- suggested verdict: pilot-now (this is the mandatory baseline check — see bottom section).

### [M-02] Smoothed/Hysteresis Rule Baseline
- hypothesis: An exponentially-smoothed or min-dwell ("sticky": regime must persist k days before switching) version of the rule labeler captures the only plausible value-add of the deep pair — soft, stable transitions — without any model.
- mechanism: The deep ensemble's soft probs over a 21-day sequence are functionally a learned low-pass filter on the rule labels; an explicit EMA over one-hot rule labels (probs = EMA, label = argmax, confidence = max prob) replicates that for free and makes the comparison three-way: raw rules vs smoothed rules vs deep pair. If smoothed-rules ≈ deep pair, the 2-layer GRU + 4-head Transformer is an expensive EMA.
- testability: replay-only (same swap path as M-01, one extra hyperparameter: EMA halflife or dwell k; sweep 3-4 values pre-holdout, pick one, confirm on holdout).
- integration: regime swap.
- lambda/training fit: trivial — a few lines of state carried in portfolio/pipeline state; no torch.
- evidence path to E2: same harness as M-01; headline = fold_score delta of {raw rules, EMA rules, deep pair} on holdout. Win = EMA rules within noise of (or above) deep pair.
- suggested verdict: pilot-now (runs in the same pilot as M-01 at near-zero marginal cost).

### [M-03] Gaussian HMM Regime Baseline (unsupervised, real structure)
- hypothesis: A 3–5 state Gaussian HMM fit on the 10 context features finds regime structure from the data itself rather than inheriting the labeler's hand thresholds, and its filtered state probs are genuinely probabilistic (sticky transition matrix = built-in hysteresis).
- mechanism: It is the only cheap candidate that can know something the rule labeler doesn't — it learns its own state boundaries and transition persistence from 11 years of context parquet, escaping the distillation ceiling entirely; states get mapped to the 5 compat labels by their fitted means (e.g., high-vol/negative-return state → high_vol_panic) so the decision engine is untouched.
- testability: needs-retraining but trivially within budget (hmmlearn, seconds on CPU, fit once on context parquet up to 2025-08, then filtered — not smoothed — forward probs per replay date; filtered probs use only past data, no lookahead). Replay consumes them via the same per-date swap as M-01.
- integration: regime swap.
- lambda/training fit: inference = one forward-algorithm step in pure numpy (no torch, layer-free); training = seconds monthly.
- evidence path to E2: fit on pre-2025-08 context; generate filtered probs for all 235 dates; replay holdout vs M-01/M-02/deep pair. Win = beats both rules and deep pair on fold_score, and regime flips/quarter ≤ deep pair (stability check).
- suggested verdict: pilot-now (second-wave: run after M-01 lands, same harness).

### [M-04] Bayesian Online Changepoint Detector as Transition Flag
- hypothesis: A BOCPD (or simpler CUSUM) on daily SPY returns emits a "regime just broke" flag that throttles new buys / tightens sizing for the following days, complementing rather than replacing the regime label.
- mechanism: Every current regime source (rules, deep pair, HMM) is a 21-day trailing window and is therefore ~2-3 weeks late at breaks; a changepoint statistic on daily returns reacts in days, and the cost of false positives is bounded because it only gates aggression, not direction. Honest caveat: with ~3 genuine breaks in the 235-day window, the replay evidence will be anecdotal.
- testability: replay-only for the signal (computable from stored prices), but needs a small decision-engine hook (a new throttle input) — wiring cost, not data cost.
- integration: sizing (new throttle factor into compute_position_size / fusion).
- lambda/training fit: pure numpy recursion, no training.
- evidence path to E2: compute flag series offline; first pass = event study (does the flag lead realized drawdowns / rule-label flips in the 235 days?); only if yes, wire a throttle and replay holdout. Win at stage 1 = flag leads label flips by ≥3 days on the observed breaks.
- suggested verdict: parked-promising (revival condition: M-01 shows regime path matters at all AND the offline event study shows lead time; don't spend wiring before that).

### [M-05] Calibration-Weighted Ensemble (replace equal weights)
- hypothesis: Weighting GRU vs Transformer by recent calibration (rolling log-loss against realized rule labels) instead of 50/50 improves the prob vector and hence confidence-driven sizing.
- mechanism: Weak, and saying so: both members distill the same teacher on the same features, so their errors are highly correlated and the gap between them is mostly noise — reweighting two copies of the same opinion has a low ceiling. Stored separate gru_prediction/transformer_prediction probs make it nearly free to check, which is the only reason to run it.
- testability: replay-only — static weights already exist (`ensemble_overrides.gru_weight/transformer_weight` in decision_engine); a per-date adaptive weight needs a ~20-line harness extension computing rolling log-loss vs realized rule labels (available at t since rules are trailing).
- integration: ensemble weighting.
- lambda/training fit: trivial (a rolling scalar in pipeline state).
- evidence path to E2: first sweep static weights {0/100, 25/75, 50/50, 75/25, 100/0} over pre-holdout (zero new code), then adaptive variant; confirm best on holdout. Win = any weighting beats 50/50 by > seed noise. Expected outcome: flat, which is itself useful (confirms member redundancy → kill one member, halve Lambda model load).
- suggested verdict: pilot-now (cheapest experiment in the set; run for the kill-one-member finding, not the win).

### [M-06] Confidence Calibration Layer (temperature/isotonic on ensemble probs)
- hypothesis: The ensemble's confidence is uncalibrated (deep distillations are notoriously overconfident), and since sizing is literally `0.5 + 0.5*confidence`, recalibrating probs against realized rule labels changes position sizes on every single day.
- mechanism: This attacks the highest-leverage consumer of model output — the sizing multiplier — without touching the label; a single temperature parameter fit on stored pre-holdout probs vs realized labels is the minimum-parameter intervention with maximum decision surface.
- testability: replay-only (fit temperature on stored probs 2025-08→2026-03; apply per date in replay via the swap path; no retraining).
- integration: fusion change (post-hoc transform on probs before the multiplier).
- lambda/training fit: one scalar (or small isotonic table) applied in the handler — no torch needed at apply time.
- evidence path to E2: report ECE/Brier before vs after on pre-holdout; replay holdout with recalibrated confidence. Win = calibration measurably improves (ECE down) AND holdout fold_score ≥ champion (sizing changes don't hurt).
- suggested verdict: pilot-now.

### [M-07] Conformal Regime Sets (prediction sets gate sizing)
- hypothesis: Replace scalar confidence with a conformal prediction set (smallest set of regimes covering 90% empirically); set size 1 = full sizing, set size ≥2 spanning risk-on and risk-off = throttle hard.
- mechanism: Gives a distribution-free uncertainty signal with an actual guarantee, unlike disagreement (which only measures how similarly two students copied the same teacher); but it calibrates against rule labels, so "coverage" means coverage of the teacher, inheriting the pseudo-label honesty problem.
- testability: replay-only (calibrate set thresholds on stored pre-holdout probs vs realized rule labels; apply per date).
- integration: sizing (set-size → multiplier mapping replaces confidence/disagreement multiplier).
- lambda/training fit: a stored quantile threshold; trivial.
- evidence path to E2: build sets offline, check empirical coverage and set-size time series; replay holdout with set-size sizing vs current multiplier. Win = same/better fold_score with fewer regime-flip whipsaws.
- suggested verdict: parked-promising (revival condition: M-06 shows confidence is miscalibrated AND M-01 shows the deep probs carry any signal worth dressing in guarantees).

### [M-08] Meta-Labeling Head for Trade Filtering
- hypothesis: A small classifier predicting "given the engine wants to BUY X, will this trade end positive net of the trailing stop?" filters or scales individual buys — the Lopez de Prado meta-labeling pattern, sitting above the engine rather than inside it.
- mechanism: It is trained on REAL outcomes (trade P&L), escaping the pseudo-label trap, and targets the exact failure mode no current model addresses: the engine's buys are score-threshold crossings with no learned notion of conditional success probability. Honest weakness: real production trades number in the dozens; training data must come from synthetic replay trades over the long parquet history (2014→2026), which imports the simulator's assumptions into the labels.
- testability: needs-retraining — first needs a label-generation pass (replay-style simulation of the buy rule over local parquets), then a small MLP/GBM; fits easily in the monthly window (well under an hour). Evaluation of the filter itself is replay-only afterward.
- integration: new head (post-engine buy filter / size scaler).
- lambda/training fit: GBM or tiny MLP — numpy-evaluable, no torch in handler; training local, fast.
- evidence path to E2: generate ~10 years of synthetic primary buys + outcomes; train; replay holdout with filter at 2-3 operating points. Win = higher win_rate and fold_score with turnover not exploding.
- suggested verdict: parked-promising (revival condition: replay shows ≥30% of holdout buys are net losers after costs — check this number first, it's one query against stored fills).

### [M-09] Continuous Risk Dial Replacing the 5-Class Output Space
- hypothesis: The 5-regime space is the wrong output space — the labeler's if/elif tree is a coarse quantization of two underlying axes (trend: spy_return_21d; stress: vol/credit/vixy); outputting two continuous dials and mapping them smoothly to exposure/threshold multipliers removes brittle label-flip discontinuities (the REGIME_SHIFT/panic gates fire on label changes).
- mechanism: Every regime consumer (compat multipliers, regime_adj, cash reserve, panic filter) is a step function of the label; replacing steps with smooth functions of (trend, stress) eliminates the day-to-day whipsaw where one basis point of spy_return_21d flips risk_on→risk_off and reprices the whole book.
- testability: replay-only for a first version — derive dials directly from the same context inputs the labeler uses (no model at all), map to the existing multiplier ranges, swap per date; the learned version (a model outputting the dials) is needs-retraining later.
- integration: fusion change (regime label retained for logging; multipliers computed from dials).
- lambda/training fit: arithmetic; nothing to load.
- evidence path to E2: implement dial→multiplier map matching current regime table at regime centroids; replay holdout. Win = fold_score ≥ champion with materially fewer forced REGIME_SHIFT/panic transitions (count them in fills).
- suggested verdict: pilot-now (second-wave; it's the structural answer to charter question (c)).

### [M-10] Learned Position Sizing Model
- hypothesis: Replace the hand-tuned multiplier chain (vol_adj × regime_adj × llm × ensemble × expert × throttle) with a model trained to output position size from the full state.
- mechanism: Weak, and saying so: there are ~235 days of decision data, the multiplier chain already has an evolutionary hyperparameter search tuning exactly these knobs with far fewer degrees of freedom, and a learned sizer on this sample is an overfitting machine with no offsetting information source.
- testability: needs-retraining, and worse, needs a differentiable/simulatable objective through the replay — weeks of work for negative expected value.
- integration: sizing.
- lambda/training fit: fine technically; irrelevant given the verdict.
- evidence path to E2: none worth running before years more data.
- suggested verdict: killed (the evolutionary param search already occupies this niche at appropriate capacity for the sample size).

### [M-11] Expert Signals as Features for a Learned Fusion / Meta-Regime Model
- hypothesis: Stack the four expert scores (macro_credit, vol_uncertainty, fragility, entropy) + ensemble probs into a small learned model (logistic/GBM) predicting forward 21d market return sign or drawdown, replacing the hand-rule `decide_regime_v3` fusion — the explicit re-examination of the production-only design.
- mechanism: This is the first place real labels (forward returns) and the richest features (expert signals) would meet, which is exactly why it's tempting and exactly why the original design excluded it: the trainable signal history is ~170 pre-holdout days, and a learned fusion on that sample will memorize the one drawdown it saw. The design decision survives re-examination for now — on sample-size grounds, not on principle.
- testability: needs-new-data effectively — either ≥2 more quarters of stored signals or a backfill of the signal series (VIX/VVIX/SKEW, fragility PCA inputs) from free sources over 2014→2026, which is feasible (CBOE/FRED) and would convert this to needs-retraining (minutes).
- integration: fusion change.
- lambda/training fit: logistic/GBM in handler is trivial; training is minutes.
- evidence path to E2 (when revived): train on backfilled signals to 2026-03; replay holdout with learned fusion replacing decide_regime_v3. Win = fold_score > champion AND coefficients/feature-importances economically sane.
- suggested verdict: parked-promising (revival condition: signal-series backfill completed — that backfill is itself a worthwhile DATA candidate; flag to the data architect).

### [M-12] GBM Tabular Regime Model on Real Forward Outcomes
- hypothesis: A gradient-boosted tree on the 10 context features (plus lags) trained against REAL targets — e.g., forward 21d SPY return bucket × forward realized vol bucket mapped to the 5 labels — replaces the deep pair with a model that learns something the rule labeler doesn't know.
- mechanism: Two fixes at once: (1) escapes distillation by using realized-future targets instead of the labeler, (2) matches model class to data (tabular, ~2,800 rows — GBM territory, not 2-layer-GRU-plus-Transformer territory); the deep pair's architecture is spending capacity a 10-feature daily problem cannot fill. Honest caveat: predicting forward regimes is genuinely hard, and the rule labeler (trailing) may still win because trailing description, not prediction, might be all the decision engine actually needs.
- testability: needs-retraining — minutes locally (lightgbm); evaluation replay-only via per-date prob swap (predict with data through t only, walk-forward refit monthly to be honest).
- integration: regime swap.
- lambda/training fit: lightgbm predict is small/no-torch (or export to pure-python trees); monthly training cost negligible.
- evidence path to E2: walk-forward fit on context parquet, emit probs for all 235 dates, replay holdout vs M-01/M-03/deep pair. Win = beats the rule baseline (the bar M-01 sets), not merely the deep pair.
- suggested verdict: pilot-now (third-wave, after M-01/M-03 establish the bar; it's the strongest "forward-looking regime" contender within budget).

### [M-13] Health Model Honesty Check (rule health vs trained AE)
- hypothesis: The health autoencoder has the same disease as the regime ensemble — `compute_health_labels_from_baseline` is a hand-rule composite (momentum/vol/drawdown ranks), so the trained model is another distillation; swapping per-date asset_health with rule-computed values tests whether the AE earns its keep.
- mechanism: Health drives buy thresholds, HEALTH_COLLAPSE sells, and HEALTH_DROP reduces — a larger decision surface than regime — yet the trained model's target is "correlate with the rule composite," so the same baseline-vs-student question applies and is even higher-stakes; snapshot.features_df contains the rank inputs per date, so the swap is mechanical.
- testability: replay-only (compute rule health from stored features per date, overwrite inference['asset_health']).
- integration: scoring feature (health source swap).
- lambda/training fit: rules already in `src/models/baseline_health.py`; if rules win, another torch artifact exits the Lambda path.
- evidence path to E2: replay holdout with rule health vs stored AE health, same bundle, 5 seeds. Win for the AE = fold_score above rules by > seed noise.
- suggested verdict: pilot-now (it's M-01's twin and shares all its harness work).

### [M-14] Ranking Blend Activation Sweep (the one real-label model already built)
- hypothesis: RankingMLP — the ONLY model in the stack trained on realized outcomes (forward 21d return ranks) — is currently inert unless ranking_blend > 0; sweeping the blend in replay may be the cheapest genuine alpha test available.
- mechanism: It injects cross-sectional real-outcome information into base_score where today only the distilled health composite lives; the replay harness already accepts ranking_model/ranking_blend (optimizer/replay.py:307-309), so the experiment is configuration, not code.
- testability: replay-only (model and wiring exist; sweep blend ∈ {0.1, 0.25, 0.5, 0.75} pre-holdout, confirm on holdout). Also report standalone rank-IC on holdout dates as a sanity check before trusting replay deltas.
- integration: scoring feature (existing ranking_blend path).
- lambda/training fit: already budgeted; MLP is tiny.
- evidence path to E2: rank-IC of predictions vs realized 21d forward ranks on holdout; replay sweep. Win = positive holdout rank-IC AND some blend > 0 beats blend = 0 on fold_score.
- suggested verdict: pilot-now.

### [M-15] Volatility-Targeting Sizing Baseline
- hypothesis: A no-model exposure dial — multiplier = clip(target_vol / realized_21d_portfolio_or_SPY_vol) — replacing the regime_adj/ensemble-multiplier sizing chain is the classic dumb baseline that regime classifiers must beat to justify existing.
- mechanism: Most of what the 5-regime apparatus does to sizing (shrink in panic, grow in calm) is a noisy proxy for inverse-vol scaling; if direct vol targeting matches the full chain in replay, the regime layer's sizing role (distinct from its gating role) is redundant.
- testability: replay-only for the signal (SPY/portfolio vol from stored prices); needs a small hook to feed a per-date sizing multiplier (same extension M-04 wants).
- integration: sizing.
- lambda/training fit: arithmetic.
- evidence path to E2: replay holdout: champion vs champion-with-vol-targeting-replacing-regime_adj. Win for vol targeting = fold_score ≥ champion; report side-by-side with M-01 to decompose regime value into gating vs sizing.
- suggested verdict: parked-promising (revival condition: M-01 lands and the per-date-multiplier hook gets built for any other candidate; then it's an afternoon).

### [M-16] Disagreement-Signal Validity Audit (possibly delete a knob)
- hypothesis: Stored gru-vs-transformer disagreement predicts nothing (not forward vol, not regime instability, not next-day rule-label flips), and the disagreement throttle is noise injected into sizing.
- mechanism: Disagreement between two students of the same teacher trained on the same features measures training stochasticity, not market uncertainty — if the audit confirms it, removing the throttle simplifies sizing and removes a random tax on position size; this is a pure offline-stats candidate with a config-level fix.
- testability: replay-only (the audit is a correlation study on 235 stored disagreement values vs forward outcomes; the fix is `ensemble_overrides.disagreement_threshold → 1.0`, already a supported override — replay confirms).
- integration: ensemble weighting (throttle removal).
- lambda/training fit: none.
- evidence path to E2: Spearman of disagreement vs forward 5d realized vol and vs label-flip-within-5d, full 235 dates; if |rho| < ~0.1, replay holdout with throttle disabled. Win = throttle-off fold_score ≥ champion, plus the negative audit result.
- suggested verdict: pilot-now (hours of work total, possible permanent simplification).

---

## ARCHITECT'S TOP PICKS
1. **M-01 + M-02 + M-13 (the distillation honesty block, run as one pilot)** — the entire torch inference path rests on students of rule-based teachers; one replay harness extension answers whether either student earns its keep, and every other regime candidate needs the bar this sets.
2. **M-14 Ranking blend sweep** — the only real-label model in the stack is sitting at blend=0 with replay wiring already finished; highest information-per-hour in the whole list.
3. **M-06 Confidence calibration (+M-16 disagreement audit alongside)** — sizing is a direct affine function of confidence, so calibration is the smallest intervention with daily decision impact, and both run entirely off stored probs.

## BASELINE CHECK DESIGN
**Baseline:** the rule labeler itself, two variants: (A) raw one-hot per date, (B) EMA-smoothed (halflife ~3d; probs = EMA of one-hots, label = argmax). Not HMM for the mandatory check — the honest question is "does the student beat its own teacher?", and the teacher is these rules; HMM (M-03) is the follow-up "can anything beat the teacher?" question. Vol-scaled momentum (M-15) answers the sizing question, not the labeling question — keep it separate.

**Wiring:** extend `optimizer/replay.py` with an optional `regime_source` hook: per date, before `decision_engine.run`, rebuild `snapshot.inference['regime']` as {probs: rule one-hot (or EMA), label: argmax, embedding: zeros}. Labeler inputs computed strictly from data at t: spy_return_21d/spy_vol_21d from SPY closes in stored prices.parquet (trailing 21d), vixy_return_21d from VIXY closes, credit_spread_proxy from signal_row's hy_spread_proxy — all trailing windows, no lookahead. Run each variant twice: (i) confidence=1.0/disagreement=0.0 (multiplier → 1.0 cap) and (ii) confidence/disagreement copied from the stored ensemble — this decomposes the deep pair's contribution into label-path vs sizing-path. Also overwrite gru_prediction/transformer_prediction probs with the same vector so `_apply_ensemble_overrides` can't silently reintroduce stored probs.

**Headline number:** holdout (2026-03-11→2026-06-10) fold_score (the existing `compute_fold_score` composite: Sharpe/Calmar/return/win-rate) for {deep ensemble, raw rules, EMA rules}, same champion bundle, mean±std over 5 slippage seeds; secondary: annualized return, max drawdown, regime flips per month, and label-agreement % between deep ensemble and rules over all 235 days (if agreement is ~95%+, the verdict nearly writes itself). Decision rule: if the deep pair does not exceed the better rule variant by more than one seed-std of fold_score, the GRU+Transformer regime path is killed-by-evidence and the rules (or M-03/M-12 successors) take over production regime duty.
