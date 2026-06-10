# Regime Confidence Diagnosis — PKT-TB-004, Model Calibration Specialist

**Role:** Model Calibration Specialist (panel seat 5)
**Date:** 2026-06-10
**Backing data:** `regime_confidence_history.json` (per-date pulled/computed series),
`calib_diagnosis_stats.json` (claims C1–C9), `calib_candidate_eval.json` (validation +
candidate evaluation). Scripts in `scripts/calib_*.py`. All S3 reads, zero production
writes.

---

## 0. Verdict up front

The 4.3% is **primarily a measurement artifact, secondarily honest uncertainty, and
not a probability-calibration problem in the packet's sense**. The published
`regime_confidence` is `1 − cosine-disagreement(GRU probs, Transformer probs)` — not a
class probability — so the packet's "uniform = 0.20" floor framing is a category error
(chair already flagged this; the data below confirms it). On 2026-06-10 both models
were individually decisive (GRU 0.957 on risk_on_trend, Transformer 0.629 on choppy);
near-one-hot vectors pointing at different classes are near-orthogonal, so the cosine
measure collapses to ≈0.04 even though nothing is "flat."

Of the two packet-named calibration candidates, **temperature scaling fails its own
calibration metric on the holdout and has adverse downstream signs in both market
phases — park it without spending an E2 replay slot**. The **5→3 class collapse is the
one candidate worth an E2 slot** (pre-registration spec in §6.2). The panel-added
disagreement-measure swap (cosine→JSD) is wrong-signed in the rally and is parked; a
zero-behavior-change **reporting fix** (publish ensemble max-prob alongside/instead of
the cosine quantity) dissolves the "4.3%" alarm itself and needs no replay because it
changes no trade.

---

## 1. Data actually available (correction to the substrate sheet)

**D1.** The substrate sheet says inference.json spans 2025-08-04 → 2026-06-10 (~235
days). True, but only **84 days (2026-01-31 → 2026-06-10) contain GRU+Transformer
ensemble output**. Everything earlier (and 2026-02-03/05) is a backfilled
baseline-heuristic regime: one-hot probs, `disagreement=0`, no per-model blocks
(verified by direct inspection of `daily/2025-10-15/inference.json` and
`daily/2026-02-03/inference.json`; the 2025 files carry 2026-03-18 S3 timestamps).
Every conclusion below therefore rests on **84 ensemble days**, of which **80 also
have decisions.json expert_metrics** (2026-02-07 →) and **26 are pre-holdout**
(< 2026-03-11). This is a small sample; treat all distributional claims accordingly.

**D2.** Stored artifacts verify internally:
- The stored `disagreement` equals the cosine formula recomputed from the stored
  per-model prob vectors to ≤1.3e-07 on all 84 days (`C1`).
- Published `regime_confidence` equals `1 − disagreement` **exactly** (max diff 0.0)
  on all 61 no-override days (`C2`). On override days it is
  `max(panic_prob, vol_uncertainty)` (panic) or `vol_uncertainty` (unstable calm), per
  `src/signals/regime_fusion.py:123,149`.
- An offline re-run of `decide_regime_v3` from stored inputs reproduces the published
  `final_regime_label` / `position_size_modifier` / `risk_throttle_factor` on
  **80/80 days** (`V_baseline_replay_era_params`), using default fusion params before
  2026-04-30 and the fragility-relax params from then on (the relax flag went live
  with the 2026-04-30 recalibration; both the param-history file and the clean break
  in the mismatch pattern confirm the date). The offline fusion chain used for all
  candidate counterfactuals is therefore validated against the canon record.

---

## 2. Is 0.0426 typical or episodic? — Episodic, bimodal, version-coupled

**D3. The published confidence series is bimodal, not flat.** Over 84 ensemble days
(`C3`): median 0.876, mean 0.644; 35 days > 0.9 but 21 days < 0.2 and 15 days < 0.1.
0.0426 sits at the ~11th percentile of published days — the low mode is **not rare,
but not the norm**.

**D4. The low mode is exactly the label-disagreement days.** Cross-model label
agreement is 61.9% (32/84 days disagree). Mean confidence is **0.927 on agree days vs
0.185 on disagree days** (`C4_summary`, `C5`). For sharp models the cosine measure is
effectively a **binary same-argmax indicator**: it carries almost no information beyond
"did the two argmaxes match."

**D5. Disagreement is episodic and clusters by model version** (`C6`):

| model version | days | span | agree rate | mean conf | mean GRU/TR max-prob |
|---|---|---|---|---|---|
| v20260205 | 39 | 02-06→04-02 | **0.41** | 0.46 | 0.87 / 0.88 |
| v20260402 | 21 | 04-03→05-01 | 0.95 | 0.93 | 0.78 / 0.52 |
| v20260501 | 14 | 05-02→05-30 | 0.93 | 0.84 | 0.54 / 0.92 |
| v20260601/0606 | 7 | 06-02→06-10 | **0.29** | 0.49 | 0.65 / 0.68 |

Two episodes: (a) Feb–Mar correction/panic under v20260205 — a genuine regime
transition where GRU leaned risk_off and the Transformer leaned high_vol_panic or
calm_uptrend; (b) June under the fresh 06-01/06-06 retrains — GRU choppy/risk_on vs
Transformer split, during a calm tape. Retraining boundaries clearly reset ensemble
behavior (the same week's market produced agreement 0.95 under the April model and
0.29 under the June model), but episode (a) coincides with a real market break, so
version and market are confounded — n is far too small to separate them.

**D6. Which failure story?** Three candidates were posed: (a) flat/uncertain
individual models, (b) confident-but-contradicting models, (c) broken disagreement
formula. The data supports **(c) amplified by (b), and rejects (a)**:
- Story (a) rejected: mean individual max-prob is 0.777 (GRU) / 0.772 (Transformer);
  only 3 and 2 days respectively fall below 0.4 (`C3`). The ensemble is not flat.
- Story (b) real: on the 32 disagree days both models average max-prob ≈0.80, and on
  22/32 days **both** exceed 0.6 (`C4_summary`). The models genuinely contradict.
- Story (c) is what turns contradiction into "4.3%": cosine on near-one-hot orthogonal
  vectors → ≈0 regardless of how informative each model is. The proper-divergence
  comparison shows it: normalized JSD on 2026-06-10 is 0.865 (high but not saturated)
  while cosine disagreement is 0.957; across history the two measures correlate 0.97
  on agree days but the cosine measure saturates to the extremes
  (`measure_divergence`).

**D7. The 2026-06-10 row, exactly** (`regime_confidence_history.json`,
`C9.days_below_0.1`): GRU risk_on_trend @ 0.957; Transformer choppy @ 0.629 with
0.315 on risk_off_trend; ensemble argmax risk_on_trend @ **0.481** (this is the number
the packet's "uniform = 0.20" framing should have been applied to — it is above
uniform, mildly uncertain, not 4.3%); cosine confidence 0.0426; the training-rule
proxy label for that day was **choppy**, i.e. the Transformer was the correct one.
The day's low psm (0.3) is *partially honest*: even under a max-prob confidence
measure (0.481) or a JSD measure (0.135), the day stays below the 0.8 fragility-relax
threshold. **No calibration fix turns 2026-06-10 into a high-confidence day; the fix
is that 0.0426 should never have been the published number.**

---

## 3. What the confidence number actually feeds (downstream map)

**D8.** Verified consumers of the regime confidence/disagreement chain:
1. **Published `regime_confidence`** (dashboards/artifacts) = the cosine quantity —
   the alarming surface, no trading effect of its own.
2. **`ensemble_multiplier`** (`ensemble_regime.py:232-252`): `0.5 + 0.5·ens_maxp`,
   penalized when cosine disagreement > 0.3, clipped [0.5, 1.0]; multiplies psm in
   fusion rule 6. Worst case −50% on new-buy size.
3. **Fragility relax gate** (`regime_fusion.py:206-212`, live since 2026-04-30):
   relax requires `1 − cosine disagreement ≥ 0.8`; fired on 15 production days
   (04-30→05-30 cluster plus 06-09), lifting psm from ~0.52 to ~0.88 in the May
   rally. On label-disagree days the gate cannot fire regardless of how sure the
   ensemble distribution is.
4. **The regime label** (not confidence) drives regime_adj sizing (1.10/0.90/0.80/
   0.50), cash-reserve tier (10%/20%/40%/40%), compatibility scoring, and sell
   triggers — so candidates matter mostly through **label flips**.

---

## 4. Calibration quality of the 5-class probs (vs. the training-rule proxy)

Ground truth used: `compute_regime_labels_from_baseline`
(`training/utils/metrics.py:268`) applied to each day's stored `context.parquet` —
the **same pseudo-label rule the models are trained on**, so this measures fit to the
models' own target, not to an external truth. Limits in §7.

**D9.** (`C8_*`) Full 84 days: GRU acc 0.70, ECE 0.089; Transformer acc 0.57,
ECE 0.260, NLL 2.55. Split by holdout: the **Transformer was catastrophically
miscalibrated pre-holdout** (acc 0.154 at mean max-prob 0.82, ECE 0.67, NLL 4.21 —
confidently wrong through the Feb correction) and merely overconfident in holdout
(acc 0.76, ECE 0.18). The GRU is acceptably calibrated in holdout (acc 0.76,
ECE 0.022). The ensemble average lands between (holdout acc 0.79, ECE 0.18, with the
miscalibration concentrated in the 0.52–0.68 bin).

**D10. Sharpness/persistence sanity** (`C7`): ensemble max-prob *is* informative —
when ens max-prob ≥ 0.6 the ensemble label holds next day 84% of the time (n=45) and
5 days later 77%; in the 0.4–0.6 band only 33%/36%. The ensemble max-prob would be a
*meaningful* published confidence; the cosine quantity is not.

---

## 5. Candidate evaluation (offline math on stored probs; fusion chain validated §1/D2)

All three candidates were run through the identical validated chain
(`calib_candidates.py`): candidate transform → ensemble label/disagreement/multiplier
→ `decide_regime_v3` with stored expert inputs and era-correct params → per-day diffs
vs the same chain on uncalibrated inputs. 80 evaluable days. **These are diagnostic
counterfactuals of the fusion outputs only — not P&L replays; the E2 battery is the
chair's.**

### 5.1 C-TEMP — temperature scaling (packet-named) → **PARK, do not spend E2**

Fit (NLL vs training-rule labels, **pre-holdout only**, 26 days 01-31→03-10):
**T_GRU = 2.158, T_Transformer = 9.951** (`C_TEMP_fit`). The Transformer fit runs to
near-uniform flattening — on this window temperature scaling degenerates into
"mostly mute the Transformer," which is a model-weighting decision wearing a
calibration costume.

- **Fails its own metric out-of-sample:** holdout ensemble NLL **worsens 0.649 →
  1.027**, accuracy 0.793 → 0.776, and the ensemble flips from overconfident
  (gap +0.10) to badly *under*confident (−0.37) (`C_TEMP_holdout_ensemble_eval`).
  Pre-holdout improvement (NLL 1.37→1.10) is in-fit and inadmissible.
- **Downstream signs are adverse in both market phases** (`candidates.TEMP`): 76/80
  days change psm, 14 change final label. Flattened panic probs **defuse the panic
  override on all 9 mid-March panic days** (psm 0.25/throttle 1.0 → ~0.48/0.2: ~+0.3
  effective exposure into the crash), while flattened max-probs shrink the multiplier
  and break the fragility relax through the Apr–May rally (−0.07 to −0.54 effective
  exposure on 20+ rally days; holdout mean Δeffective-exposure **−0.040**). More
  invested in the crash, less in the rally.
- Predicted E2 signs if replayed anyway: return ↓, Sharpe ↓, maxDD worse, exposure ↓
  in holdout. Disposition input: **parked-with-evidence** at this diagnostic grade.

### 5.2 C-COLLAPSE — 5→3 class collapse (packet-named) → **the one E2 slot worth using**

Mapping {calm_uptrend, risk_on_trend}→risk_on, {choppy}→neutral, {risk_off_trend,
high_vol_panic}→risk_off; collapse applied to label, disagreement, and multiplier;
**hard-override inputs (5-class panic_prob, trend prob) stay raw**, so the panic
override is untouched (verified: 0 throttle-diff days).

- Cross-model label agreement rises 0.62 → 0.75; days with confidence < 0.2 drop
  21 → 10 (most Feb risk_off-vs-panic splits are adjacent-class noise; June
  risk_on-vs-choppy splits remain genuine and stay penalized).
- Downstream (`candidates.COLLAPSE`): 13 final-label diff days (9 are
  high_vol_panic→risk_off_trend), 56 psm diff days, throttle unchanged. Mean
  Δeffective-exposure **+0.041 full / +0.015 holdout**; June rally days +0.07 to
  +0.11. Cash-reserve tier nearly unchanged (panic and risk_off share the 40% tier;
  mean Δ −0.001).
- Direction: more invested through the late-Feb/Mar washout via psm and via
  regime_adj 0.80-instead-of-0.50 on the 9 panic→risk_off days (note 03-24/03-26/
  04-01/04-02 fall in holdout near the bottom — buying there preceded the rally),
  and slightly more invested in the Mar→Jun rally. Risk: the same mechanism buys
  more in any *future* panic that keeps falling; the E2 replay prices exactly that
  trade-off.

### 5.3 C-MEASURE — cosine→JSD disagreement (panel-added) → **PARK**

Swapping `compute_disagreement` to normalized JSD (labels unchanged): 24 psm-diff
days (`candidates.MEASURE`). It softens the spurious Feb penalties (+0.02..0.08
exposure) but **breaks the fragility relax on 8 May-rally days** (JSD notices the
GRU-sharp/Transformer-flat shape mismatch that cosine forgives; conf drops just
below the 0.8 gate): −0.39..−0.41 effective exposure each; holdout mean **−0.050**.
Wrong-signed in the rally; would need a re-tuned gate threshold = a second tuned
parameter. Parked.

**The cheap correct piece of C-MEASURE is presentational:** publish ensemble
max-prob (0.481 on 06-10) as `regime_confidence` (or alongside, as
`ensemble_agreement` vs `ensemble_confidence`) in `publish_artifacts.py`. With gates
left on the cosine quantity this changes **zero trades** — it needs no E2, just an
operator nod, and it removes the recurring "4.3%" false alarm from every future
surface read.

---

## 6. Pre-registration spec for the chair's E2 battery

### 6.1 K (looks) honesty
This panel seat examined **3 candidate configurations** (TEMP, COLLAPSE, MEASURE)
against holdout-period stored decisions, descriptively. Only one is proposed for an
E2 replay. Count these 3 looks in the packet-wide K.

### 6.2 C-COLLAPSE (proposed for E2) — exact spec
- **Transform** (replay flag, applied where the inference regime block enters the
  decision engine, i.e. alongside `_apply_ensemble_overrides` in
  `src/steps/decision_engine.py`, operating on stored `gru_prediction.probs` /
  `transformer_prediction.probs`):
  - g3/t3 = [calm_uptrend+risk_on_trend, choppy, risk_off_trend+high_vol_panic]
  - ensemble e3 = (g3+t3)/2; label = argmax mapped back {risk_on→`risk_on_trend`,
    neutral→`choppy`, risk_off→`risk_off_trend`} (back-mapping is required because
    every downstream table is keyed on the 5 labels; this is part of the candidate,
    state it in the manifest)
  - disagreement = 1 − cosine(g3, t3); multiplier = clip(0.5+0.5·max(e3) with the
    existing >0.3 penalty, 0.5, 1.0)
  - fusion inputs `panic_prob` and `trend_risk_on_prob` remain the **raw 5-class**
    values (panic/unstable-calm overrides must keep their trigger).
- **No fitted parameters** (no calibration-window leakage possible).
- **Predicted signs (holdout 2026-03-11→):** average gross exposure **↑** (≈+1.5pt
  of effective-exposure multiplier on changed days), total return **↑**, Sharpe
  **↑ or flat**, maxDD **slightly worse** (more exposure held through the
  late-March panic tail), trade count ~flat, transaction costs ~flat-to-↑.
- **Falsifier:** if holdout return/Sharpe do not improve, or maxDD degrades by more
  than the return gain justifies on the daily-delta distribution, disposition is
  parked-with-evidence.

### 6.3 C-TEMP (spec recorded for completeness; recommendation: do NOT replay)
T_GRU=2.1583, T_Transformer=9.9512, fitted 2026-01-31→2026-03-10 (26 days,
pre-holdout only) by NLL vs the training-rule labels; apply `p ∝ p^(1/T)` per model
before averaging, then recompute label/disagreement/multiplier. Predicted holdout
signs: return ↓, Sharpe ↓, maxDD worse, exposure ↓. It already failed holdout NLL —
the calibration claim it ships under is empirically false on this data.

### 6.4 Reporting fix (no E2 needed)
`src/steps/publish_artifacts.py:304,464`: publish `regime.confidence` (ensemble
max-prob) as the headline confidence; keep/rename the cosine quantity as
`ensemble_agreement`. Zero trade deltas by construction (no decision input changes).

---

## 7. Honesty section

1. **Tiny sample.** 84 ensemble days total; 26 pre-holdout; the entire low-confidence
   phenomenon is concentrated in two episodes. Every rate quoted has wide error bars;
   the v20260205 row of D5 is one market event, not a model property estimate.
2. **Constructed ground truth.** The "accuracy/ECE/NLL" figures use the training
   pseudo-label rule (trailing SPY return/vol/credit heuristic). It is circular by
   design (it is what the models are trained to match) and it is **trailing** — a
   model that disagrees with it may be early rather than wrong. The 2026-06-10
   Transformer-was-right call (D7) relies on this rule.
3. **In-sample evaluation everywhere except the T-fit.** Candidate downstream counts
   are computed on the same history that motivated the candidates. Only the chair's
   pre-registered E2 replay counts as proof; my "predicted signs" are the
   pre-registration commitments, not findings.
4. **One market path.** Feb–Mar 2026 correction + Mar–Jun rally. C-COLLAPSE's
   favorable read leans on the rally following the panic; a continued crash flips its
   sign. Nothing here estimates that distribution.
5. **Retraining confound.** Model version changes land at month starts and visibly
   reset agreement behavior (D5); disagreement episodes cannot be cleanly attributed
   to market vs. retrain with n=2 episodes.
6. **Fusion-params archaeology.** Pre-2026-04-30 replays assume default fusion params
   (no relax); supported by the param history file and an exact 80/80 reproduction,
   but the historical configs themselves were not snapshotted per-day in S3.
7. **Days 2026-01-31→02-06** (4 ensemble days) lack decisions.json expert inputs and
   are excluded from fusion counterfactuals (84 vs 80).

---

## 8. Verdict input (for the committee report)

- **Is 4.3% a calibration problem?** No — it is a **measurement artifact** (cosine
  similarity between near-one-hot prob vectors masquerading as a confidence) sitting
  on top of **honest, episodic ensemble disagreement** (both models individually
  confident, pointing at different classes, concentrated in the Feb–Mar transition
  and post-retrain June). The individual models are not flat, and the ensemble
  max-prob (0.48 that night) is informative (84% next-day label persistence when
  ≥0.6). The "5-class, uniform=0.20" framing in the packet is the category error the
  chair flagged.
- **E2 slot:** spend it on **C-COLLAPSE** (spec §6.2, no fitted parameters, predicted
  exposure/return ↑ in holdout, maxDD slightly worse). **C-TEMP: park** — it fails
  holdout NLL and is adverse-signed in both market phases (spec §6.3 recorded if the
  panel wants the negative proven). **C-MEASURE (JSD): park** — wrong-signed in the
  rally via the fragility-relax gate.
- **Independent of any replay:** adopt the zero-trade-delta reporting fix (§6.4) so
  the published confidence is the ensemble max-prob and the cosine quantity is
  labeled as what it is — agreement. The operator's alarm was triggered by a number
  that was never a probability.
