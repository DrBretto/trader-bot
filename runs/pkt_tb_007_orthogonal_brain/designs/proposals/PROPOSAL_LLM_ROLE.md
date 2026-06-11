# PROPOSAL — LLM Organ Role (PKT-TB-007, panel role 6: LLM/Event Role-Finder)

**Packet:** PKT-TB-007-ORTHOGONAL-BRAIN (assignment item 4) — **Date:** 2026-06-10
**Question:** the LLM organ produces genuinely distinct signal ($0.14/mo live, not a tone
proxy: corr to V2TONE 0.116) that did NOT convert to utility as generic per-bucket sentiment
features (TB-006 E1 −1.08 indeterminate negative-leaning, E2 dSharpe −0.86, paired t −1.50).
Find its value-maximizing role in the orthogonal brain, or establish honestly that none
exists at measurable size.

**Answer up front:** none exists at measurable size. 46 pre-registered-style screens across
6 feature families × 5 target classes on 386 pre-holdout decision dates produce **zero
BH-FDR(10%) survivors**. The single boundary-grade signal (cross-bucket sentiment
disagreement → forward cross-sectional dispersion, ρ=0.122 raw) converts, under the most
generous arithmetic, to <1 bp/day against a 21 bp/day holdout MDE. The value-maximizing
role is **R5: honest retirement to a $0.14/mo monitoring artifact with a pre-registered
zero**, with the disagreement-conditioner specced as the one admissible falsifier arm if
the committee wants the organ physically present in the battery. Every number below is
reproducible from `runs/pkt_tb_007_orthogonal_brain/prototype/llm_role_probe.py` →
`llm_role_probe.json` (and `regime_labels_probe.py` for §3.6).

**HONESTY RAIL respected:** all probing uses decision dates D with open(D)→open(D+5)
ending strictly before 2026-03-11. Window: 2024-08-16 → 2026-03-03 (386 dates), the
contiguous LLM coverage run (orphan week 2024-01-02..08 dropped). Zero holdout reads.

---

## 1. Substrate decomposed

671 scored artifacts in `runs/pkt_tb_006_clean_sheet_brain/prototype/store/llm/`
(+ `store/llm_features.parquet`, 890×154). Families probed:

| family | columns | n |
|---|---|---|
| sent level | `llm_sent_<bucket>` ×27 | level |
| sent dynamics | `_ema3` ×27, `_d1` ×27 | smoothed / news-momentum |
| conf / sal | `llm_conf_*`, `llm_sal_*` ×27 each | confidence, salience |
| global axes | `risk_appetite`, `rates_pressure`, `geopol_risk` | 3 |
| event flags | `llm_event_*` ×12 (binary) + per-flag severity from raw JSON | 12 |
| derived | sal-weighted cross-bucket disagreement, sal_total, conf_mean, ev_count, onsets | — |

Lag discipline in the probe: signal joined to decision date D via `visible_from ≤ D`
(merge_asof, 4-day tolerance) — the artifact for calendar day X carries
`visible_from = X+1`, so D consumes news through D−1's late-evening GKG files, ≥10.5h
before D's execution. Same convention the brain would use live.

## 2. Method and multiplicity (the screen ledger)

- Targets: (a) bucket direction — daily cross-bucket Spearman IC vs 1d and 5d abnormal
  bucket returns (buckets with sal>0.05); (b) cross-sectional dispersion — forward 5d
  cross-symbol return std, raw AND rank-residualized on trailing dispersion; (c) vol
  expansion — fwd_book_vol5 / trailing-21d book vol, raw + residualized; (d) regime-label
  flips (separate probe, §3.6); (e) conditional member skill — CAST/GBM daily OOF
  cross-sectional IC (folds 5–6, 365 overlapping days) median-split on each conditioner.
- Stats: 5d-overlap-corrected t (n_eff = n/5) for daily-IC means; circular moving-block
  bootstrap (block 10 ≥ 2× target overlap, B=2,000, seed 4242) for CIs and Spearman sign
  tests; per-segment sign consistency over S1 (F5 tail 2024-08→2025-02), S2 (F6), S3
  (2026-02→03-03).
- **Multiplicity, stated:** 46 ledgered screens (6 families × ~5 targets + variants).
  Benjamini–Hochberg at FDR 10% across the full ledger: **0 survivors**. Raw p-values
  below are reported with that fact attached; nothing here is a "discovery."

## 3. Where the signal lives (it doesn't, at this sample size)

### 3.1 Direction — dead
Mean daily cross-bucket IC vs 5d bucket returns: sent −0.014 (t=−0.50), ema3 −0.018,
sent×conf −0.016, d1 +0.014 — all |mean| < 0.022 against a family MDE of ≈0.057 at t=2.
Sign flips across segments (sent level: +0.07 S1, −0.05 S2, −0.06 S3). The only
sign-consistent line is news-momentum d1 → next-day (+0.021, CI90 [+0.003,+0.039],
p=0.11, positive in all 3 segments) — noted for the record, FDR-dead, and at +0.02 daily
IC it is ~⅕ of CAST's +0.105. Per-bucket time-series ICs scatter ±0.12 with incoherent
signs (industrials_defense −0.118, europe +0.109) — exactly the null envelope for 27
buckets at n≈250–350 (null sd ≈ 0.06).

### 3.2 Global axes — dead
risk_appetite vs book 5d: ρ=+0.006. −rates_pressure vs rates_duration bucket: +0.070
(p=0.14, the best of four). geopol vs gold/volatility buckets: +0.03 / −0.00.

### 3.3 Cross-bucket disagreement → dispersion — the one boundary signal
Unweighted cross-bucket sent std vs forward 5d cross-sectional dispersion: **ρ=+0.122
(p=0.029, CI90 [+0.03,+0.21])**; sal-weighted +0.103 (p=0.029); after rank-residualizing
on trailing dispersion: +0.113 (p=0.055) / +0.084 (p=0.11). Direction-consistent, sits
exactly at the screen's own MDE (ρ≈0.12), does not survive FDR. **The conversion test
fails where it matters:** disagreement does NOT forecast when relative-strength bets pay —
vs CAST daily IC ρ=−0.028 (null); vs GBM +0.074 (p=0.08, FDR-dead). So even if real, it
forecasts the *opportunity set's width*, not the brain's *hit rate* in it.

### 3.4 Vol/range expansion — dead
geopol −0.04, sal_total +0.05, ev_count −0.05, sal_volatility +0.04, n_clusters +0.01 vs
trailing-controlled vol expansion (all p>0.2). Severity-thresholded event days (any flag
≥0.7) vs vol expansion: ρ=−0.000. The organ does not see vol coming.

### 3.5 Conditional value (confidence conditioner) — dead, with a wrong-signed tease
CAST IC on high-conf_mean days 0.066 vs low 0.126 (diff −0.060, p=0.13): **high LLM
confidence weakly coincides with WORSE transformer skill** — the opposite of a usable
confidence conditioner, and FDR-dead anyway. All ten conditioner×member splits ≤|0.06|
against an MDE of **0.155** IC points (CAST daily IC sd=0.389, n_eff≈73): the test is
honest about being underpowered, but the point estimates aren't even pointing anywhere.
Best raw line: geopol→GBM +0.050 (p=0.115, FDR-dead).

### 3.6 Regime-transition early warning — target is degenerate, role impossible
Pre-holdout regime-label series from the daily cache (153 labeled days <2026-03-11):
the entire 2025 backfill leg is constant `risk_on_trend` (one-hot heuristic backfill
written 2026-03-18 — **zero flips, nothing to predict**), and the live pre-holdout leg
flips **20 times in ~28 trading days** (2026-01-31→03-10) — a ~74% per-day change rate.
Flipping is the picker's normal gait, not an event; "early warning of the next flip" is
not a definable target on this record. Candidate (b) is dead on the target side before
any LLM feature is consulted. (Side-finding for the Orthogonality Engineer/Skeptic: the
live picker's label volatility is itself notable.)

## 4. Role candidates, ranked by this evidence

| rank | role | evidence | verdict |
|---|---|---|---|
| 1 | **(e) Honest retirement** to monitoring artifact + pre-registered zero | 0/46 FDR survivors; TB-006 E1/E2 negative-leaning; conf conditioner wrong-signed | **RECOMMENDED** |
| 2 | (a) tilt-aggressiveness conditioner via disagreement→dispersion | the one boundary signal (ρ=0.12, raw p=0.03) but conversion to member IC null; expected uplift <1 bp/day vs 21 bp/day holdout MDE | admissible only as a falsifier arm, expected outcome zero |
| 3 | (d) salience-weighted gating for EventHead | TB-006: LLM flowed through EventHead and the result leaned negative; §3.5 conditioners null | not supported |
| 4 | (c) per-name event-risk veto | flags predict neither next-day |move| nor vol expansion (level, onset, AND severity≥0.7 variants all p>0.2); organ is bucket-grain, not name-grain | not supported |
| 5 | (b) regime-flip early warning | target degenerate (§3.6) | impossible on this record |

## 5. Top role spec — R5, retirement with a pre-registered zero

**Integration:** none into the decision path. The brain trains and runs with **no llm_*
features in any member's partition** (this also returns ~80 columns of partition budget
to the Orthogonality Engineer and removes the TB-006 negative-leaning channel for free).

**What ships instead:**
1. **Monitoring artifact (live, $0.14–0.20/mo, unchanged pipeline):** the nightly organ
   keeps writing `store/llm/<D>.json`. Its output is surfaced on the dashboard/watchlist
   only (disagreement index, geopol axis, de-chattered event onsets per §6) — operator
   eyes, zero decision influence. Rationale: the record keeps accruing ~22 scored
   days/mo so the n≈386 that was insufficient here roughly doubles in 18 months.
2. **Pre-registered zero (the scorecard line):** "LLM organ: 0 (measured) — retired by
   role-finding; 46-screen pre-holdout decomposition, 0 BH-FDR(10%) survivors; strongest
   raw candidate (disagreement→dispersion ρ=0.122) converts to <1 bp/day vs 21 bp/day
   MDE." This is the assignment-item-4 deliverable in the "honest zero with the arm to
   prove it" branch the packet explicitly sanctions.
3. **The arm to prove it (battery slot, only if the committee spends one):** R-LLM arm =
   brain + the §5.1 conditioner vs brain, paired, pre-holdout E1 read first; promote to a
   holdout look ONLY if E1 paired t ≥ +2. Expected outcome: not promoted.
4. **Re-test trigger (pre-registered, not a soak):** re-run `llm_role_probe.py` verbatim
   when the scored-artifact count crosses 900 trading days (~mid-2027) or before any
   future packet proposes LLM features; promotion bar = the same screen surviving
   BH-FDR(10%) on the extended window.

### 5.1 Runner-up spec (the falsifier arm's exact shape, so it is buildable)
- **Feature:** `llm_disag = std over 27 buckets of llm_sent_<b>` (unweighted — it beat
  the sal-weighted variant), z-scored on a trailing 252d window, lag = `visible_from ≤ D`.
- **Consumer:** the **expression layer**, not a member: scale the brain's tilt width
  (the rank-permutation distance / tilt budget, NOT direction, NOT gross — C1-safe by
  construction) by `g(z) = clip(1 + 0.25·tanh(z), 0.75, 1.25)`.
- **Falsifier:** paired E1 vs the unconditioned brain; kill at t < +2.0. **Expected
  effect vs MDE, stated honestly:** tilt-sleeve PnL is bounded by IC×dispersion×budget;
  crediting the full ρ=0.12 and a ±25% width swing, the uplift is O(0.1–0.5) bp/day —
  **~50× under the 21 bp/day holdout MDE and far under any E1 MDE.** This arm exists to
  discharge the operator's "every model type present" directive with evidence, not
  because the evidence says it will pass.

**Reconciliation with operator directive 4** ("all model types present, each at max
value, wherever it is"): the LLM's measured max value on this record is **as a monitored
sense with a pre-registered re-entry path, not as a wired-in signal** — "wherever it is"
honestly resolves to "not in the decision path yet." Wiring it in anyway would repeat
TB-006: a negative-leaning garnish that costs orthogonality budget and attribution power.

## 6. The chattiness fix (works on existing artifacts; $0 re-scoring)

The "fires 98% of days" Stage-1 finding is **one flag, not twelve**:
`geopolitical_escalation` fires 82.9% of days with mean run length 7.8 days. The other
11 flags are already event-like: fire rates 1.8–20.2%, mean runs 1.0–1.6 (cb_decision
6.5%, jobs_print 2.8%, credit_event 1.8%, inflation_print 13.7%, natural_disaster 20.2%).

De-chattering spec (post-hoc transform, no prompt change, no Phase C Bedrock spend):
1. **Demote `geopolitical_escalation` from flag to axis** — it duplicates the
   `llm_geopol_risk` global axis conceptually; if an event form is wanted, use **onset**
   (0→1 transition: 10.6% of days) or severity ≥0.7 (10% of its firings).
2. **Onset encoding for all flags** (`flag_t ∧ ¬flag_{t−1}`) — collapses persistence runs.
3. **Severity floor 0.5** where present (severities cluster at 0.4–0.5; ≥0.7 is the
   genuine-tail cut: 23% of natural_disaster, 16% of election_political, 10% of geopol).
A prompt-side fix (asking for "new or escalating only") would need full re-scoring
(~$2.9 for the 671-day window) and is **not** recommended: §3.4 shows even the
de-chattered encodings predict nothing, so the fix is documentation + monitoring-surface
hygiene, not signal rescue.

## 7. Coverage reality, per role (with one correction to the packet brief)

**Correction:** the brief states the 2025 backfill leg has NO LLM coverage. That is true
only of the **chassis's** `llm_risk.json` (absent from `daily/2025-08-04..2026-01-28`,
hence the incumbent's LLM veto/size-adj runs dark there — identically in both arms). The
**organ's** artifacts (`store/llm/`) cover that leg fully: 178 scored days 2025-08-04→
2026-01-28 (Tier-2) plus Tier-1 from 2026-01-29 — verified on disk. Consequences:
- **R5 retirement (recommended):** coverage moot for the verdict; monitoring artifact is
  live-era forward only.
- **R2 disagreement conditioner (falsifier arm):** usable over the FULL ~189-day verdict
  window — no coverage asterisk, no respend. (Only true gaps: the 2026-05-11..20 outage
  hole, which removes replay days for both arms anyway.)
- **Any chassis-side LLM role** (e.g., modifying llm_adj sizing): dark on the 2025 leg
  in both arms; effects measurable only on the ~67-day live leg → MDE worsens ~×1.7.
- TB-007 training discipline: if any member were (against this recommendation) given
  llm_* features, the `m_llm_available` mask must stay in its partition — coverage
  starts 2024-08-15, i.e. folds F1–F4 are all-masked.

## 8. Cost line

Retirement keeps the live pipeline at ≈$0.14–0.20/mo (monitoring) with **$0 new Bedrock
spend in Phase C** (all candidate roles were designed against existing artifacts; the
probe itself made zero model calls). Killing the live emission entirely would save the
$0.14/mo but forfeit the accruing re-test record; recommended: keep emitting.

## 9. Files

- `runs/pkt_tb_007_orthogonal_brain/prototype/llm_role_probe.py` — the 46-screen
  decomposition (seed 4242, B=2,000 block bootstrap), writes `llm_role_probe.json`.
- `runs/pkt_tb_007_orthogonal_brain/prototype/llm_role_probe.json` — machine record:
  full ledger with CIs, per-segment means, per-bucket ICs, chattiness + severity tables.
- `runs/pkt_tb_007_orthogonal_brain/prototype/regime_labels_probe.py` +
  `regime_labels_preholdout.json` — §3.6 regime-label series and flip list.

**Bottom line for synthesis:** retire the LLM from the decision path with the
pre-registered zero above; keep the $0.14/mo monitoring emission and the verbatim re-test
trigger; spend at most one E1 battery slot on the disagreement-width conditioner if the
panel wants the directive-4 box discharged by measurement rather than by this report.
The orthogonality budget the organ vacates should go to senses that showed per-name CI
significance (Universe Selector §3) — that is where "every model at max value" is real.
