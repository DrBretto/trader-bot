# BUILD_SPEC_007 — ORB-1 (PKT-TB-007, Phase C build specification)

**Author:** Synthesis Chair, 2026-06-10. Companion to `../TOURNAMENT_007.md` (the frozen
pre-registration; where this spec and that document could disagree, TOURNAMENT_007 §4
wins and the disagreement is a logged deviation).
**Contract:** buildable end-to-end with zero builder decisions. Every constant is in this
file or in TOURNAMENT_007 §4. Anything not specified here is NOT built.
**Write surface:** everything under `runs/pkt_tb_007_orthogonal_brain/prototype/`;
imports from `runs/pkt_tb_006_clean_sheet_brain/prototype/` are read-only or copied in.
Production `src/` and `training/` are never modified (runtime monkeypatches only,
try/finally, the sanctioned `run_replay.py:258-269` pattern). No deploys, no new AWS
resources, $0 new Bedrock.

---

## 0. Imports (TB-006 assets, per ANALYST_AUDIT §C — no re-ingestion)

| Import | From (under tb_006/prototype/) | Use |
|---|---|---|
| Daily artifact cache 2025-08-04..present | `cache/s3/daily/` | replays (live era only) |
| Deployed ranking model + normalization | `models/ranking_expanded_unconditioned/`, `cache/s3/models/...` | incumbent arm + member zero + surrogate base |
| OHLCV 64 syms 2014→ | `cache/ohlcv/` | β̂/σ̂, M6 features, dispersion lags |
| CBOE / FRED / COT edges | `cache/cboe/`, `cache/fred/`, `cache/cot/` | M3 / M2 / M5 partitions |
| GDELT features | `store/gdelt_features.parquet` | M4 |
| LLM features + artifacts | `store/llm_features.parquet`, `store/llm/` | M4-B variant, R-LLM arm |
| Panel + meta | `store/panel.npz`, `store/panel_meta.json` | alignment; target slices regenerated here |
| Machinery | `feature_store.py`, `train_members.py`, `members.py`, `folds.py`, `books.py`, `ea.py`, `battery.py`, `stats.py`, `e1_reads.py`, `assemble_evidence.py`, `run_replay.py`, `precompute_nightly.py`, `strategy_adapter.py` (lineage) | copy/adapt into this dir |

NOT used: TB-006 trained members, OOF matrices, EA weights, nightly dirs (all rebuilt);
the 2025 backfill leg (no epoch shim exists in this build).

## 1. Organ specifications (post-forced-simplification)

Common: folds F1–F6 (2020-02-03→2026-01-30), walk-forward OOF, holdout firewall
≥ 2026-03-11 absolute (no training, gate, mask, or EA input touches it), fitness data
ends 2026-02-06. Embargo/purge: **21 td for all members except M2 = 26 td** (21+5,
TOURNAMENT G1). Every member writes a manifest (`walk_forward: true`, seeds, params hash,
data range, code SHA).

### 1.1 M1 CAST-XS (transformer; tilt-directional)
- Target: `y5_rank` — per-day Gaussian rank of open(D)→open(D+5) return minus
  cross-sectional mean (exists in panel).
- Partition (fast block, 10 X-cols + embeddings): return_1d, return_5d, return_21d,
  vol_21d, drawdown_21d, rel_strength_21d, range_pct, volume_z_21d, gap_open_pct,
  dist_from_52w_high + sector/asset-class embeddings. NO 63d stats, macro, event, COT.
  (Chair note: M1 keeps this full known-good block; M6's raw overlap on 3 cols is
  disclosed report-only — TOURNAMENT G9.)
- Config: TB-006 shrink-rung CAST-Small as shipped — ≤16.5k params, wd 1e-3, rank target
  + aux head, **2 OOF seeds {4242, 4243}**, anti-memorization spine unchanged. Ridge twin
  re-trained on the same partition as referee (reported, not gated).
- Acceptance: TOURNAMENT §4.4.1 row M1. Effective sample ≈ 7,100 (189k × 12/64 ÷ 5).

### 1.2 M2 ROT-GBM (gradient-boosted trees; tilt-directional; status OPEN)
- Target: per-day Gaussian rank of **open(D+5)→open(D+21)** cross-sectional excess
  (zero shared bars with M1) — new panel slice `y_rot` (§6.1).
- Partition (slow + macro): return_63d, vol_63d, drawdown_63d, rel_strength_63d;
  ctx_rate_2y/10y, yield_slope, credit_spread, risk_off, spy_ret_21d, spy_vol_21d,
  fred_nfci, stlfsi4, hy_oas(+d5), icsa_z, t10yie_d21, dfii10(+d63); br_adv_dec,
  br_pct_above_50d/200d, br_rsp_spy_21. NO ≤21d momentum, vol-surface, event, COT.
- Config: HistGB — max_iter 60, max_leaf_nodes 10, max_depth 4, lr 0.05, L2 1.0,
  monotone constraints as in TB-006 gbm_cond, purged early stop. ≤~600 leaf values vs
  ~1,750 effective (150k × 12/64 ÷ 16).
- Embargo 26 td. Acceptance row M2 (OPEN; power band 29–62% printed).

### 1.3 M3 DISP-HAR (linear; gain organ, never direction)
- Target: log of forward 5d realized cross-sectional dispersion — sd over the **support
  tier set** (§2.2) of open(D)→open(D+5) demeaned returns — new slice `y_disp`.
- Predictors (13 + intercept): realized dispersion lags 1d/5d/21d (from open_px);
  s1_cor3m_z, vvix_z, skew_z, vix_term_slope, vxn_minus_vix, s1_curvature,
  br_dispersion; calendar dummies turn_of_month, fomc_week, opex_week. OLS, log link.
- Baseline twin for the gate: lags-only HAR (3 coefs + intercept). Acceptance = the
  INCREMENT F-test (TOURNAMENT §4.4.1 row M3). ~558 effective obs.
- Fail ⇒ dispersion term in c_t becomes the constant = trailing-21d mean (arm B-disp).

### 1.4 M4 EVT-NET (elastic net; per-name width/veto, direction-free)
- Target: per-bucket exceedance 1{|5d market-removed bucket return| > trailing 80th pct
  for that bucket} — new slice `y_evt` (27 buckets).
- **M4-A (baseline, expected to ship):** GDELT-only B-columns (G1–G5 families + availability
  masks, mask-interaction form) + trailing bucket abnormal-vol control. **M4-B (variant):**
  M4-A + llm_* columns masked pre-2024-08-15 (`m_llm_available` stays in the partition).
- Model: logistic elastic net; pre-registered hyperparameter grid
  l1_ratio ∈ {0.5, 0.8, 0.95}, C ∈ {0.03, 0.1, 0.3, 1.0}, chosen by inner-fold CV
  (within training folds only). Control net: same form, vol-control columns only (the
  AUC baseline).
- A/B decision: TOURNAMENT §4.4.8 (ΔAUC > 2×se_Δ on F5–F6 pooled OOF; one ledger look).
- Expression: per-name width multiplier (§2.5). Acceptance row M4. ~4,500 effective.
- Bucket→symbol map: the TB-006 `bucket_map` carried verbatim.

### 1.5 M5 POS-Z (rule; 0 trained params; sleeve tilt)
- Inputs: cot_es_lev_net_z, cot_ust10y_lev_net_z, cot_vix_lev_net_z — z over 156-week
  trailing window, publication-keyed (`m_cot_fresh` respected). Constants fixed:
  threshold |z| > 2, decay linear over 21 td, magnitude unit = min(|z|−2, 1).
- Sleeves (within the support set only): equity sleeve {ITA, SOXX, XRT, RSP, KRE, IYR,
  FXI}; duration sleeve {TLT, AGG, MUB, SHY}. Mapping: ES complex → equity sleeve;
  UST10Y complex → duration sleeve; VIX complex → equity sleeve at weight 0.5.
- Signal: `mu_M5[s] = −sign(z_c) · min(|z_c|−2, 1) · decay_t · sleeve_w` for s in
  sleeve(c); 0 elsewhere; emitted only while an episode is live.
- Enable bar: TOURNAMENT §4.4.1 row M5 (binomial one-sided p ≤ 0.05 on F1–F6 episodes;
  expected DISABLED; tally printed either way). Threshold tuning is denied by the
  capacity row — any change to 156w/2.0/21d is a deviation.

### 1.6 M6 GAP-GRU (recurrent; FORECAST ALTITUDE ONLY — no gene, no tilt path, no replay arm)
- Target: `y1_z` — next-day (D→D+1) cross-sectional excess, z-scored (exists in panel).
- Features (8, engineered from `cache/ohlcv/`, new block §6.1): overnight_ret_1d,
  intraday_ret_1d, overnight_minus_intraday_5d_sum, gap_fill_rate_21d,
  amihud_illiq_21d_z, gap_open_pct, range_pct, volume_z_21d. Sequence length 10 days.
- Model: GRU(input 8, hidden 24, 1 layer) + linear head ≈ 2.5k params; MSE on z; seed
  4242; standard early stop on fold-val.
- Verdict line: TOURNAMENT §4.4.1 row M6 (IC ≥ 0.02 AND ≥4/6 fold sign-positive).
  Utility line pre-printed: `structurally unresolvable at this T_max — forecast-altitude
  verdict only`. C2 rows report-only.

## 2. The tilt adapter (the one expression channel; new module `tilt_adapter.py`)

### 2.1 Placement and arms
A `Strategy.post_decision` implementation passed to `run_variant` (hook
`replay_engine.py:656-658`). It EDITS the chassis's own intents at the margin — never
replaces them. Incumbent arm runs with no adapter. One OS process per arm.

### 2.2 Support set and tiers (FROZEN — Phase-0 table, TOURNAMENT G15/A9)
- support(D) = chassis buy set(D) ∪ held names(D) ∪ TILT_CORE ∪ TILT_COND.
- `TILT_CORE` (10): ITA SOXX XRT TLT AGG MUB FXE USO FXI RSP. Per-name |Δw| cap 2.5% NAV;
  FXE additionally capped 1.25% (ADV $14M).
- `TILT_COND` (3): IYR SHY KRE. Per-name cap 1.25% NAV.
- **VIXY excluded from support entirely. Ballast (all remaining names) tilt cap = 0** —
  held ballast may be trimmed/topped only as projection funding legs within its existing
  position (no new ballast positions opened by the brain).
- Names being fully sold by the chassis that day: no tilt. No shorting: Δw_i ≥ −w_i.

### 2.3 Per-organ per-name tier masks (FROZEN rule text — also re-derived per rotation, §4.6)
Rule text (applied to a per-name OOF stats table T): `mask[o,s] = 1.0 if T shows organ
o's per-symbol 90% CI fully > 0 for s; 0.5 if ≥5/6 folds positive but CI spans 0; 0.0
otherwise; mask forced 0 outside TILT_CORE ∪ TILT_COND.`
Production masks (rule applied to the frozen Phase-0 table, M2 inheriting the GBM tags):
- M1 (CAST column): ITA SOXX TLT AGG MUB FXE USO FXI RSP = 1.0; XRT IYR SHY = 0.5; KRE = 0.
- M2 (GBM column): XRT = 1.0; KRE IYR = 0.5; all others = 0.
- M5: masks not applied (sleeve definition is its mask). M3/M4: not directional-per-name.
The shrinkage sentence (honest core IC ≈ 0.04–0.06) prints wherever the masks print.
M2's new-target per-name OOF stats are computed and PRINTED report-only (dissent D7).

### 2.4 Combiner and conviction (constants fixed here)
- Directional score over support: `s_i = Σ_k τ_k · mask[k,i] · rank_z(mu_k)_i`,
  k ∈ shipped ∩ {M1, M2, M5}; τ = softmax(organ_trust genes) over shipped directional
  organs.
- Conviction: `c_t = σ( (a·agree_t + b·disp_t + Σ_k τ_k·q_k − θ) / temp )` with FIXED
  a = 1.0, θ = 0.5; b = `disp_gain` gene; temp = `conviction_temp` gene; q_k = organ
  self-confidence in [0,1] (M1/M2: rank-z dispersion of mu scaled to [0,1]; M5: episode
  magnitude); agree_t = mean pairwise rank-corr of active directional organs' mu vectors
  (1 organ active ⇒ agree_t = q of that organ); disp_t = M3's forecast, z-scored on its
  trailing 252d window, squashed σ(·) (B-disp: constant 0.5).
- Budget: `T_t = tilt_gain · c_t · T_max`, **T_max = 8% NAV one-sided (FIXED)**;
  c_t < dead_zone ⇒ T_t = 0 ⇒ neutral.
- Raw tilt: Δw_raw ∝ s_i (demeaned over support), scaled so Σ|Δw|/2 = T_t.
- M4 width multiplier (if shipped): `Δw_i ← Δw_i · (1 − event_damp_strength ·
  clip((p̂_exceed,i − 0.2)/0.8, 0, 1))` (0.2 = base rate), p̂ via bucket_map.
- defensive_fraction: the {TLT, AGG, MUB, SHY} share of Σ|Δw|/2 is clipped to the gene
  value (excess redistributed pro-rata to non-defensive legs) BEFORE projection.

### 2.5 Parity projection (C1 by construction)
- Constraints: {Σ Δw_i = 0} ∩ {Σ Δw_i·β̂_i = 0} ∩ {|Σ Δw_i·σ̂_i| ≤ ε_σ} with
  **ε_σ = 0.10 × Σ|Δw_i|·σ̂_i** (scales with the tilt; both sides recomputed each
  iteration). β̂_i = trailing 126d OLS beta to SPY; σ̂_i = trailing 21d realized vol —
  both point-in-time from `cache/ohlcv/`, computed through D−1's close.
- Algorithm: closed-form projection onto the two equality constraints, then clip to box
  bounds (tier caps, −w_i floors), iterate project→clip ≤5 times; then σ-budget check —
  if violated, shrink the offending side pro-rata and re-project once.
- Order conversion: scale chassis BUY dollars; add SELL trims of held names; add BUYs for
  core/conditional names (b-extended). min_order respected; legs under min_order dropped;
  residual cash imbalance (≤ one min_order) absorbed by shrinking the largest buy leg.
  Sub-min_order total tilt (< ~1.5% NAV at $100k) ⇒ neutral day, `dead_zone_quantized`
  logged.
- **Neutral recovery (load-bearing):** T_t = 0 or B0 genome ⇒ post_decision returns the
  incumbent's intent objects UNCHANGED (same list, same objects) ⇒ bit-for-bit identical
  series. Verified by B0-EXPR (§8 step 3).

### 2.6 R-LLM conditioner (battery arm only, never in ORB)
`llm_disag` = std over 27 `llm_sent_*` buckets (unweighted), z on trailing 252d,
joined `visible_from ≤ D`; arm applies `T_t ← T_t · clip(1 + 0.25·tanh(z), 0.75, 1.25)`.
Direction/gross untouched. Config flag in the arm manifest.

### 2.7 Observability
Per-decision `expression_log/<D>.json` (JSONL mirror for batch reads); exact schema:

```
{ "date": D, "channel": "tilt", "neutral": false,
  "conviction": {"c": 0.62, "agree": 0.41, "disp_forecast_z": 1.3,
                 "organ_q": {"M1": 0.7, "M2": 0.5, "M5": 0.0}},
  "trust": {"M1": 0.38, "M2": 0.29, "M4": 0.18, "M5": 0.15},
  "organ_mu": {"M1": {"ITA": 1.2, ...}, ...},          # rank_z over support
  "combined": {"ITA": 0.9, ...},                        # s_i
  "m4_damp": {"FXI": 0.85, ...},                        # width multipliers (if shipped)
  "budget": {"T_t": 0.043, "T_max": 0.08, "tilt_gain": 0.6,
             "defensive_share_pre": 0.31, "defensive_share_post": 0.25},
  "projection": {"beta_resid": 0.0, "sigma_resid": ..., "eps_sigma": ...,
                 "iters": 3, "clipped": ["FXE"], "infeasible": false},
  "tilt": {"ITA": 0.025, "SPY": -0.011, ...},           # final Δw
  "organ_attribution": {"M1": {"ITA": 0.017, ...}, ...},
  "intents_edits": {"scaled": [...], "added": [...], "suppressed": [...],
                    "rounding_absorber": "ITA"},
  "rllm": {"z": null, "g": null},                       # R-LLM arm only
  "book_overlap_vs_chassis": 0.93,
  "parity_expost": {"gross_gap": ..., "beta_gap_21d": ...} }
```

`organ_attribution`: per decision, recompute Δw with organ k zeroed and τ renormalized;
log the difference (≤4 extra ≤16-name projections/day) — exact w.r.t. the projection used.

### 2.8 PERM-DESC arm (descriptive only; built LAST under the §8 step-11 timebox)
Mechanics per PROPOSAL_EXPRESSION §1a, fixed parameters, no genes:
- Patched `RE._compute_ranking_scores` calls the original, locally replays the candidate
  pipeline (0.65/0.35 blend → regime multiplier → clip → threshold/health/vol-bucket/
  LLM-veto/panic), permutes the incumbent's `final_score` multiset over the feasible set
  by M1's mu ordering (max Kendall distance fixed at 3), solves back
  r'_i = (target_i/mult_i − 0.65·h_i)/0.35, cap |r'| ≤ 3 with greedy re-repair.
- Guard (exact, pre-trade): predicted buy lists both assignments via the full
  `compute_position_size` formula; |Δβ$| and |Δσ$| ≤ 10% of the unpermuted buy-list mass
  (floor 0.02% NAV·β); predicted gross within ±1% NAV; high-vol-gate names whose
  feasibility depends on final_score > 0.80 excluded from the support unless already
  admitted unpermuted; greedy revert-worst-swap repair; identity fallback logged
  `guard_infeasible`.
- Exactness gate (timebox): predicted incumbent buy list must match the real engine's on
  EVERY smoke date; any mismatch ⇒ arm cut, forfeit printed.
- Products: guard-infeasible rate, Kendall distances, swap inventory, realized parity
  diagnostics. The pre-committed no-claims sentence (TOURNAMENT §4.8.7) prints with all
  of them.

## 3. Lot fixes (C5; both sites; applied identically to BOTH arms' processes)

1. **Adapter aggregation:** build `held_shares` and `w_prev` by accumulation
   (`held_shares[sym] += shares`; `w_prev[idx] += shares·mark/nav`) — never dict-overwrite.
2. **Harness monkeypatch** (runtime, try/finally, production untouched): in
   `_execute_intents`, BUY into a held symbol increments that Position (shares-weighted
   entry_price, peak_price = max) instead of appending; SELL iterates ALL lots of the
   symbol (full exit sells every lot).
3. **`test_lot_aggregation.py` (must pass before step 3 of §8):** (i) SCHD 2×738-share
   lots, mark $25, cash $10k ⇒ w_prev and held_shares see 1,476 shares; (ii) full-exit
   SELL of 1,476 leaves 0 lots and credits 1476×open; (iii) BUY of a held symbol yields
   ONE lot, summed shares, weighted entry; (iv) regression: single-lot behavior
   byte-identical pre/post patch on a 3-day synthetic replay.

## 4. Evolution (adapted `ea.py`)

### 4.1 Genome (12 genes max; genes exist ONLY for shipped organs)

| Gene | Range (encoding) | B0 | Governs |
|---|---|---|---|
| organ_trust[M1] / [M2] / [M4] / [M5] | logit [−2, +2] each | 0.0 | softmax listen-weights (gene present iff organ ships; M4's trust gene = weight on its damp inside c_t's q-sum) |
| tilt_gain | [0, 1] | **0.0** | THE zero point |
| conviction_temp | log [0.5, 2.0] | 1.0 | conviction curve shape |
| dead_zone | [0, 0.3] | 0.1 | turnover control |
| disp_gain (b) | [0, 2.0] | 0.5 | M3's weight in c_t (B-disp: gene removed) |
| cap_core | [0, 2.5%] NAV | 1.25% | per-name cap, core tier |
| cap_conditional | [0, 1.25%] | 0.625% | per-name cap, conditional tier |
| defensive_fraction | [0, 0.5] | 0.25 | defensive-sleeve budget share |
| event_damp_strength | [0, 1] | 0.0 | M4 width damp (iff M4 ships; veto gene CUT — damp→0 subsumes it) |

FIXED, never genes: T_max, parity bounds/ε_σ, tiers/masks, ballast 0, VIXY exclusion,
acceptance/C2 gates, all fitness constants. No gate_strength, m6_timing_gain, channel_mix.

### 4.2 Fitness (verbatim PROPOSAL_EVOLUTION_007 §1)
Per fold f, cost scenario c ∈ {1.0, 1.5}: Δr_t = paired daily tilt return (genome arm −
neutral arm, same surrogate, incremental transaction costs × c);
U_f,c = √252·mean(Δr)/max(sd(Δr), s_min); U_f = min_c U_f,c;
FITNESS = mean_f U_f − 1.0·max(0, −min_f U_f) − λ_reg·Σ(g_i − g_B0,i)².
Constants FIXED: s_min = 2 bp/day, λ_reg = 0.05 (unit² gene space, float genes).
FITNESS(B0) ≡ 0 identically.

### 4.3 Search, controls, gates
- Production run (ONE): (μ+λ) GA, P=28, G=14, K ≤ 400 evals, 4-of-6 fold subsampling per
  generation (retained, not load-bearing), B0 = individual #0 every generation.
- Champion: top-8 full-fold-rescored, L2-dedupe < 0.05 (unit space), per-gene median
  (majority for binaries), re-evaluated fresh on all 6 folds; if below the best
  max-min-over-folds top-8 member, ship that member.
- B1: budget-matched uniform random search, same fitness/folds, seeded; best taken.
- **Adoption gate:** champion > max(1.4, √(2·ln K_eff)) × cross-fold sd of its U_f,
  K_eff = deduped distinct genomes evaluated, computed at run time and printed.
- **Rotation gate:** 6 rotations (P=28, G=8, K≈190 each), fold f fully excluded from
  fitness, subsampling, elite re-scoring; each rotation **re-derives masks + acceptance
  decisions + pooled-only C2 from its own 5 training folds via the frozen rule texts
  (§2.3, TOURNAMENT §4.4.1)**; rotation champion evaluated on its unseen fold under its
  own masks/roster. Pass: ≥4/6 ΔU_f ≥ 0 AND pooled unseen mean > 1×se(ΔU_f).
- Gates are AND; B0 ships on any failure. Boundary-pin audit line in the champion
  manifest (fraction of genes within 5% of an edge; >⅓ ⇒ noise flag).
- Logging: every variant to `ea/generation_*.jsonl` (rotations included); deterministic
  replay from master seed 4242; wall cap 90 min — overrun shrinks rotation G, then
  production G, then P; rotations are never dropped.
- Budget caps: ≤3 full EA cycles across the whole packet (remediation-triggered reruns
  included). Blend-0 sensitivity (§5.3) = one extra reduced-budget run (P=28, G=8).

### 4.4 C2 gate computation spec (`c2_gate.py`)
- **G-signal:** for each directional pair (m,n) ∈ {member zero, M1, M2, M4, M5}: per-day
  cross-sectional Spearman of standardized signals over the common 64-symbol space (M4
  mapped bucket→symbol as negated exceedance prob; M5 mapped sleeve→symbol), averaged
  over F1–F6 OOF days; abstention days (M4 no event mass; M5 |z|≤2) excluded from that
  pair with the inclusion rate printed (silence-orthogonality caught by the conditional
  read).
- **G-book:** pairwise Pearson of daily solo-book returns under the TB-006 fixed
  solo-book rule (BUILD_SPEC TB-006 §6 unchanged), pooled F1–F6 OOF.
- **G-scalar:** M3's aggressiveness series vs (a) each member's daily |tilt| series,
  (b) the regime multiplier series, (c) VIX level; row (b) also for M5. Pooled ≤ 0.7,
  per-fold printed.
- Gate logic, ceilings, replication trigger, ladder, and the member-zero special row:
  TOURNAMENT §4.4.2 verbatim. M6 rows computed and printed REPORT-ONLY (including its
  3-column raw-input overlap with M1). Output: `designs/c2_gate_matrix.json` with pooled
  + per-fold values, both spaces, inclusion rates, trigger evaluations, ladder history.

## 5. Training procedure, surrogate, fidelity

### 5.1 Folds and order of operations
F1–F6 carried verbatim; embargo 21 td (M2: 26); holdout firewall absolute. Order:
panel slices → members train (+referee/control twins) → OOF matrices → per-name OOF
stats printed → production masks asserted == frozen Phase-0 application → acceptance
gates (ledgered) → C2 gate (+ ladder if triggered, ≤2 passes/pair, ≤4 rungs total) →
roster frozen → surrogate built → fidelity checks + anchor → EA (production + rotations
+ B1) → gates → champion-or-B0 → bake-off + battery.

### 5.2 The chassis surrogate (fast paired walk)
Vectorized two-pass walk per fold: base arm computed ONCE per fold and cached; tilt arm
per genome; daily difference = Δr. The base arm replicates the incumbent's selection
policy: `base_score = 0.65·health_score + 0.35·MLP_score`, regime multiplier, clip,
threshold-by-regime, top-N selection, vol-adjusted sizing, matched gross by construction;
costs = half-spread table + 1 bp expectation slippage (the seeded full model only at
bake-off). Health scores and regime labels from the artifact/feature record where they
exist; where the surrogate diverges from the engine (e.g., filters not replicated), the
divergence is itemized in `surrogate_manifest.json` (L2c).

### 5.3 Fidelity checks (run BEFORE the first production EA generation)
1. **MLP memorization check:** deployed-MLP per-fold rank-IC F1–F6 vs live-era
   (2026-01-31→03-10) rank-IC; both printed in the surrogate manifest.
2. **Blend-0 sensitivity:** reduced-budget EA re-run with the surrogate base at
   ranking_blend 0 (health-only); print champion trust-weight deltas; instability ⇒
   pre-registered finding "fitness measures the fiction" (stop, escalate).
3. **Anchor (TOURNAMENT §4.4.7):** champion + B0 through the REAL engine,
   2026-01-31→2026-03-10 (~26 paired deltas): Pearson(Δr_surr, Δr_real) ≥ 0.8 AND
   real-on-surrogate OLS slope ∈ [0.5, 2.0] AND |meanΔU_surr − meanΔU_real| ≤ 2×se_real.
   Failure = stop-and-fix finding. Slope in [0.5,0.7]∪[1.3,2.0] ⇒ magnitude caution
   printed next to s_min/λ_reg in the champion manifest.
4. **Neutral-recovery unit check** = B0-EXPR (§8 step 3), which precedes all of this.

### 5.4 Fine-tune discipline
No member is retrained after seeing any acceptance/C2/EA outcome except through the
remediation ladder (each rung ledgered). No hyperparameter changes outside the
pre-registered grids/configs in §1. Any retrain increments the retrain ledger (≤9).

## 6. Precompute / nightly flow

### 6.1 One-time panel work (`make_targets_007.py`, `make_m6_features.py`)
New slices into `store/panel_007.npz`: `y_rot` (M2), `y_disp` (M3, support-set
dispersion), `y_evt` (M4 exceedance flags + trailing 80th-pct thresholds), M6 feature
block (8 cols × seq window) from `cache/ohlcv/`; `y5_rank`, `y1_z` referenced from the
imported panel. Manifest with construction SHAs.

### 6.2 Nightly precompute (`precompute_nightly_007.py`, adapted)
Per decision date D (replay-time, from cached artifacts): organ inputs assembled with
point-in-time rules (FRED `visible_from`, COT publication keys, GDELT/LLM
`visible_from ≤ D`); each shipped organ emits `mu_k` over support + `q_k`; M3 emits
disp_t; M4 emits p̂_exceed; written to `store/nightly_007/<D>.json`. The tilt adapter
consumes ONLY these files (no model forward passes inside the replay loop). Deployed
shape (if ever promoted, not this packet): same flow inside the Lambda night, +$0.21/mo
marginal, ≈$9.5/mo absolute carry.

## 7. Artifact and manifest conventions

- Every replay arm: own out dir `runs_battery_007/<ARM>/` with `manifest.json` (code
  SHA, params hash, genome hash, seeds, data snapshot range `daily/<d0>..<d1>`,
  cost-model version, exact command, wall-clock), `timeline.json`, `daily_series.csv`,
  expression logs. One OS process per arm; try/finally patches.
- Gate artifacts: `designs/c2_gate_matrix.json` (full matrix, pooled + per-fold, both
  spaces, committed before any bake-off replay); `gates/acceptance_007.json` (per-gate
  statistic, threshold, power pair, outcome, ledger index); `ea/` generation logs +
  `champion_manifest.json` (incl. boundary-pin line, K_eff, gate margins, anchor result).
- Ledgers (initialized 0, append-only): `ledgers/looks_holdout.json` (budget 1),
  `ledgers/gates.json`, `ledgers/remediation.json` (≤4), `ledgers/ea_cycles.json` (≤3),
  `ledgers/replay_arms.json` (≤18), `ledgers/retrains.json` (≤9).
- Every E1/E2 read: one JSON + one short MD per EVIDENCE_PROTOCOL (deltas on the canon
  line, paired-distribution summary, manifest, every variant logged).

## 8. Build order + wall-clock (FA-audited; the exactness chain FIRST) + shrink ladder

| Step | Item | Wall-clock | Gate to proceed |
|---|---|---|---|
| 1 | Lot fix (both sites) + `test_lot_aggregation.py` | 1–2 h | tests green |
| 2 | Tilt adapter + projection + expression logs (no organs yet: zero-mu path) | 0.5–1 d | unit tests on projection (Σ=0, β=0, σ-band, caps, no-short, min_order rounding) green |
| 3 | **B0-EXPR exactness chain:** INC arm replay + ORB arm under B0 genome, live window; assert `sha256(timeline.json)` equal | 1 h run + **0.5 d debug reserve** | HASH MATCH — nothing downstream is trusted before this |
| 4 | Panel slices + M6 feature block (§6.1) | 2–4 h | slice manifests written |
| 5 | Member training M1 (16k/2-seed) + ridge twin, M2, M3 + lags-twin, M4-A/B + control net, M5 (computed), M6 | CAST ≤1 h cold; rest <30 min; M6 ≤1 h | manifests + OOF matrices written |
| 6 | Per-name OOF stats (printed), acceptance gates, M4-A/B decision, C2 gate (+ ladder if triggered) | 0.5 d code; minutes run | roster frozen; `c2_gate_matrix.json` committed |
| 7 | Nightly precompute for the roster (§6.2) | 2–4 h code | nightly files for the live window present |
| 8 | Surrogate + fidelity checks 1–2 + anchor + neutral check | 0.5–1 d | anchor PASS (else stop-and-fix finding) |
| 9 | EA: production + 6 rotations (per-rotation re-derivation) + B1 + blend-0 sensitivity | 0.5 d step; 20–90 min compute | gates evaluated; champion-or-B0 + manifest |
| 10 | Bake-off + battery (§ TOURNAMENT 4.5 arms; E1 full-live + the ONE E2 read fired by the orchestrator) | <1 h compute; 0.5 d attention | all arms manifested; ledgers consistent |
| 11 | **PERM-DESC timebox (LAST, optional): 1 day hard cap** — exact local pipeline replica + solve-back + guard; exactness gate = predicted buy list matches the real engine on every smoke date | ≤1 d | pass ⇒ run the descriptive arm; fail ⇒ cut + print the forfeit |
| 12 | Evidence assembly (mechanical), panel review inputs | 0.5 d | final line per TOURNAMENT §4.7 |

**Total ≈ 3–3.5 working days** (≤2.5 d if step 11 is skipped); compute <3 h end to end.
Cheaper than TB-006 on every line, as the packet requires. Cost: $0 new Bedrock (cap
$3.00 stands unspent), $0 AWS, deployed-shape marginal +$0.21/mo.

**Shrink ladder (if wall-clock or caps bind, in order — drop from the top):**
1. PERM-DESC timebox (print the forfeit sentence).
2. M6 build (its directive-4 line then reads `GRU: not built this packet — forecast
   measurement deferred`, a logged deviation requiring chair sign-off).
3. SEED-A/SEED-B sensitivity arms (descriptive).
4. Challenger arms beyond the first gate-failer.
5. EA rotation G (8→6), then production G (14→10), then P (28→20) — **the rotation gate
   itself is never dropped; the verdict pair (INC, ORB, B0-EXPR) and the E2 read are
   never dropped.**

## 9. Phase C exit checklist (all must be true before Phase D)

- [ ] test_lot_aggregation green; B0-EXPR hash match recorded.
- [ ] All planned members trained with manifests; OOF rebuilt; per-name stats printed.
- [ ] Acceptance outcomes + power pairs in `gates/acceptance_007.json`; roster mechanical.
- [ ] `c2_gate_matrix.json` committed pre-bake-off; ladder ledger ≤4 rungs.
- [ ] Surrogate manifest with L2 fidelity results; anchor PASS recorded.
- [ ] EA: every variant logged; gates + boundary-pin + K_eff in champion manifest.
- [ ] Battery arms ≤18 / retrains ≤9 / EA cycles ≤3; one process per arm; ledgers append-only.
- [ ] Holdout ledger = 1 (the E2 read) and nothing else touched ≥ 2026-03-11.
- [ ] Every surrogate-space number in any artifact carries "(surrogate space)".
- [ ] Final line emitted in the TOURNAMENT §4.7 format with §4.1 caveats.

— end —
