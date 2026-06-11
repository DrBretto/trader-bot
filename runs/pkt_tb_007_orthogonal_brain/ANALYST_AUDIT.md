# ANALYST_AUDIT — PKT-TB-007-ORTHOGONAL-BRAIN (Phase 0, role 1)

**Scope:** (A) ranking-socket mechanics end-to-end; (B) artifact depth + caveat map for the
2025-08-04→present paired window, with the cache extended back to 2025-08-04; (C) TB-006
asset reuse inventory. Every claim cites code or measured data. Audit scripts live in
`runs/pkt_tb_007_orthogonal_brain/prototype/` (`list_s3_daily.py`, `extend_cache.py`,
`audit_artifacts.py`, `audit_artifacts2.py`); the S3 listing snapshot is
`prototype/s3_daily_listing.json`, write-lag table `prototype/write_lag_epochs.csv`.

---

## A. The ranking socket, end-to-end

### A1. Where scores enter, and what they blend against

Entry point: `decision_engine.run(..., ranking_scores: Dict[str,float], ranking_blend: float)`
(`src/steps/decision_engine.py:979-988`) → passed only to `score_candidates()`
(`:1161-1169`). The blend (`:243-251`):

```python
merged['base_score'] = merged['health_score']            # default
if ranking_scores and ranking_blend > 0:
    merged['ranking_score'] = merged['symbol'].map(lambda s: ranking_scores.get(s, 0.5))
    blend = min(max(ranking_blend, 0.0), 1.0)
    merged['base_score'] = (1-blend)*merged['health_score'] + blend*merged['ranking_score']
```

- Blends against **health_score** (per-symbol, from `inference.json` `asset_health`), NOT a
  composite. Then `final_score = (base_score * regime_multiplier).clip(0,1)` (`:254`).
- **Missing symbols silently default to 0.5** (`:245`) — a brain that scores a subset
  leaves the rest at neutral.
- Stage: **buy-candidate scoring only.** `final_score` is consumed by
  `filter_buy_candidates()` (threshold + sort, `:294-296,:326`) and the buy loop ordering
  (`:1264`); plus the high-vol-bucket exception `final_score > 0.80` in calm_uptrend (`:307`).

### A2. Expected scale; out-of-scale behavior

`RankingMLP` ends in `nn.Sigmoid()` (`training/models/ranking_mlp.py:35`); `forward()`
returns scores in **(0,1)** (`:40`). Inputs are z-scored per `ranking_normalization.json`
(`predict_scores`, `:51-60`). The deployed dir `models/ranking_expanded_unconditioned/`
has normalization keys = exactly the 10 base `RANKING_FEATURES` (verified) — no
regime/asset-class/sector one-hots, so the optional conditioning branches (`:62-111`) are
inert. Out-of-scale external scores: `ranking_blend` is clamped to [0,1] (`:247`) but
**`ranking_scores` values are NOT clamped** — a score outside [0,1] propagates linearly
into `base_score`; only `final_score` is clipped to [0,1] after the regime multiplier
(`:254`). Contract for any brain score: emit in [0,1], cross-sectionally meaningful
against health scores on the same scale.

### A3. WHICH vs HOW MUCH — does score touch dollars?

**Directly: selection-only.** `compute_position_size()` (`:767-890`) takes no score:
`dollars = PV × max_position_weight(0.30) × vol_adj(0.80–1.10) × regime_adj(0.50–1.10) ×
llm_adj × ensemble_adj/PSM × throttle_adj` (`:861-863`). The regime PSM path
(`decide_regime_v3` → `position_size_modifier`, `:1049-1066`) never sees ranking.

**Indirectly: yes, four channels (C1 is NOT structural through this socket):**
1. **Threshold count.** `final_score >= buy_score_threshold_by_regime` (0.62/0.62/0.65/
   0.68/0.72; `:294-296`). Different scores ⇒ different COUNT of passers ⇒ marginal buys.
   One marginal buy moves gross by up to `0.30 × PV` pre-adjustment (deployed
   max_position_weight = 0.30), bounded by `max_positions 8 − holdings` slots (`:1186-1188`)
   and `cash − min_cash_reserve_by_regime` (`:1191-1193`).
2. **Composition.** A different symbol in the same slot sizes differently via
   `vol_adj` (0.80 vs 1.10 = ±15% swing) and that symbol's `llm_confidence_adj`.
3. **High-vol gate.** `final_score > 0.80` admits high-vol-bucket symbols in calm_uptrend
   (`:307`) — score can unlock a riskier candidate class.
4. **Path dependence.** Different holdings ⇒ different later sell/trim/cash trajectories.

**Sells: zero direct influence.** `evaluate_holdings()` (`:379-634`) and
`compute_exposure_trims()` (`:637-764`) have no ranking input; no sell trigger reads
`final_score`. The watchlist (`_build_watchlist`, `:893`) is dashboard-only.

### A4. Config keys and injection

- Bundle block `decision_engine: {ranking_blend, ranking_model_dir}` in
  `config/decision_params.active.json`. Production: `src/handler.py:100` maps it to
  `config['decision_engine_overrides']`; step-10 loads the model (local `models/` first,
  S3 fallback) and computes per-symbol scores (`handler.py:256-305`).
- Replay: `load_variant_configs()` maps `active['decision_engine']` →
  `VariantConfig.decision_engine_overrides` (`src/utils/three_line_replay/replay_engine.py:94-115`);
  per-day the engine reads `ranking_blend`/`ranking_model_dir` from the variant
  (`:617-625`) and computes scores via `_compute_ranking_scores(features, cache, model_dir)`
  (`:174-187`) on the features' latest date; failure ⇒ blend forced 0 (health-only).
- **Hazard:** `_get_ranking_model` caches in module globals and returns the cached model
  *ignoring `model_dir`* on second call (`:147-149`). Two variants with different model
  dirs in ONE process would silently share the first model. Run arms in separate
  processes (TB-006's `run_replay.py` pattern) or patch.
- **Deployed values** (`config/decision_params.active.json`, version
  `beat-champion-participation-2026-06-06-v1`): `ranking_blend: 0.35`,
  `ranking_model_dir: "models/ranking_expanded_unconditioned"`. Model artifacts exist in
  S3 (written 2026-03-21), in local `models/`, and already in the TB-006 disk cache
  (`cache/s3/models/ranking_expanded_unconditioned/`).

### A5. Clean experiment contract through this socket

For "incumbent+brain vs incumbent" with ONLY the cross-sectional ranking input changed:

| Field | Incumbent arm | Brain arm |
|---|---|---|
| `decision_engine_overrides.ranking_blend` | 0.35 (deployed) | **0.35 — unchanged** |
| ranking-score SOURCE | `_compute_ranking_scores` on deployed MLP | brain's per-date per-symbol dict in [0,1] |
| everything else (`decision_params`, `regime_compatibility`, `signals`, `regime_fusion`, `ensemble`, `transaction_costs`, seed 4242, start date, universe, harness code) | identical | identical |

Mechanically the swap is a runtime monkeypatch of `RE._compute_ranking_scores` (same
sanctioned pattern as the `START_PORTFOLIO_DATE` patch,
`runs/pkt_tb_006_clean_sheet_brain/prototype/run_replay.py:258-269`) reading precomputed
per-date score files — the engine-level API (`ranking_scores` dict) is already external.
Changing `ranking_blend` itself is a SECOND knob (how loudly the chassis listens); if the
committee wants it trained, pre-register it as part of the brain arm and disclose it as a
deviation from the pure only-the-input-changed contract.

**C1 verdict for this socket: selection-mostly, not selection-only.** Exposure parity
holds only approximately (channels A3.1–A3.4). Strongest structural mitigation available:
**rank-permutation expression** — the brain reorders the incumbent's own daily score
multiset (same values, brain's assignment to symbols), which preserves the
count-above-threshold exactly (modulo the independent `min_health_buy 0.60`, LLM-veto,
and vol-bucket filters) and converts the socket into a near-pure WHICH channel. The
Expression Architect should treat raw-score substitution vs rank-permutation as a design
decision with C1 consequences quantified above.

---

## B. Artifact depth + caveat map (2025-08-04 → present)

### B0. Cache extension (done in this audit)

S3 `daily/` listing: **235 date dirs, first = exactly 2025-08-04, last = 2026-06-10.**
Extension download into the TB-006 cache (`runs/pkt_tb_006_clean_sheet_brain/prototype/
cache/s3/daily/`): **745 files, 6.9 MB, 59.7 s** (skip-existing 633; 267 file-slots absent
in S3, dominated by 2025-era `llm_risk.json`). Listing pass ≈ 25 s. Cache now spans
2025-08-04..2026-06-10.

### B1. Replay-input completeness

- 88 dirs have all 7 files (live era). **2025-08-04..2026-01-28 (119 dirs): everything
  except `llm_risk.json`** — the replay engine defaults `llm_risks={}` on a missing file
  (`replay_engine.py:560-564`), so 2025 replays run with no LLM veto/size-adj, identically
  in both arms.
- 24 dirs are portfolio_state-only (live-era Mondays/holidays + the mid-May outage below).

### B2. TB-004 backfill boundary — verified empirically

Full-window classification of `inference.json` (audit_artifacts.py):
- **2025-08-04..2026-01-30: one-hot heuristic backfill** — `probs` max = 1.0, no
  per-model (`gru_prediction`/`transformer_prediction`) probs, no confidence field,
  legacy key set, **asset_health = 27 symbols**, files written 2026-03-18 (LastModified).
- **2026-01-31 onward: full model output** (soft probs, per-model probs, confidence,
  asset_health = 64). Two stragglers inside the live era are one-hot with conf=1.0:
  **2026-02-03 and 2026-02-05** (no per-model probs). One day (2026-05-06) has 63 health rows.
- Counts: 127 one-hot / 84 full-model / 24 absent.
- Knock-on: `_apply_regime_collapse_3` and `_apply_ensemble_overrides` are inert on
  one-hot days (no per-model probs; `decision_engine.py:133-137,:42-45`) — identical both arms.

### B3. The 2025-era substrate is materially thinner (NEW findings)

1. **27-symbol universe.** Backfill-era prices/features/asset_health carry 27 symbols
   (SPY, QQQ, IWM, DIA, sector SPDRs, EEM/EFA/VEU, TLT/IEF/SHY/TIP/LQD/HYG, GLD/SLV/USO,
   VIXY). **VEU and XLRE are not in `config/universe.csv` (64 rows)** — the
   `eligible==1` merge filter (`decision_engine.py:214,:223`) drops them, so the 2025 leg
   trades a **25-name buyable universe**, not 64. Both arms identically.
2. **Prices-file convention break.** Backfill `daily/<D>/prices.parquet` = a single-date
   file containing **D's own bar** (27 syms, full OHLC cols). Live files contain
   ~1 year of history **through D−1** (64 syms; e.g. 2026-03-12 file spans
   2025-03-13..2026-03-11). The native `run_variant` plan prices decision date
   `td[i+1]` out of `td[i+2]`'s file (`replay_engine.py:535-538`) — in the backfill era
   that lookup finds nothing and the SPY check (`:574-578`) skips the day.
   **As-coded, the 2025 era replays ZERO days. TB-007 needs an epoch-aware shim:**
   backfill pairing = inputs@D (features are through D's close), fill at D+1's own-bar
   file — semantically identical to the live era's decide-on-yesterday's-close /
   fill-at-next-open. One careful stitch at 2026-01-28..2026-02-02.
3. **Schema drift, benign:** backfill features.parquet has 16 cols incl. `health_score` +
   `vol_bucket`; live has 14 (health from inference.json). All 10 `RANKING_FEATURES`
   present in both eras, single-date rows — `_compute_ranking_scores` works across the
   whole window. `signals.parquet` carries every column `_build_expert_signals` needs
   (incl. entropy block) in both eras (verified 4 probe dates).

### B4. Repaired-timeline epochs (S3 LastModified vs dir date)

167/235 dirs have ≥1 file written >3 days after the dir date. Per-file repair events:

| File | Repair write date | Dirs affected |
|---|---|---|
| prices/context/features.parquet | **2026-01-31** | 122 (all of 2025-08-04..2026-01-28) |
| portfolio_state.json | **2026-02-06** | 127 |
| inference.json | **2026-03-18** | 125 (the TB-004 one-hot backfill event) |
| signals.parquet | **2026-04-01** | **167 — includes live-era dirs through ~2026-03-31** |

The 2026-04-01 signals rebuild is the widest repair: the expert-signal inputs (macro,
vol-uncertainty, fragility, entropy) for the ENTIRE record before April 2026 are a
retrospective rebuild. Operator's caveat (packet directive 6) applies; paired-only reads;
print with every full-record number. Same-week (organic) writes exist for 89 dirs
(prices/features) / 86 (inference) / 44 (signals) — the genuinely-live tail.

### B5. Expected paired-day counts (executable decisions, SPY-bar-verified)

- **Backfill era (with shim): 122 decision dates** (2025-08-04..2026-01-27; +2 more
  recoverable at the boundary stitch). Includes 24 Mondays — the synthetic era has a
  uniform Mon–Fri cadence, unlike the live Tue–Sat record.
- **Live era (native plan): 67 decision dates** (2026-01-30..2026-06-09). Saturday dirs
  (18) never execute (no Saturday bar); live Mondays absent by cadence.
- **Full window total ≈ 189** (packet estimated "~170+" — confirmed, slightly better).
- **Holdout (≥2026-03-11): 44** — exactly reproduces TB-006's E2 n=44 paired dates.
- Paired daily-return deltas = decision dates − 1 per contiguous segment.

### B6. Exclusions / holes (disclose up front, both arms identical)

- **2026-05-11..2026-05-20: 7 trading days with no analysis dirs** (inside the holdout)
  — flat-hold hole; also 2026-03-30..2026-03-31 (2 days) and live-era Mondays/holidays
  (2026-02-16 etc.). No further exclusions required: every dir with prices has full inputs.
- 2025 era: no LLM artifacts (B1), 25-name universe (B3.1), all four repair epochs (B4).

### B7. Seeding from 2025-08-04 — verified

`daily/2025-08-04/portfolio_state.json` = **cash 100000, holdings [], portfolio_value
100000** — a clean $100k zero-position seed (no seed-lot hazards; the TB-006 SCHD
double-lot entered via the 2026-03-11 seed's two lots). `seed_portfolio()` hardcodes
`START_PORTFOLIO_DATE='2026-03-11'` (`replay_engine.py:40,:302-309`); the sanctioned
runtime override is the try/finally monkeypatch in TB-006 `run_replay.py:258-269` —
set it to `2025-08-04`, identical for both arms, restore after. `benchmark_shares`
missing from the 2025 state → defaults 0.0 (`:322-326`), harmless.

---

## C. TB-006 asset reuse inventory

All paths under `runs/pkt_tb_006_clean_sheet_brain/prototype/` unless noted.
State: **F** = final-grade (full-window/production discipline), **S** = smoke/TB-006-specific.

| Asset | Path | State | TB-007 action |
|---|---|---|---|
| Daily artifact cache 2025-08-04..2026-06-10 | `cache/s3/daily/` (235 dirs, ~67 MB after extension) | F | **Import as-is** (extended by this audit) |
| Deployed ranking model + norm | `models/ranking_expanded_unconditioned/` + `cache/s3/models/...` | F (deployed 2026-03-21) | Import (incumbent arm) |
| Deep OHLCV, 64 syms, 2014→present | `cache/ohlcv/` (10 MB + coverage_report.json, seed-4242 S3 cross-check) | F | Import |
| CBOE indices (VIX, VIX9D, VIX3M, VVIX, SKEW, COR3M, VXN) | `cache/cboe/` | F | Import |
| FRED 12 series with `visible_from` lag rules | `cache/fred/` | F | Import |
| CFTC TFF positioning (ES/UST10Y/VIX), publication-keyed | `cache/cot/` | F | Import |
| GDELT raw event/theme cache 2015→present | `gdelt_cache/` (3.4 GB, 4131 days) | F | Import (do NOT re-ingest) |
| GDELT engineered features | `store/gdelt_features.parquet` (4131×114, `gdelt_date`+`visible_from`) | F | Import |
| LLM organ artifacts | `store/llm/` (671 day-files, 2024-01-02→) + `llm_seen_state.json` | F ($2.873 spent) | Import — no respend for same window |
| LLM engineered features | `store/llm_features.parquet` (890×155, 2024-01-02..2026-06-09) | F | Import |
| Consolidated panel + targets | `store/panel.npz` (2845 dates × 64 syms: X[·,64,63,14], T[·,64,56], B[·,27,26], Z[·,24], y5_rank/y1_z/fwd_vol5/w_rec, open/close px) + `panel_meta.json` | F | Import alignment machinery; **regenerate target/partition slices** for retargeted members (C2) |
| Feature-store builder | `feature_store.py` | F | Import |
| Member training + folds | `train_members.py`, `members.py`, `folds.py` | F machinery | Reuse code; **retrain all members** (TB-006 members are the 0.941-corr set — the thing C2 exists to kill) |
| Trained members + metrics | `models_out/` (cast, gbm_cond, event_head, ridge_twin, risknet + ablations) | S for TB-007 | Reference only (CAST IC .105 / GBM E1 HAC t +2.48 baselines) |
| OOF matrices | `oof/` (84 npz, 22 MB) | S for TB-007 | Reference for decorrelation baselines; rebuild |
| Books / executive / EA | `books.py`, `executive.py`, `ea.py`, `ea/`, `exec_out*/` | F machinery / S weights | Reuse code; re-aim fitness (C3); frozen SYN-1 gate is TB-006-specific |
| Nightly precompute | `precompute_nightly.py`, `store/nightly*/` (93 dirs) | S (member-specific) | Rebuild after retargeting |
| Harness adapter | `strategy_adapter.py` | F machinery | Reuse for the secondary socket; **apply per-symbol lot aggregation fix** (SCHD double-lot, COMMITTEE_REPORT.md:72-79; C5) |
| Replay runner | `run_replay.py` (start-date patch, cost overlay seed 4242, holdout env guard) | F | Extend: 2025-08-04 start + **epoch-aware prices shim (B3.2)** + ranking-score injection |
| Battery + stats + evidence | `battery.py`, `battery_plan.json`, `stats.py`, `e1_reads.py`, `run_e1_reads.py`, `assemble_evidence.py`, `gap_diagnostic.py` | F | Reuse; new plan + fresh look ledgers |
| Battery runs (R01–R19) | `runs_battery/` (full-window, 65 dec dates 2026-02-04..2026-06-10) | F (TB-006 reads) | Reference baselines |
| Smoke replays | `replay_out/` (19 dec dates) | S | Reference only |
| Deep training panels | `training/data/asset_features_history.parquet` (2014-08-29→2026-06-05), `historical_context.parquet`, `historical_gdelt.parquet` | F | Import |

**Rebuild list is short:** retargeted/decorrelated members + their OOF + nightly
precompute + EA fitness; the ranking-socket injection shim; the backfill-era prices shim;
the lot fix. Everything data-shaped imports.

---

## Bottom line for the design phase

1. **Socket:** real and clean at the engine API (`ranking_scores` in [0,1], blend 0.35
   deployed), buy-side only, no sell influence — but **selection-mostly, not
   selection-only**: threshold-count, composition, and the 0.80 high-vol gate leak into
   gross exposure. Rank-permutation expression is the structural C1 fix candidate.
2. **Window:** 2025-08-04 start is viable (clean $100k seed; monkeypatch pattern proven);
   **~189 paired decision dates full-window, 44 holdout** — but the 2025 leg requires an
   epoch-aware prices shim (else 0 days), runs a 25-name universe, no LLM artifacts, and
   sits entirely on the 4-epoch repaired record (caveat printed with every read).
3. **Reuse:** ~3.5 GB of final-grade data assets (GDELT, LLM, OHLCV, edges, panel,
   harness, battery) import as-is; TB-007's new spend is members + expression + two shims.
