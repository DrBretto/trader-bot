# PKT-TB-007 Phase C wave 2b — ORB-1 evolution stack

## Context
BUILD_SPEC_007 §4 (EA) + §5 (surrogate/fidelity/anchor) + TOURNAMENT_007 §4.4.3–4.4.7.
Wave-1 (expression layer: tilt_adapter, genome_007, lot fixes, B0-EXPR PASS) is on
ai/orthogonal-brain. Wave-2a (organs/OOF/masks) has NOT landed — build against the
documented contracts with synthetic inputs, flag clearly.

## Plan
- [x] Surgical refactor: extract tilt_adapter `_pipeline` closure → module-level
  `solve_tilt(...)` so the EA fitness walk and the replay adapter share ONE
  implementation of the expression solve (direction→demean→budget→M4→defensive→projection).
- [x] `surrogate_007.py` — chassis surrogate (§5.2): ohlcv feature panel (production
  feature defs), deployed-MLP batch scorer, base-arm walk (0.65/0.35 blend, regime
  multiplier, threshold-by-regime, top-N, vol-adjusted sizing), per-fold day-packs,
  fidelity L2a (fold vs live rank-IC), blend fidelity, L7 firewall verification,
  synthetic organ/mask generators, surrogate_manifest.json.
- [x] `ea_007.py` — (μ+λ) GA (TB-006 mechanics) on the 12-gene Genome007; floored
  paired-IR fitness (§4.2) through the shared solve path; B0 = individual #0 with
  FITNESS(B0) ≡ 0 asserted; median-of-top-8 champion; adoption gate
  max(1.4, √(2lnK_eff))×sd; 6-rotation gate (≥4/6 + pooled mean > 1×se) with
  per-rotation masks/roster (wave-2a contract); B1 random search; boundary-pin audit;
  every genome logged; blend-0 sensitivity entrypoint.
- [x] `anchor_007.py` — L3/FM5: surrogate vs real engine paired Δr on the pre-holdout
  leg (reuses B0_EXPR/incumbent + SMOKE_TILT/orb1_run1 replays); Pearson ≥ 0.8,
  slope ∈ [0.5,2.0], mean-equivalence ≤ 2×se_real.
- [x] `tests/test_ea_007.py` — B0≡0, walk-vs-adapter consistency 1e-9, floor behavior,
  rotation withholding, gate arithmetic, clamps + boundary-pin detector.
- [x] Smoke: P=8 G=2 + 2 rotations on synthetic organs; report fitness improvement,
  gates exercising, anchor + fidelity numbers; findings → validation_looks_007.jsonl.

## Execution Log
- Refactor done; expression tests stay green (16/16) + B0-EXPR re-verified by hash.
- Surrogate built; packs cached under prototype/store/surrogate_007/.
- L7 resolved by repo forensics (no formal manifest exists — ledgered): deployed
  ranking_mlp.pt SHA == repo commit 30fb365 blob; training cut at date < 2026-01-15.
- L2a live-era IC computed at the 5d horizon (21d would need prices ≥ 2026-03-11,
  violating the firewall pre-E2) — deviation ledgered.
- Anchor run on the fixed test genome + synthetic wave-1 organ inputs (the registered
  final form re-runs with champion + real organs at production time via the same command).
- EA smoke green: fitness improves over B0, B0 present every generation, all gates
  exercise; production entrypoint handed to the orchestrator (NOT run — organs pending).

## Follow-ups
- Wave-2a must land: store/nightly_007_folds/<D>.json (organ_inputs_007.v1, OOF-true),
  store/rotation_masks_007/rotation_<f>.json, store/roster_007.json.
- Orchestrator runs: `ea_007.py --mode full` (production EA + rotations + B1 + gates)
  after anchor re-run with real organs.
