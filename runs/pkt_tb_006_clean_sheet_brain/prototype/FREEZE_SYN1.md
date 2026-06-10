# SYN-1 CONFIGURATION FREEZE — the ONE pre-registered bake-off configuration

**Frozen:** 2026-06-10, before any holdout replay, per TOURNAMENT.md §4.1 ("one
pre-registered configuration") and §5 precondition (ii). Every rung decision below was
produced mechanically by a pre-registered rule; provenance cited. Changes after this
commit are protocol violations.

| Decision | Frozen value | Pre-registered rule that decided it |
|---|---|---|
| Genome | **EA champion** `ea/genome_2026-02-06.json` (eps 0.382, feature_gates OFF: {G1_themes, G3_tone}, member_gates all ON, vol_target 0.086, gross 0.516, no_trade_band 0.026, abstain 0.244, event_weight_cap 0.88, conviction_temp 4.0) | §8 adoption gate: champion − B0 = +0.876 > 1.0 × cross-fold sd 0.638; champion +1.0919 > B1 best +0.838 |
| Executive gate | **linear twin** (`exec_out/linear_twin.pt`, 121 params) | §7.3 ladder: MLP 5-seed val util 1.688e-4 did NOT beat twin 1.701e-4 → "ships if the MLP cannot beat it" |
| Executive MLP seeds | retained for LOFO fitness walks only (TR S2) | §8 fitness spec |
| Fine-tune | **none applied** (delta exactly 0 on all 5 seeds; changed_params=[]) | §7.6 + §4.6#9 de-claim rule |
| Transfer-B (record weights) | **CAST: weighted** (4/6 fold win); **GBM: uniform** (3/6, mean −0.0044) | §9.2 "ships only on a purged-validation win", per-learner |
| Transfer-A | **conjunctive gate DEAD** (0/36 family×fold); **R3-only screening ships** for EventHead/GBM routing | §9.1 falsifier: "if the conjunctive gate adds no walk-forward lift, the gate is dead and R3 screening remains" |
| LLM organ | ships (Stage-1: variance PASS, tone-proxy PASS 0.116, truncation PASS 1.5%); event-flag chattiness partial fail logged (fires 98% of days; TOURNAMENT kill condition "misses scheduled events" not met — 8/8 hit) | §3 Stage-1 + TOURNAMENT §4.3 LLM row |
| GDELT G1 dictionary | enters per genome (EA gated G1_themes OFF anyway); scorecard prints `0 (measured)` regardless of block arm | §4.6.3 placebo FAIL (28th percentile of 50 permutations) |
| Members | CAST-Small 16,483p (record-weighted), GBM-Cond uniform-w R3-only, EventHead R3-only (abstain 8-13%), RiskNet+ ridge | BUILD_SPEC §5 + rules above |
| Sigma source (replay) | trailing-21 proxy (the executive's training convention); `--sigma-source` knob names it in every manifest | wiring finding, journaled |
| Cost overlay | post-hoc, seed 4242 paired runs / 4243,4244 sensitivity | §4.1 + wiring adjudication |

**Reductions taken to date (final-line items):** CAST OOF seeds 3→2; deploy ensemble
5→3; 8-day gradient minibatches (semantics-preserving); LLM Tier-2 window reduced to
2024-08-15→2026-01-28 (latest contiguous fit under the $3.10 cap; $2.873 spent).

**Validation looks consumed at freeze:** see `validation_looks.jsonl` (count printed in
the dossier). **Holdout looks consumed: 0.**
