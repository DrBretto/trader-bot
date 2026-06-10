# PKT-TB-006 Run Journal

Chronological record. Load-bearing for the sealed-incumbent rule: the design registration
commit must appear in this journal BEFORE any incumbent-strategy read.

- 2026-06-10 :: Branch `ai/clean-sheet-traders-brain` created from 8311170. Run dir scaffolded.
- 2026-06-10 :: PANEL.md — full 12-role panel declared before Phase 0.
- 2026-06-10 :: Orchestrator sealed-read state: NO reads so far of decision_engine.py,
  src/signals/, regime fusion, optimizer/fitness.py, summary.md strategy sections, prior
  committee returns under runs/, or incumbent parameter bundles.
- 2026-06-10 :: Phase 0 complete: ASSIGNMENT_BRIEF.md written by Analyst subagent (seal incidents: none).
  Key findings: universe.csv = 64 symbols (packet said 65); GDELT dark in live pipeline (daily counts
  endpoint 404; GKG v2 15-min files alive); deep history local (2014→2026); holdout ≈ 62 trading days.
- 2026-06-10 :: Infotropy Canon Liaison (role 6) returned designs/INFOTROPY_TRANSFER.md:
  6 mechanisms, 2 TRANSFERS / 3 RESTATEMENT / 1 NO-TRANSFER.
- 2026-06-10 :: Dispatching roles 7-10 (Meta-Evaluator Designer, Evolution Engineer, LLM Sentiment
  Engineer, Data Edge Scout) for blind mechanism proposals. Seal still intact: no incumbent reads.
- 2026-06-10 :: Roles 7-10 returned blind proposals (designs/proposals/): META_EVALUATOR (gating MLP,
  differentiable decision replay, ~600 params), EVOLUTION (29-gene disposition genome, OOF-frozen-head
  fitness, B0/B1/B2 ceremonial-proof battery), LLM_SENTIMENT (GKG slug+quotation organ, ~$0.15/mo,
  $2.40 backfill, Bedrock Haiku verified live), DATA_EDGE (GDELT-rich backfill + CBOE/COT/FRED shortlist).
  CRITICAL: GDELT local history ends 2026-02-04 — top-up backfill is a Phase C prerequisite.
  Seal intact: no incumbent reads by any role.
- 2026-06-10 :: Dispatching Architects Alpha/Beta/Gamma (independent complete designs, sealed rule absolute).
- 2026-06-10 :: Architects returned three independent complete designs (no cross-reading):
  DESIGN_ALPHA "FORECAST FIRST" (cross-sectional rank forecasting, 5-organ ensemble, CAST transformer
  45k params, meta-labeling gate), DESIGN_BETA "BOOKWRIGHT" (direct-allocation policy brain, 4 policy
  families emitting books, differentiable replay, evolution evolves reward-shaping library), DESIGN_GAMMA
  (information-funnel brain, event-theme transformer over GDELT/LLM event tokens, info-health-aware
  executive). All three adopted the specialist proposals with stated adaptations.
- 2026-06-10 :: REGISTRATION COMMIT FOLLOWS THIS ENTRY. Until this point, NO panel role and not the
  orchestrator has read: decision_engine.py, src/signals/, regime fusion, optimizer/fitness.py,
  summary.md strategy sections, prior committee returns under runs/, strategies.py, or parameter
  bundles. Seal incidents across all Phase 0/A subagents: none. After this commit the incumbent
  becomes readable as the bake-off opponent only.
- 2026-06-10 :: Phase B attacks complete (tournament/ATTACK_{SKEPTIC,TRAINING_REALIST,FEASIBILITY}.md).
  Headline kills: Gamma record-gate look-ahead (G-K1), holdout multiplicity arithmetic (E1 cross-fold
  becomes primary organ read), Gamma LLM-annotator token-budget break. All deployed-cost verdicts PASS.
- 2026-06-10 :: Synthesis converged on SYN-1 (graft/kill tables in TOURNAMENT.md; 18 killed components).
  Transformer slot: CAST-Small 16k params (Training Realist arithmetic governed over Feasibility cost
  preference; dissent + vindication path recorded). Battery: planned 14 replays / 5 retrains, caps 20/9.
  BEATS iff paired holdout t >= +1.0 and dSharpe > 0; TIES |t| < 1; LOSES t <= -1.0.
- 2026-06-10 :: PRE-REGISTRATION COMMIT FOLLOWS. TOURNAMENT.md bake-off criteria + BUILD_SPEC.md are
  committed BEFORE any build work or holdout read. Holdout-look ledger: 0. Validation-look ledger: 0.
- 2026-06-10 :: Phase C wave 1 done. data_layer.py (S3 snapshot cache 87 prices-days in window,
  OHLCV 64/64 via yfinance with split table, CBOE 7/7, FRED 12/12, COT 3 markets, 22 tests green).
  GDELT top-up 2026-02-05->2026-06-10 COMPLETE (124/124 days, 0 failed) -- holdout GDELT-dark gap closed.
  Deep backfill (2015->2026) running, ETA minutes. FINDINGS: snapshot gap 2026-05-11->05-22 (8 missing
  decision days, both bake-off arms affected identically); HY-OAS only 2023->; morning_prices rare in S3.
- 2026-06-10 :: HARNESS WIRING FINDINGS (post-registration, allowed): replay_engine._execute_intents
  applies NO transaction-cost model (fills at raw open); seed_portfolio hardcodes 2026-03-11 (the live
  book at holdout start). Adjudication, identical for both arms: E2 verdict pair = native 03-11-seeded
  runs; full-period context pair via identical runtime seed-date override (2 contingency slots); the
  pre-registered "same cost model, seeded rng 4242" implemented as an identical post-hoc cost overlay
  on both arms' executed fills, raw and cost-adjusted both reported, paired stats on cost-adjusted.
- 2026-06-10 :: DICTIONARY FREEZE COMMIT FOLLOWS (theme_to_sector, ACTOR_MAP, bucket_map, THEMES_FIN
  frozen before any validation-fold model selection, per TR G4 / TOURNAMENT 4.6.3).
- 2026-06-10 :: Phase C wave 2+3 done. Feature store (2,845 dates, 6/6 alignment tests), GDELT shift
  gate PASS (worst bucket energy_oil 4.83sd of 5.0 limit -- flagged to placebo read). Deep GDELT
  backfill COMPLETE 2015-02-18->2026-06-10, 4,131 days, 0 failures. Members trained: CAST-Small FINAL
  16,483 params, purged Spearman mean .106 / weekly rank-IC .105 (bar 0.02); ridge twin at noise
  (-.009). Shrink rungs taken (logged): OOF seeds 3->2, deploy 5->3, 8-day grad minibatches.
  INFOTROPY-A: conjunctive R1^R2^R3 gate fails all families all folds (R1 at base rate vs 0.5 bar);
  per pre-registered 9.1 falsifier the gate is DEAD, R3-only screening ships; verdict to scorecard.
  Transfer-B A/B: CAST marginal win (ships weighted), GBM negative on smoke (re-read post-LLM-merge).
  Executive 561 params + linear twin 121 + EA (27-gene, B0/B1 controls) built, 70/70 tests green.
  FINDINGS logged: beta_to=0.5 scale concern (registered constant; watch final training), COT-in-Z
  spec inconsistency (sec2.3 vs sec4 -- sec4 tensor spec governs, COT stays member-side), q95
  self-erasure on sparse nights. LLM Tier-1 resumed after rate-limit kill (seen-cache repaired by
  re-scoring 7 pilot-thread days ~$0.03).
- 2026-06-10 :: Phase C wave 4 done. Harness adapter + replay runners (holdout-guarded; smoke on
  pre-holdout dates only: both arms run clean under pandas 3.0, byte-identical reruns; syn1-B0 smoke
  turnover ~3.4%/day inside the 6% budget). Battery runner + stats + evidence reporting (134/134
  tests). PLACEBO GATE (4.6.3): FAIL -- real G1 dictionary 28th pct of 50 permutations; pre-registered
  consequence applies: G1 dictionary scores `0 (measured)` regardless of block arm. LLM Tier-1
  COMPLETE (132 artifacts, cum $0.726). STAGE-1: variance PASS, tone-proxy PASS (corr 0.116 << 0.8),
  truncation PASS (1.5%); event-flag chattiness partial fail (fires 98% of days; TOURNAMENT kill
  condition "misses scheduled events" NOT met -- 8/8 hit; organ ships, chattiness reported, flag
  channel expected ~0 attribution). Tier-2 launched (cum $0.726 < $3.00 gate). Pre-registration
  ambiguities logged: 4.2 BEATS rule gap (UNDEFINED branch printed, not coerced); 4.7 ensemble slot
  printed as member-tagged compound.
