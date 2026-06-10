# 2026-06-09 — Expansion Atlas (PKT-TB-005)

## Context

PKT-TB-005-EXPANSION-ATLAS-V1-20260609: the operator is open to widening the
system's aperture (more information, more models, different angles) if and only
if a candidate can be shown to add real decision value. This run produces a
wide, honestly-triaged atlas of expansion candidates, carries the top few to
counterfactual replay evidence (E1+E2 per committee/EVIDENCE_PROTOCOL.md),
including the mandatory dumb-baseline-vs-ML-ensemble check, and parks the rest
with named revival conditions. Evidence-only: nothing lands in production paths.

## Plan

- [x] Read-first list + substrate survey (replay harnesses, S3 daily artifact
      depth 2025-08-04→2026-06-09, holdout boundary 2026-03-11, per-model probs
      stored in inference.json, regime/health models trained on rule pseudo-labels)
- [x] Declare 6-member panel (PANEL.md) before deliberation
- [x] Enumeration pass: Information Scout (D-01..D-24, sources live-probed),
      Model Architect (M-01..M-16), Portfolio Strategist (S-01..S-16) as real
      subagents; returns saved verbatim under panel_returns/
- [x] Honesty pass: Skeptic (graveyard prior, holdout-looks rule, baseline-design
      audit) + Feasibility Auditor (envelope grades) on the combined list
- [x] Analyst synthesis → ATLAS.md triage ledger (56 → 7 piloted / 40 parked / 9 killed)
- [x] Pilots to evidence (5 pilots incl. mandatory baseline check; 11/12 holdout
      arm-reads), pinned data cache, manifests per EVIDENCE_PROTOCOL → PILOT_EVIDENCE.md
- [x] ROADMAP.md (Wave 0 riders / Wave 1 next-window reads / Wave 2 operator-gated)
- [x] COMMITTEE_REPORT.md (PKT-TB-002 field set + required final line)
- [x] Commit run dir on ai/expansion-atlas (temp-index plumbing; concurrent
      PKT-TB-004 session owned the checkout's branch)

## Execution Log

- 2026-06-09: Branch ai/expansion-atlas created. Run dir scaffolded.
- 2026-06-09: Pilot harness built (runs/pkt_tb_005_expansion_atlas/pilots/common.py)
  — faithful copy of optimizer/replay.py loop + recording/hooks; validated
  byte-identical to production run_replay_for_dates on the holdout segment.
- 2026-06-09: Data cache pinned (data_cache.pkl: 194 aligned snapshots
  2025-08-04→2026-06-09 + per-date context rows, from S3 daily artifacts).
- 2026-06-10: Enumeration returns in (56 candidates). Notable substrate
  discoveries: GDELT production fetch path 404s (Scout D-10, live-probed);
  engine cannot top up held winners (Strategist S-03); all 65 symbols have
  leverage_flag=0 so the leveraged param stack is dead code (S-11); regime AND
  health models are distillations of rule-based labelers (Architect M-01/M-13).
- 2026-06-10: Baseline-check battery v1 ran without the active bundle's
  ranking_blend (not production-faithful) — preserved as a logged variant;
  the v1-vs-v2 contrast became the run's headline finding (ranking layer).
- 2026-06-10: Skeptic audit forced battery v2 (rules_storedsizing arm,
  deep-era conditioning, t≥3.0 standard) after discovering 128/194 stored
  inference days are rule-fallback one-hots.
- 2026-06-10: Five pilots to E1+E2: baseline tie; health rules > AE
  (noise-with-direction); ranking layer unsupported + worse path; sleeve
  evidence-incomplete (found production partial-SELL landmine in
  paper_trader); universe removals harmless (win by tie-favors-removal).
- 2026-06-10: Concurrent PKT-TB-004 session switched the checkout branch
  mid-run and committed flags-off changes to imported modules; control-replay
  byte-identity verified across the boundary; this run committed via
  temp-index plumbing without touching the other session's checkout.

## Follow-ups

- See runs/pkt_tb_005_expansion_atlas/ROADMAP.md for the sequenced
  parked-promising list (written at close).
- Production integration of any proven pilot is a future packet authored from
  this return (packet hard constraint).
