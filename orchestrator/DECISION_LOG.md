# Decision Log

## 2026-03-12

### D-2026-03-12-01 Orchestrator Control Plane Bootstrapped
- Context: repo had active broker/cutover work but no project-native orchestration registry.
- Decision: create `orchestrator/` control plane with bootstrap, timeline, tracker, self-update trigger, and daily logs.
- Impact: sequence and evidence discipline become explicit and durable across sessions.

### D-2026-03-12-02 Single Active Packet Policy Enforced
- Context: multiple in-flight artifacts existed; risk of parallel ambiguous execution.
- Decision: enforce one active row (`T020`) and event-driven state `S3_WAIT_EXECUTION`.
- Impact: prevents scope creep and forces reconcile before next unlock.

### D-2026-03-12-03 Continuity Deploy Verification Prioritized
- Context: continuity logic patch exists locally and requires deploy verification for production truth.
- Decision: set continuity deploy packet as immediate next execution (`T020`) before docs closure/commit packaging.
- Impact: acceptance evidence precedes closure and checkpoint tasks.

## 2026-03-13

### D-2026-03-13-01 Tracker Backfill Required for Verified Checkpoint
- Context: git checkpoint commit `7781143` existed, but tracker row `T050` had not yet been reconciled to evidence.
- Decision: allow administrative tracker backfill when a completed task has durable evidence and the stale tracker state would otherwise create drift.
- Impact: tracker now matches actual repo state without reopening already-finished work.

### D-2026-03-13-02 Trusted User Sharing Direction Recorded
- Context: future sharing is intended for a small trusted circle, not a public product, and needs a durable architecture note before implementation.
- Decision: record the recommended future path as `Cognito + Google sign-in` for app identity and `Alpaca OAuth / Connect` for broker authorization.
- Impact: future multi-user work now has a written source of truth and can be sequenced without rediscovering auth assumptions.

## 2026-03-18

### D-2026-03-18-01 T040 Closed and Cutover Continuity Epic Finalized
- Context: T040 (close cutover continuity epic and normalize plan docs) had been pending since 2026-03-13. The work it describes was functionally complete but the tracker row and plan docs had not been formally closed.
- Decision: close T040, mark plan docs as complete, add dashboard truth-surface labels and cutover marker as part of Phase 1 truth stabilization.
- Impact: the continuity epic is formally closed. Dashboard now honestly labels the dual truth surface created by the Alpaca cutover bridge.

### D-2026-03-18-02 External Takeover Governance Adopted
- Context: the repo is now managed as a Lab external project (ULP-TRB) with structured execution packets.
- Decision: accept the Phase 1 truth stabilization pass as the first externally governed execution step.
- Impact: future execution work follows Lab-side packet discipline while the local orchestrator handles session mechanics.
