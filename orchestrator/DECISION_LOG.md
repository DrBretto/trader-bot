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
