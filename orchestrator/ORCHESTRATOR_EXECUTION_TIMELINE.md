# Orchestrator Execution Timeline

Event-driven state machine for this repo.

## States

### S0_BOOTSTRAP
Entry condition:
- Control-plane files missing or stale.

Exit event:
- Bootstrap files created/updated and tracker seeded.

Next state:
- `S1_RECONCILE`

### S1_RECONCILE
Entry condition:
- Tracker exists with at least one unlocked task.

Actions:
- Reconcile tracker, decision log, git state, and artifact evidence.

Exit event:
- Exactly one unlocked task selected.

Next state:
- `S2_SCAFFOLD_PACKET`

### S2_SCAFFOLD_PACKET
Actions:
- Build bounded execution packet:
  - objective
  - inputs
  - required outputs
  - acceptance checks
  - stop condition

Exit event:
- Packet committed to file and linked in tracker evidence.

Next state:
- `S3_WAIT_EXECUTION`

### S3_WAIT_EXECUTION
Actions:
- No speculative changes.
- Wait for execution evidence from external runner or local run completion.

Exit event:
- Evidence artifact arrives, or blocker threshold reached.

Next state:
- `S4_RECONCILE_RESULTS` or `S5_BLOCKED`

### S4_RECONCILE_RESULTS
Actions:
- Verify acceptance checks against evidence.
- Mark row done only if checks pass.

Exit event:
- Row closed with evidence path.

Next state:
- `S1_RECONCILE` (for next unlocked task)

### S5_BLOCKED
Entry condition:
- Same blocker persisted for 2 consecutive sessions.

Actions:
- Mark blocker explicitly in tracker.
- Create fallback path packet.
- Log decision entry.

Exit event:
- Fallback task unlocked.

Next state:
- `S2_SCAFFOLD_PACKET`

## Current Sequence State
`S3_WAIT_EXECUTION` for continuity hotfix deploy verification packet.
