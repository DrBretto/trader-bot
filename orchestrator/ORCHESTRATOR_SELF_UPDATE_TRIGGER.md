# Orchestrator Self-Update Trigger Protocol

## Trigger Conditions
Update control-plane documents when any of the following occurs:

1. Workflow drift:
   - Tracker sequence no longer reflects actual dependency order.
2. Repeated blocker:
   - Same task blocked in 2 consecutive sessions.
3. New operating policy:
   - Source-of-truth order, guardrails, or handoff contract changes.
4. Structural repo change:
   - New major subsystem changes task graph (broker mode, deployment path, optimizer path).
5. Evidence contract mismatch:
   - A task cannot provide evidence under current tracker schema.

## Mandatory Update Steps
1. Add a decision entry in `orchestrator/DECISION_LOG.md`.
2. Update impacted control-plane docs:
   - `ORCHESTRATOR_BOOTSTRAP.md`
   - `ORCHESTRATOR_EXECUTION_TIMELINE.md`
   - `ORCHESTRATOR_PROGRESS_TRACKER.tsv`
3. Record update in `orchestrator/daily/<YYYY-MM-DD>.md`.
4. Link changed files as evidence in the active tracker row.

## Integrity Rules
- No retroactive status rewrites without explicit decision log note.
- No removal of blocker history.
- Keep one active task row maximum after update.
