# Daily Orchestrator Logs

Purpose:
- Session-by-session control-plane audit trail.
- Evidence index for tracker state transitions.

Naming:
- One file per day: `YYYY-MM-DD.md`

Required sections per file:
1. `Advanced` (what actually moved)
2. `Blockers` (explicit task IDs)
3. `Evidence` (paths/commands)
4. `Next Task ID`
