> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Claude Code Prompt: Cutover Continuity Bridge (Simulated -> Alpaca Paper)

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

## Goal
Preserve pre-cutover performance continuity in dashboard metrics while keeping Alpaca paper as execution source of truth.

Context:
- Pre-cutover period used simulated execution.
- Post-cutover period uses `BROKER_MODE=alpaca_paper`.
- Alpaca account was fresh/reset at cutover, causing a discontinuity in raw portfolio value.

We need to carry forward cumulative metrics cleanly (no destructive deletes), and keep benchmark comparison coherent.

## Requirements
1. Do not hard-delete historical S3 artifacts.
2. Keep broker truth for current cash/positions.
3. Restore continuity for cumulative return metrics across cutover.
4. Keep SPY benchmark comparison coherent with the same effective capital stack.
5. Make the operation idempotent (safe to re-run).

## Implementation Tasks
1. Add a migration script:
   - `scripts/bridge_cutover_continuity.py`
   - Inputs:
     - `--bucket` (default `investment-system-data`)
     - `--region` (default `us-east-1`)
     - `--cutover-date YYYY-MM-DD` (required)
     - `--dry-run` (default true unless `--apply`)
     - `--apply`
   - Behavior:
     - Load `daily/<cutover-date>/portfolio_state.json` (cutover day) and previous trading day's portfolio state.
     - Compute one-time external cashflow bridge:
       - `external_cashflow = cutover_value - previous_value`
       - This neutralizes the cutover jump in cashflow-adjusted return math.
     - Set explicit continuity marker on cutover and future state:
       - e.g. `metrics_reset_id="continuity-bridge-v1"` (or a more precise ID including cutover date).
       - Important: use marker strategy that preserves desired segment continuity; do not accidentally isolate to one row.
     - Ensure benchmark continuity fields exist and are sensible:
       - preserve/repair `benchmark_start_price` and `benchmark_shares` if missing
       - avoid resetting benchmark to a new baseline unless explicitly requested.
     - Write back patched cutover-day state (and only what is required).
     - Print a before/after diff summary.

2. Add helper module for deterministic patching:
   - `src/utils/cutover_bridge.py`
   - Pure functions with tests for:
     - computing bridge cashflow
     - idempotent patch merge
     - benchmark field preservation

3. Add tests:
   - `tests/test_cutover_bridge.py`
   - Cover:
     - large discontinuity case
     - already-patched idempotent case
     - missing benchmark fields case
     - dry-run output case

4. Add docs:
   - Update `docs/OPERATIONS.md` with a “Cutover Continuity Bridge” section:
     - when to use
     - command examples
     - cautions (do not use in live without understanding accounting impact)

5. Validation steps (run and report):
   - Unit tests for new module/tests.
   - Dry-run against current bucket/date.
   - Apply mode once.
   - Recompute/publish dashboard artifacts by invoking morning phase once (or documented equivalent), then verify:
     - continuity in return metrics
     - broker-reconciled state still true
     - no secret leakage

## Output Format
At completion provide:
1. Files changed.
2. Dry-run and apply summaries.
3. Verification evidence from resulting `portfolio_state.json` and `daily/latest.json`.
4. Any residual caveats (especially chart-level raw value discontinuity vs cashflow-adjusted metrics).