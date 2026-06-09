> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Claude Code Prompt: Alpaca Paper Cutover Operations

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

The Alpaca broker integration is already merged. Do NOT re-implement broker code.

## Goal
Prepare and validate paper-broker execution operationally in AWS (not live money).

## Context
- Paper secrets already exist in AWS Secrets Manager:
  - `investment-system/alpaca-paper-key-id`
  - `investment-system/alpaca-paper-secret-key`
- Live secrets may remain unset.
- This run must stay paper-only.

## Tasks
1. Confirm current Lambda environment on `investment-system-daily-pipeline`.
2. Update Lambda environment variables to:
   - `BROKER_MODE=alpaca_paper`
   - `BROKER_TRADING_ENABLED=true`
   - Keep existing env vars intact (merge, do not overwrite unrelated keys).
3. Verify function configuration update completes successfully.
4. Run a manual **morning-execution** Lambda invoke:
   - payload should include `{"bucket":"investment-system-data","source":"morning-execution","region":"us-east-1"}`
5. Tail CloudWatch logs and extract broker execution evidence:
   - resolved broker mode
   - account checks / trade routing messages
   - trade count and any validation skips
6. Pull latest S3 artifacts and confirm morning report integrity:
   - `daily/latest.json`
   - today’s `daily/<date>/morning_execution_report.json` (or equivalent published morning report file)
   - portfolio state includes new trades or explicit no-trade rationale
7. If errors occur, diagnose and fix only operational/config issues (no broad refactors).

## Safety Constraints
- Do not enable `alpaca_live`.
- Do not introduce or print secret values.
- Do not change existing trading strategy logic.
- Keep edits minimal and scoped to ops/config/docs if needed.

## Validation Output Required
Return a concise report with:
1. Lambda env diff (keys changed only, values for non-sensitive keys only).
2. Morning invoke result summary.
3. CloudWatch proof lines for broker mode + execution.
4. S3 artifact checks.
5. Final status: "paper cutover operational" or "blocked" with concrete blocker.