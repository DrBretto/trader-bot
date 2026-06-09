> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Claude Code Prompt: Continuity Hotfix Deploy + Verification

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

## Goal
Deploy the continuity hotfix so dashboard equity/value remain continuous across the simulated->Alpaca cutover while preserving raw broker truth.

## Scope
Use only the already-implemented local changes in:
- `src/utils/dashboard_metrics.py`
- `src/steps/publish_artifacts.py`
- tests and docs touched in this patch

Do not redesign logic; deploy + verify.

## Required Steps
1. Run full tests:
   - `.venv/bin/python -m pytest -q`
2. Deploy Lambda container:
   - `./infrastructure/lambda_deploy_container.sh`
3. Trigger morning rebuild once (request/response invoke).
4. Validate output from:
   - `s3://investment-system-data/dashboard/dashboard.json`
   - `s3://investment-system-data/daily/latest.json`
   - `s3://investment-system-data/daily/2026-03-12/portfolio_state.json`

## Acceptance Checks
1. `dashboard.metrics.total_value` is continuity-adjusted (around prior continuity, not raw ~100k reset).
2. `dashboard.metrics.broker_total_value` equals broker-reconciled raw equity.
3. `dashboard.equity_curve[-1].value` equals continuity-adjusted value.
4. `dashboard.equity_curve[-1].raw_value` equals raw broker equity.
5. `dashboard.reset_boundary` should remain `null` unless a separate real reset marker exists.

## Guardrails
1. Do not hardcode default-profile CLI usage anywhere.
2. Do not mutate/de-delete historical S3 keys.
3. Do not alter broker account cash/positions as part of this deploy task.

## Output Required
1. Commands run.
2. Test result summary.
3. Before/after key fields from dashboard JSON.
4. Any residual mismatch if acceptance checks fail.