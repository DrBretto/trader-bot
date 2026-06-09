> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Claude Code Prompt: Drift Lockdown (AWS + Broker Cutover Paths)

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

## Objective
Harden the broker cutover toolchain so behavior cannot silently drift due to local AWS profile assumptions or undocumented fallbacks.

## Non-Negotiable Constraints
1. Never hardcode `default` AWS profile in code, docs, or examples.
2. Any CLI `--profile` input must fail fast with a clear error if the profile does not exist.
3. If `--profile` is omitted, rely on normal AWS credential chain (env/role/shared config) without overriding session behavior.
4. Keep all operations idempotent and non-destructive.
5. Do not print or store secrets.

## Files In Scope
- `scripts/bridge_cutover_continuity.py`
- `scripts/bootstrap_alpaca_from_sim_state.py`
- `docs/OPERATIONS.md`
- `docs/DEPLOY.md`
- `docs/plans/*.md` (only broker/cutover prompts)
- `tests/` (add regression guards)

## Required Work
1. Profile hardening:
   - Ensure both scripts gracefully handle invalid `--profile` (`ProfileNotFound` with actionable message + non-zero exit).
   - Ensure no silent fallback to `default`.
2. Docs hardening:
   - Replace any hardcoded default-profile examples with `--profile your-aws-profile`.
   - Add one line clarifying `--profile` is optional when using Lambda role or environment credentials.
3. Drift guard tests:
   - Add a test that fails if hardcoded `default` profile usage appears in broker/cutover scripts/docs.
   - Keep test scope focused to broker/cutover files to avoid unrelated legacy tooling churn.
4. Validation:
   - Run targeted tests for broker/cutover paths.
   - Report exact pass/fail counts.

## Acceptance Criteria
1. No hardcoded `default` profile strings remain in broker cutover code/docs.
2. Invalid `--profile` exits cleanly with explicit remediation text.
3. Broker/cutover tests pass after hardening.
4. A new regression guard exists to prevent profile-assumption drift from reappearing.

## Output Required
1. Changed files list.
2. Commands run.
3. Test results.
4. Residual risks, if any.