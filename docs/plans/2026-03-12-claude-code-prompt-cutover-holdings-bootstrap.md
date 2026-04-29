# Claude Code Prompt: Simulated -> Alpaca Holdings Bootstrap

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

## Goal
After continuity metrics are bridged, bootstrap Alpaca paper holdings so broker execution starts from approximately the same portfolio state as the last simulated day.

## Scope
This prompt handles **position/bootstrap continuity**.
Use it after (or together with) the continuity-metrics bridge prompt.

## Requirements
1. Non-destructive, idempotent behavior.
2. Dry-run first, apply second.
3. Do not hard-code or print secret values.
4. Respect safety rails (`max_order_notional`, symbol allowlist, kill switch).

## Implementation Tasks
1. Add bootstrap script:
   - `scripts/bootstrap_alpaca_from_sim_state.py`
   - Inputs:
     - `--bucket` (default `investment-system-data`)
     - `--region` (default `us-east-1`)
     - `--source-date YYYY-MM-DD` (optional: defaults to latest pre-cutover simulated date)
     - `--mode scaled|exact` (default `scaled`)
     - `--dry-run` / `--apply`
   - Behavior:
     - Read source simulated portfolio holdings from `daily/<source-date>/portfolio_state.json`.
     - Read current Alpaca paper account cash/equity and positions.
     - Compute target bootstrap orders:
       - `scaled`: match source holdings by weight using current Alpaca equity.
       - `exact`: match source-dollar exposure (not exact share counts).
     - Generate proposed orders as notional buys/sells (fractional-aware).
     - Skip symbols unavailable/not tradable with clear reasons.
     - In apply mode, submit orders with deterministic `client_order_id`.
     - Write run artifact:
       - `daily/<today>/alpaca_bootstrap_plan.json` (dry-run)
       - `daily/<today>/alpaca_bootstrap_result.json` (apply)

2. Add guardrails:
   - default max bootstrap notional per order
   - max total notional cap for a single bootstrap run
   - optional `--symbol-allowlist` override

3. Add tests:
   - `tests/test_bootstrap_alpaca.py`
   - Cover:
     - scaled target computation
     - insufficient cash handling
     - idempotent repeat behavior (already-positioned symbols)
     - dry-run artifact generation

4. Add docs:
   - `docs/OPERATIONS.md`:
     - section: “Bootstrapping Alpaca to Simulated Portfolio”
     - commands for dry-run/apply
     - warnings about market drift/slippage and partial fills

## Validation Steps
1. Run tests.
2. Dry-run with current bucket.
3. Apply with conservative caps.
4. Reconcile and verify:
   - Alpaca positions roughly match source allocation intent.
   - Morning execution continues normally in `alpaca_paper`.

## Output Format
Provide:
1. File change summary.
2. Dry-run plan summary (counts, skipped symbols, total notional).
3. Apply results (submitted/filled/rejected counts).
4. Any residual drift vs source simulated holdings.
