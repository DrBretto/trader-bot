## Context
Move this repo from simulated paper execution (`paper_trader`) to broker-connected execution, with Alpaca paper trading first and live trading later. User requirements:
- Fully automated (no manual order entry)
- Fractional/notional buying support for small account size
- Low-fee broker path
- Safe rollout (paper first, live later)

Current state:
- System is documented as paper-only.
- Buy sizing floors to whole shares in `decision_engine` and `morning_executor`.
- Trade execution is currently simulated in `paper_trader.execute_trade`.

## Target Outcome
- Night phase still produces intents.
- Morning phase can execute through a broker adapter.
- First broker adapter is Alpaca in paper mode.
- Fractional/notional order flow is supported end-to-end.
- Live mode exists but stays disabled by default.

## Plan
- [x] Add broker abstraction with explicit modes:
  - `simulated` (existing behavior)
  - `alpaca_paper`
  - `alpaca_live`
- [x] Implement Alpaca client wrapper for account/order/position calls with environment-safe base URL and credentials.
- [x] Add execution router in morning flow to choose simulated vs Alpaca execution.
- [x] Add fractional-aware sizing:
  - keep intent dollars as canonical
  - generate `qty` (decimal) and/or `notional` for Alpaca
  - avoid `int(...)` floors for buys
- [x] Add safety rails:
  - global kill switch
  - max order notional cap
  - symbol allowlist toggle
  - idempotent `client_order_id` format
  - account tradability check before orders
- [x] Add brokerage reconciliation:
  - account cash/equity status checks before orders
  - position sync after fills
  - local portfolio state updated from broker truth when broker mode is enabled
- [x] Add tests:
  - unit tests for sizing, adapter routing, payload building
  - mocked Alpaca API tests for submit/fill/retry/error handling
- [x] Add smoke test command for paper:
  - health check (`/v2/account`)
  - submit tiny fractional buy in paper mode
  - optional immediate close
- [x] Update docs and deployment secrets setup for Alpaca keys + mode controls.

## Acceptance Criteria
- Paper mode can place and confirm at least one fractional/notional order via Alpaca sandbox/paper environment.
- End-to-end morning run completes in `alpaca_paper` mode without manual intervention.
- Existing `simulated` mode remains functional and backward compatible.
- Live mode requires explicit config switch and separate live keys.
- Test suite passes for touched modules.

## Rollout Sequence
1. Ship adapter + router + tests.
2. Run paper smoke test with tiny notional.
3. Run full morning execution in paper mode for multiple sessions.
4. Validate logs, fills, and state reconciliation.
5. Enable live mode only after explicit user decision and funding.

## Execution Log

- 2026-03-12: Starting implementation. All source files read. Building broker abstraction first.
- 2026-03-12: Created `src/brokers/` package (base.py, alpaca.py, router.py, __init__.py).
- 2026-03-12: Updated decision_engine position sizing to keep dollars as canonical (min check before int floor).
- 2026-03-12: Updated morning_executor to accept broker adapter, route execution through broker or paper_trader.
- 2026-03-12: Added broker reconciliation (portfolio sync from broker positions after fills).
- 2026-03-12: Updated handler.py to load Alpaca secrets and create broker adapter for morning phase.
- 2026-03-12: Updated infrastructure/secrets_setup.sh with Alpaca paper + live key prompts.
- 2026-03-12: Created scripts/alpaca_paper_smoke_test.py with account-check, place-order, close-after modes.
- 2026-03-12: Created 3 test files: test_broker_router.py (16 tests), test_alpaca_broker.py (18 tests), test_morning_executor_broker.py (6 tests).
- 2026-03-12: All 180 tests pass (40 new + 140 existing, 0 regressions).
- 2026-03-12: Updated docs: OPERATIONS.md (broker modes section), DEPLOY.md (Alpaca setup), .env.example.
