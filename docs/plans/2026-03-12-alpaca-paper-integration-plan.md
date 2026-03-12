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
- [ ] Add broker abstraction with explicit modes:
  - `simulated` (existing behavior)
  - `alpaca_paper`
  - `alpaca_live`
- [ ] Implement Alpaca client wrapper for account/order/position calls with environment-safe base URL and credentials.
- [ ] Add execution router in morning flow to choose simulated vs Alpaca execution.
- [ ] Add fractional-aware sizing:
  - keep intent dollars as canonical
  - generate `qty` (decimal) and/or `notional` for Alpaca
  - avoid `int(...)` floors for buys
- [ ] Add safety rails:
  - global kill switch
  - max order notional cap
  - symbol allowlist toggle
  - idempotent `client_order_id` format
  - dry-run validation mode
- [ ] Add brokerage reconciliation:
  - account cash/equity status checks before orders
  - position sync after fills
  - local portfolio state updated from broker truth when broker mode is enabled
- [ ] Add tests:
  - unit tests for sizing, adapter routing, payload building
  - mocked Alpaca API tests for submit/fill/retry/error handling
- [ ] Add smoke test command for paper:
  - health check (`/v2/account`)
  - submit tiny fractional buy in paper mode
  - optional immediate close
- [ ] Update docs and deployment secrets setup for Alpaca keys + mode controls.

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

