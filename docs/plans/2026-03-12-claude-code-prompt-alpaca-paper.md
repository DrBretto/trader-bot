> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Claude Code Prompt: Alpaca Paper Integration (Fractional-Ready)

Use this prompt with Claude Code to implement broker connectivity and paper-first rollout in this repo.

## Prompt To Paste Into Claude Code

You are working in `/Users/drbretto/Desktop/Projects/trader-bot`.

Implement Alpaca paper-broker execution with strict safety controls, while preserving existing simulated mode.

### Objectives
- Add broker integration so morning execution can route orders to Alpaca.
- Keep paper mode first-class and default-safe.
- Add fractional/notional trading support for small-account operation.
- Preserve backward compatibility with existing simulation paths.

### Current Repo Facts (important)
- The repo currently executes only simulated trades via `src/steps/paper_trader.py`.
- Buy sizing currently floors to whole shares in:
  - `src/steps/decision_engine.py` (`int(adjusted_dollars / current_price)`)
  - `src/steps/morning_executor.py` (`int(target_dollars / morning_price)`)
- Morning flow is currently in `src/steps/morning_executor.py`.
- Config is loaded in `src/handler.py`.

### Hard Requirements
- No live trading by default.
- Default behavior remains simulated unless explicitly enabled.
- Alpaca paper and Alpaca live must be separate modes with separate credentials.
- Fractional buys must be supported (notional and/or decimal qty).
- Do not hard-code account numbers, API keys, or secrets.
- Keep deterministic/idempotent order submission semantics.

### Implementation Tasks
1. Add broker abstraction
- Create a broker interface layer under `src/brokers/`:
  - `src/brokers/base.py`
  - `src/brokers/alpaca.py`
  - `src/brokers/router.py`
  - `src/brokers/__init__.py`
- Include methods for:
  - account health check
  - submit buy/sell
  - list/open positions
  - fetch fills/orders for reconciliation

2. Add configuration and mode controls
- Add a broker mode config with allowed values:
  - `simulated`
  - `alpaca_paper`
  - `alpaca_live`
- Make `simulated` the default if mode is missing.
- Add env/secrets config support for:
  - Alpaca paper key id/secret
  - Alpaca live key id/secret
- Update `.env.example` with placeholders.
- Update `infrastructure/secrets_setup.sh` to optionally store Alpaca secrets.

3. Wire morning execution to broker router
- Update `src/steps/morning_executor.py`:
  - route executions through broker adapter when mode is `alpaca_paper` or `alpaca_live`
  - keep existing simulated path using `paper_trader.execute_trade`
- Add robust pre-trade checks in broker mode:
  - account tradable/not blocked
  - buying power/cash checks
  - skip with clear logs on failure

4. Implement fractional/notional support
- Remove whole-share flooring assumptions for broker mode.
- Keep intent dollars as canonical for BUY intents.
- For Alpaca buys, submit `notional` (string/decimal-safe) when fractional mode is enabled.
- For sells/reductions, submit decimal qty based on held position.
- Add configurable precision and minimum notional guardrail.
- Ensure trade records and portfolio fields tolerate float shares.

5. Reconciliation and state updates
- In broker mode, reconcile local portfolio from broker account/positions after order cycle.
- Keep `portfolio_state` and dashboard artifacts coherent with broker truth.
- Maintain transaction logging with clear status fields (submitted/accepted/filled/rejected/canceled).

6. Safety rails
- Add kill switch env/config (e.g. `BROKER_TRADING_ENABLED=false` default false for non-sim modes unless explicitly true).
- Add per-order max notional cap.
- Add optional symbol allowlist.
- Use deterministic `client_order_id` to avoid duplicate submissions.

7. Add smoke test tooling
- Add `scripts/alpaca_paper_smoke_test.py` that:
  - validates auth (`/v2/account`)
  - optionally places a tiny paper order (default `$1` notional SPY)
  - optionally closes the test position
  - prints pass/fail summary and exits non-zero on failure
- Add command examples in docs.

8. Tests
- Update impacted tests for fractional behavior:
  - `tests/test_decision_engine.py`
  - `tests/test_paper_trader.py`
- Add new tests for broker routing and Alpaca payload behavior with mocked HTTP:
  - `tests/test_broker_router.py`
  - `tests/test_alpaca_broker.py`
- Add at least one morning-execution broker-mode test.

9. Documentation
- Update:
  - `README.md` (execution modes and safety defaults)
  - `docs/OPERATIONS.md` (paper/live mode operations + rollback)
  - `docs/DEPLOY.md` (Alpaca secrets and smoke test steps)

### Validation Commands
Run and report output summaries:
- `pytest tests/ -q`
- `python scripts/alpaca_paper_smoke_test.py --help`

If credentials are present locally, also run:
- `python scripts/alpaca_paper_smoke_test.py --account-check`

### Deliverable Format
At the end, provide:
1. A concise change summary by file.
2. Test results.
3. Any follow-up actions required before enabling live mode.