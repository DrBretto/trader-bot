## Context

After switching from simulated to Alpaca paper execution (Phase 8), the portfolio shows a discontinuity:
- Pre-cutover (2026-03-11): $103,025.53 with 8 holdings
- Post-cutover (2026-03-12): $100,000.00 with 0 holdings (fresh Alpaca paper account)

Two problems: (1) dashboard metrics show false ~3% loss, (2) Alpaca has no positions.

## Plan

### Part A: Cutover Continuity Bridge
- [x] `src/utils/cutover_bridge.py` — Pure functions for cashflow computation and patch logic
- [x] `scripts/bridge_cutover_continuity.py` — CLI script for dry-run/apply
- [x] `tests/test_cutover_bridge.py` — 13 tests

### Part B: Holdings Bootstrap
- [x] `scripts/bootstrap_alpaca_from_sim_state.py` — CLI script for dry-run/apply
- [x] `tests/test_bootstrap_alpaca.py` — 15 tests

### Part C: Docs
- [x] `docs/OPERATIONS.md` — Added Cutover Bridge and Bootstrap sections

### Part D: Execution
- [x] Bridge dry-run — cashflow -$3,025.53 confirmed
- [x] Bridge apply — patched portfolio_state.json in S3 with external_cashflow and marker
- [x] Bootstrap dry-run — 8 orders, $37,682.52 total
- [x] Bootstrap apply — 8/8 orders filled (pending_new → filled)
- [x] Bootstrap top-up attempted — correctly blocked by deterministic client_order_id (same-day idempotency)
- [ ] Lambda morning re-run to reconcile and rebuild dashboard

## Execution Log

- All code and tests implemented. 28/28 tests pass.
- Bridge applied: external_cashflow=-3025.53, marker=continuity-bridge-v1:2026-03-12
- Bootstrap applied: 8 positions filled, $37,678 invested, $62,317 cash (62.3%)
- Three positions (XLV, XLI, IYT) capped at $5k per-order vs targets of $6.8-8.4k — acceptable for initial bootstrap.
- Top-up run correctly rejected by Alpaca's client_order_id uniqueness constraint (safety rail working).

## Follow-ups

- After bootstrap, run Lambda morning execution to reconcile and rebuild dashboard.
- Verify equity curve is smooth across cutover in dashboard.
