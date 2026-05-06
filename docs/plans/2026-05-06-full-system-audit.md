# Plan — Trader Bot full-system audit + repair (2026-05-06)

Branch: `ai/full-system-audit-20260506`
Packet: `Infotropy Book/book-factory/use_lane_outputs/runs/20260506_trader-bot-full-system-algorithm-audit-repair-v1/handoff/PKT-ULP-TRB-FULL-SYSTEM-ALGORITHM-AUDIT-REPAIR-20260506.md`

## Context

Operator emergency: live dashboard reports ~75% cash / 25% invested while expert metrics confidently report `risk_on_trend` with `position_size_modifier ≈ 0.85` (target_gross_exposure 0.85). Public dashboard is internally inconsistent. Algorithm itself is fine; the surrounding plumbing is broken. Seven prior audits returned "good to go" without actually testing the live behavior.

May 4-5 sell-down liquidated profitable positions on single-day health dips:

- May 2 night queued SELL ARKK (h=0.26), SLV (h=0.15), XBI (h=0.29), XLK (h=0.29 — held +19% gain) → executed May 4 morning. All HEALTH_COLLAPSE.
- May 4 night queued SELL VUG (full position, ~$8.5k gain), XRT (full position, +4% gain), and the now-dust SLV/XLK → executed May 5 morning. Two SELLs failed broker-side because position was already dust; VUG/XRT got partially filled.
- Net: ~$31,500 of profitable positions liquidated; 75% of equity sitting in cash in a confirmed risk-on regime.

Live broker truth (Alpaca paper) on 2026-05-06: equity $97,053; positions = FXI (213 sh material), XLF (152 sh material), and 16 dust positions ≤ $0.0001 each. App's portfolio_state still claims VUG=28.77 sh / $2,425 and XRT=73.09 sh / $6,169 — neither exists at the broker. Dashboard is rendering fictional holdings.

## Three confirmed root causes

### F-NEW-1 (P0) — `sell_health_days` is dead code

`config/decision_params.active.json` carries `sell_health_days: 3` and the optimizer's `param_inventory.json` registers it as a tunable parameter. Source code references: zero. `src/steps/decision_engine.py:299-334` reads only `sell_health_threshold`; the persistence gate it implies has never been wired. SELL/HEALTH_COLLAPSE fires on the first day a holding's health dips ≤ 0.35, regardless of regime confidence or duration.

**Fix:** wire a per-holding `consecutive_below_health_days` counter in portfolio holding state. Increment when current_health ≤ threshold; reset when above; only emit SELL when count ≥ `sell_health_days`. Preserve through broker reconciliation. Mutate the holding dict in `evaluate_holdings` so the increment persists into the published portfolio_state.

### F-NEW-2 (P0) — Partial-fill order handling treats `partially_filled` as terminal

`src/steps/morning_executor.py:36-44` lists `partially_filled` in `BROKER_TERMINAL_ORDER_STATUSES`. Alpaca's `partially_filled` is **not** terminal — the order continues working. The `_await_broker_order_update` loop exits on the first partial, the reconciliation snapshots the broker mid-fill, and subsequent fills (even within the same minute) are never reconciled by the trader-bot until the next morning run. With dust-cleanup behavior in Alpaca's matching, the remainder gets filled in seconds, leaving Alpaca with effectively zero quantity while the app still believes it holds the partial-fill snapshot.

**Fix:** remove `partially_filled` from the terminal-status set. Increase `_await_broker_order_update` deadline to 30s. After the morning execution loop, before reconciliation, poll `broker.list_orders(status='open')` until every submitted order has settled (or a hard deadline is reached) so the post-execution snapshot reflects final state.

### F-NEW-3 (P1) — Night phase never reconciles against broker truth

`src/handler.py:117` loads portfolio_state via `paper_trader.load_portfolio_state` (S3 file read) and passes it to the decision engine. Step 11 (`handler.py:442-446`) reloads the same S3 file and updates valuations from prices_df. **No call to `_reconcile_portfolio_from_broker` anywhere on the night path.** The decision engine evaluates holdings the broker no longer holds; the dashboard publishes those phantoms.

**Fix:** if broker mode is active, reconcile portfolio_state against broker positions/account immediately after `load_portfolio_state` in the night phase, **before** decision-engine. Drop the redundant `load_portfolio_state` reload in Step 11 so per-day mutations from `evaluate_holdings` (peak_price, consecutive_below_health_days) survive into the published state.

## Plan

1. ✅ Read packet + read-first list (CLAUDE.md, PLAN.md, POSTMORTEMS.md, DEPLOY.md, prior RETURN docs, decision_params.active, decision_engine.py).
2. ✅ Capture current truth: git status, EventBridge state, Lambda image, S3 daily/2026-05-{02,04,05,06}, dashboard.json + Alpaca paper account/positions, Playwright baseline screenshot.
3. ✅ Reconstruct May 4-6 timeline from artifacts (May 2 night → May 4 morning sells; May 4 night → May 5 morning sells; broker partial-fill divergence; May 5/6 night = 0 actions because health recovered).
4. **Wire `sell_health_days` persistence gate** in `src/steps/decision_engine.py::evaluate_holdings`.
5. **Treat Alpaca `partially_filled` as non-terminal** and add post-execution settle-wait in `src/steps/morning_executor.py`.
6. **Add night-phase broker reconciliation** in `src/handler.py::_run_night_phase`; drop the redundant Step 11 reload so holding-level state mutations survive.
7. **Tests** for each fix:
   - `tests/test_decision_engine.py::TestSellHealthPersistence` — single-day dip does NOT sell; three-day persistent low DOES sell; recovery resets counter.
   - `tests/test_morning_executor_broker.py::TestPartialFillHandling` — `partially_filled` is non-terminal; await loop continues; settlement wait observed.
   - `tests/test_morning_executor_broker.py::TestReconcilePreservesHealthCounter` — `_reconcile_portfolio_from_broker` carries the new counter through.
8. **Dry-run replay** of decision_engine against today's live signals + broker-truth holdings; confirm no liquidation under risk-on with persistent-low recovery cycle.
9. **Deploy** via `infrastructure/lambda_deploy_container.sh` (existing container path) + frontend rebuild + S3 sync (no frontend code changes; build still runs to keep tooling honest).
10. **Playwright verification** post-deploy: holdings now match broker truth (FXI + XLF only); cash matches Alpaca; total_value matches broker equity; trade log unchanged.
11. **Reports** in run dir; commit; push; close_run with git_state.

## Constraints honored

- No new deployment method.
- No new persistent infra.
- No manual cosmetic totals or hidden chart overrides.
- Paper mode only.
- Scheduler cadence unchanged.
- Playwright MCP used for visible verification.

## Execution Log

- 2026-05-06 09:30 — read-first list completed.
- 2026-05-06 09:35 — branch created, plan doc written.
- 2026-05-06 09:40 — fix-1 sell_health_days persistence wired in evaluate_holdings.
- 2026-05-06 09:45 — fix-2 partial_filled handling + settle-wait in morning_executor.
- 2026-05-06 09:50 — fix-3 night-phase broker reconciliation in handler.
- 2026-05-06 09:55 — tests written, run.
- 2026-05-06 10:00 — dry-run replay against today's data.
- 2026-05-06 10:10 — container build + Lambda deploy + frontend sync.
- 2026-05-06 10:25 — Playwright post-deploy verification.

## Follow-ups

- **Buy gate starvation** in confirmed risk-on: candidate health scores currently 0.39-0.50 vs `min_health_buy=0.60`, producing voluntary cash drag. Health-model recalibration is out of this packet's scope; the consistency-guard log line will surface the contradiction so it stops looking like "the algorithm decided cash."
- **Continuity bridge `total_value` vs `broker_total_value` gap** ($106,049 vs $97,054) is the F-10 carryover from the 2026-04-29 audit. Frontend caption work is operator-side WIP. With night-phase reconciliation landed, `broker_total_value` will refresh daily and the gap will be visible against truth.
- **Optimizer sweep** for `sell_health_threshold` × `sell_health_days` once persistence is live; current `sell_health_days=3` is the doc-implied default but never empirically validated.
