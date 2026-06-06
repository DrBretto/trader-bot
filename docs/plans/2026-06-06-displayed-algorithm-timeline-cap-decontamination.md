# 2026-06-06 — Displayed algorithm/fantasy line: cap decontamination + durability

## Context
The displayed optimized-champion `value` line on https://trader-bot.infotrope.io was
contaminated by the live broker's $5,000-per-order-capped, cash-heavy account. Two leaks:
1. `extender.py::_broker_return_extend` scaled interior-gap and tail dates of the champion
   line by the **broker's raw_value daily return** — a chronically cash-heavy, capped book.
   Live-active on 2026-06-01 and 2026-06-06 (value-return == raw_value-return to 1e-9).
2. Durability: `extend_dashboard` (the three-line replay) had **no committed call site** in
   the production publish path — the deployed Lambda ran it only from a working-tree
   `COPY src/`. A fresh checkout + rebuild would silently drop the corrected line.

Operator decision (2026-06-06): keep the **2026-03-11 broker anchor** (do NOT replay from
true inception — that was a slightly different model and ~6 weeks of real history is the
honest window). Fix the leak + make it durable.

## Plan
- `src/utils/three_line_replay/extender.py`: replace `_broker_return_extend` with
  `_market_return_extend` — scale gap/tail dates by the **benchmark (fully-invested SPY)**
  daily return already stored on each equity_curve row, not the broker raw_value. Skip
  (flat-hold) where benchmark is absent/<=0. Update the single call site + comment.
- `src/steps/publish_artifacts.py`: wire `extend_dashboard(s3.s3, dashboard_data)` into the
  night (~:724→:733) and morning (~:873→:879) publish paths, after build, before the publish
  guard. (`s3.s3` is the boto3 client per s3_client.py:16.)
- `tests/test_dashboard_metrics.py`: rewrite `TestBrokerReturnExtendInteriorGap` for
  benchmark-return semantics.
- `tests/test_publish_calls_extender.py`: new structural guardrail asserting both publish
  functions call `extend_dashboard` (survives `COPY src/` rebuilds; fails CI on regression).

## Execution Log
- (in progress) edits being applied.

## Follow-ups
- Rebuild + push the Lambda container so future nightly runs auto-produce the corrected,
  durable line (immediate fix lands via one-shot re-extend of live dashboard.json).
- `target_gross_exposure`/`effective_exposure_multiplier` are computed but never applied
  (dead governor); `max_sector_weight`, `leveraged_constraints.*`, `reduce_health_drop` are
  dead config. Out of scope here; flagged for the operator.
