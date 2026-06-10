# 2026-06-09 — Dead-broker purge (PKT-TB-001)

## Context

A dead paper-broker book (removed Alpaca integration) still gets a portfolio value
computed and published fresh every day (`daily/latest.json` `portfolio_value` ≈ $95.5k
vs canon line ≈ $114k), making every fresh reader conclude there are two live books.
Packet: `committee/packets/PKT-TB-001-DEAD-BROKER-PURGE-V1-20260609.md`. Full trace:
`runs/pkt_tb_001_dead_broker_purge/FLOW_MAP.md` (sightings S1–S10).

## Plan

- [x] Trace lineage + readers; commit FLOW_MAP.md before code changes
- [ ] S1: drop `portfolio_value`/`positions_count` from `daily/latest.json` (both writers)
- [ ] S2: boundary rename `portfolio_value` → `sim_book_value` + `book_role` marker in
      freshly written `portfolio_state.json` (publish night/morning + midday writer);
      loader maps back; `dashboard_metrics._load_daily_states` accepts either name
- [ ] S3/S4: publish functions return `canon_total_value`; night/morning emails report
      the canon book (or "unavailable — dashboard held")
- [ ] S5/S6: midday email reads canon from dashboard.json; Lambda response bodies stop
      carrying dead `portfolio_value`
- [ ] S7: strip dead dollar values from log lines (handler, morning_executor, paper_trader)
- [ ] S8: drop `portfolio_value` from timeseries row builder; strip column from rolling
      timeseries republish; remove dead field from frontend timeseries type
- [ ] S9: weather prompt portfolio block sourced from canon dashboard metrics (omit when
      unavailable)
- [ ] S10: delete dead legacy loaders in publish_artifacts
- [ ] Single-book invariant test (new): published daily artifacts must not carry a
      non-canon portfolio-value-shaped top-level field
- [ ] Fix stale tests in blast radius (dashboard_metrics ×3, publish_artifacts_dates ×2;
      canon_promotion fake as test-side drive-by). Never touch test_transaction_costs.py
- [ ] Verify: night + morning phases against local fixtures; canon advances; suite green
- [ ] RETURN.md + verification notes in run dir; commit on ai/dead-broker-purge; NO deploy

## Execution Log

- 2026-06-09: trace complete; FLOW_MAP committed. Baseline 6 failed / 291 passed
  (matches packet's stale set; flaky PKT-TB-003 test passed this run).

## Follow-ups

- `automation/check_pipeline.py` prints latest.json `portfolio_value` (outside packet
  write surface) — will show $0.00 until updated; flagged in RETURN.md.
