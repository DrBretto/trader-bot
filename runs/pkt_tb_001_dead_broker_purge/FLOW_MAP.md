# FLOW_MAP — Dead-broker (paper-account) value lineage trace

**Packet:** PKT-TB-001-DEAD-BROKER-PURGE-V1-20260609
**Traced at:** 2026-06-09, branch `ai/dead-broker-purge` (base `60b128c`)
**Method:** full read of morning_executor / paper_trader / publish_artifacts / extender /
handler / dashboard_metrics / midday_checker / sns_alerts / llm_weather / replay_engine
(read surface), plus repo-wide grep for `portfolio_value` / `positions_count` /
`latest.json` readers in `src/`, `tests/`, `scripts/`, `automation/`, `frontend/`, and
read-only S3 spot-checks of the live artifacts.

## 1. What the dead lineage IS

The removed Alpaca paper account survives as an **internal intent-sizing simulation
book**: `paper_trader.load_portfolio_state` loads `daily/<date>/portfolio_state.json`
(via the `daily/latest.json` date pointer), the night decision engine sizes intents
against its cash/holdings, the morning executor simulates fills into it, the midday
checker re-marks it. Its `portfolio_value` (~$95.5k as of 2026-06-09, confirmed live in
S3: `daily/latest.json` carries `portfolio_value: 95565.856…`) is a **second book** that
has nothing to do with the displayed canon line (~$114k, optimized-champion replay).

## 2. What the CANON line actually is (must not break)

- `extender.extend_dashboard` re-runs the optimized-champion replay from
  `daily/*/{inference.json,features.parquet,signals.parquet,prices.parquet,
  morning_prices.parquet}` + configs, and **overwrites** `dashboard.json`'s
  `equity_curve.value`, metrics, holdings, trades for all dates ≥ 2026-03-12.
- The publish-time advance guard (`_verify_extension_or_alarm`) blocks any
  dashboard/latest publish when the extender didn't stamp/advance.
- **Canon dependencies on the dead lineage (load-bearing, keep):**
  1. **Replay seed (historical, fixed):** `replay_engine` reads
     `daily/2026-03-11/portfolio_state.json` (START_PORTFOLIO_DATE). Historical S3
     record — never rewritten; unaffected by a forward-only purge.
  2. **Date scaffolding + benchmark:** `dashboard_metrics._load_daily_states` builds
     the base equity-curve rows from each fresh `daily/<date>/portfolio_state.json`
     (`portfolio_value` → placeholder `value`, `benchmark_value` → SPY benchmark line,
     plus cash/holdings_count/cashflow). The extender patches rows ≥ 2026-03-12 with
     champion values, but **a row must exist for the date to be displayed** → the state
     file must keep being written daily with a parseable value. This is the packet's
     case (b): persist under a renamed, non-live-looking field.
  3. **`daily/latest.json` `date`/`intents_date` pointers:** read by
     `paper_trader.load_portfolio_state`, `morning_executor.load_trade_intents`,
     handler morning/midday, `scripts/verify_canon_promotion.py`. Pointers only —
     the `portfolio_value`/`positions_count` fields in that file have **no reader in
     src/** (only `automation/check_pipeline.py` prints them).
  4. **`morning_prices.parquet`** (provisional canon pricing) is produced by the
     morning phase from quotes — not from the dead book's values. Unaffected.
  5. **Historical cutover/bridge machinery** (`cutover_bridge`,
     `canonical_replay_anchor`, `historical_corrections`) reads historical states at
     fixed past dates. Records; unaffected by forward-only purge.

**Conclusion vs the STOP condition:** the dead lineage is load-bearing only via the
state file's existence (scaffolding/benchmark) and historical records. The packet
explicitly authorizes rename-and-keep for exactly this; no re-plumbing needed → no STOP.

## 3. Sightings — every daily-produced surface carrying a dead-book value

| # | Surface (produced fresh daily) | Dead value | Producer | Readers | Fix |
|---|---|---|---|---|---|
| S1 | `daily/latest.json` | `portfolio_value` (95565.86 live), `positions_count` | `publish_artifacts.run` step 14; `publish_morning_artifacts` step 6 | date/intents_date pointers read by src/ + scripts (value fields read by NOTHING in src/; `automation/check_pipeline.py` prints them) | **Drop both fields.** (check_pipeline is outside the write surface — deferred, noted in RETURN) |
| S2 | `daily/<date>/portfolio_state.json` | `portfolio_value` + the file looks like a live book | `publish_artifacts.run` #7; `publish_morning_artifacts` #1; `midday_checker.run` L372 | `paper_trader.load_portfolio_state` (sim continuity); `dashboard_metrics._load_daily_states` (canon scaffolding/benchmark); historical-only: replay seed, cutover_bridge, analysis scripts | **Boundary rename on write:** `portfolio_value` → `sim_book_value` + `book_role`/`book_note` markers. Loader maps back; `_load_daily_states` accepts either name (old files = records, new files = renamed). In-memory key unchanged (no strategy-code churn). |
| S3 | Morning email (subject + body) | `${portfolio_value:,.0f}` of dead book | `handler._run_morning_phase` → `sns_alerts.format_morning_summary` | operator inbox | **Report canon book**: publish returns `canon_total_value` (extender-stamped `metrics.total_value`); email reports it, "unavailable (dashboard held)" when guard holds. |
| S4 | Night email (body) | "Portfolio Value: $dead" | `handler._run_night_phase` → `format_night_summary` | operator inbox | Same canon plumbing as S3. |
| S5 | Midday email (body) | "Portfolio Value: $dead" | `handler._run_midday_check` → `format_midday_summary` | operator inbox | Midday doesn't rebuild the dashboard → read `dashboard/dashboard.json` `metrics.total_value` (canon) defensively; omit on failure. |
| S6 | Lambda response bodies (night/morning/midday) | `'portfolio_value': dead` | handler L499/649/729 | CloudWatch / manual invokes | Replace with `canon_total_value` (None when held); midday uses the canon read. |
| S7 | CloudWatch log lines | `print` of dead $ | handler L444-446; morning_executor L397-398; paper_trader L403-405 | logs | **Remove the dollar values** from log lines (objective says no log line carries the dead value); keep counts. |
| S8 | `daily/<date>/signals.parquet`, `dashboard/timeseries.parquet`, `dashboard/timeseries.json`, `dashboard/data/timeseries.json` | `portfolio_value` column per row | `publish_artifacts._build_timeseries_row` | replay reads signals.parquet **signal columns only**; `compute_signals` reads entropy columns; frontend: type field only, no component renders it | **Drop the field** from the row builder; strip the column from the rolling timeseries on republish (rolling files are produced-fresh-daily, not dated records); remove the dead field from the frontend timeseries type. Historical `daily/<date>/signals.parquet` = records, untouched. |
| S9 | Weather blurb (published narrative) | LLM prompt embeds "Total value: $dead", dead cash_pct/num_positions → can steer published text | `llm_weather.run` prompt snapshot | `daily/<date>/weather_blurb.json`, dashboard weather panel | Feed the prompt the **canon** posture (last published `dashboard.json` metrics — total_value/cash_pct/holdings count) via handler; omit the portfolio block when unavailable. Dead book no longer feeds the prompt. |
| S10 | Dead legacy loaders (read dead-book values, zero callers) | `load_recent_trades`, `load_historical_equity`, `load_historical_drawdowns`, `load_monthly_returns`, `_build_equity_curve_from_daily` in publish_artifacts | n/a (dead code) | none (grep: no callers in src/tests/scripts) | **Delete** — pure residue that teaches fresh readers the dead lineage. |

Non-sightings checked and cleared: `morning_execution.json` (currently value-free —
the packet's "$96,761.88" string is the S7 log line / S3 email, same number family);
`run_report.json`, `midday_check_report.json`, `circuit_breaker.json` (counts/VIX only);
`trade_intents.json` / `shadow_*` (order-sizing dollars, not portfolio values);
`dashboard.json` (extender overwrites all value surfaces with canon; guard holds
otherwise); equity-curve `benchmark` (SPY total-return proxy, labeled benchmark);
`decision_engine` / in-memory `portfolio['portfolio_value']` (internal sizing state,
never published under that name after the boundary rename).

## 4. Readers of each sighting (acceptance-test inventory)

- **Dashboard/frontend:** reads `dashboard.json` (canon after extend; guard-held
  otherwise) and `timeseries.json` (S8 fix). Type residue `portfolio_value` removed.
- **Replay seeding:** historical 2026-03-11 state only → unaffected.
- **Midday checker:** reads state file via the same loader (rename-aware) → unaffected.
- **Emails:** S3/S4/S5 all switch to canon.
- **Tests:** `test_publish_artifacts_dates.py` (latest.json stamps; stale FakeS3),
  `test_dashboard_metrics.py` (3 stale failures, in blast radius via `_load_daily_states`
  change), `test_timeseries_json_sanitization.py` (timeseries row shape),
  `test_paper_trader.py` / `test_morning_executor_simulated.py` (state lineage),
  `test_three_line_replay_canon_promotion.py` (stale FakeCache; drive-by candidate).
- **Outside write surface (note-only):** `automation/check_pipeline.py` prints
  `latest.json portfolio_value` (will print $0.00 after purge until updated);
  `scripts/backfill_historical.py` writes a legacy-shaped latest.json (historical
  backfill tool, dormant).

## 5. Test baseline (pre-change, this machine)

`297 collected; 6 failed, 291 passed`. The 6 = the packet's known stale set
(3× test_dashboard_metrics, 2× test_publish_artifacts_dates, 1× canon_promotion fake).
The 7th known failure (flaky `tests/test_transaction_costs.py`, PKT-TB-003-owned)
passed this run and will not be touched.
