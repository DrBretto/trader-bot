# Settled NY trading-day grid (forward stamp fix) — PKT-4

## Context

Every canon leaf/decision/`daily/<D>/` folder is stamped by UTC `datetime.now()`
(`src/handler.py`). The night job fires `cron(0 3 ? * TUE-SAT)` = 03:00 UTC = the
prior ET evening, so `datetime.now()` returns the *next* calendar day. Result:

- A Friday-night settle lands on a **Saturday** leaf (phantom weekend leaves
  06-13 / 06-20 / 06-27).
- The `daily/<D>/` folder date drifts one day ahead of the market day it contains.

Fix the **forward** stamp so every leaf/decision is keyed to the **settled NY
trading day** (the actual close date the book is marked at). Weekends/holidays
produce no leaf; SPY (benchmark) + canon ride ONE real-trading-day grid.

This is PKT-4 of the 6-packet trader-bot audit rebuild. **Forward stamping only** —
historical phantom leaves are removed by PKT-5's seam-free reconstruction, NOT here.

## Plan

- [x] `src/utils/market_calendar.py` — settled-NY-trading-day helpers:
  - `ny_today(now_utc=None)` — ET calendar date (DST-aware via zoneinfo, fixed-offset
    fallback safe for the cron hours).
  - `latest_settled_session(now_utc=None)` — latest NY weekday on/before ET-today
    (weekend walk-back; conservative on holidays). The clock-derived grid key used by
    the intraday phases.
  - `settled_day_from_prices(prices_df, symbol='SPY', now_utc=None)` — max settled bar
    date for SPY in the freshly-ingested panel = "the actual close date the book is
    marked at" (holiday+weekend aware by construction). The night's source of truth.
- [x] `src/handler.py` — rekey every stamp that dates a leaf/decision/folder:
  - Night phase: `run_date` seeded to `latest_settled_session()` (ET, not UTC), then
    set to `settled_day_from_prices(prices_df)` right after price ingest — the leaf,
    decisions.json, `daily/<D>/`, trade_intents, and the equity append all key off it.
  - Morning / midday / republish / shadow phases: default `run_date` from
    `latest_settled_session()` instead of UTC `datetime.now()` (event override kept).
- [x] `src/handler.py` — `source='date-grid-diag'` NON-DESTRUCTIVE route: for simulated
  `asof` instants (Fri 22:00 ET, Sat, Sun, a holiday) compute the settled day and the
  append-vs-noop decision against the LIVE ledger frontier (read-only, no S3 write).
  Proves acceptance 1/2/4.

## SPY + canon one grid (already structural)

`equity_append.append_settled_point_for_publish()` writes ONE leaf per `run_date`
carrying both `value` (canon) and `benchmark` (SPY). Its live no-op guard
(`run_date <= front_date`, backed by `EquityLedger` G-APPEND-ONLY-FRONTIER) makes a
weekend/holiday/duplicate run a no-op. So once `run_date` is the settled day, SPY +
canon cannot drift and non-trading days append nothing. This packet corrects the KEY;
the guard was already live (PKT-2).

## Execution Log

- Read handler.py, brain/runtime.py, canon/equity_append.py, canon/equity_ledger.py,
  publish_artifacts.py, infra crons, deploy script. Root cause = UTC `datetime.now()`
  at the 03:00-UTC night hour; append/grid machinery already correct.

## Follow-ups

- Historical phantom leaves (06-13/20/27) + the 06-25 splice → **PKT-5** (seam-free
  reconstruction replays the corrected forward path onto this grid). Not touched here.
</content>
</invoke>
