# 2026-07-09 — Restore settled-bar production (CL-708001)

## Context
The P9 cutover (2026-07-05, commit 580e349) pointed prod at the clean thin router
(`app.handler` → `app.night`). The clean night path is a pure CONSUMER of settled
bars: `production_forecaster` → `store.ohlcv_store._extend_ohlcv_from_s3` downloads
`daily/<D>/prices.parquet` from S3 and splices it, then the freshness gate checks
currency. Nothing in the clean path PRODUCES those files — that was the old chassis
step `src/steps/publish_artifacts.py` (fed by `src/steps/ingest_prices.py`), which
the cutover left un-rebuilt (same class as the earlier morning-path gap).

Result: the last `prices.parquet` was written the night of 07-03 (bars → 07-02).
Since then no settled bar is produced; the OHLCV store froze at 07-02 and every
scheduled night correctly ABORTS stale (live proof: the 07-09 night extended the
store fully to 07-02, 16/16 dates, 0 errors, then aborted — max settled bar 07-02
is 4 trading days behind run date 07-08). The engine is correct; it is starved.

## Plan
- Add `feeds/produce.py::produce_settled_prices` — fetch settled OHLCV for the
  universe via `feeds.prices` (Yahoo v8 primary + yfinance/AV/stooq fallbacks) and
  write `daily/<D>/prices.parquet` to S3. Idempotent; never fabricates a bar (a
  genuine feed outage writes nothing → the freshness gate aborts and surfaces).
- Wire it into `app/night.py::run_night` BEFORE `run_cutover`, so the store extend
  picks up the fresh bar and the freshness gate passes.
- Do NOT touch the freshness/abort gate — fix the substrate it reads.
- Deploy the container image; prove with a real night cycle that the store advances
  and the canon line appends a forward leaf off 07-02.

## Execution Log
- Root cause established from code + live S3/CloudWatch evidence (above).
- Added `feeds/produce.py`; wired into `app/night.py`. Local dry-run: producer
  writes a correct-schema frame (SPY bar through 07-08) via the yfinance fallback
  (Yahoo v8 429'd from the laptop IP; expected — the Lambda AWS IP is the supported
  path, and the fallback chain covers either way).
- Deploy + real-night proof: see run receipt
  `Infotropy Book/book-factory/runs/20260709_trader-bot-substrate-freshness-restore/`.

## Follow-ups
- The next scheduled night (07-09 23:00 ET) will advance to 07-09 on its own via the
  now-deployed producer (one leaf/day, forward-only).
