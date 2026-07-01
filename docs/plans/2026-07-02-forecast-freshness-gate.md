# 2026-07-02 — Forecast freshness + fail-loud staleness gate (PKT-2, THE DRIVER)

## Context
The in-Lambda OHLCV store never advanced past 2026-06-10: `_extend_ohlcv_from_s3`
(`src/brain/runtime.py`) was a silently-swallowed no-op (two bare `except: continue`
with zero logging) → frozen `mu` → identical `held_symbols` every night → the frozen
concentrated 10-name line, for ~2 weeks, with **no alarm** — because every monitor
keyed on PUBLISH recency (a fresh leaf written nightly) not SUBSTRATE currency (the
OHLCV panel behind `mu`). Confirmed root per the 2026-07-01 audit committee DOSSIER
(ISSUE-01/02/08/13). This is external-project build-packet PKT-2 of 6.

## Plan
- Surface the swallowed failure: instrument `_extend_ohlcv_from_s3` to LOG the exact
  caught exception per date + a store watermark before/after (no more silent swallow).
- Add a governed, **non-destructive** `forecast-diag` Lambda route that runs the
  forward path (extend → panel → inference) and returns substrate-currency
  diagnostics + the staleness-gate verdict WITHOUT writing any S3 object.
- Root-cause via a governed in-Lambda invoke + CloudWatch (the F-D1 unknown).
- Fix the root; replace the bare excepts with a **fail-loud substrate-currency
  staleness gate**: a stale OHLCV store ABORTS + ALARMS the night (incumbent
  retained) instead of shipping a frozen forecast. Invariants: max settled bar within
  K trading days of run date, and bar-count advanced when the S3 gap was non-empty.
- Re-key `monitors.py` `check_stale_publish` / `check_canon_fresh` off substrate
  currency, not publish recency.
- Redirect GDELT `MANIFEST_PATH` under the writable `/tmp` state tree (ISSUE-08).
- Carry the committed `9dad6c3` never-destructive publish guard into the deployed
  image BEFORE any live verify; build a clean SHA-tagged image; deploy; verify live
  (guarded, non-destructive).

## Execution Log
- Instrumented `_extend_ohlcv_from_s3` (dict return + `[FRESHNESS]` logging); added the
  gate machinery + `diagnose_forecast_freshness`; wired the `forecast-diag` route;
  redirected GDELT `MANIFEST_PATH`; re-baked. (see run receipt in the Infotropy repo)

## Follow-ups
- PKT-3 (regime chassis) runs strictly AFTER this — activating the chassis over a
  fresh `mu` is the point; over a frozen `mu` it just re-orders a dead vector.
