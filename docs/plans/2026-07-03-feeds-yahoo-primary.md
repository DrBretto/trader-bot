# 2026-07-03 — feeds: Yahoo v8 primary (clean rebuild P1)

Executes PKT-TRADER-BOT-FEEDS-YAHOO-PRIMARY-V1-20260703 (governed run
`20260703_trader-bot-feeds-yahoo-primary` in the Infotropy Book repo). First
clean-folder step of the `trader-bot-core/` rebuild.

## Context
Live `src/steps/ingest_prices.py` uses **stooq as PRIMARY**, which is bot-blocked
from AWS IPs (404s all 64 symbols). Prices only survive because the yfinance
fallback carries the universe; VVIX/SKEW come from the dead stooq-index source
(`fetch_stooq_index`), so `vvix_value`/`skew_value` were degraded. Yahoo v8 chart
raw endpoint is confirmed working from AWS.

## Plan
- [x] Scaffold `trader-bot-core/` skeleton (folders per architecture output 5;
      only `feeds/` populated by this packet).
- [x] `feeds/contracts.py` — `OHLCVBar` dtype contract + `coerce_to_ohlcv`
      (coerce-or-RAISE, no bare-except swallow; the boundary that kills the
      June-10 dtype-mismatch freeze class).
- [x] `feeds/prices.py` — Yahoo v8 chart raw endpoint PRIMARY (urllib, no
      yfinance dep), explicit named fallbacks (yfinance → AV → stooq), 429
      backoff + inter-symbol pacing, `run_with_report` for served-source
      accounting; `fetch_vol_index`/`latest_vol_index` (Yahoo v8) replace the
      dead stooq VVIX/SKEW source.
- [x] Non-destructive `source='feeds-diag'` branch in `src/handler.py` + bake
      `trader-bot-core/` into `Dockerfile.lambda`; add to deploy BAKE_PATHS.
- [x] Deploy the image; invoke `feeds-diag` from AWS-IP Lambda; capture the five
      acceptance criteria.

## Reality-based acceptance (NOT unit tests)
From an AWS-IP Lambda invoke: (1) full-64 non-empty OHLCV; (2) stored settled SPY
close vs Yahoo v8 < 0.5%; (3) stooq blocked → no freeze + fallback visible;
(4) vvix/skew nonzero; (5) OHLCVBar dtypes conform, forced mismatch RAISES.

## Constraints
Build in `trader-bot-core/` only. Do NOT modify the live `ingest_prices.py` path
or cut prod over (that is P9). ~$97k `sim_book_value` is NOT the line.

## Execution Log
- Built contracts + prices; local smoke test green (dtypes exact, raises fire,
  fallback chain visible). Local Yahoo hit 429 from the laptop IP (expected after
  repeated manual probes) → added retry/backoff; the AWS-IP invoke is the real gate.

## Follow-ups
- P2 (gdelt) branches from this `feeds/contracts.py` boundary.
- Cutover of the live night path to `trader-bot-core/feeds` is P9, not here.
