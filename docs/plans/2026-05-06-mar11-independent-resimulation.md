> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# 2026-05-06 — Mar 11 Independent Resimulation

## Context

Operator emergency directive 2026-05-06: prior post-Alpaca timeline /
correction attempts are not trusted. Build a fresh independent
resimulation of the Trader Bot performance line from EOD 2026-03-11
forward, free of dashboard / corrected-equity / Lambda / prior-failed-
attempt inputs, and surface what the public chart *should* look like
under clean execution and clean accounting.

Driven from packet
`PKT-ULP-TRB-MAR11-INDEPENDENT-RESIMULATION-20260506`. Run output lives
under
`Infotropy Book/book-factory/use_lane_outputs/runs/20260506_trader-bot-mar11-independent-resimulation-v1/`.

## Plan

- [x] Read PACKET_EXECUTION + trader-bot CLAUDE.md + DEPLOY.md +
  OPERATIONS.md + chart_markers.json + decision_engine.py + regime_fusion.py
- [x] Capture baseline (live dashboard.json, timeseries.json, S3 latest.json)
  for comparison only — never as input truth
- [x] Probe data sources: S3 daily/<date>/, Alpaca paper API,
  yfinance, Stooq via daily/<D+1>/prices.parquet
- [x] Reconstruct 2026-03-11 EOD starting state from
  `daily/2026-03-11/portfolio_state.json` (cash/shares anchor) +
  yfinance + Stooq verified close prices (re-mark)
- [x] Choose and document fill-price rule (next-session open from
  yfinance, Stooq cross-check, fallbacks documented)
- [x] Build independent simulator
  `scripts/resimulate_mar11_forward_independent.py`
- [x] Run simulator 2026-03-12 → 2026-05-06 (39 trading days)
- [x] Produce 6 evidence tables + 10 verdict reports
- [x] March recovery window day-by-day
- [x] VUG/accounting trouble spot inspection
- [x] Cross-check vs live dashboard ONLY after independent sim built
- [x] Decide on production integration (deferred — operator decision)
- [ ] Commit simulator script to feature branch (this commit)

## Execution Log

- Independent sim 2026-05-06 final: $106,291.08 (vs dashboard chart
  $107,182.58, diff -0.84%). Within tight noise budget — chart
  magnitude is approximately right.
- Found: bridge-cashflow correction on 2026-04-22 has no Alpaca
  counterparty (verified by querying every non-FILL activity type).
  Compensates for production's mishandling of the VUG 6:1 split on
  2026-04-21.
- Found: production's panic-day SELL orders fail at Alpaca with HTTP
  403 because of a 6-decimal-place rounding artifact (requested shares
  exceed Alpaca-available by sub-microsecond rounding). XLI and XLU
  were never sold; remained held through panic and recovery.
- Found: yfinance `auto_adjust=False` for VUG returns split-adjusted
  prices retroactively. Stooq returns raw prices with the actual
  ex-date discontinuity. Simulator uses yfinance as primary source and
  records splits as audit-only events.

## Follow-ups

These are recommended for separate packets / sessions, NOT part of this
run's deliverable:

- Production code: reconcile `portfolio_state.json` from Alpaca's actual
  positions list after every morning execution.
- Production code: apply ETF splits to held positions' `shares` and
  `entry_price` fields on ex-date (using yfinance split metadata + Stooq
  ex-date detection).
- Production code: clamp SELL `shares` to `min(intent_shares, broker_available)`
  or round DOWN to a safer precision (4 decimals) to prevent Alpaca 403
  rejections.
- Production code: tag failed-execution intents so the next-night
  decision engine doesn't infinitely reissue the same SELL.
- Dashboard: remove the second bridge-cashflow event (no real Alpaca
  counterparty) AFTER the production code fixes above are in place.

Reference equity series for any future fix's validation:
`Infotropy Book/.../runs/20260506_trader-bot-mar11-independent-resimulation-v1/reports/tables/DAILY_INDEPENDENT_RESIMULATION.tsv`.