# Dashboard Metrics Audit + Canonical Snapshot Cohesion

## Context

Dashboard metrics currently come from mixed computation paths and timestamps (precomputed portfolio fields + independently rebuilt chart tables), which can produce visible mismatches in YTD/MTD, Sharpe/drawdown behavior, trade counts, and "last updated" semantics. This task aligns analytics and regime explainability around one canonical snapshot and one return/trade accounting definition.

Post-deploy UI feedback (same task scope) identified:
- new exposure cards appearing as zeros in the live dashboard view
- a remaining gap in the hero grid (wants one additional metric card)
- request for educational, detailed tooltips across dashboard panels

## Plan

- [x] Orient on roadmap and prior plans (`docs/PLAN.md`, `docs/plans/*`, `docs/POSTMORTEMS.md`, `CLAUDE.md`)
- [x] Implement canonical metrics engine for:
  - [x] cashflow-adjusted daily returns
  - [x] reset boundary handling
  - [x] Sharpe/drawdown from the same series
  - [x] deterministic FIFO round-trip accounting (fills vs round-trips)
  - [x] lightweight exposure/risk transparency metrics
- [x] Route `paper_trader` stats and dashboard publication through the canonical engine
- [x] Add snapshot metadata (`snapshot_id`, timestamp, date, phase) and enforce panel cohesion
- [x] Improve regime fusion explainability payload (ordered rule trail with thresholds/inputs/effects)
- [x] Update frontend to consume new semantics:
  - [x] Sharpe `N/A` for short windows
  - [x] monthly YTD compounding
  - [x] trade summary clarity (fills vs round-trips)
  - [x] regime rule trail sourced from backend explanation
- [x] Add automated validation tests:
  - [x] metrics reconciliation
  - [x] cashflow exclusion behavior
  - [x] snapshot cohesion
- [x] Document definitions + local validation in `docs/metrics_audit.md`
- [x] Run checks/tests
- [x] Commit in logical groups
- [x] Diagnose and fix hero metric zero rendering in live dashboard view
- [x] Add one more hero metric: portfolio total return vs SPY total return
- [x] Add detailed explanatory tooltips across dashboard UI
- [x] Rebuild + redeploy frontend and verify live payload/render consistency

## Execution Log

- Created branch `codex/feat/dashboard-metrics-audit`.
- Mapped current metric paths:
  - backend metrics currently in `src/steps/paper_trader.py`
  - dashboard assembly in `src/steps/publish_artifacts.py`
  - frontend render paths in `frontend/src/components/*`
- Confirmed key mismatch risks:
  - dashboard uses precomputed portfolio stats plus separately rebuilt monthly/equity tables
  - monthly table YTD is additive not compounded
  - trade summary currently mixes total fills with win/loss counts
  - dashboard timestamp uses `datetime.now()` rather than a stable snapshot ID
- Implemented canonical metrics engine in `src/utils/dashboard_metrics.py`:
  - cashflow-adjusted return series
  - reset boundary handling (explicit marker + heuristic fallback)
  - Sharpe/drawdown from canonical equity curve
  - FIFO round-trip pairing and trade summary
  - exposure metrics (cash/gross/net/top/beta proxy)
- Routed `paper_trader.compute_portfolio_stats()` and dashboard assembly through canonical metrics.
- Added snapshot cohesion metadata in dashboard payload (`snapshot`, `panel_snapshot_ids`, `metrics.snapshot_id`) and synced `latest.json` snapshot timestamp/id.
- Extended regime fusion output with ordered `fusion_rules` trail and exposure-mapping explainability fields.
- Updated frontend for:
  - Sharpe `N/A` rendering
  - compounded monthly-table YTD
  - trade summary semantics from canonical round-trip stats
  - fusion-rule rendering from backend diagnostics
- Added `tests/test_dashboard_metrics.py` with reconciliation, cashflow exclusion, and snapshot cohesion checks.
- Added `docs/metrics_audit.md` with definitions and local validation steps.
- Ran checks:
  - `pytest -q` (117 passed, 1 skipped)
  - `npm run build` in `frontend` (success)
- Created logical commits:
  - `6c01973` — canonical metrics engine + snapshot cohesion + tests/docs
  - `74ac424` — frontend alignment with canonical metric semantics
- Follow-up diagnostics for live "zero" report:
  - verified both `s3://investment-system-data/dashboard/dashboard.json` and `s3://investment-system-data/dashboard/data/dashboard.json`
  - confirmed non-zero canonical values are present in payload (`cash_pct=0.7955`, `gross_exposure=0.2045`, `net_exposure=0.2045`, `top_position_pct=0.0684`)
  - likely issue narrowed to frontend fetch/render path (stale cache or missing-field fallback), not backend metric computation
- Implemented frontend follow-up polish/fixes:
  - `HeroMetrics` now computes robust fallback exposure values from current holdings if optional metrics are missing
  - added cache-busting + `cache: "no-store"` for dashboard/timeseries fetches to avoid stale snapshot rendering
  - added new hero card: `Portfolio vs SPY` (relative total-return spread)
  - added reusable `InfoTooltip` component and applied detailed explanatory tooltips across dashboard cards/tables/charts
- Validation:
  - `cd frontend && npm run build` (success)
- Deployed frontend update:
  - `cd frontend && VITE_DATA_URL=dashboard.json npm run build`
  - `aws s3 sync dist/ s3://investment-system-data/dashboard/ --exclude "dashboard.json" --exclude "timeseries.json" --exclude "timeseries.parquet" --exclude "data/*" --delete --region us-east-1`
  - verified upload and object timestamps via `aws s3 ls s3://investment-system-data/dashboard/ --region us-east-1`

## Follow-ups

- If real brokerage fills/commissions are integrated later, extend round-trip accounting inputs to explicit commission/fee fields and reconcile with estimated slippage costs.
- Optional enhancement: compute a realized portfolio beta proxy from rolling symbol return history when symbol-level history is available during publish.
