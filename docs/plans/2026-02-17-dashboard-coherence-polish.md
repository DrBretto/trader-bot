# Dashboard Coherence + UI Correctness Polish

## Context

Latest UI review surfaced polish-level coherence issues:

- Regime probability highlight semantics are ambiguous when post-fusion selection differs from raw model argmax.
- Ensemble section wording makes model votes read like duplicate regimes.
- Drawdown chart can render visual spikes when x-axis labels collapse timestamps or series has ordering/duplicate issues.
- Monthly returns should distinguish true `0.0%` from `no data`.
- Win rate and return displays should remain explicitly consistent with canonical realized metrics.

The goal is a minimal, targeted polish pass with consistent semantics and clearer UX copy.

## Plan

- [x] Create branch for this pass (`codex/feat/dashboard-coherence-polish`)
- [x] Regime probabilities panel:
  - [x] highlight argmax probability
  - [x] tag selected post-fusion regime explicitly
  - [x] add explanatory tooltip/copy
- [x] Ensemble vote wording:
  - [x] update model rows to “vote/suggests” language
  - [x] avoid duplicate regime interpretation
- [x] Drawdown chart hygiene:
  - [x] sort + dedupe series timestamps before rendering
  - [x] break line on large gaps/reset-like discontinuities using nulls
  - [x] use non-colliding x-axis date key
- [x] Monthly returns no-data handling:
  - [x] add monthly observation count in canonical backend payload
  - [x] render `—` for zero-observation months
  - [x] exclude zero-observation months from YTD compounding
- [x] Win-rate semantics/copy:
  - [x] ensure dashboard text states realized round-trip basis
  - [x] keep counts and formula deterministic/reconciled
- [x] Add lightweight dev warnings:
  - [x] argmax != selected with no override reason
  - [x] monthly zero return with zero observations
  - [x] non-strictly-increasing timestamp series
- [x] Run checks/build and commit in small scoped units

## Execution Log

- Created branch `codex/feat/dashboard-coherence-polish`.
- Reviewed current frontend and backend data-flow points:
  - `frontend/src/App.tsx` (Regime Probabilities)
  - `frontend/src/components/EnsembleStatus.tsx` (Regime Decision)
  - `frontend/src/components/DrawdownChart.tsx` (render path)
  - `frontend/src/components/MonthlyReturnsHeatmap.tsx` (monthly rendering/YTD)
  - `src/utils/dashboard_metrics.py` (canonical metrics, trade/win-rate semantics)
- Implemented regime-probability semantics:
  - argmax probability bar is now the visual highlight
  - post-fusion regime receives explicit `Selected` tag
  - added explanatory tooltip/copy for raw model probs vs final regime selection
- Updated `Regime Decision` ensemble wording to read as model votes/suggestions.
- Added `frontend/src/utils/timeseries.ts` for sort/dedupe/gap-break handling and strict-order checks.
- Updated drawdown chart to:
  - normalize timestamps before render
  - break on large gaps
  - use true date key + formatted tick labels to avoid duplicate-category spikes
- Added monthly no-data sentinel path:
  - backend publishes `observations` per month
  - frontend renders `—` for zero-observation months
  - YTD compounds only observed months
- Added dev-only console warnings in `App.tsx` for:
  - argmax/selected mismatch without override reason
  - zero-observation 0.0% monthly rows
  - non-increasing drawdown timestamps
  - header YTD vs monthly compounded YTD mismatch
- Updated win-rate copy to explicitly state realized round-trip semantics.
- Checks run:
  - `pytest -q` passed (`117 passed, 1 skipped`)
  - `cd frontend && npm run build` passed
- Scoped commits:
  - `72de7a0` Clarify regime probability highlight and selected semantics
  - `218168f` Rename ensemble outputs as model votes in regime panel
  - `d8115bf` Normalize drawdown series before chart rendering
  - `23ede08` Render monthly no-data cells from observation counts
  - `5021f82` Explicitly label win rate as realized round-trip metric

## Follow-ups

- If desired later: move ad-hoc dev warnings into a small shared diagnostics module and optionally surface a non-intrusive debug panel in local dev.
