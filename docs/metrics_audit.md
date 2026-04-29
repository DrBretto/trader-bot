# Metrics Audit (Dashboard + Trading Analytics)

## Canonical Snapshot Model

Dashboard metrics now come from a single canonical computation path in `src/utils/dashboard_metrics.py`, with one explicit snapshot context:

- `snapshot.id` = `<run_date>:<phase>:<timestamp>`
- `snapshot.date` = artifact date used for all period calculations
- `snapshot.phase` = `night` or `morning`
- `snapshot.timestamp` = portfolio state timestamp used for "Last updated"

`src/steps/publish_artifacts.py` now publishes:

- `snapshot` (top-level)
- `panel_snapshot_ids` (per-panel snapshot references)
- `metrics.snapshot_id`

All major dashboard panels are built from the same canonical snapshot ID.

## Metric Definitions

### Returns (Cashflow-Adjusted)

Canonical daily return:

```text
r_t = (equity_t - equity_{t-1} - external_cashflow_t) / equity_{t-1}
```

Where `external_cashflow_t` is net external deposits/withdrawals/transfers (if present in state).

Period returns (MTD/YTD) are time-weighted (compounded daily returns), not simple P&L sums.

### Monthly Returns

Monthly returns are compounded from daily canonical returns within each month:

```text
R_month = Π(1 + r_t) - 1
```

Frontend YTD in the monthly table is now compounded from monthly returns (not additive).

### Sharpe Ratio

- Input series: canonical daily returns (cashflow-adjusted)
- Risk-free rate: fixed at `0.0` annual for now
- Annualization: `Sharpe_ann = mean(excess_daily) / std(excess_daily) * sqrt(252)`
- Robustness guard: if fewer than 60 daily observations, Sharpe is `null` and UI shows `N/A`

### Drawdown

Drawdown uses the same canonical equity curve as returns:

```text
drawdown_t = equity_t / running_peak_t - 1
```

- `max_drawdown` = min(drawdown series)
- `current_drawdown` = latest drawdown

### Trade Lifecycle + Win Rate

Entity definitions:

- Fill: one executed trade record from `daily/<date>/trades.jsonl`
- Round-trip: FIFO pairing of entry fill(s) and exit fill(s), realized-only

Win-rate basis:

- `wins` = round-trips with realized P&L > 0
- `losses` = round-trips with realized P&L < 0
- `win_rate = wins / (wins + losses)`
- `total_trades` is counted-trade basis (`wins + losses`) for denominator consistency
- `breakeven_trades` tracked separately

Transaction costs are derived from fills:

- `abs(fill_price - market_price) * shares + commission`

### Exposure Transparency

Published backend metrics now include:

- `cash_pct`
- `gross_exposure`
- `net_exposure`
- `top_position_pct`
- `beta_proxy` (when holding-level beta exists, otherwise `null`)
Dashboard hero metrics additionally include `portfolio_vs_spy` (frontend-derived from canonical equity curve):

```text
(portfolio_end / portfolio_start - 1) - (spy_end / spy_start - 1)
```

Positive values indicate benchmark outperformance over the displayed history window.

## Reset Handling

To avoid silently mixing pre/post-reset history:

1. Preferred: explicit reset markers in state (`metrics_reset_id`, `reset_id`, `portfolio_reset_id`)
2. Fallback heuristic: detect reset-like discontinuity (large jump to near-initial, mostly-cash state), then restart aggregation from that boundary

Detected boundary is exposed as `reset_boundary` in dashboard payload.

## Regime Decision Explainability

`src/signals/regime_fusion.py` now emits ordered rule diagnostics:

- `fusion_rules[]` with `order`, `label`, `fired`, `inputs`, `threshold`, `effect`
- `target_gross_exposure`
- `effective_exposure_multiplier`
- `throttle_mapping`

Frontend `Regime Decision` renders this backend-provided rule trail directly, so display logic matches execution logic.

## Dashboard Explainability UX

- Added detailed on-hover tooltips across:
  - hero metrics
  - chart cards
  - table headers
  - regime/fusion panels
- Dashboard data fetch now uses cache-busting query params + `cache: "no-store"` to reduce stale S3 object caching issues in browsers.

## Local Validation

Run:

```bash
pytest -q
pytest -q tests/test_dashboard_metrics.py
cd frontend && npm run build
```

Key audit checks in `tests/test_dashboard_metrics.py`:

- Metrics reconciliation:
  - header YTD equals compounded monthly-table YTD
  - panel max drawdown equals series-derived max drawdown
  - win-rate math and counts reconcile
- Cashflow exclusion:
  - deposits/withdrawals do not spike return series
- Snapshot cohesion:
  - all panel snapshot IDs match the same snapshot ID and timestamp
