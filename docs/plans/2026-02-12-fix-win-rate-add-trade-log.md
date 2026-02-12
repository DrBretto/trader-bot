# Fix Win Rate Bug + Add Trade Log Panel

## Context

Win rate on the dashboard was jumping between 100% and 0% between days. Root cause: the night phase pipeline sets `portfolio_state['trades_today'] = []` in `handler.py:264` before publishing to S3, overwriting the morning's actual trade records in `daily/{date}/portfolio_state.json`. Since `compute_portfolio_stats()` read `trades_today` from those daily files, all trade history was being silently lost.

User also requested a visible trade log on the dashboard to verify wins/losses directly.

## Plan

- [x] Add `read_jsonl()` method to `S3Client`
- [x] Fix `compute_portfolio_stats()` to read from `trades.jsonl` (append-only, never clobbered) instead of `trades_today` field
- [x] Add `load_recent_trades()` helper in `publish_artifacts.py`
- [x] Include `trades` array in `dashboard.json` output
- [x] Add `Trade` TypeScript interface and update `DashboardData`
- [x] Create `TradeLog` component with summary stats + trade table
- [x] Wire into `App.tsx`

## Execution Log

- Fixed win rate root cause by switching data source from `portfolio_state.trades_today` to `daily/{date}/trades.jsonl`
- Added `read_jsonl()` to S3Client for JSONL file reading
- Added `load_recent_trades()` to publish_artifacts, loads last 90 days of trades
- Built TradeLog component: summary bar (wins/losses/win rate) + color-coded trade table
- Columns: Date, Symbol, Action, Shares, Price, P&L, P&L%, Days Held, Reason

## Follow-ups

- The night phase still sets `trades_today = []` on the portfolio state — this is now harmless since nothing reads it for stats, but could be cleaned up
- REDUCE trades still don't record P&L (partial close has no entry_price tracking)
