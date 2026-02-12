## Context
Paper trading results overstate real-world performance: trades fill at exact prices (no spread/slippage), SPY benchmark is price-only (no dividends), and no transaction costs are tracked. This enhancement adds realistic friction to make paper P&L more predictive of real trading results.

## Plan
- [x] Create `src/utils/transaction_costs.py` — spread tiers by asset class + slippage
- [x] Integrate costs into `src/steps/paper_trader.py` → `execute_trade()`
- [x] Add dividend-adjusted SPY benchmark in `paper_trader.py` and `morning_executor.py`
- [x] Add `cumulative_transaction_costs` to dashboard metrics in `publish_artifacts.py`
- [x] Update frontend Trade type, TradeLog component, and fix spacing bug
- [x] Run tests + build frontend

## Execution Log
- Created `src/utils/transaction_costs.py` with spread tiers for all 65 universe sectors + slippage
- Integrated into `paper_trader.execute_trade()` — fills at adjusted price, tracks cumulative costs
- Dividend-adjusted SPY benchmark in both `paper_trader.py` and `morning_executor.py` (~1.3% annual yield reinvested)
- Added `cumulative_transaction_costs` to dashboard metrics
- Frontend: new Cost column in TradeLog, cumulative costs in summary bar, Trade type updated
- Fixed `.tables-row` spacing (added `margin-bottom: 24px`)
- Updated EquityCurve legend to "SPY Total Return"
- All 134 tests pass, frontend builds clean

## Follow-ups
- Could pull actual SPY dividend data from FRED or yfinance for exact amounts instead of estimated 1.3% yield
- Could add market impact model for larger orders (not needed at current portfolio size)
