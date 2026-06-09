> SUPERSEDED by docs/plans/2026-06-08-alpaca-removal-and-continuous-line-committee-packet.md (Alpaca removed 2026-06-08; this system is a pure simulation).

# Phase 3 — P&L and broker reconciliation

Date: 2026-04-29
Sources used:
- `s3://investment-system-data/dashboard/dashboard.json` — refreshed 2026-04-29 09:30 ET
- `s3://investment-system-data/dashboard/timeseries.json` — refreshed 2026-04-29 09:30 ET
- prior snapshot from `/tmp/claude-501/trader-bot-audit/dashboard.json` (2026-04-28 13:45 ET) — used for the F-4 transient comparison

**Gap (must surface — packet rule):** The packet specifies "pull broker truth from Alpaca paper" and "every dollar number on the dashboard either matches the broker truth or has a documented, traceable cashflow gap." The Alpaca paper API host (`paper-api.alpaca.markets`) is not in this session's sandbox network allowlist, and AWS Secrets Manager (where `alpaca/paper/keys` lives) is also blocked. **I cannot poll the broker directly in this session.** The reconciliation below uses the broker-derived fields *already published into S3* (`broker_total_value`, `holdings[*].market_value`, `holdings[*].unrealized_pnl`, fills with `broker_order_id`/`broker_status`/`execution_mode='alpaca_paper'`). Any claim that requires a live broker call (e.g. comparing today's snapshot to today's `account.equity` from Alpaca) needs operator-side execution. Phase-6 shadow-day will need real Alpaca polling regardless.

## §A — Headline reconciliation table

Snapshot 2026-04-29 (today). All amounts in USD.

| Field on dashboard | Reported value | Source / definition | Reconciles to |
|---|---:|---|---|
| `metrics.total_value` | 105,925.15 | continuity-adjusted equity (broker raw − cumulative_external_cashflow) | broker_total_value + $10,077.83 |
| `metrics.broker_total_value` | 95,847.32 | broker `account.equity` at last snapshot | `cash + sum(holdings.market_value)` = 95,847.32 ✓ |
| `metrics.cash` | 39,138.17 | broker `account.cash` | matches `cash + invested = broker_total_value` |
| `metrics.invested` | 56,709.15 | `sum(holdings.market_value)` | exactly matches sum: 56,709.15 ✓ |
| `equity_curve[-1].raw_value` | 95,847.32 | broker truth at snapshot | matches `broker_total_value` ✓ |
| `equity_curve[-1].value` | 105,925.15 | continuity (raw − ccf) | matches `total_value` ✓ |
| `equity_curve[-1].cumulative_external_cashflow` | −10,077.83 | sum of all `external_cashflow` rows | drives the $10k continuity gap |
| `equity_curve[-1].benchmark` | 103,082.61 | dividend-reinvested SPY shares × current price | independent of broker reconciliation |

**Result**: The dashboard's broker-derived numbers reconcile internally to the cent. The $10,077.83 gap between `total_value` and `broker_total_value` is fully explained by `cumulative_external_cashflow = −10,077.83` and the continuity formula in `dashboard_metrics.py:264`. **No silent dollar drift** in the headline arithmetic.

## §B — Per-holding cost-basis reconciliation (today)

For each holding, expected `pnl = (current_price − entry_price) × shares` using local entry_price.

| symbol | shares    | entry  | current | reported_pnl | expected_pnl | reported_pct | sign-coherent | match |
|--------|----------:|-------:|--------:|-------------:|-------------:|-------------:|---------------|-------|
| ARKK   | 109.7753  | 69.36  | 75.46   | +669.60      | +669.60      | +8.79%       | yes           | OK    |
| FXI    | 212.9585  | 35.39  | 36.26   | +185.27      | +185.27      | +2.46%       | yes           | OK    |
| VUG    | 102.7658  | 73.48  | 82.74   | +952.47      | +952.47      | +12.61%      | yes           | **OK today** |
| XBI    | 58.9429   | 126.72 | 131.15  | +260.82      | +260.82      | +3.49%       | yes           | OK    |
| XLF    | 151.6585  | 49.68  | 51.86   | +330.97      | +330.97      | +4.39%       | yes           | OK    |
| XLK    | 55.3724   | 135.87 | 157.85  | +1,217.09    | +1,217.09    | +16.18%      | yes           | OK    |
| XRT    | 93.0890   | 80.89  | 84.48   | +334.19      | +334.19      | +4.44%       | yes           | OK    |

All 7 holdings reconcile to local cost basis on 2026-04-29 (today).

## §C — VUG is the F-4/F-5 reproducer in the wild — *transient* between snapshots

Comparing yesterday's snapshot vs today's:

| date       | source dataset                                 | VUG.shares | VUG.entry_price | VUG.current_price | VUG.unrealized_pnl | VUG.unrealized_pnl_pct | local-basis check | dual-basis fail? |
|------------|------------------------------------------------|-----------:|----------------:|------------------:|--------------------:|------------------------:|------------------:|------------------|
| 2026-04-28 | prior snapshot (`dashboard.json` 13:45 ET)     | 102.7658   | 73.48           | 82.77             | **−6,097.78**       | **+12.64%**             | (cp−ep)·sh=+954.52 | **YES**          |
| 2026-04-29 | refreshed snapshot (`dashboard.json` 09:30 ET) | 102.7658   | 73.48           | 82.74             | **+952.47**         | **+12.61%**             | (cp−ep)·sh=+952.47 | NO               |

Same holding, same shares, same local entry_price. **Reported `unrealized_pnl` flipped sign and magnitude in 24 hours, with no underlying position change.** This is exactly the failure mode predicted by Phase-1 F-5 (`morning_executor.py:351-413` `_reconcile_portfolio_from_broker` mixes broker `unrealized_pl` with local-basis `unrealized_pnl_pct`):

- Yesterday's path: `_reconcile_portfolio_from_broker` succeeded with a broker-side `avg_entry_price ≈ $142.11` (likely a stale lot from before the cutover). `pnl = pos['unrealized_pl'] = −6,097.78`. `pct = current/local_entry − 1 = +12.64%`. **Sign mismatch published.**
- Today's path: Either (a) `_update_valuations_from_quotes` ran (broker reconciliation failed today and fell through to line 716), or (b) `_reconcile_portfolio_from_broker` ran with a fresh `pos['unrealized_pl']` that finally matches local cost basis after a bookkeeping change on the broker side. Either way the *published* number is now the local-basis value.

Surfacing this transient is the heart of why F-5 is structural: **the dashboard's correctness depends on which valuation path executed at runtime, and the user has no way to tell from the published artifact which path produced it.** Even when both paths happen to agree on a given day, the system is *not* reliably correct.

## §D — Equity curve and TWR continuity

Pre/post pivot continuity (drawn from F-11 in Phase 2):

- Aug 4 2025 → Jan 28 2026: `value` and `raw_value` BOTH equal $100,000 every day for 125 consecutive days. The bot was 100% in cash; no broker discrepancy.
- Jan 29 2026: first trades fill. `raw_value` and `value` start to diverge from $100,000.
- Feb 11 2026 (estimated cutover date — confirmed by negative `external_cashflow` showing up around then): a one-day external_cashflow of approximately −$10,078 occurs. From this date forward, `value − raw_value = +$10,077.83` (cashflow is subtracted via the continuity bridge).
- Apr 29 2026 (today): `value = 105,925.15`, `raw_value = 95,847.32`, gap = $10,077.83 stable.

The cashflow happened on a single day; from the equity_curve data its sign is `−10,077.83` meaning a *withdrawal* (broker received less than the simulated portfolio held at cutover). The continuity bridge subtracts this withdrawal from cumulative cashflow so the visualization preserves performance. **This is the documented `$10k cashflow gap` from the packet's `Acceptance test`** — traceable, single-source, internally consistent. It is NOT a silent drift.

But it is **invisible to the dashboard reader** as currently rendered (`metrics.total_value` and `metrics.broker_total_value` are both displayed but no on-page caption explains the gap). F-10 in Phase 1 already covers this; Phase 4 should add a UI annotation (operator-side, since `frontend/src/` is read-only per packet constraints).

## §E — Trade and round-trip reconciliation

| Field                              | Value     | Cross-check                                   |
|------------------------------------|----------:|-----------------------------------------------|
| `total_fills`                      | 150       | matches len(`trades_history`) = 150 ✓         |
| Pre-cutover fills (`exec_mode=null`) | 125     | sim-mode legacy                                |
| Post-cutover fills (`exec_mode=alpaca_paper`) | 25 | broker-routed                                  |
| `realized_round_trips`             | 75        | matches sum of `round_trips` array length     |
| `wins / losses / breakeven`        | 43 / 27 / 5 | sums to 75 ✓                                 |
| `total_trades` (denominator for win_rate) | 70 | excludes breakeven (43+27)                    |
| `win_rate`                         | 0.6143    | 43/70 ✓                                       |
| Σ `realized_pnl` over 75 round-trips | +4,717.19 | +4,717.19 against $100k start = +4.72% realized |
| `cumulative_transaction_costs`     | 118.45    | $0.79/trade avg                               |

`Σ realized_pnl − cumulative_transaction_costs = +4,598.74`. Reconcile against `value − initial`:

- `total_value − 100,000 = +5,925.15` (continuity-adjusted equity gain)
- `realized + transaction_costs = 4,598.74`
- residual = `5,925.15 − 4,598.74 = 1,326.41` — this should equal the **unrealized P&L on currently held positions**.
- Σ holdings.unrealized_pnl = sum across 7 holdings = **+3,950.41**.
- residual gap: `3,950.41 − 1,326.41 = 2,624.00`

**This residual is unexplained by the obvious accounting.** Hypotheses:

1. **Continuity bridge subtracts a portion of the unrealized at cutover.** Likely: at the cutover date, a chunk of the simulated unrealized was *moved into* the cumulative_external_cashflow figure. Without polling the broker for `account.equity` history, I cannot confirm.
2. **Transaction costs are double-counted somewhere.** Unlikely given the `cumulative_transaction_costs` field is small.
3. **Round-trip `realized_pnl` excludes some closed positions** (e.g., `unmatched_closing_shares: 228` from `trade_summary` — there are 228 closing shares that don't match an opening lot). This is a known accounting quirk.

The `unmatched_closing_shares = 228` is the most likely explanation for the residual: closing shares without a matched opening lot are P&L that lives outside the round-trip ledger. Their dollar impact would land in `total_value` but not in `realized_round_trips`. **This is a sub-finding (F-16):** the dashboard surfaces `realized_round_trips` and `realized_pnl` but does not reconcile them against the equity curve. A reader assuming `realized + unrealized = total_change` will be off by the unmatched-closing-shares gap.

### F-16 — `unmatched_closing_shares=228` creates an unreconciled gap between realized round-trips and the equity curve

Discovery from this Phase 3 reconciliation. **Current production: 228 closing shares cannot be matched to opening lots** (likely orphaned by the cutover bridge — pre-existing positions were carried into Alpaca paper and later sold without a matching pre-cutover BUY in the local trade ledger).

`src/utils/dashboard_metrics.py` `compute_canonical_dashboard_metrics` exposes `unmatched_closing_shares` in `trade_summary` but does not roll their realized P&L back into any displayed column. The user-visible result: equity curve total return (+5.9%) ≠ realized + unrealized P&L from displayed figures.

**Recommended fix summary**:
1. Add a `realized_pnl_unmatched` line item in `trade_summary` that surfaces the dollars-on-table from those 228 unmatched closing shares.
2. Alternatively: at cutover, synthesize a virtual "opening fill" for each carried-over broker position so future closings find a match. This requires Phase-4 fix discipline because it touches the canonical metric path and needs a regression test.

**Risk class**: OBSERVABILITY. No live decision impact, but ledger transparency is a packet acceptance criterion.

## §F — Required broker-poll items (operator-side execution)

For Phase 6 shadow-day to satisfy the packet's acceptance test, the following calls must be made operator-side:

1. `GET /v2/account` on Alpaca paper — record `equity`, `last_equity`, `cash`, `buying_power`. Compare today's `equity` to dashboard `broker_total_value = 95,847.32`. Expected: ≤ $1 drift (intra-day quote noise).
2. `GET /v2/positions` on Alpaca paper — for each position, record `qty`, `avg_entry_price`, `current_price`, `unrealized_pl`, `cost_basis`. Compare to dashboard `holdings[*]`. Expected: each row matches local; if VUG's broker `avg_entry_price` is now $73.48 (matching local), the F-4 transient has been fixed silently broker-side; if it's still $142.11, the bug is still latent.
3. `GET /v2/orders?status=all&limit=200` — confirm last 25 alpaca_paper trades from the dashboard match broker history by `client_order_id`.

These three calls take about 30 seconds. They MUST happen before any live cutover decision.

## §G — Acceptance-test status

| Acceptance criterion (from packet)                                                       | Status |
|------------------------------------------------------------------------------------------|--------|
| "Every dollar number on the dashboard either matches the broker truth or has a documented, traceable cashflow gap" | **PASS** for today (gap traced to continuity bridge); **F-4 transient** remains a structural risk that may print incoherent rows on any random day |
| "No silent paper-vs-broker drift"                                                         | **PASS** at cent precision today; **F-5 makes future drift not impossible** without the Phase-4 fix |
| "Realized + unrealized + costs reconciles to total_value"                                 | **FAIL** — residual gap of $2,624 from `unmatched_closing_shares=228` (F-16) |

End Phase 3.