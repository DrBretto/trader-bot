import { Trade, TradeSummary } from '../types';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  trades: Trade[];
  cumulativeCosts?: number;
  tradeSummary?: TradeSummary;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function formatDate(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

const actionColors: Record<string, string> = {
  BUY: '#3b82f6',
  SELL: '#f59e0b',
  REDUCE: '#8b5cf6',
};

export function TradeLog({ trades, cumulativeCosts, tradeSummary }: Props) {
  const sellTrades = trades.filter((t) => t.action === 'SELL' && t.pnl !== undefined);
  const wins = tradeSummary?.wins ?? sellTrades.filter((t) => (t.pnl ?? 0) > 0).length;
  const losses = tradeSummary?.losses ?? sellTrades.filter((t) => (t.pnl ?? 0) < 0).length;
  const breakeven = tradeSummary?.breakeven ?? (sellTrades.length - wins - losses);
  const roundTrips = tradeSummary?.realized_round_trips ?? sellTrades.length;
  const fillsTotal = tradeSummary?.fills_total ?? trades.length;
  const winRate = tradeSummary?.win_rate ?? (wins + losses > 0 ? wins / (wins + losses) : 0);
  const txnCosts = tradeSummary?.cumulative_transaction_costs ?? cumulativeCosts;

  if (trades.length === 0) {
    return (
      <div className="card">
        <div className="card-title">
          <span>Trade Log</span>
          <InfoTooltip
            content={`Execution-level fill history. A fill is one executed order event.
Round-trips are realized entry+exit pairings (FIFO), used for win/loss accounting and realized performance stats.`}
            label="Trade log"
          />
        </div>
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>
          No trades recorded yet
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        <span>Trade Log</span>
        <InfoTooltip
          content={`Execution-level fill history. A fill is one executed order event.
Round-trips are realized entry+exit pairings (FIFO), used for win/loss accounting and realized performance stats.`}
          label="Trade log"
        />
      </div>
      {(roundTrips > 0 || fillsTotal > 0) && (
        <div
          style={{
            display: 'flex',
            gap: '24px',
            padding: '8px 0 16px',
            fontSize: '0.85rem',
            color: '#94a3b8',
            flexWrap: 'wrap',
          }}
        >
          <span title="Total executed fills (buys + sells + reductions).">
            {fillsTotal} fills
          </span>
          <span title="Realized entry/exit pairs used for win/loss analytics.">
            {roundTrips} round-trips
          </span>
          <span style={{ color: '#22c55e' }} title="Count of round-trips with positive realized P&L.">
            {wins} wins
          </span>
          <span style={{ color: '#ef4444' }} title="Count of round-trips with negative realized P&L.">
            {losses} losses
          </span>
          {breakeven > 0 && (
            <span style={{ color: '#f8fafc' }} title="Round-trips with near-zero realized P&L.">
              {breakeven} breakeven
            </span>
          )}
          <span title="wins / (wins + losses), excluding breakeven outcomes.">
            Win rate:{' '}
            <span
              style={{
                color: winRate >= 0.5 ? '#22c55e' : '#ef4444',
                fontWeight: 600,
              }}
            >
              {(winRate * 100).toFixed(0)}%
            </span>
          </span>
          {txnCosts != null && txnCosts > 0 && (
            <span title="Cumulative estimated transaction costs from fills (commissions + slippage model).">
              Txn costs: <span style={{ color: '#f59e0b', fontWeight: 600 }}>{formatCurrency(txnCosts)}</span>
            </span>
          )}
          {(tradeSummary?.unmatched_closing_shares ?? 0) > 0 && (
            <span
              style={{ color: '#eab308' }}
              title="Closing shares that could not be paired to open inventory lots; investigate data integrity if non-zero."
            >
              Unmatched close qty: {tradeSummary?.unmatched_closing_shares}
            </span>
          )}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th title="Execution date of the fill.">Date</th>
              <th title="Ticker symbol.">Symbol</th>
              <th style={{ textAlign: 'center' }} title="Execution action type: BUY, SELL, or REDUCE.">Action</th>
              <th style={{ textAlign: 'right' }} title="Executed share quantity.">Shares</th>
              <th style={{ textAlign: 'right' }} title="Executed fill price per share.">Price</th>
              <th style={{ textAlign: 'right' }} title="Per-trade transaction-cost assumption in basis points (bps).">Cost</th>
              <th style={{ textAlign: 'right' }} title="Realized dollar P&L for closing fills; blank for opening fills.">P&L</th>
              <th style={{ textAlign: 'right' }} title="Realized percent P&L for closing fills; blank for opening fills.">P&L %</th>
              <th style={{ textAlign: 'right' }} title="Holding period in days for realized exits.">Days</th>
              <th title="Execution reason from the decision/risk pipeline.">Reason</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade, i) => (
              <tr key={`${trade.timestamp}-${trade.symbol}-${i}`}>
                <td style={{ whiteSpace: 'nowrap' }}>{formatDate(trade.timestamp)}</td>
                <td style={{ fontWeight: 600 }}>{trade.symbol}</td>
                <td style={{ textAlign: 'center' }}>
                  <span
                    style={{
                      color: actionColors[trade.action] || '#94a3b8',
                      fontWeight: 600,
                      fontSize: '0.8rem',
                    }}
                  >
                    {trade.action}
                  </span>
                </td>
                <td style={{ textAlign: 'right' }}>{trade.shares}</td>
                <td style={{ textAlign: 'right' }}>{formatCurrency(trade.price)}</td>
                <td style={{ textAlign: 'right', color: '#94a3b8', fontSize: '0.8rem' }}>
                  {trade.transaction_cost_bps != null
                    ? `${trade.transaction_cost_bps.toFixed(1)} bps`
                    : '\u2014'}
                </td>
                <td
                  style={{
                    textAlign: 'right',
                    color:
                      trade.pnl !== undefined
                        ? trade.pnl >= 0
                          ? '#22c55e'
                          : '#ef4444'
                        : '#64748b',
                  }}
                >
                  {trade.pnl !== undefined ? formatCurrency(trade.pnl) : '\u2014'}
                </td>
                <td
                  style={{
                    textAlign: 'right',
                    color:
                      trade.pnl_pct !== undefined
                        ? trade.pnl_pct >= 0
                          ? '#22c55e'
                          : '#ef4444'
                        : '#64748b',
                  }}
                >
                  {trade.pnl_pct !== undefined ? formatPercent(trade.pnl_pct) : '\u2014'}
                </td>
                <td style={{ textAlign: 'right', color: '#94a3b8' }}>
                  {trade.days_held !== undefined ? trade.days_held : '\u2014'}
                </td>
                <td
                  style={{
                    color: '#94a3b8',
                    fontSize: '0.8rem',
                    maxWidth: '150px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={trade.reason}
                >
                  {trade.reason || '\u2014'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
