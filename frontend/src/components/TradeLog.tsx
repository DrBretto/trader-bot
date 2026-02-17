import { Trade, TradeSummary } from '../types';

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
        <div className="card-title">Trade Log</div>
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>
          No trades recorded yet
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Trade Log</div>
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
          <span>
            {fillsTotal} fills
          </span>
          <span>
            {roundTrips} round-trips
          </span>
          <span style={{ color: '#22c55e' }}>
            {wins} wins
          </span>
          <span style={{ color: '#ef4444' }}>
            {losses} losses
          </span>
          {breakeven > 0 && (
            <span style={{ color: '#f8fafc' }}>
              {breakeven} breakeven
            </span>
          )}
          <span>
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
            <span>
              Txn costs: <span style={{ color: '#f59e0b', fontWeight: 600 }}>{formatCurrency(txnCosts)}</span>
            </span>
          )}
          {(tradeSummary?.unmatched_closing_shares ?? 0) > 0 && (
            <span style={{ color: '#eab308' }}>
              Unmatched close qty: {tradeSummary?.unmatched_closing_shares}
            </span>
          )}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Symbol</th>
              <th style={{ textAlign: 'center' }}>Action</th>
              <th style={{ textAlign: 'right' }}>Shares</th>
              <th style={{ textAlign: 'right' }}>Price</th>
              <th style={{ textAlign: 'right' }}>Cost</th>
              <th style={{ textAlign: 'right' }}>P&L</th>
              <th style={{ textAlign: 'right' }}>P&L %</th>
              <th style={{ textAlign: 'right' }}>Days</th>
              <th>Reason</th>
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
