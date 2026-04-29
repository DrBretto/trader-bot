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
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>Trade Log</span>
          <InfoTooltip
            content={`Execution-level fill history. A fill is one executed order event.\nRound-trips are realized entry+exit pairings (FIFO), used for win/loss accounting and realized performance stats.`}
            label="Trade log"
          />
        </div>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#64748b', textAlign: 'center' }}>No trades recorded yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="lower-deck-header-row">
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>Trade Log</span>
          <InfoTooltip
            content={`Execution-level fill history. A fill is one executed order event.\nRound-trips are realized entry+exit pairings (FIFO), used for win/loss accounting and realized performance stats.`}
            label="Trade log"
          />
        </div>
        {(roundTrips > 0 || fillsTotal > 0) && (
          <div className="lower-deck-header-detail">
            <span style={{ color: '#22c55e' }}>{wins}w</span>
            <span style={{ color: '#ef4444' }}>{losses}l</span>
            {breakeven > 0 && <span>{breakeven}be</span>}
            {txnCosts != null && txnCosts > 0 && (
              <span>costs: <span style={{ color: '#f59e0b' }}>{formatCurrency(txnCosts)}</span></span>
            )}
          </div>
        )}
      </div>
      {(roundTrips > 0 || fillsTotal > 0) && (
        <div className="lower-deck-summary">
          <div className="lower-deck-summary-row">
            <span className="lower-deck-summary-stat">
              <span className="lower-deck-summary-value">{fillsTotal}</span>
              <span className="lower-deck-summary-label">fills</span>
            </span>
            <span className="lower-deck-summary-stat">
              <span className="lower-deck-summary-value">{roundTrips}</span>
              <span className="lower-deck-summary-label">round-trips</span>
            </span>
            <span className="lower-deck-summary-stat">
              <span className="lower-deck-summary-value" style={{ color: winRate >= 0.5 ? '#22c55e' : '#ef4444' }}>
                {(winRate * 100).toFixed(0)}%
              </span>
              <span className="lower-deck-summary-label">win rate</span>
            </span>
          </div>
        </div>
      )}
      <div className="bounded-scroll" style={{ overflowX: 'auto' }}>
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
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
