import { Holding, PortfolioMetrics } from '../types';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

const ALPACA_CUTOVER_DATE = '2026-03-12';

interface Props {
  holdings: Holding[];
  snapshotTimestamp?: string;
  metrics?: PortfolioMetrics;
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

function formatShares(shares: number): string {
  if (shares >= 1) return shares.toFixed(2);
  if (shares >= 0.001) return shares.toFixed(4);
  return shares.toExponential(2);
}

const DUST_THRESHOLD = 0.01;

export function PortfolioTable({ holdings, snapshotTimestamp, metrics }: Props) {
  const isLive = snapshotTimestamp ? snapshotTimestamp >= ALPACA_CUTOVER_DATE : false;
  const eraLabel = isLive ? 'Live' : 'Backtest';
  let dateLabel = '';
  try { if (snapshotTimestamp) dateLabel = format(parseISO(snapshotTimestamp), 'MMM d'); } catch { /* */ }

  const realHoldings = holdings.filter(h => h.market_value >= DUST_THRESHOLD);
  const dustCount = holdings.length - realHoldings.length;
  const cashPct = metrics?.cash_pct ?? (metrics && metrics.total_value > 0 ? metrics.cash / metrics.total_value : null);

  const headerRow = (
    <div className="lower-deck-header-row">
      <div className="card-title" style={{ marginBottom: 0 }}>
        <span>Current Holdings</span>
        {dateLabel && <span style={{ fontSize: 10, fontWeight: 400, color: '#64748b', textTransform: 'none', letterSpacing: 0 }}>{dateLabel} · {eraLabel}</span>}
        <InfoTooltip
          content={`Current open positions in the portfolio. P&L is unrealized (mark-to-market vs entry price). Health is the model's quality score (0-100) — lower scores may trigger sell signals. Vol bucket determines position sizing limits.`}
          label="Current holdings"
        />
      </div>
      {dustCount > 0 && (
        <div className="lower-deck-header-detail">
          <span style={{ color: '#475569' }}>{dustCount} dust filtered</span>
        </div>
      )}
    </div>
  );

  // Persistent summary — always visible above scroll
  const summaryBlock = metrics ? (
    <div className="lower-deck-summary">
      <div className="lower-deck-summary-row">
        <span className="lower-deck-summary-stat">
          <span className="lower-deck-summary-value">{realHoldings.length}</span>
          <span className="lower-deck-summary-label">positions</span>
        </span>
        {cashPct != null && (
          <span className="lower-deck-summary-stat">
            <span className="lower-deck-summary-value">{(cashPct * 100).toFixed(1)}%</span>
            <span className="lower-deck-summary-label">cash</span>
          </span>
        )}
        <span className="lower-deck-summary-stat">
          <span className="lower-deck-summary-value">{formatPercent(metrics.gross_exposure ?? 0).replace('+', '')}</span>
          <span className="lower-deck-summary-label">exposure</span>
        </span>
      </div>
    </div>
  ) : null;

  if (realHoldings.length === 0) {
    return (
      <div className="card">
        {headerRow}
        {summaryBlock}
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#64748b', textAlign: 'center' }}>
            No current holdings
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      {headerRow}
      {summaryBlock}
      <div className="bounded-scroll-fill" style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th title="Ticker symbol for the holding.">Symbol</th>
              <th style={{ textAlign: 'right' }} title="Current share count held in inventory.">Shares</th>
              <th style={{ textAlign: 'right' }} title="Current market value of the position.">Value</th>
              <th style={{ textAlign: 'right' }} title="Unrealized percent gain/loss since entry cost basis.">P&L</th>
              <th style={{ textAlign: 'center' }} title="Model health score (0-100), where higher implies stronger modeled quality.">Health</th>
              <th style={{ textAlign: 'center' }} title="Volatility bucket used by sizing and risk controls.">Vol</th>
            </tr>
          </thead>
          <tbody>
            {realHoldings.map((holding) => (
              <tr key={holding.symbol}>
                <td style={{ fontWeight: 600 }}>{holding.symbol}</td>
                <td style={{ textAlign: 'right' }}>{formatShares(holding.shares)}</td>
                <td style={{ textAlign: 'right' }}>{formatCurrency(holding.market_value)}</td>
                <td
                  style={{
                    textAlign: 'right',
                    color: holding.unrealized_pnl >= 0 ? '#22c55e' : '#ef4444',
                  }}
                >
                  {formatPercent(holding.unrealized_pnl_pct)}
                </td>
                <td style={{ textAlign: 'center' }}>
                  <span
                    style={{
                      color:
                        holding.health_score >= 0.7
                          ? '#22c55e'
                          : holding.health_score >= 0.4
                          ? '#eab308'
                          : '#ef4444',
                    }}
                  >
                    {(holding.health_score * 100).toFixed(0)}
                  </span>
                </td>
                <td style={{ textAlign: 'center' }}>
                  <span
                    className={`badge badge-${
                      holding.vol_bucket === 'low'
                        ? 'low'
                        : holding.vol_bucket === 'high'
                        ? 'high'
                        : 'medium'
                    }`}
                  >
                    {holding.vol_bucket}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
