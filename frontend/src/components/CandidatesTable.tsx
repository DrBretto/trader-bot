import { BuyCandidate } from '../types';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

const ALPACA_CUTOVER_DATE = '2026-03-12';

interface Props {
  candidates: BuyCandidate[];
  snapshotTimestamp?: string;
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(1)}%`;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function CandidatesTable({ candidates, snapshotTimestamp }: Props) {
  const isLive = snapshotTimestamp ? snapshotTimestamp >= ALPACA_CUTOVER_DATE : false;
  const eraLabel = isLive ? 'Live' : 'Backtest';
  let dateLabel = '';
  try { if (snapshotTimestamp) dateLabel = format(parseISO(snapshotTimestamp), 'MMM d'); } catch { /* */ }

  const top = candidates[0];
  const totalSize = candidates.reduce((acc, c) => acc + c.suggested_size, 0);

  const titleRow = (
    <div className="card-title" style={{ marginBottom: 0 }}>
      <span>Buy Candidates</span>
      {dateLabel && <span style={{ fontSize: 10, fontWeight: 400, color: '#64748b', textTransform: 'none', letterSpacing: 0 }}>{dateLabel} · {eraLabel}</span>}
      <InfoTooltip
        content={`Assets the system would buy right now, ranked by a blended score (65% health model + 35% ranking MLP when hybrid is active). Size is the dollar amount after regime-based position sizing and any active throttles.`}
        label="Buy candidates"
      />
    </div>
  );

  const summaryBlock = candidates.length > 0 ? (
    <div className="lower-deck-summary">
      <div className="lower-deck-summary-row">
        <span className="lower-deck-summary-stat">
          <span className="lower-deck-summary-value">{candidates.length}</span>
          <span className="lower-deck-summary-label">candidates</span>
        </span>
        {top && (
          <span className="lower-deck-summary-stat">
            <span className="lower-deck-summary-value">{top.symbol}</span>
            <span className="lower-deck-summary-label">top pick</span>
          </span>
        )}
        <span className="lower-deck-summary-stat">
          <span className="lower-deck-summary-value">{formatCurrency(totalSize)}</span>
          <span className="lower-deck-summary-label">total size</span>
        </span>
      </div>
    </div>
  ) : null;

  if (candidates.length === 0) {
    return (
      <div className="card">
        {titleRow}
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: '#64748b', textAlign: 'center' }}>
            No buy candidates at this time
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      {titleRow}
      {summaryBlock}
      <div className="bounded-scroll-fill" style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th title="Ticker symbol for the candidate asset.">Symbol</th>
              <th title="Decision score: blend of health (65%) and ranking model (35%). Bar shows decomposition.">Score</th>
              <th style={{ textAlign: 'right' }} title="Recent 21-day trailing return.">21d</th>
              <th style={{ textAlign: 'center' }} title="Behavior classification inferred by the health model.">Type</th>
              <th style={{ textAlign: 'right' }} title="Suggested dollar size after all sizing modifiers and constraints.">Size</th>
            </tr>
          </thead>
          <tbody>
            {candidates.slice(0, 10).map((candidate) => (
              <tr key={candidate.symbol}>
                <td style={{ fontWeight: 600 }}>{candidate.symbol}</td>
                <td style={{ minWidth: 130 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div
                      className="score-bar"
                      title={`Health: ${(candidate.health_score * 100).toFixed(0)} · Ranking est: ${((candidate.score - candidate.health_score * 0.65) / 0.35 * 100).toFixed(0)} · Final: ${candidate.score.toFixed(3)}\nApproximate decomposition — actual scoring includes regime compatibility adjustments.`}
                    >
                      <div className="score-bar-health" style={{ width: `${candidate.health_score * 65}%` }} />
                      <div className="score-bar-ranking" style={{ width: `${Math.max(0, candidate.score - candidate.health_score * 0.65) / 0.35 * 35}%` }} />
                    </div>
                    <span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums', minWidth: 32 }}>{candidate.score.toFixed(2)}</span>
                  </div>
                </td>
                <td
                  style={{
                    textAlign: 'right',
                    color: candidate.return_21d >= 0 ? '#22c55e' : '#ef4444',
                  }}
                >
                  {formatPercent(candidate.return_21d)}
                </td>
                <td style={{ textAlign: 'center' }}>
                  <span
                    className={`badge badge-${
                      candidate.behavior === 'momentum' ? 'low' : candidate.behavior === 'mean_reversion' ? 'high' : 'medium'
                    }`}
                  >
                    {candidate.behavior}
                  </span>
                </td>
                <td style={{ textAlign: 'right' }}>{formatCurrency(candidate.suggested_size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
