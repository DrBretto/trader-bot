import { BuyCandidate } from '../types';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  candidates: BuyCandidate[];
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

export function CandidatesTable({ candidates }: Props) {
  if (candidates.length === 0) {
    return (
      <div className="card">
        <div className="card-title">
          <span>Buy Candidates</span>
          <InfoTooltip
            content={`Ranked watchlist produced by the decision engine.
Score combines model health and decision filters, while Size is the suggested dollar allocation after regime and risk throttles.`}
            label="Buy candidates"
          />
        </div>
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>
          No buy candidates at this time
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        <span>Buy Candidates</span>
        <InfoTooltip
          content={`Ranked watchlist produced by the decision engine.
Score combines model health and decision filters, while Size is the suggested dollar allocation after regime and risk throttles.`}
          label="Buy candidates"
        />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th title="Ticker symbol for the candidate asset.">Symbol</th>
              <th style={{ textAlign: 'right' }} title="Decision score used for candidate ranking; higher is more favorable.">Score</th>
              <th style={{ textAlign: 'right' }} title="Recent 21-day trailing return.">21d</th>
              <th style={{ textAlign: 'center' }} title="Model health score (0-100).">Health</th>
              <th style={{ textAlign: 'center' }} title="Behavior classification inferred by the health model.">Type</th>
              <th style={{ textAlign: 'right' }} title="Suggested dollar size after all sizing modifiers and constraints.">Size</th>
            </tr>
          </thead>
          <tbody>
            {candidates.slice(0, 10).map((candidate) => (
              <tr key={candidate.symbol}>
                <td style={{ fontWeight: 600 }}>{candidate.symbol}</td>
                <td style={{ textAlign: 'right' }}>{candidate.score.toFixed(2)}</td>
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
                    style={{
                      color:
                        candidate.health_score >= 0.7
                          ? '#22c55e'
                          : candidate.health_score >= 0.4
                          ? '#eab308'
                          : '#ef4444',
                    }}
                  >
                    {(candidate.health_score * 100).toFixed(0)}
                  </span>
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
