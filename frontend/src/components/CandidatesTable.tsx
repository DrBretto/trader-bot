import { BuyCandidate } from '../types';

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
        <div className="card-title">Buy Candidates</div>
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>
          No buy candidates at this time
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Buy Candidates</div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th style={{ textAlign: 'right' }}>Score</th>
              <th style={{ textAlign: 'right' }}>21d</th>
              <th style={{ textAlign: 'center' }}>Health</th>
              <th style={{ textAlign: 'center' }}>Type</th>
              <th style={{ textAlign: 'right' }}>Size</th>
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
