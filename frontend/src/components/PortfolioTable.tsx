import { Holding } from '../types';

interface Props {
  holdings: Holding[];
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

export function PortfolioTable({ holdings }: Props) {
  if (holdings.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Current Holdings</div>
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px 0' }}>
          No current holdings
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Current Holdings</div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th style={{ textAlign: 'right' }}>Shares</th>
              <th style={{ textAlign: 'right' }}>Value</th>
              <th style={{ textAlign: 'right' }}>P&L</th>
              <th style={{ textAlign: 'center' }}>Health</th>
              <th style={{ textAlign: 'center' }}>Vol</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((holding) => (
              <tr key={holding.symbol}>
                <td style={{ fontWeight: 600 }}>{holding.symbol}</td>
                <td style={{ textAlign: 'right' }}>{holding.shares}</td>
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
