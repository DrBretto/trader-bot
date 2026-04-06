interface Props {
  totalValue: number;
  ytdReturn: number;
  sharpe: number | null;
  maxDrawdown: number;
  vsSpy: number | null;
}

function formatCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

function formatPct(v: number): string {
  const sign = v >= 0 ? '+' : '';
  return `${sign}${(v * 100).toFixed(2)}%`;
}

export function MobileHeroMetrics({ totalValue, ytdReturn, sharpe, maxDrawdown, vsSpy }: Props) {
  return (
    <div className="mobile-hero">
      <div className="mobile-hero-value">{formatCurrency(totalValue)}</div>
      <div className="mobile-hero-label">Total Value</div>
      <div className="mobile-hero-metrics">
        <div className="mobile-hero-metric">
          <span className={`mobile-hero-metric-value ${ytdReturn >= 0 ? 'positive' : 'negative'}`}>{formatPct(ytdReturn)}</span>
          <span className="mobile-hero-metric-label">YTD</span>
        </div>
        <div className="mobile-hero-metric">
          <span className="mobile-hero-metric-value">{sharpe != null ? sharpe.toFixed(2) : 'N/A'}</span>
          <span className="mobile-hero-metric-label">Sharpe</span>
        </div>
        <div className="mobile-hero-metric">
          <span className={`mobile-hero-metric-value ${maxDrawdown >= 0 ? 'positive' : 'negative'}`}>{formatPct(maxDrawdown)}</span>
          <span className="mobile-hero-metric-label">Max DD</span>
        </div>
        <div className="mobile-hero-metric">
          <span className={`mobile-hero-metric-value ${(vsSpy ?? 0) >= 0 ? 'positive' : 'negative'}`}>{vsSpy != null ? formatPct(vsSpy) : 'N/A'}</span>
          <span className="mobile-hero-metric-label">vs SPY</span>
        </div>
      </div>
    </div>
  );
}
