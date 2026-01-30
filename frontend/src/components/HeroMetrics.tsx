import { PortfolioMetrics } from '../types';

interface Props {
  metrics: PortfolioMetrics;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(2)}%`;
}

export function HeroMetrics({ metrics }: Props) {
  const metricItems = [
    {
      label: 'Total Value',
      value: formatCurrency(metrics.total_value),
      isPercentage: false,
    },
    {
      label: 'YTD Return',
      value: formatPercent(metrics.ytd_return),
      isPercentage: true,
      rawValue: metrics.ytd_return,
    },
    {
      label: 'MTD Return',
      value: formatPercent(metrics.mtd_return),
      isPercentage: true,
      rawValue: metrics.mtd_return,
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio.toFixed(2),
      isPercentage: false,
    },
    {
      label: 'Max Drawdown',
      value: formatPercent(metrics.max_drawdown),
      isPercentage: true,
      rawValue: metrics.max_drawdown,
    },
    {
      label: 'Win Rate',
      value: `${(metrics.win_rate * 100).toFixed(0)}%`,
      isPercentage: false,
    },
  ];

  return (
    <div className="metrics-grid">
      {metricItems.map((item) => (
        <div key={item.label} className="metric-card">
          <div className="metric-label">{item.label}</div>
          <div
            className={`metric-value ${
              item.isPercentage
                ? (item.rawValue ?? 0) >= 0
                  ? 'positive'
                  : 'negative'
                : ''
            }`}
          >
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}
