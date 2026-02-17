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

function formatUnsignedPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function HeroMetrics({ metrics }: Props) {
  const metricItems = [
    {
      label: 'Total Value',
      value: formatCurrency(metrics.total_value),
      colorBySign: false,
    },
    {
      label: 'YTD Return',
      value: formatPercent(metrics.ytd_return),
      colorBySign: true,
      rawValue: metrics.ytd_return,
    },
    {
      label: 'MTD Return',
      value: formatPercent(metrics.mtd_return),
      colorBySign: true,
      rawValue: metrics.mtd_return,
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio != null ? metrics.sharpe_ratio.toFixed(2) : 'N/A',
      colorBySign: false,
    },
    {
      label: 'Max Drawdown',
      value: formatPercent(metrics.max_drawdown),
      colorBySign: true,
      rawValue: metrics.max_drawdown,
    },
    {
      label: 'Win Rate',
      value: `${(metrics.win_rate * 100).toFixed(0)}%`,
      colorBySign: false,
    },
    {
      label: 'Cash %',
      value: formatUnsignedPercent(metrics.cash_pct ?? 0),
      colorBySign: false,
    },
    {
      label: 'Gross Exposure',
      value: formatUnsignedPercent(metrics.gross_exposure ?? 0),
      colorBySign: false,
    },
    {
      label: 'Net Exposure',
      value: formatPercent(metrics.net_exposure ?? 0),
      colorBySign: true,
      rawValue: metrics.net_exposure ?? 0,
    },
    {
      label: 'Top Position',
      value: formatUnsignedPercent(metrics.top_position_pct ?? 0),
      colorBySign: false,
    },
    {
      label: 'Beta Proxy',
      value: metrics.beta_proxy != null ? metrics.beta_proxy.toFixed(2) : 'N/A',
      colorBySign: false,
    },
  ];

  return (
    <div className="metrics-grid">
      {metricItems.map((item) => (
        <div key={item.label} className="metric-card">
          <div className="metric-label">{item.label}</div>
          <div
            className={`metric-value ${
              item.colorBySign
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
