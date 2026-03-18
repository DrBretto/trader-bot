import { EquityCurvePoint, Holding, PortfolioMetrics } from '../types';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  metrics: PortfolioMetrics;
  holdings: Holding[];
  equityCurve: EquityCurvePoint[];
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number, digits = 2): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

function formatUnsignedPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function computeVsSpySpread(equityCurve: EquityCurvePoint[]): number | null {
  if (equityCurve.length < 2) return null;

  const first = equityCurve.find((point) => point.value > 0 && point.benchmark > 0);
  const last = [...equityCurve].reverse().find((point) => point.value > 0 && point.benchmark > 0);
  if (!first || !last) return null;

  const portfolioReturn = last.value / first.value - 1;
  const benchmarkReturn = last.benchmark / first.benchmark - 1;
  return portfolioReturn - benchmarkReturn;
}

export function HeroMetrics({ metrics, holdings, equityCurve }: Props) {
  const totalValue = metrics.total_value > 0 ? metrics.total_value : 0;
  const holdingsGrossValue = holdings.reduce((sum, holding) => sum + Math.abs(holding.market_value), 0);
  const holdingsNetValue = holdings.reduce((sum, holding) => sum + holding.market_value, 0);
  const maxHoldingValue = holdings.reduce(
    (maxValue, holding) => Math.max(maxValue, holding.market_value),
    0,
  );

  const cashPct =
    metrics.cash_pct ??
    (totalValue > 0 ? Math.max(metrics.cash, 0) / totalValue : 0);
  const grossExposure =
    metrics.gross_exposure ??
    (totalValue > 0 ? holdingsGrossValue / totalValue : 0);
  const netExposure =
    metrics.net_exposure ??
    (totalValue > 0 ? holdingsNetValue / totalValue : 0);
  const topPositionPct =
    metrics.top_position_pct ??
    (totalValue > 0 ? maxHoldingValue / totalValue : 0);
  const vsSpySpread = computeVsSpySpread(equityCurve);

  const metricItems = [
    {
      label: 'Total Value',
      value: formatCurrency(metrics.total_value),
      colorBySign: false,
      tooltip:
        'Current marked-to-market account equity: cash + market value of all open positions at this snapshot. Note: this value includes the continuity bridge adjustment from the pre-Alpaca simulated period. The broker-only account value may differ. Use the broker-only toggle above charts for the Alpaca-only view.',
    },
    {
      label: 'YTD Return',
      value: formatPercent(metrics.ytd_return),
      colorBySign: true,
      rawValue: metrics.ytd_return,
      tooltip:
        'Time-weighted return from the first trading day of the calendar year through this snapshot. External deposits/withdrawals are excluded from performance. Includes both the simulated pre-cutover period and the Alpaca paper period, bridged by a cashflow adjustment on 2026-03-12.',
    },
    {
      label: 'MTD Return',
      value: formatPercent(metrics.mtd_return),
      colorBySign: true,
      rawValue: metrics.mtd_return,
      tooltip:
        'Time-weighted return from the first trading day of the current month through this snapshot, using the same canonical return series as Sharpe and drawdown. Post-cutover months reflect Alpaca broker truth only.',
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio != null ? metrics.sharpe_ratio.toFixed(2) : 'N/A',
      colorBySign: false,
      tooltip:
        'Annualized Sharpe ratio from daily cashflow-adjusted returns: mean(excess return) / std(return) * sqrt(252). Displays N/A when observation count is too short for stability.',
    },
    {
      label: 'Max Drawdown',
      value: formatPercent(metrics.max_drawdown),
      colorBySign: true,
      rawValue: metrics.max_drawdown,
      tooltip:
        'Worst peak-to-trough decline on the canonical equity curve during the active history window (or post-reset segment when a reset boundary exists).',
    },
    {
      label: 'Realized Win Rate',
      value: `${(metrics.win_rate * 100).toFixed(0)}%`,
      colorBySign: false,
      tooltip:
        'Realized round-trip win rate: wins / (wins + losses), where each round-trip is a deterministic FIFO pairing of entry and exit fills.',
    },
    {
      label: 'Cash %',
      value: formatUnsignedPercent(cashPct),
      colorBySign: false,
      tooltip:
        'Uninvested capital share of total equity. High cash means lower market exposure and lower strategy risk/return sensitivity.',
    },
    {
      label: 'Gross Exposure',
      value: formatUnsignedPercent(grossExposure),
      colorBySign: false,
      tooltip:
        'Sum of absolute position exposures divided by equity. Long-only portfolios typically range from 0% to 100%; leverage pushes this above 100%.',
    },
    {
      label: 'Net Exposure',
      value: formatPercent(netExposure, 1),
      colorBySign: true,
      rawValue: netExposure,
      tooltip:
        'Directional market exposure (long minus short) as a fraction of equity. Positive values are net long; negative values are net short.',
    },
    {
      label: 'Top Position',
      value: formatUnsignedPercent(topPositionPct),
      colorBySign: false,
      tooltip:
        'Concentration metric: largest single-position market value divided by total equity. Larger values imply idiosyncratic concentration risk.',
    },
    {
      label: 'Beta Proxy',
      value: metrics.beta_proxy != null ? metrics.beta_proxy.toFixed(2) : 'N/A',
      colorBySign: false,
      tooltip:
        'Approximate sensitivity of portfolio returns to SPY returns over a recent rolling window. Higher beta means stronger market co-movement.',
    },
    {
      label: 'Portfolio vs SPY',
      value: vsSpySpread != null ? formatPercent(vsSpySpread) : 'N/A',
      colorBySign: true,
      rawValue: vsSpySpread ?? undefined,
      tooltip:
        'Relative return spread over the displayed equity window: (Portfolio total return - SPY total return). Positive means outperformance.',
    },
  ];

  return (
    <div className="metrics-grid">
      {metricItems.map((item) => (
        <div key={item.label} className="metric-card">
          <div
            className="metric-label"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <span>{item.label}</span>
            <InfoTooltip content={item.tooltip} label={item.label} align="right" />
          </div>
          <div
            className={`metric-value ${
              item.colorBySign && item.rawValue != null
                ? item.rawValue >= 0
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
