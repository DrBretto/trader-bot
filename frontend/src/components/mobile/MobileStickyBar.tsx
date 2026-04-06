const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

const REGIME_SHORT: Record<string, string> = {
  calm_uptrend: 'Calm',
  risk_on_trend: 'Trending',
  choppy: 'Choppy',
  risk_off_trend: 'Risk Off',
  high_vol_panic: 'Panic',
};

interface Props {
  regime: string;
  totalValue: number;
  dailyChange: number;
  scrolled: boolean;
}

function formatCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

export function MobileStickyBar({ regime, totalValue, dailyChange, scrolled }: Props) {
  const color = REGIME_COLORS[regime] || '#64748b';
  const label = REGIME_SHORT[regime] || regime.replace(/_/g, ' ');
  const changeColor = dailyChange >= 0 ? '#22c55e' : '#ef4444';
  const changeSign = dailyChange >= 0 ? '+' : '';

  return (
    <header
      className={`mobile-sticky-bar ${scrolled ? 'mobile-sticky-bar--scrolled' : ''}`}
      aria-label="Portfolio status"
    >
      <div className="mobile-sticky-regime">
        <span className="mobile-sticky-dot" style={{ backgroundColor: color }} />
        <span style={{ color }}>{label}</span>
      </div>
      <span className="mobile-sticky-value">{formatCurrency(totalValue)}</span>
      <span className="mobile-sticky-change" style={{ color: changeColor }}>
        {changeSign}{(dailyChange * 100).toFixed(2)}%
      </span>
    </header>
  );
}
