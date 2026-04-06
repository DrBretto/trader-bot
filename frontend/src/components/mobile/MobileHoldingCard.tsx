import { Holding } from '../../types';

interface Props {
  holding: Holding;
}

export function MobileHoldingCard({ holding }: Props) {
  const healthColor = holding.health_score >= 0.7 ? '#22c55e' : holding.health_score >= 0.4 ? '#eab308' : '#ef4444';
  const pnlColor = holding.unrealized_pnl_pct >= 0 ? '#22c55e' : '#ef4444';
  const pnlSign = holding.unrealized_pnl_pct >= 0 ? '+' : '';

  return (
    <div className="mobile-holding-card">
      <div className="mobile-holding-top">
        <span className="mobile-holding-symbol">{holding.symbol}</span>
        <span className="mobile-holding-health" style={{ color: healthColor }}>{(holding.health_score * 100).toFixed(0)}</span>
      </div>
      <div className="mobile-holding-bottom">
        <span className="mobile-holding-shares">{holding.shares.toFixed(2)} shares</span>
        <span className="mobile-holding-value">
          {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(holding.market_value)}
        </span>
      </div>
      <div className="mobile-holding-bottom">
        <span className="mobile-holding-pnl" style={{ color: pnlColor }}>{pnlSign}{(holding.unrealized_pnl_pct * 100).toFixed(2)}%</span>
        <span className="mobile-holding-vol">{holding.vol_bucket}</span>
      </div>
    </div>
  );
}
