import { useState, useEffect, useRef } from 'react';
import { DashboardData, TimeseriesPoint } from '../../types';
import { MobileStickyBar } from './MobileStickyBar';
import { MobileHeroMetrics } from './MobileHeroMetrics';
import { MobileStoryPreview } from './MobileStoryPreview';
import { MobileChart } from './MobileChart';
import { MobileExpandableSection } from './MobileExpandableSection';
import { MobileHoldingCard } from './MobileHoldingCard';

interface Props {
  data: DashboardData;
  timeseries: TimeseriesPoint[];
}

function formatCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

function formatPct(v: number): string {
  const sign = v >= 0 ? '+' : '';
  return `${sign}${(v * 100).toFixed(1)}%`;
}

function computeVsSpySpread(equityCurve: { value: number; benchmark: number }[]): number | null {
  if (equityCurve.length < 2) return null;
  const first = equityCurve.find(p => p.value > 0 && p.benchmark > 0);
  const last = [...equityCurve].reverse().find(p => p.value > 0 && p.benchmark > 0);
  if (!first || !last) return null;
  return (last.value / first.value - 1) - (last.benchmark / first.benchmark - 1);
}

export function MobileDashboard({ data, timeseries }: Props) {
  const [openSection, setOpenSection] = useState<string | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setScrolled(!entry.isIntersecting),
      { threshold: 0 }
    );
    if (heroRef.current) observer.observe(heroRef.current);
    return () => observer.disconnect();
  }, []);

  const toggle = (id: string) => setOpenSection(prev => prev === id ? null : id);

  const m = data.metrics;
  const signals = data.expert_signals;
  const regime = signals?.final_regime_label || 'unknown';
  const vsSpySpread = computeVsSpySpread(data.equity_curve);
  const realHoldings = data.holdings.filter(h => h.market_value >= 0.01);
  const cashPct = m.cash_pct ?? (m.total_value > 0 ? m.cash / m.total_value : 0);

  const gru = data.weather.regime.ensemble?.gru_prediction;
  const trans = data.weather.regime.ensemble?.transformer_prediction;
  const agreement = data.weather.regime.ensemble?.agreement ?? 0;

  const firedRules = (signals?.fusion_rules ?? []).filter(r => r.fired);
  const tradeSummary = data.trade_summary;
  const trades = data.trades ?? [];
  const fillsTotal = tradeSummary?.fills_total ?? trades.length;
  const roundTrips = tradeSummary?.realized_round_trips ?? 0;
  const winRate = tradeSummary?.win_rate ?? 0;

  return (
    <div className="mobile-dashboard">
      <MobileStickyBar
        regime={regime}
        totalValue={m.total_value}
        dailyChange={m.mtd_return}
        scrolled={scrolled}
      />

      <div className="mobile-content">
        <div ref={heroRef}>
          <MobileHeroMetrics
            totalValue={m.total_value}
            ytdReturn={m.ytd_return}
            sharpe={m.sharpe_ratio}
            maxDrawdown={m.max_drawdown}
            vsSpy={vsSpySpread}
          />
        </div>

        <MobileStoryPreview
          headline={data.weather.headline}
          firedRules={firedRules}
          posture={getPosture(regime)}
          fullSummary={data.weather.summary}
          expanded={openSection === 'story'}
          onToggle={() => toggle('story')}
        />

        <MobileChart
          equityData={data.equity_curve}
          timeseries={timeseries}
        />

        <div className="mobile-sections">
          <MobileExpandableSection
            id="holdings"
            summary={`${realHoldings.length} position${realHoldings.length !== 1 ? 's' : ''} · ${(cashPct * 100).toFixed(1)}% cash · ${formatPct(m.gross_exposure ?? 0).replace('+', '')} exposure`}
            open={openSection === 'holdings'}
            onToggle={() => toggle('holdings')}
          >
            {realHoldings.length === 0 ? (
              <p className="mobile-empty">No current holdings</p>
            ) : (
              realHoldings.map(h => <MobileHoldingCard key={h.symbol} holding={h} />)
            )}
          </MobileExpandableSection>

          <MobileExpandableSection
            id="brain"
            summary={`Brain: ${gru ? `GRU ${gru.label.replace(/_/g, ' ')} ${(gru.confidence * 100).toFixed(0)}%` : '—'} · ${trans ? `Trans ${trans.label.replace(/_/g, ' ')} ${(trans.confidence * 100).toFixed(0)}%` : '—'} · Agree ${(agreement * 100).toFixed(0)}%`}
            open={openSection === 'brain'}
            onToggle={() => toggle('brain')}
          >
            <div className="mobile-brain-detail">
              {gru && <div className="mobile-brain-row">GRU: <strong>{gru.label.replace(/_/g, ' ')} {(gru.confidence * 100).toFixed(0)}%</strong></div>}
              {trans && <div className="mobile-brain-row">Transformer: <strong>{trans.label.replace(/_/g, ' ')} {(trans.confidence * 100).toFixed(0)}%</strong></div>}
              <div className="mobile-brain-row" style={{ color: agreement >= 0.95 ? '#22c55e' : agreement >= 0.7 ? '#eab308' : '#ef4444' }}>
                Agreement: <strong>{(agreement * 100).toFixed(0)}%</strong>
              </div>
              {firedRules.length > 0 && (
                <div className="mobile-brain-rules">
                  {(signals?.fusion_rules ?? []).map(r => (
                    <span key={r.code} className={`mobile-fusion-dot ${r.fired ? 'fired' : ''}`}>
                      {r.label.replace(' Override', '').replace(' Gate', '').replace(' Sizing', '')}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </MobileExpandableSection>

          <MobileExpandableSection
            id="candidates"
            summary={`${data.candidates.length} candidates · Top: ${data.candidates[0]?.symbol ?? '—'} · ${formatCurrency(data.candidates.reduce((a, c) => a + c.suggested_size, 0))} total`}
            open={openSection === 'candidates'}
            onToggle={() => toggle('candidates')}
          >
            {data.candidates.slice(0, 5).map(c => (
              <div key={c.symbol} className="mobile-candidate-row">
                <span className="mobile-candidate-symbol">{c.symbol}</span>
                <span className="mobile-candidate-score">{c.score.toFixed(2)}</span>
                <span style={{ color: c.return_21d >= 0 ? '#22c55e' : '#ef4444' }}>{formatPct(c.return_21d)}</span>
                <span>{formatCurrency(c.suggested_size)}</span>
              </div>
            ))}
          </MobileExpandableSection>

          <MobileExpandableSection
            id="trades"
            summary={`${fillsTotal} fills · ${roundTrips} round-trips · ${(winRate * 100).toFixed(0)}% win rate`}
            open={openSection === 'trades'}
            onToggle={() => toggle('trades')}
          >
            {trades.slice(0, 5).map((t, i) => (
              <div key={`${t.timestamp}-${t.symbol}-${i}`} className="mobile-trade-row">
                <span className="mobile-trade-date">{new Date(t.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
                <span className="mobile-trade-symbol">{t.symbol}</span>
                <span className="mobile-trade-action" style={{ color: t.action === 'BUY' ? '#3b82f6' : t.action === 'SELL' ? '#f59e0b' : '#8b5cf6' }}>{t.action}</span>
                <span>{t.shares} @ {formatCurrency(t.price)}</span>
              </div>
            ))}
          </MobileExpandableSection>
        </div>
      </div>
    </div>
  );
}

const REGIME_POSTURE: Record<string, string> = {
  high_vol_panic: 'Defensive — high cash reserve',
  risk_off_trend: 'Cautious — reduced equity',
  choppy: 'Neutral — balanced exposure',
  risk_on_trend: 'Constructive — equity-favored',
  calm_uptrend: 'Growth — full exposure',
};

function getPosture(regime: string): string {
  return REGIME_POSTURE[regime] || 'Unknown posture';
}
