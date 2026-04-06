import { BuyCandidate, ExpertSignals, PortfolioMetrics } from '../types';
import { InfoTooltip } from './InfoTooltip';

interface FusionRule {
  order: number;
  code: string;
  label: string;
  fired: boolean;
  effect: string;
}

interface WeatherData {
  headline: string;
  summary: string;
  risks?: string[];
}

interface Props {
  signals?: ExpertSignals;
  metrics: PortfolioMetrics;
  candidates: BuyCandidate[];
  fusionRules?: FusionRule[];
  weather: WeatherData;
}

const CHANGE_HINTS: Record<string, string> = {
  high_vol_panic: 'Panic prob < 70% would release the override.',
  risk_off_trend: 'Macro credit > +0.50 or vol calming would ease posture.',
  choppy: 'Stable trend in either direction would shift regime.',
  risk_on_trend: 'Rising vol or credit stress could trigger caution.',
  calm_uptrend: 'Sharp vol spike could trigger defensive posture.',
};

export function TodaysStoryCard({ signals, metrics, candidates, fusionRules, weather }: Props) {
  if (!signals) return null;

  const regime = signals.final_regime_label || 'unknown';
  const cashPct = metrics.cash_pct ?? (metrics.total_value > 0 ? metrics.cash / metrics.total_value : 0);
  const firedRules = (fusionRules ?? []).filter(r => r.fired);
  const top3 = candidates.slice(0, 3);
  const changeHint = CHANGE_HINTS[regime] || '';

  return (
    <div className="card todays-story-card">
      <div className="story-header-row">
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>Today's Story</span>
          <InfoTooltip
            content="What the system decided today and why. The headline comes from an LLM summarizing current market conditions. Cash %, position sizing, and throttle levels reflect the fusion rules acting on live signal data."
            label="Today's story"
          />
        </div>
        <div className="story-meta-cluster">
          <span className="story-meta-chip">Cash {(cashPct * 100).toFixed(1)}%</span>
          <span className="story-meta-chip">Size {((signals.position_size_modifier ?? 1) * 100).toFixed(0)}%</span>
          {signals.risk_throttle_factor > 0 && (
            <span className="story-meta-chip">Throttle {(signals.risk_throttle_factor * 100).toFixed(0)}%</span>
          )}
        </div>
      </div>

      <div className="story-body">
        <div className="story-copy-block">
          <div className="story-headline">
            {weather.headline}
          </div>
          <div className="story-summary">
            {weather.summary}
          </div>
        </div>

        <div className="story-bottom-block">
          {firedRules.length > 0 && (
            <div className="story-rules-row">
              {firedRules.map(r => (
                <span key={r.code} className="fired-rule-badge" title={`${r.label}: ${r.effect}`}>
                  {r.label}
                </span>
              ))}
            </div>
          )}

          <div className="story-footer-line">
            {top3.length > 0 && `Top: ${top3.map(c => c.symbol).join(', ')}`}
          </div>

          {(changeHint || (weather.risks && weather.risks.length > 0)) && (
            <div className="story-hint-block">
              {changeHint}
              {weather.risks && weather.risks.length > 0 && (
                <details style={{ marginTop: 2, fontStyle: 'normal' }}>
                  <summary style={{ cursor: 'pointer' }}>{weather.risks.length} risk{weather.risks.length !== 1 ? 's' : ''}</summary>
                  <ul style={{ margin: '2px 0 0 14px' }}>
                    {weather.risks.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </details>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
