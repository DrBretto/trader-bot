import { BuyCandidate, ExpertSignals, PortfolioMetrics } from '../types';
import { InfoTooltip } from './InfoTooltip';

interface FusionRule {
  order: number;
  code: string;
  label: string;
  fired: boolean;
  inputs: string;
  threshold: string;
  effect: string;
}

interface Props {
  signals?: ExpertSignals;
  metrics: PortfolioMetrics;
  candidates: BuyCandidate[];
  fusionRules?: FusionRule[];
}

const REGIME_THRESHOLDS: Record<string, string> = {
  high_vol_panic: 'Panic probability falling below 70% would release the override, allowing equity re-entry.',
  risk_off_trend: 'Macro credit improving above +0.50 or vol regime calming would release the risk-off posture.',
  choppy: 'Stable trending behavior in either direction would shift the regime.',
  risk_on_trend: 'Rising vol or deteriorating macro could shift toward caution.',
  calm_uptrend: 'A sharp vol spike or credit deterioration could trigger defensive posture.',
};

export function DecisionStoryCard({ signals, metrics, candidates, fusionRules }: Props) {
  if (!signals) return null;

  const regime = signals.final_regime_label || 'unknown';
  const confidence = (signals.regime_confidence * 100).toFixed(1);
  const cashPct = metrics.cash_pct ?? (metrics.total_value > 0 ? metrics.cash / metrics.total_value : 0);
  const firedRules = (fusionRules ?? []).filter(r => r.fired);
  const top3 = candidates.slice(0, 3);
  const changeHint = REGIME_THRESHOLDS[regime] || 'Regime change would alter the portfolio posture.';

  return (
    <div className="card decision-story-card">
      <div className="card-title">
        <span>Decision Story</span>
        <InfoTooltip
          content="Synthesized explanation of today's portfolio posture. All data from the nightly pipeline — no separate LLM generation."
          label="Decision story"
        />
      </div>

      <div className="decision-story-body">
        <div className="decision-story-regime">
          <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>
            {regime.replace(/_/g, ' ')}
          </span>
          <span style={{ color: '#64748b', marginLeft: 6 }}>
            ({confidence}% confidence)
          </span>
        </div>

        {firedRules.length > 0 && (
          <div className="decision-story-rules">
            {firedRules.map(r => (
              <span key={r.code} className="fired-rule-badge" title={`${r.label}: ${r.effect}`}>
                {r.label}
              </span>
            ))}
          </div>
        )}

        <div className="decision-story-effect">
          Cash {(cashPct * 100).toFixed(1)}% · Position sizing at {((signals.position_size_modifier ?? 1) * 100).toFixed(0)}%
          {signals.risk_throttle_factor > 0 && ` · Throttle ${(signals.risk_throttle_factor * 100).toFixed(0)}%`}
        </div>

        {top3.length > 0 && (
          <div className="decision-story-candidates">
            Top candidates: {top3.map(c => `${c.symbol} (${c.score.toFixed(2)})`).join(', ')}
          </div>
        )}

        <div className="decision-story-change">
          {changeHint}
        </div>
      </div>
    </div>
  );
}
