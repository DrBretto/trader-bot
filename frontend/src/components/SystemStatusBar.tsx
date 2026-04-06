import { CandidateBundleSummary, ExpertSignals, PortfolioMetrics } from '../types';
import { InfoTooltip } from './InfoTooltip';

const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

const REGIME_POSTURE: Record<string, string> = {
  high_vol_panic: 'Defensive — equity sold, fixed-income ranked by blend, high cash reserve',
  risk_off_trend: 'Cautious — reduced equity, bond-favored, elevated cash',
  choppy: 'Neutral — balanced exposure, tighter sizing',
  risk_on_trend: 'Constructive — equity-favored, normal sizing',
  calm_uptrend: 'Growth — full equity exposure, normal cash',
};

interface Props {
  signals?: ExpertSignals;
  metrics: PortfolioMetrics;
  candidateBundle: CandidateBundleSummary | null;
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function PipelineNode({ label, active, color }: {
  label: string; active: boolean; color: string;
}) {
  return (
    <span className="pipeline-node">
      <span className="pipeline-dot" style={{ backgroundColor: active ? color : '#334155' }} />
      <span className="pipeline-node-label">{label}</span>
    </span>
  );
}

export function SystemStatusBar({ signals, metrics, candidateBundle }: Props) {
  const regime = signals?.final_regime_label || 'unknown';
  const regimeColor = REGIME_COLORS[regime] || '#64748b';
  const isLive = candidateBundle?.promotion_status === 'live_active';
  const version = isLive ? candidateBundle?.version_id : 'opt-bootstrap';
  const posture = REGIME_POSTURE[regime] || 'Unknown posture';
  const cashPct = metrics.cash_pct ?? (metrics.total_value > 0 ? metrics.cash / metrics.total_value : 0);

  const hasThrottle = signals && signals.risk_throttle_factor > 0;
  const hasOverride = signals?.override_reason != null;

  // Build a single comprehensive pipeline summary for the tooltip
  const pipelineSummary = signals
    ? `Pipeline state (live values):

Signals → Macro: ${signals.macro_credit_score.toFixed(2)} · Vol: ${signals.vol_uncertainty_score.toFixed(2)} · Frag: ${signals.fragility_score.toFixed(2)} · Entropy: ${signals.entropy_score.toFixed(2)}

Regime → ${regime.replace(/_/g, ' ')} (${(signals.regime_confidence * 100).toFixed(0)}% confidence)

Fusion → ${hasThrottle ? `Throttle: ${(signals.risk_throttle_factor * 100).toFixed(0)}%` : 'No throttle'}${hasOverride ? ` · Override: ${signals.override_reason}` : ''}

Scoring → ${isLive ? '65% health + 35% ranking MLP (hybrid)' : 'Health-only scoring'}

Actions → ${metrics.total_trades} trades · ${(metrics.win_rate * 100).toFixed(0)}% win rate`
    : 'Signal data not loaded';

  return (
    <div className="system-status-bar" style={{ borderLeftColor: regimeColor }}>
      <div className="status-bar-row">
        <div className="status-item">
          <span className="status-label">Model</span>
          <span className="status-value" style={{ fontFamily: 'monospace', color: isLive ? '#22c55e' : '#94a3b8' }}>
            {version}
          </span>
          <InfoTooltip
            content={isLive
              ? `Active promoted model: ${version}\nPromoted: ${candidateBundle?.promotion_date}\nBlend: 35% ranking MLP + 65% health score\nRollback: ${candidateBundle?.rollback_version ?? 'available'}`
              : `Bootstrap model active. No hybrid promotion yet — using health-only scoring.`}
            label="Active model version"
          />
        </div>

        <div className="status-divider" />

        <div className="status-item">
          <span className="status-label">Regime</span>
          <span className="status-value" style={{ textTransform: 'capitalize' }}>
            <span style={{
              display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
              backgroundColor: regimeColor, marginRight: 6, verticalAlign: 'middle',
            }} />
            {regime.replace(/_/g, ' ')}
          </span>
        </div>

        <div className="status-divider" />

        <div className="status-item">
          <span className="status-label">Cash</span>
          <span className="status-value">{formatPct(cashPct)}</span>
        </div>

        <div className="status-divider" />

        <div className="status-item">
          <span className="status-label">Ranking</span>
          <span className="status-value" style={{ color: isLive ? '#22c55e' : '#64748b' }}>
            {isLive ? 'Active' : 'Off'}
          </span>
        </div>

        {isLive && candidateBundle?.rollback_available && (
          <>
            <div className="status-divider" />
            <div className="status-item">
              <span className="status-label">Rollback</span>
              <span className="status-value" style={{ color: '#94a3b8' }}>Ready</span>
            </div>
          </>
        )}
      </div>

      {/* Pipeline mini-diagram */}
      <div className="pipeline-row">
        <PipelineNode label="Signals" active={true} color="#3b82f6" />
        <span className="pipeline-arrow">→</span>
        <PipelineNode label="Regime" active={true} color={regimeColor} />
        <span className="pipeline-arrow">→</span>
        <PipelineNode label="Fusion" active={true} color={hasOverride || hasThrottle ? '#eab308' : '#22c55e'} />
        <span className="pipeline-arrow">→</span>
        <PipelineNode label="Score" active={true} color={isLive ? '#22c55e' : '#94a3b8'} />
        <span className="pipeline-arrow">→</span>
        <PipelineNode label="Actions" active={true} color="#3b82f6" />
        <InfoTooltip
          content={pipelineSummary}
          label="Decision pipeline"
        />
      </div>

      {/* Regime-Action Annotation */}
      <div className="status-posture">
        {posture}
      </div>
    </div>
  );
}
