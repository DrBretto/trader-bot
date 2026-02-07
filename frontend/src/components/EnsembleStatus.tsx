import { EnsembleMetrics, ExpertSignals } from '../types';

interface Props {
  ensemble?: EnsembleMetrics;
  signals?: ExpertSignals;
}

const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

const OVERRIDE_LABELS: Record<string, string> = {
  panic_override: 'Panic Override',
  unstable_calm_override: 'Unstable Calm',
  macro_downgrade: 'Macro Downgrade',
  macro_upgrade: 'Macro Upgrade',
};

function ScoreBar({ value, min, max, color, threshold }: {
  value: number; min: number; max: number; color: string; threshold?: number;
}) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const threshPct = threshold !== undefined
    ? Math.max(0, Math.min(100, ((threshold - min) / (max - min)) * 100))
    : undefined;
  return (
    <div style={{ position: 'relative', height: 6, backgroundColor: '#334155', borderRadius: 3, overflow: 'visible' }}>
      <div style={{ width: `${pct}%`, height: '100%', backgroundColor: color, borderRadius: 3 }} />
      {threshPct !== undefined && (
        <div style={{
          position: 'absolute', top: -2, left: `${threshPct}%`,
          width: 2, height: 10, backgroundColor: '#ef4444',
        }} />
      )}
    </div>
  );
}

function FusionRule({ step, label, active, detail }: {
  step: number; label: string; active: boolean; detail: string;
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 0',
      opacity: active ? 1 : 0.4,
    }}>
      <span style={{
        fontSize: 11, fontWeight: 600, color: '#64748b',
        width: 16, textAlign: 'right', flexShrink: 0,
      }}>{step}</span>
      <span style={{
        fontSize: 12,
        fontWeight: active ? 600 : 400,
        color: active ? '#eab308' : '#94a3b8',
      }}>
        {label}
      </span>
      <span style={{
        fontSize: 11, color: '#64748b', marginLeft: 'auto', textAlign: 'right',
      }}>
        {detail}
      </span>
    </div>
  );
}

export function EnsembleStatus({ ensemble, signals }: Props) {
  // If no expert signals, show legacy view
  if (!signals) {
    if (!ensemble || !ensemble.is_ensemble) {
      return (
        <div className="card ensemble-card">
          <div className="card-title">Model Status</div>
          <div style={{ color: '#94a3b8', fontSize: '14px' }}>
            Single model (baseline)
          </div>
        </div>
      );
    }
    return <LegacyEnsembleView ensemble={ensemble} />;
  }

  const regime = signals.final_regime_label;
  const regimeColor = REGIME_COLORS[regime] || '#64748b';
  const sizePct = (signals.position_size_modifier * 100).toFixed(0);
  const throttlePct = (signals.risk_throttle_factor * 100).toFixed(0);

  // Fusion rule evaluation
  const panicProb = signals.panic_prob ?? 0;
  const volLabel = signals.vol_regime_label;
  const macroScore = signals.macro_credit_score;
  const fragScore = signals.fragility_score;
  const entropyFlag = signals.entropy_shift_flag;
  const ensDisagreement = signals.ensemble_disagreement ?? 0;
  const overrideReason = signals.override_reason;

  const isPanicActive = overrideReason === 'panic_override';
  const isUnstableActive = overrideReason === 'unstable_calm_override';
  const isMacroActive = overrideReason === 'macro_downgrade' || overrideReason === 'macro_upgrade';
  const isFragilityActive = fragScore > 0.75;
  const isEntropyActive = entropyFlag;
  const isDisagreementActive = ensDisagreement > 0.3;

  return (
    <div className="card" style={{ minHeight: 200 }}>
      <div className="card-title">Regime Decision</div>

      {/* A) Final Decision Banner */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 16px', backgroundColor: '#0f172a',
        borderRadius: 8, marginBottom: 16,
      }}>
        <div style={{
          width: 14, height: 14, borderRadius: '50%',
          backgroundColor: regimeColor,
          boxShadow: `0 0 8px ${regimeColor}`,
        }} />
        <span style={{ fontWeight: 600, fontSize: 16, textTransform: 'capitalize' }}>
          {regime.replace(/_/g, ' ')}
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            Size: <span style={{ color: Number(sizePct) < 60 ? '#ef4444' : '#22c55e', fontWeight: 600 }}>{sizePct}%</span>
          </span>
          {Number(throttlePct) > 0 && (
            <span style={{ fontSize: 12, color: '#eab308' }}>
              Throttle: {throttlePct}%
            </span>
          )}
          {overrideReason && OVERRIDE_LABELS[overrideReason] && (
            <span style={{
              fontSize: 11, padding: '2px 8px', borderRadius: 4,
              backgroundColor: 'rgba(234, 179, 8, 0.15)',
              color: '#eab308', fontWeight: 500,
            }}>
              {OVERRIDE_LABELS[overrideReason]}
            </span>
          )}
        </div>
      </div>

      {/* B) Input Sources - Two Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>

        {/* Left: Ensemble Models */}
        <div>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
            Ensemble Models
          </div>
          {ensemble?.gru_prediction && ensemble?.transformer_prediction ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <ModelRow label="GRU" prediction={ensemble.gru_prediction.label} confidence={ensemble.gru_prediction.confidence} />
              <ModelRow label="XFMR" prediction={ensemble.transformer_prediction.label} confidence={ensemble.transformer_prediction.confidence} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  backgroundColor: ensemble.agreement >= 0.8 ? '#22c55e' : ensemble.agreement >= 0.6 ? '#eab308' : '#ef4444',
                }} />
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  {(ensemble.agreement * 100).toFixed(0)}% agreement
                </span>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: '#64748b' }}>
              Raw: {(signals.ensemble_regime_label || 'unknown').replace(/_/g, ' ')}
            </div>
          )}
        </div>

        {/* Right: Expert Signals */}
        <div>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
            Expert Signals
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <SignalRow
              label="Macro/Credit"
              value={macroScore.toFixed(2)}
              bar={<ScoreBar value={macroScore} min={-1} max={1} color={macroScore > 0 ? '#22c55e' : '#ef4444'} />}
            />
            <SignalRow
              label="Vol Regime"
              value={volLabel}
              bar={<ScoreBar value={signals.vol_uncertainty_score} min={0} max={1} color={signals.vol_uncertainty_score > 0.6 ? '#ef4444' : '#22c55e'} />}
            />
            <SignalRow
              label="Fragility"
              value={fragScore.toFixed(2)}
              bar={<ScoreBar value={fragScore} min={0} max={1} color={fragScore > 0.75 ? '#ef4444' : '#a855f7'} threshold={0.75} />}
              alert={isFragilityActive}
            />
            <SignalRow
              label="Entropy"
              value={entropyFlag ? 'SHIFT' : 'normal'}
              bar={<ScoreBar value={signals.entropy_score} min={0} max={1} color={entropyFlag ? '#ef4444' : '#06b6d4'} />}
              alert={isEntropyActive}
            />
          </div>
        </div>
      </div>

      {/* C) Fusion Logic Trail */}
      <div style={{ borderTop: '1px solid #334155', paddingTop: 12 }}>
        <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 }}>
          Fusion Rules
        </div>
        <FusionRule step={1} label="Panic Override" active={isPanicActive}
          detail={isPanicActive ? `panic ${(panicProb * 100).toFixed(0)}%` : `panic ${(panicProb * 100).toFixed(1)}%`} />
        <FusionRule step={2} label="Unstable Calm" active={isUnstableActive}
          detail={`vol: ${volLabel}`} />
        <FusionRule step={3} label="Macro Modulation" active={isMacroActive}
          detail={`score: ${macroScore.toFixed(2)}`} />
        <FusionRule step={4} label="Fragility Gate" active={isFragilityActive}
          detail={isFragilityActive ? `${fragScore.toFixed(2)} > 0.75 → cap 60%` : `${fragScore.toFixed(2)}`} />
        <FusionRule step={5} label="Entropy Shift" active={isEntropyActive}
          detail={isEntropyActive ? 'shift detected → size ×0.7' : 'no shift'} />
        <FusionRule step={6} label="Ensemble Disagreement" active={isDisagreementActive}
          detail={isDisagreementActive ? `${(ensDisagreement * 100).toFixed(0)}% → size ×${((signals.ensemble_multiplier ?? 1) * 100).toFixed(0)}%` : `${(ensDisagreement * 100).toFixed(0)}%`} />
      </div>
    </div>
  );
}

function ModelRow({ label, prediction, confidence }: {
  label: string; prediction: string; confidence: number;
}) {
  const regimeColor = REGIME_COLORS[prediction] || '#64748b';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: '#64748b', width: 36 }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 500, color: regimeColor, textTransform: 'capitalize', flex: 1 }}>
        {prediction.replace(/_/g, ' ')}
      </span>
      <span style={{ fontSize: 11, color: '#94a3b8' }}>{(confidence * 100).toFixed(0)}%</span>
    </div>
  );
}

function SignalRow({ label, value, bar, alert }: {
  label: string; value: string; bar: React.ReactNode; alert?: boolean;
}) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: alert ? '#eab308' : '#94a3b8' }}>
          {alert ? '! ' : ''}{label}
        </span>
        <span style={{ fontSize: 11, fontWeight: 500, color: alert ? '#eab308' : '#f8fafc', textTransform: 'capitalize' }}>
          {value}
        </span>
      </div>
      {bar}
    </div>
  );
}

// Legacy view for when expert signals are not available
function LegacyEnsembleView({ ensemble }: { ensemble: EnsembleMetrics }) {
  const agreementPct = (ensemble.agreement * 100).toFixed(0);
  const confidencePct = (ensemble.confidence * 100).toFixed(0);
  const sizingPct = (ensemble.position_size_multiplier * 100).toFixed(0);

  return (
    <div className="card ensemble-card">
      <div className="card-title">Ensemble Model Status</div>
      <div className="ensemble-status-row" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '12px', height: '12px', borderRadius: '50%',
            backgroundColor: ensemble.agreement >= 0.8 ? '#22c55e' : ensemble.agreement >= 0.6 ? '#eab308' : '#ef4444',
          }} />
          <span style={{ fontWeight: 600 }}>
            {ensemble.agreement >= 0.8 ? 'High Agreement' : ensemble.agreement >= 0.6 ? 'Moderate Agreement' : 'Models Disagree'}
          </span>
        </div>
        <span style={{ color: '#94a3b8' }}>{agreementPct}% aligned</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div className="ensemble-metric">
          <div className="ensemble-metric-label">Confidence</div>
          <div className="ensemble-metric-value">{confidencePct}%</div>
        </div>
        <div className="ensemble-metric">
          <div className="ensemble-metric-label">Position Sizing</div>
          <div className="ensemble-metric-value">{sizingPct}%</div>
        </div>
      </div>
    </div>
  );
}
