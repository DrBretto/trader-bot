import { InfoTooltip } from './InfoTooltip';

interface ModelPrediction {
  label: string;
  confidence: number;
  probs: Record<string, number>;
}

interface EnsembleData {
  confidence: number;
  disagreement: number;
  agreement: number;
  gru_prediction?: ModelPrediction;
  transformer_prediction?: ModelPrediction;
}

interface Props {
  ensemble?: EnsembleData;
}

const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

function ProbBar({ regime, prob, maxProb }: { regime: string; prob: number; maxProb: number }) {
  const color = REGIME_COLORS[regime] || '#64748b';
  const pct = Math.max(1, (prob / Math.max(maxProb, 0.01)) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0' }}>
      <span style={{ fontSize: 10, color: '#64748b', width: 55, textTransform: 'capitalize', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {regime.replace(/_/g, ' ')}
      </span>
      <div style={{ flex: 1, height: 4, backgroundColor: '#1e293b', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', backgroundColor: color, borderRadius: 2, transition: 'width 0.3s ease' }} />
      </div>
      <span style={{ fontSize: 10, color: '#94a3b8', width: 32, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
        {(prob * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function ModelColumn({ title, prediction }: { title: string; prediction?: ModelPrediction }) {
  if (!prediction) return null;
  const sorted = Object.entries(prediction.probs).sort((a, b) => b[1] - a[1]);
  const maxProb = sorted[0]?.[1] ?? 1;
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 500 }}>{title}</div>
      {sorted.map(([regime, prob]) => (
        <ProbBar key={regime} regime={regime} prob={prob} maxProb={maxProb} />
      ))}
    </div>
  );
}

export function ModelPredictionsPanel({ ensemble }: Props) {
  if (!ensemble?.gru_prediction && !ensemble?.transformer_prediction) return null;

  const agreeColor = ensemble.agreement >= 0.95 ? '#22c55e' : ensemble.agreement >= 0.7 ? '#eab308' : '#ef4444';

  return (
    <div className="card" style={{ padding: 12 }}>
      <div className="card-title" style={{ marginBottom: 8 }}>
        <span>Model Predictions</span>
        <InfoTooltip
          content={`GRU and Transformer independently predict regime probabilities.\nEnsemble weights (GRU ~55%, Transformer ~68%) are fixed from the optimizer — they do not adapt at runtime.\nAgreement shows how aligned the two models are.`}
          label="Model predictions"
        />
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <ModelColumn title="GRU" prediction={ensemble.gru_prediction} />
        <ModelColumn title="Transformer" prediction={ensemble.transformer_prediction} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, paddingTop: 6, borderTop: '1px solid #334155', fontSize: 11 }}>
        <span style={{ color: agreeColor }}>
          Agreement: {(ensemble.agreement * 100).toFixed(0)}%
        </span>
        <span style={{ color: '#64748b' }}>
          Disagreement: {(ensemble.disagreement * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
