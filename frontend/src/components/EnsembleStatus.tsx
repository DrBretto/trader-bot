import { EnsembleMetrics } from '../types';

interface Props {
  ensemble?: EnsembleMetrics;
}

export function EnsembleStatus({ ensemble }: Props) {
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

  const agreementPct = (ensemble.agreement * 100).toFixed(0);
  const confidencePct = (ensemble.confidence * 100).toFixed(0);
  const sizingPct = (ensemble.position_size_multiplier * 100).toFixed(0);

  // Determine status color based on agreement
  const getStatusColor = () => {
    if (ensemble.agreement >= 0.8) return '#22c55e'; // green
    if (ensemble.agreement >= 0.6) return '#eab308'; // yellow
    return '#ef4444'; // red
  };

  const getStatusText = () => {
    if (ensemble.agreement >= 0.8) return 'High Agreement';
    if (ensemble.agreement >= 0.6) return 'Moderate Agreement';
    return 'Models Disagree';
  };

  return (
    <div className="card ensemble-card">
      <div className="card-title">Ensemble Model Status</div>

      {/* Agreement Indicator */}
      <div className="ensemble-status-row" style={{ marginBottom: '16px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <div
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: getStatusColor(),
              boxShadow: `0 0 8px ${getStatusColor()}`,
            }}
          />
          <span style={{ fontWeight: 600, color: getStatusColor() }}>
            {getStatusText()}
          </span>
        </div>
        <span style={{ color: '#94a3b8' }}>{agreementPct}% aligned</span>
      </div>

      {/* Metrics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div className="ensemble-metric">
          <div className="ensemble-metric-label">Confidence</div>
          <div className="ensemble-metric-value">{confidencePct}%</div>
          <div
            className="ensemble-bar"
            style={{ backgroundColor: '#334155' }}
          >
            <div
              style={{
                width: `${confidencePct}%`,
                height: '100%',
                backgroundColor: '#3b82f6',
                borderRadius: '2px',
              }}
            />
          </div>
        </div>

        <div className="ensemble-metric">
          <div className="ensemble-metric-label">Position Sizing</div>
          <div className="ensemble-metric-value">{sizingPct}%</div>
          <div
            className="ensemble-bar"
            style={{ backgroundColor: '#334155' }}
          >
            <div
              style={{
                width: `${sizingPct}%`,
                height: '100%',
                backgroundColor: sizingPct === '100' ? '#22c55e' : '#eab308',
                borderRadius: '2px',
              }}
            />
          </div>
        </div>
      </div>

      {/* Individual Model Predictions */}
      {ensemble.gru_prediction && ensemble.transformer_prediction && (
        <div style={{ borderTop: '1px solid #334155', paddingTop: '12px' }}>
          <div
            style={{
              fontSize: '12px',
              color: '#94a3b8',
              marginBottom: '8px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            Individual Models
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '12px',
            }}
          >
            <div className="model-prediction">
              <div className="model-name">GRU</div>
              <div className="model-regime">
                {ensemble.gru_prediction.label?.replace(/_/g, ' ') || 'N/A'}
              </div>
              <div className="model-confidence">
                {(ensemble.gru_prediction.confidence * 100).toFixed(0)}% conf
              </div>
            </div>

            <div className="model-prediction">
              <div className="model-name">Transformer</div>
              <div className="model-regime">
                {ensemble.transformer_prediction.label?.replace(/_/g, ' ') || 'N/A'}
              </div>
              <div className="model-confidence">
                {(ensemble.transformer_prediction.confidence * 100).toFixed(0)}% conf
              </div>
            </div>
          </div>

          {/* Visual disagreement indicator */}
          {ensemble.gru_prediction.label !== ensemble.transformer_prediction.label && (
            <div
              style={{
                marginTop: '12px',
                padding: '8px 12px',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '6px',
                fontSize: '12px',
                color: '#f87171',
              }}
            >
              ⚠️ Models predict different regimes — position sizes reduced to {sizingPct}%
            </div>
          )}
        </div>
      )}
    </div>
  );
}
