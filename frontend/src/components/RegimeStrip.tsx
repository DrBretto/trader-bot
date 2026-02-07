import { TimeseriesPoint, ExpertSignals } from '../types';

interface Props {
  signals?: ExpertSignals;
  timeseries: TimeseriesPoint[];
}

const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

function SignalGauge({ label, value, min, max, color }: {
  label: string; value: number; min: number; max: number; color: string;
}) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div style={{ flex: 1, minWidth: 120 }}>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>{value.toFixed(2)}</div>
      <div style={{ height: 4, backgroundColor: '#334155', borderRadius: 2 }}>
        <div style={{ width: `${pct}%`, height: '100%', backgroundColor: color, borderRadius: 2 }} />
      </div>
    </div>
  );
}

export function RegimeStrip({ signals, timeseries }: Props) {
  if (!signals && timeseries.length === 0) return null;

  const regime = signals?.final_regime_label || 'unknown';
  const regimeColor = REGIME_COLORS[regime] || '#64748b';

  return (
    <div className="card" style={{ marginBottom: 24 }}>
      <div className="card-title">Market Intelligence</div>

      {/* Current regime banner */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 16px', backgroundColor: '#0f172a',
        borderRadius: 8, marginBottom: 16,
      }}>
        <div style={{
          width: 12, height: 12, borderRadius: '50%',
          backgroundColor: regimeColor,
        }} />
        <span style={{ fontWeight: 600, textTransform: 'capitalize' }}>
          {regime.replace(/_/g, ' ')}
        </span>
        {signals && (
          <span style={{ color: '#94a3b8', fontSize: 13, marginLeft: 'auto' }}>
            Confidence: {(signals.regime_confidence * 100).toFixed(0)}%
            {signals.risk_throttle_factor > 0 && (
              <span style={{ color: '#eab308', marginLeft: 12 }}>
                Throttle: {(signals.risk_throttle_factor * 100).toFixed(0)}%
              </span>
            )}
          </span>
        )}
      </div>

      {/* Expert score gauges */}
      {signals && (
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <SignalGauge label="Macro/Credit" value={signals.macro_credit_score} min={-1} max={1}
            color={signals.macro_credit_score > 0 ? '#22c55e' : '#ef4444'} />
          <SignalGauge label="Vol Uncertainty" value={signals.vol_uncertainty_score} min={0} max={1}
            color={signals.vol_uncertainty_score > 0.6 ? '#ef4444' : '#22c55e'} />
          <SignalGauge label="Fragility" value={signals.fragility_score} min={0} max={1}
            color={signals.fragility_score > 0.6 ? '#ef4444' : '#22c55e'} />
          <SignalGauge label="Entropy" value={signals.entropy_score} min={0} max={1}
            color={signals.entropy_shift_flag ? '#ef4444' : '#22c55e'} />
          <SignalGauge label="Position Size" value={signals.position_size_modifier} min={0} max={1}
            color={signals.position_size_modifier < 0.6 ? '#ef4444' : '#22c55e'} />
        </div>
      )}

      {/* Regime strip (last 30 days from timeseries) */}
      {timeseries.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>Regime History</div>
          <div style={{ display: 'flex', gap: 1, height: 8 }}>
            {timeseries.slice(-60).map((pt, i) => (
              <div
                key={i}
                title={`${pt.date}: ${pt.final_regime_label}`}
                style={{
                  flex: 1,
                  backgroundColor: REGIME_COLORS[pt.final_regime_label] || '#64748b',
                  borderRadius: i === 0 ? '2px 0 0 2px' : i === Math.min(59, timeseries.length - 1) ? '0 2px 2px 0' : 0,
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
