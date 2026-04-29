import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, TooltipProps,
} from 'recharts';
import { ExpertSignals, TimeseriesPoint } from '../types';
import { format, parseISO } from 'date-fns';
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

interface FusionRule {
  order: number;
  code: string;
  label: string;
  fired: boolean;
  effect: string;
}

interface Props {
  ensemble?: EnsembleData;
  signals?: ExpertSignals;
  fusionRules?: FusionRule[];
  timeseries: TimeseriesPoint[];
  isHybridLive: boolean;
  mostFiredSignalKey?: string;
}

const REGIME_COLORS: Record<string, string> = {
  calm_uptrend: '#22c55e',
  risk_on_trend: '#3b82f6',
  choppy: '#eab308',
  risk_off_trend: '#f97316',
  high_vol_panic: '#ef4444',
};

type SignalKey = 'macro' | 'vol' | 'frag' | 'entropy';

const SIGNALS: Record<SignalKey, { key: keyof TimeseriesPoint; label: string; color: string; threshold?: number; thresholdLabel?: string }> = {
  macro: { key: 'macro_credit_score', label: 'Macro/Credit', color: '#3b82f6' },
  vol: { key: 'vol_uncertainty_score', label: 'Vol Uncertainty', color: '#06b6d4', threshold: 0.8, thresholdLabel: 'Panic zone' },
  frag: { key: 'fragility_score', label: 'Fragility', color: '#a855f7', threshold: 0.75, thresholdLabel: 'Sizing cap' },
  entropy: { key: 'entropy_score', label: 'Entropy', color: '#f59e0b' },
};

const SIGNAL_ORDER: SignalKey[] = ['macro', 'vol', 'frag', 'entropy'];

function SignalTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const pt = payload[0]?.payload;
  if (!pt) return null;
  let dateStr: string;
  try { dateStr = format(parseISO(pt.date), 'MMM d, yyyy'); } catch { dateStr = pt.date; }
  return (
    <div style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, padding: '8px 12px', fontSize: 11, lineHeight: 1.6 }}>
      <div style={{ color: '#94a3b8', fontWeight: 600, marginBottom: 2 }}>{dateStr}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <span style={{ fontWeight: 600 }}>{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
      {pt.final_regime_label && (
        <div style={{ color: '#64748b', marginTop: 2, borderTop: '1px solid #1e293b', paddingTop: 2 }}>
          Regime: {pt.final_regime_label.replace(/_/g, ' ')}
        </div>
      )}
    </div>
  );
}

export function SystemBrainPanel({ ensemble, fusionRules, timeseries, isHybridLive, mostFiredSignalKey }: Props) {
  const gru = ensemble?.gru_prediction;
  const trans = ensemble?.transformer_prediction;
  const agreement = ensemble?.agreement ?? 0;
  const agreeColor = agreement >= 0.95 ? '#22c55e' : agreement >= 0.7 ? '#eab308' : '#ef4444';

  // Signal chart state — default to All
  const [showAll, setShowAll] = useState(true);
  const [activeSignals, setActiveSignals] = useState<Set<SignalKey>>(new Set(SIGNAL_ORDER));

  const visibleSignals = showAll ? SIGNAL_ORDER : Array.from(activeSignals);

  const toggleSignal = (key: SignalKey) => {
    if (showAll) {
      setShowAll(false);
      setActiveSignals(new Set([key]));
    } else if (activeSignals.has(key) && activeSignals.size === 1) {
      return;
    } else if (activeSignals.has(key)) {
      const next = new Set(activeSignals);
      next.delete(key);
      setActiveSignals(next);
    } else {
      setActiveSignals(new Set([...activeSignals, key]));
    }
  };

  const last90 = timeseries.slice(-90);
  let yMin = -0.2, yMax = 1.1;
  if (visibleSignals.includes('macro')) { yMin = -1.1; }
  const hasSignalHistory = last90.length > 1;

  return (
    <div className="card system-brain-panel">
      <div className="brain-header-row">
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>System Brain</span>
          <InfoTooltip
            content="How the system sees the market right now. Two neural networks (GRU + Transformer) predict the regime. Fusion rules override sizing when risk signals fire. The signal chart shows the raw expert inputs — the highlighted signal is currently most active."
            label="System brain"
          />
        </div>
        <div className="brain-header-meta">
          <span className="brain-meta-chip">{isHybridLive ? '65/35 hybrid' : 'health-only'}</span>
          <span className="brain-meta-chip" data-learn-target="agreement-pct" style={{ color: agreeColor }}>
            Agree {(agreement * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Ensemble predictions — one compact row */}
      {(gru || trans) && (
        <div className="brain-prediction-row">
          {gru && (
            <span style={{ color: '#94a3b8' }}>
              GRU: <span style={{ color: '#f8fafc', fontWeight: 600 }}>{gru.label.replace(/_/g, ' ')} {(gru.confidence * 100).toFixed(0)}%</span>
            </span>
          )}
          {trans && (
            <span style={{ color: '#94a3b8' }}>
              Trans: <span style={{ color: '#f8fafc', fontWeight: 600 }}>{trans.label.replace(/_/g, ' ')} {(trans.confidence * 100).toFixed(0)}%</span>
            </span>
          )}
        </div>
      )}

      {/* Blend info */}
      <div className="brain-blend-line">
        Blend: {isHybridLive ? '65% health + 35% ranking MLP' : 'health-only'} · Weights fixed from optimizer
      </div>

      {/* Fusion rule dots */}
      {fusionRules && fusionRules.length > 0 && (
        <div data-learn-target="fusion-dots" style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 10 }}>
          {fusionRules.map(r => (
            <span
              key={r.code}
              title={`${r.label}: ${r.effect}`}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 3,
                fontSize: 10, color: r.fired ? '#f8fafc' : '#475569',
              }}
            >
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                backgroundColor: r.fired ? '#ef4444' : '#334155',
                transition: 'background-color 0.3s',
              }} />
              {r.label.replace(' Override', '').replace(' Gate', '').replace(' Sizing', '')}
            </span>
          ))}
        </div>
      )}

      {/* Integrated signal chart (was UnifiedSignalMonitor) */}
      {timeseries.length > 0 && (
        <div className="brain-signal-block">
          <div className="brain-signal-header">
            <span className="brain-signal-label">Signals (90d)</span>
            <div className="brain-signal-toggles">
              {SIGNAL_ORDER.map(key => (
                <button
                  key={key}
                  className={`signal-toggle ${(showAll || activeSignals.has(key)) ? 'active' : ''}`}
                  style={{
                    borderColor: (showAll || activeSignals.has(key)) ? SIGNALS[key].color : undefined,
                    opacity: showAll && key !== mostFiredSignalKey ? 0.45 : 1,
                  }}
                  onClick={() => toggleSignal(key)}
                >
                  {SIGNALS[key].label.split('/')[0]}
                </button>
              ))}
              <button
                className={`signal-toggle ${showAll ? 'active' : ''}`}
                onClick={() => { setShowAll(!showAll); if (!showAll) setActiveSignals(new Set(SIGNAL_ORDER)); }}
              >
                All
              </button>
            </div>
          </div>
          {hasSignalHistory ? (
            <div className="brain-signal-chart-wrap">
              <ResponsiveContainer width="100%" height={124}>
                <LineChart data={last90} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="#475569"
                    tick={{ fill: '#64748b', fontSize: 9 }}
                    tickFormatter={(d) => { try { return format(parseISO(d), 'MMM d'); } catch { return d; } }}
                    interval="preserveStartEnd"
                    tickLine={false}
                    axisLine={{ stroke: '#1e293b' }}
                  />
                  <YAxis
                    stroke="#475569"
                    tick={{ fill: '#64748b', fontSize: 9 }}
                    domain={[yMin, yMax]}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip content={<SignalTooltip />} />
                  {visibleSignals.map(key => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={SIGNALS[key].key}
                      name={SIGNALS[key].label}
                      stroke={SIGNALS[key].color}
                      dot={false}
                      strokeWidth={showAll && key === mostFiredSignalKey ? 2 : 1.5}
                      strokeOpacity={showAll && visibleSignals.length > 1 ? (key === mostFiredSignalKey ? 1 : 0.35) : 1}
                    />
                  ))}
                  {visibleSignals.map(key => {
                    const sig = SIGNALS[key];
                    if (!sig.threshold) return null;
                    return (
                      <ReferenceLine
                        key={`thresh-${key}`}
                        y={sig.threshold}
                        stroke={sig.color}
                        strokeDasharray="4 4"
                        strokeOpacity={0.4}
                        label={{ value: sig.thresholdLabel || '', position: 'right', fill: sig.color, fontSize: 8 }}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="brain-signal-fallback">Signal history unavailable for this snapshot.</div>
          )}
        </div>
      )}

      {/* Regime history strip */}
      {timeseries.length > 0 && (
        <div className="brain-regime-history">
          <div className="brain-regime-label">Regime history (60d)</div>
          <div className="brain-regime-strip">
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
