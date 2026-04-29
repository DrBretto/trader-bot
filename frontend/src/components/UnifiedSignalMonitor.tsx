import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, TooltipProps } from 'recharts';
import { TimeseriesPoint } from '../types';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  timeseries: TimeseriesPoint[];
  lastFiredSignal?: string;
}

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

export function UnifiedSignalMonitor({ timeseries, lastFiredSignal }: Props) {
  const defaultSignal: SignalKey = (lastFiredSignal as SignalKey) || 'frag';
  const [activeSignals, setActiveSignals] = useState<Set<SignalKey>>(new Set([defaultSignal]));
  const [showAll, setShowAll] = useState(false);

  if (timeseries.length === 0) return null;

  const visibleSignals = showAll ? SIGNAL_ORDER : Array.from(activeSignals);

  const toggleSignal = (key: SignalKey) => {
    if (showAll) {
      setShowAll(false);
      setActiveSignals(new Set([key]));
    } else if (activeSignals.has(key) && activeSignals.size === 1) {
      return; // Don't remove last signal
    } else if (activeSignals.has(key)) {
      const next = new Set(activeSignals);
      next.delete(key);
      setActiveSignals(next);
    } else {
      setActiveSignals(new Set([...activeSignals, key]));
    }
  };

  // Compute Y domain from visible signals
  const last90 = timeseries.slice(-90);
  let yMin = -0.2, yMax = 1.1;
  if (visibleSignals.includes('macro')) { yMin = -1.1; }

  return (
    <div className="card" style={{ padding: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>Signal Monitor</span>
          <InfoTooltip
            content="Expert signal history. Each signal feeds into the fusion rule chain. Threshold markers show where rules activate. Toggle signals to compare or isolate."
            label="Signal monitor"
          />
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {SIGNAL_ORDER.map(key => (
            <button
              key={key}
              className={`signal-toggle ${(showAll || activeSignals.has(key)) ? 'active' : ''}`}
              style={{ borderColor: (showAll || activeSignals.has(key)) ? SIGNALS[key].color : undefined }}
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
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={last90} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={(d) => { try { return format(parseISO(d), 'MMM d'); } catch { return d; } }}
            interval="preserveStartEnd"
            tickLine={false}
          />
          <YAxis
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
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
              strokeWidth={1.5}
              dot={false}
              strokeOpacity={showAll && visibleSignals.length > 1 ? 0.6 : 1}
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
                strokeOpacity={0.5}
                label={{ value: sig.thresholdLabel || '', position: 'right', fill: sig.color, fontSize: 9 }}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
