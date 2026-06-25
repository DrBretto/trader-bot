import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, TooltipProps,
} from 'recharts';
import { EquityCurvePoint, TimeseriesPoint } from '../../types';
import { ShadowTimeseries } from '../../hooks/useShadowData';
import { format, parseISO } from 'date-fns';

interface Props {
  equityData: EquityCurvePoint[];
  timeseries: TimeseriesPoint[];
  shadow?: ShadowTimeseries | null;
}

interface MergedPoint {
  date: string;
  value: number;
  benchmark: number;
  shadowA?: number | null;
  regimeLabel?: string | null;
}

const REGIME_SHADER_COLORS: Record<string, string> = {
  calm_uptrend: 'rgba(34, 197, 94, 0.03)',
  risk_on_trend: 'rgba(59, 130, 246, 0.03)',
  choppy: 'rgba(234, 179, 8, 0.03)',
  risk_off_trend: 'rgba(249, 115, 22, 0.035)',
  high_vol_panic: 'rgba(239, 68, 68, 0.035)',
};

function formatCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

function MobileTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload as MergedPoint | undefined;
  if (!point) return null;
  let dateStr: string;
  try { dateStr = format(parseISO(point.date), 'MMM d'); } catch { dateStr = point.date; }
  return (
    <div style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, padding: '6px 10px', fontSize: 11 }}>
      <div style={{ color: '#94a3b8' }}>{dateStr}</div>
      <div style={{ color: '#f59e0b' }}>{formatCurrency(point.value)}</div>
      <div style={{ color: '#64748b' }}>SPY: {formatCurrency(point.benchmark)}</div>
    </div>
  );
}

export function MobileChart({ equityData, timeseries, shadow }: Props) {
  const regimeByDate = new Map(timeseries.map(pt => [pt.date, pt.final_regime_label]));

  // Dual forward shadow (paper book A). Absent/armed payloads draw nothing;
  // the caption below the chart reports the armed state.
  const shadowAByDate = new Map(shadow?.shadow_A ?? []);
  const hasShadowA = shadowAByDate.size > 0;
  const shadowArmed = !!shadow && !hasShadowA;

  const startVal = equityData[0]?.value ?? 0;
  const firstMoveIdx = equityData.findIndex(p => Math.abs(p.value - startVal) / (startVal || 1) > 0.001);
  const trimmed = firstMoveIdx > 1 ? equityData.slice(firstMoveIdx - 1) : equityData;

  const merged: MergedPoint[] = trimmed.map(p => ({
    date: p.date,
    value: p.value,
    benchmark: p.benchmark,
    shadowA: shadowAByDate.get(p.date) ?? null,
    regimeLabel: regimeByDate.get(p.date) ?? null,
  }));

  const allValues = merged.flatMap(p => [p.value, p.benchmark]);
  const yMin = Math.floor(Math.min(...allValues) / 1000) * 1000;
  const yMax = Math.ceil(Math.max(...allValues) / 1000) * 1000;

  const regimeSegments: { regime: string; x1: string; x2: string }[] = [];
  if (merged.length > 1) {
    let startIdx = 0;
    let currentRegime = regimeByDate.get(merged[0].date) ?? null;
    let carryRegime = currentRegime;
    for (let i = 1; i <= merged.length; i++) {
      const next = i < merged.length ? regimeByDate.get(merged[i].date) ?? null : null;
      if (i === merged.length || next !== currentRegime) {
        const span = i - startIdx;
        const reg = span < 4 ? carryRegime : currentRegime;
        if (reg && REGIME_SHADER_COLORS[reg]) {
          regimeSegments.push({ regime: reg, x1: merged[startIdx].date, x2: merged[Math.max(startIdx, i - 1)].date });
        }
        if (currentRegime) carryRegime = currentRegime;
        startIdx = i;
        currentRegime = next;
      }
    }
    if (regimeSegments.length > 0 && merged.length > 0) {
      const last = regimeSegments[regimeSegments.length - 1];
      const lastDate = merged[merged.length - 1].date;
      if (last.x2 < lastDate) last.x2 = lastDate;
    }
  }

  return (
    <div className="mobile-chart">
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={merged} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
          <defs>
            <linearGradient id="mobileEquityGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickFormatter={d => { try { return format(parseISO(d), 'MMM d'); } catch { return d; } }}
            interval="preserveStartEnd"
            axisLine={{ stroke: '#1e293b' }}
            tickLine={false}
          />
          <YAxis
            yAxisId="eq"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
            domain={[yMin, yMax]}
            axisLine={false}
            tickLine={false}
            width={42}
          />
          <Tooltip content={<MobileTooltip />} />
          {regimeSegments.map((seg, i) => (
            <ReferenceArea key={i} yAxisId="eq" x1={seg.x1} x2={seg.x2} fill={REGIME_SHADER_COLORS[seg.regime]} strokeOpacity={0} ifOverflow="extendDomain" />
          ))}
          <Area yAxisId="eq" type="monotone" dataKey="value" fill="url(#mobileEquityGlow)" stroke="none" />
          <Line yAxisId="eq" type="monotone" dataKey="benchmark" stroke="#64748b" strokeWidth={1.5} strokeDasharray="6 4" dot={false} />
          {hasShadowA && (
            <Line yAxisId="eq" type="monotone" dataKey="shadowA" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="3 4" dot={false} connectNulls />
          )}
          <Line yAxisId="eq" type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 3, fill: '#f59e0b', stroke: '#0f172a', strokeWidth: 2 }} />
        </ComposedChart>
      </ResponsiveContainer>
      {(hasShadowA || shadowArmed) && (
        <div style={{ fontSize: 10, color: '#94a3b8', padding: '2px 8px 0' }}>
          <span style={{ color: '#3b82f6' }}>—</span>{' '}
          {hasShadowA
            ? 'Shadow: new brain tilt (paper), accruing'
            : 'Shadow (paper): armed — accruing from tonight'}
        </div>
      )}
    </div>
  );
}
