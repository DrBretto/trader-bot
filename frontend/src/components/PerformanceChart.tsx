import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  TooltipProps,
} from 'recharts';
import { EquityCurvePoint, DrawdownPoint } from '../types';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

const ALPACA_CUTOVER_DATE = '2026-03-12';

interface Props {
  equityData: EquityCurvePoint[];
  drawdownData: DrawdownPoint[];
  brokerOnly?: boolean;
}

interface MergedPoint {
  date: string;
  dateLabel: string;
  value: number;
  benchmark: number;
  drawdownPct: number;
  peak: number;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  const point = payload[0]?.payload as MergedPoint | undefined;
  if (!point) return null;

  let dateStr: string;
  try {
    dateStr = format(parseISO(point.date), 'MMM d, yyyy');
  } catch {
    dateStr = point.date || String(label);
  }

  const dd = point.drawdownPct;
  const ddColor = dd <= -10 ? '#ef4444' : dd <= -5 ? '#f97316' : dd < 0 ? '#eab308' : '#22c55e';
  const spread = point.value && point.benchmark ? point.value - point.benchmark : null;

  return (
    <div
      style={{
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: 10,
        padding: '10px 14px',
        fontSize: 12,
        lineHeight: 1.7,
        minWidth: 200,
      }}
    >
      <div style={{ color: '#94a3b8', marginBottom: 4, fontWeight: 600 }}>{dateStr}</div>
      <div style={{ color: '#60a5fa' }}>
        Portfolio: <span style={{ fontWeight: 600 }}>{formatCurrency(point.value)}</span>
      </div>
      <div style={{ color: '#94a3b8' }}>
        SPY Benchmark: <span style={{ fontWeight: 600 }}>{formatCurrency(point.benchmark)}</span>
      </div>
      {spread !== null && (
        <div style={{ color: spread >= 0 ? '#22c55e' : '#f97316' }}>
          vs SPY: <span style={{ fontWeight: 600 }}>{spread >= 0 ? '+' : ''}{formatCurrency(spread)}</span>
        </div>
      )}
      <div style={{ color: ddColor, marginTop: 2, borderTop: '1px solid #1e293b', paddingTop: 4 }}>
        Drawdown: <span style={{ fontWeight: 600 }}>{dd.toFixed(2)}%</span>
      </div>
    </div>
  );
}

export function PerformanceChart({ equityData, drawdownData, brokerOnly = false }: Props) {
  const filtered = brokerOnly
    ? equityData.filter((p) => p.date >= ALPACA_CUTOVER_DATE)
    : equityData;

  // Build drawdown lookup
  const ddMap = new Map(drawdownData.map((d) => [d.date, d.drawdown]));

  // Merge and compute running peak
  let peak = 0;
  const merged: MergedPoint[] = filtered.map((point) => {
    peak = Math.max(peak, point.value);
    return {
      date: point.date,
      dateLabel: format(parseISO(point.date), 'MMM yyyy'),
      value: point.value,
      benchmark: point.benchmark,
      drawdownPct: (ddMap.get(point.date) ?? 0) * 100,
      peak,
    };
  });

  const cutoverIdx = !brokerOnly
    ? merged.findIndex((p) => p.date >= ALPACA_CUTOVER_DATE)
    : -1;
  const cutoverLabel = cutoverIdx >= 0 ? merged[cutoverIdx]?.dateLabel : undefined;

  // Compute Y domains
  const allValues = merged.flatMap((p) => [p.value, p.benchmark]);
  const yMin = Math.floor(Math.min(...allValues) / 1000) * 1000;
  const yMax = Math.ceil(Math.max(...allValues) / 1000) * 1000;

  const ddMin = Math.min(...merged.map((p) => p.drawdownPct));
  const ddFloor = Math.floor(ddMin / 5) * 5;

  return (
    <div className="card performance-chart-card">
      <div className="card-title">
        <span>Performance</span>
        <InfoTooltip
          content={`Blue: portfolio equity. Gray dashed: SPY benchmark (same start value).
Red shading: drawdown from peak — deeper red means further underwater.
Hover for exact values, benchmark spread, and drawdown percentage.`}
          label="Integrated performance chart"
        />
      </div>

      {/* Main equity chart */}
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={merged} margin={{ top: 8, right: 20, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="underwaterGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.0} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.25} />
            </linearGradient>
            <linearGradient id="equityGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="dateLabel"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 11 }}
            interval="preserveStartEnd"
            axisLine={{ stroke: '#334155' }}
            tickLine={false}
          />
          <YAxis
            yAxisId="equity"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            domain={[yMin, yMax]}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#475569', strokeDasharray: '3 3' }} />

          {/* Cutover reference */}
          {cutoverLabel && (
            <ReferenceLine
              yAxisId="equity"
              x={cutoverLabel}
              stroke="#eab308"
              strokeDasharray="4 4"
              strokeOpacity={0.6}
              label={{
                value: 'Live',
                position: 'insideTopRight',
                fill: '#eab308',
                fontSize: 10,
                fontWeight: 600,
              }}
            />
          )}

          {/* Equity fill — subtle glow under the portfolio line */}
          <Area
            yAxisId="equity"
            type="monotone"
            dataKey="value"
            fill="url(#equityGlow)"
            stroke="none"
          />

          {/* Peak reference — faint line showing high-water mark */}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="peak"
            stroke="#334155"
            strokeWidth={1}
            strokeDasharray="2 4"
            dot={false}
            legendType="none"
          />

          {/* SPY benchmark */}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="benchmark"
            stroke="#64748b"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            dot={false}
            legendType="none"
          />

          {/* Portfolio equity — main line */}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={false}
            legendType="none"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Drawdown strip — tightly coupled below */}
      <div className="drawdown-strip">
        <ResponsiveContainer width="100%" height={80}>
          <ComposedChart data={merged} margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="drawdownFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.15} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal vertical={false} />
            <XAxis dataKey="dateLabel" hide />
            <YAxis
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(v) => `${v}%`}
              domain={[ddFloor, 0]}
              axisLine={false}
              tickLine={false}
              width={48}
            />
            <ReferenceLine y={0} stroke="#334155" />
            <Area
              type="monotone"
              dataKey="drawdownPct"
              fill="url(#drawdownFill)"
              stroke="#ef4444"
              strokeWidth={1.5}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Inline legend */}
      <div className="performance-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#3b82f6' }} /> Portfolio
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-dashed" style={{ background: '#64748b' }} /> SPY Benchmark
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#334155' }} /> Peak
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#ef4444', opacity: 0.5 }} /> Drawdown
        </span>
      </div>
    </div>
  );
}
