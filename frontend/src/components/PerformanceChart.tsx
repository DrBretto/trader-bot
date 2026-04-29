import { useState } from 'react';
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
  ReferenceArea,
  TooltipProps,
} from 'recharts';
import { ChartMarker, EquityCurvePoint, DrawdownPoint, MonthlyReturn, TimeseriesPoint } from '../types';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  equityData: EquityCurvePoint[];
  drawdownData: DrawdownPoint[];
  monthlyReturns?: MonthlyReturn[];
  timeseries?: TimeseriesPoint[];
  chartMarkers?: ChartMarker[];
}

const MARKER_COLORS: Record<string, string> = {
  milestone: '#22c55e',
  infrastructure: '#eab308',
  data_plane: '#06b6d4',
  model: '#a855f7',
  code_fix: '#f97316',
};
const MARKER_DEFAULT_COLOR = '#94a3b8';

interface MergedPoint {
  date: string;
  value: number;
  benchmark: number;
  drawdownPct: number;
  peak: number;
  isLive: boolean;
  regimeLabel?: string | null;
}

type EraView = 'all' | 'backtest' | 'live';

const REGIME_SHADER_COLORS: Record<string, string> = {
  calm_uptrend: 'rgba(34, 197, 94, 0.03)',
  risk_on_trend: 'rgba(59, 130, 246, 0.03)',
  choppy: 'rgba(234, 179, 8, 0.03)',
  risk_off_trend: 'rgba(249, 115, 22, 0.035)',
  high_vol_panic: 'rgba(239, 68, 68, 0.035)',
};

const REGIME_LABELS: Record<string, string> = {
  calm_uptrend: 'Calm',
  risk_on_trend: 'Risk On',
  choppy: 'Choppy',
  risk_off_trend: 'Risk Off',
  high_vol_panic: 'Panic',
};

const REGIME_KEY_DOT_COLORS: Record<string, string> = {
  calm_uptrend: 'rgba(34, 197, 94, 0.5)',
  risk_on_trend: 'rgba(59, 130, 246, 0.5)',
  choppy: 'rgba(234, 179, 8, 0.5)',
  risk_off_trend: 'rgba(249, 115, 22, 0.5)',
  high_vol_panic: 'rgba(239, 68, 68, 0.5)',
};

const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function heatColor(value: number): string {
  if (value >= 0.05) return 'rgba(34, 197, 94, 0.85)';
  if (value >= 0.02) return 'rgba(34, 197, 94, 0.55)';
  if (value > 0.001) return 'rgba(34, 197, 94, 0.3)';
  if (value >= -0.001) return 'rgba(51, 65, 85, 0.4)';
  if (value >= -0.02) return 'rgba(239, 68, 68, 0.3)';
  if (value >= -0.05) return 'rgba(239, 68, 68, 0.55)';
  return 'rgba(239, 68, 68, 0.85)';
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
  const era = point.isLive ? 'Live (Alpaca)' : 'Backtest';
  const regime = point.regimeLabel ? (REGIME_LABELS[point.regimeLabel] ?? point.regimeLabel.replace(/_/g, ' ')) : null;

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
      <div style={{ color: '#94a3b8', marginBottom: 4, fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
        <span>{dateStr}</span>
        <span style={{ fontSize: 10, color: point.isLive ? '#22c55e' : '#64748b', fontWeight: 500 }}>{era}</span>
      </div>
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
      {regime && (
        <div style={{ color: '#cbd5e1' }}>
          Regime band: <span style={{ fontWeight: 600 }}>{regime}</span>
        </div>
      )}
      <div style={{ color: ddColor, marginTop: 2, borderTop: '1px solid #1e293b', paddingTop: 4 }}>
        Drawdown: <span style={{ fontWeight: 600 }}>{dd.toFixed(2)}%</span>
      </div>
    </div>
  );
}

const ALPACA_CUTOVER_DATE = '2026-03-12';
const HYBRID_PROMOTION_DATE = '2026-03-28';

export function PerformanceChart({ equityData, drawdownData, monthlyReturns, timeseries = [], chartMarkers }: Props) {
  const [eraView, setEraView] = useState<EraView>('all');

  // Build drawdown lookup
  const ddMap = new Map(drawdownData.map((d) => [d.date, d.drawdown]));
  const regimeByDate = new Map(timeseries.map((pt) => [pt.date, pt.final_regime_label]));

  // Trim leading flat zone
  const startVal = equityData[0]?.value ?? 0;
  const firstMoveIdx = equityData.findIndex(p => Math.abs(p.value - startVal) / (startVal || 1) > 0.001);
  const trimmedEquity = firstMoveIdx > 1 ? equityData.slice(firstMoveIdx - 1) : equityData;

  // Merge and compute running peak
  let peak = 0;
  const allMerged: MergedPoint[] = trimmedEquity.map((point) => {
    peak = Math.max(peak, point.value);
    return {
      date: point.date,
      value: point.value,
      benchmark: point.benchmark,
      drawdownPct: (ddMap.get(point.date) ?? 0) * 100,
      peak,
      isLive: point.date >= ALPACA_CUTOVER_DATE,
      regimeLabel: regimeByDate.get(point.date) ?? null,
    };
  });

  // Filter by era view
  const merged = eraView === 'all'
    ? allMerged
    : eraView === 'live'
      ? allMerged.filter(p => p.isLive)
      : allMerged.filter(p => !p.isLive);

  const cutoverIdx = merged.findIndex((p) => p.date >= ALPACA_CUTOVER_DATE);
  const cutoverDate = cutoverIdx >= 0 ? merged[cutoverIdx]?.date : undefined;

  const hybridIdx = merged.findIndex((p) => p.date >= HYBRID_PROMOTION_DATE);
  const hybridDate = hybridIdx >= 0 ? merged[hybridIdx]?.date : undefined;

  const backtestStartDate = merged[0]?.date;
  const backtestEndDate = cutoverDate;

  // Compute Y domains
  const allValues = merged.flatMap((p) => [p.value, p.benchmark]);
  const yMin = Math.floor(Math.min(...allValues) / 1000) * 1000;
  const yMax = Math.ceil(Math.max(...allValues) / 1000) * 1000;

  const ddMin = Math.min(...merged.map((p) => p.drawdownPct));
  const ddFloor = Math.floor(ddMin / 5) * 5;

  // Counts
  const liveDays = allMerged.filter(p => p.isLive).length;
  const backtestDays = allMerged.length - liveDays;

  const regimeSegments: Array<{ regime: string; x1: string; x2: string }> = [];
  if (merged.length > 1) {
    let startIdx = 0;
    let currentRegime = regimeByDate.get(merged[0].date) ?? null;
    let carryRegime = currentRegime;

    for (let i = 1; i <= merged.length; i += 1) {
      const nextRegime = i < merged.length ? regimeByDate.get(merged[i].date) ?? null : null;
      if (i === merged.length || nextRegime !== currentRegime) {
        const span = i - startIdx;
        const regimeForSegment = span < 4 ? carryRegime : currentRegime;
        if (regimeForSegment && REGIME_SHADER_COLORS[regimeForSegment]) {
          regimeSegments.push({
            regime: regimeForSegment,
            x1: merged[startIdx].date,
            x2: merged[Math.max(startIdx, i - 1)].date,
          });
        }
        if (currentRegime) carryRegime = currentRegime;
        startIdx = i;
        currentRegime = nextRegime;
      }
    }
  }

  // Extend last regime shader to cover equity curve dates beyond timeseries range
  if (regimeSegments.length > 0 && merged.length > 0) {
    const lastSeg = regimeSegments[regimeSegments.length - 1];
    const lastDataDate = merged[merged.length - 1].date;
    if (lastSeg.x2 < lastDataDate) {
      lastSeg.x2 = lastDataDate;
    }
  }

  // Last point for live-edge dot
  const lastPoint = merged[merged.length - 1];

  // Monthly returns heatmap data — group by year
  const monthlyByYear = new Map<number, Map<number, MonthlyReturn>>();
  (monthlyReturns ?? [])
    .filter(mr => mr.year >= 2026)
    .forEach(mr => {
    if (!monthlyByYear.has(mr.year)) monthlyByYear.set(mr.year, new Map());
    monthlyByYear.get(mr.year)!.set(mr.month, mr);
  });
  const years = [...monthlyByYear.keys()].sort();

  return (
    <div className="card performance-chart-card">
      <div className="performance-chart-header">
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>Performance</span>
          <InfoTooltip
            content={`Portfolio equity (blue) vs SPY benchmark (gray dashed), both starting from the same dollar amount.
Drawdown strip shows how far below the peak the portfolio has fallen.
Background color bands show the detected market regime at each point in time.
Switch between All / Backtest / Live to isolate historical vs broker-connected periods.`}
            label="Performance chart"
          />
        </div>
        <div className="performance-chart-controls">
          <button
            className={`signal-toggle ${eraView === 'all' ? 'active' : ''}`}
            onClick={() => setEraView('all')}
          >
            All
          </button>
          <button
            className={`signal-toggle ${eraView === 'backtest' ? 'active' : ''}`}
            style={{ borderColor: eraView === 'backtest' ? '#64748b' : undefined }}
            onClick={() => setEraView('backtest')}
          >
            Backtest ({backtestDays}d)
          </button>
          <button
            className={`signal-toggle ${eraView === 'live' ? 'active' : ''}`}
            style={{ borderColor: eraView === 'live' ? '#22c55e' : undefined }}
            onClick={() => setEraView('live')}
          >
            Live ({liveDays}d)
          </button>
        </div>
      </div>

      {/* Main equity chart */}
      <div className="performance-main-chart">
        <ResponsiveContainer width="100%" height={292}>
          <ComposedChart data={merged} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="equityGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={(d) => {
              try { return format(parseISO(d), 'MMM d'); } catch { return d; }
            }}
            interval="preserveStartEnd"
            axisLine={{ stroke: '#1e293b' }}
            tickLine={false}
          />
          <YAxis
            yAxisId="equity"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            domain={[yMin, yMax]}
            axisLine={false}
            tickLine={false}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#475569', strokeDasharray: '3 3' }} />

          {regimeSegments.map((segment, index) => (
            <ReferenceArea
              key={`${segment.regime}-${index}`}
              yAxisId="equity"
              x1={segment.x1}
              x2={segment.x2}
              fill={REGIME_SHADER_COLORS[segment.regime]}
              strokeOpacity={0}
              ifOverflow="extendDomain"
            />
          ))}

          {/* Backtest region tint */}
          {eraView === 'all' && backtestStartDate && backtestEndDate && (
            <ReferenceArea
              yAxisId="equity"
              x1={backtestStartDate}
              x2={backtestEndDate}
              fill="#64748b"
              fillOpacity={0.12}
              strokeOpacity={0}
            />
          )}

          {/* Cutover reference */}
          {eraView === 'all' && cutoverDate && (
            <ReferenceLine
              yAxisId="equity"
              x={cutoverDate}
              stroke="#22c55e"
              strokeDasharray="4 4"
              strokeOpacity={0.6}
              label={{
                value: 'Live',
                position: 'insideTopRight',
                fill: '#22c55e',
                fontSize: 10,
                fontWeight: 600,
              }}
            />
          )}

          {/* Hybrid promotion reference */}
          {hybridDate && (
            <ReferenceLine
              yAxisId="equity"
              x={hybridDate}
              stroke="#a855f7"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
              label={{
                value: 'Hybrid',
                position: 'insideTopRight',
                fill: '#a855f7',
                fontSize: 10,
                fontWeight: 600,
              }}
            />
          )}

          {/* Operator-editable timeline markers from config/chart_markers.json */}
          {(chartMarkers ?? [])
            .filter((m) => merged.some((p) => p.date >= m.date))
            .map((m) => {
              const matchPoint =
                merged.find((p) => p.date >= m.date) ?? merged[merged.length - 1];
              const color = MARKER_COLORS[m.category ?? ''] ?? MARKER_DEFAULT_COLOR;
              return (
                <ReferenceLine
                  key={`marker-${m.date}-${m.label}`}
                  yAxisId="equity"
                  x={matchPoint.date}
                  stroke={color}
                  strokeDasharray="2 4"
                  strokeOpacity={0.55}
                  label={{
                    value: m.label,
                    position: 'insideTopLeft',
                    fill: color,
                    fontSize: 9,
                    fontWeight: 500,
                  }}
                />
              );
            })}

          {/* Latest value reference — faint horizontal guide */}
          {lastPoint && (
            <ReferenceLine
              yAxisId="equity"
              y={lastPoint.value}
              stroke="#3b82f6"
              strokeDasharray="2 6"
              strokeOpacity={0.3}
            />
          )}

          {/* Equity fill */}
          <Area
            yAxisId="equity"
            type="monotone"
            dataKey="value"
            fill="url(#equityGlow)"
            stroke="none"
          />

          {/* Peak reference */}
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

          {/* Portfolio equity — live-edge dot on last point */}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="value"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={false}
            legendType="none"
            activeDot={{ r: 4, fill: '#3b82f6', stroke: '#0f172a', strokeWidth: 2 }}
          />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown strip */}
      <div className="drawdown-strip">
        <ResponsiveContainer width="100%" height={48}>
          <ComposedChart data={merged} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="drawdownFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal vertical={false} />
            <XAxis dataKey="dateLabel" hide />
            <YAxis
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 9 }}
              tickFormatter={(v) => `${v}%`}
              domain={[ddFloor, 0]}
              axisLine={false}
              tickLine={false}
              width={52}
            />
            <ReferenceLine y={0} stroke="#334155" />
            <Area
              type="monotone"
              dataKey="drawdownPct"
              fill="url(#drawdownFill)"
              stroke="#ef4444"
              strokeWidth={1.2}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="performance-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#3b82f6' }} /> Portfolio
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-dashed" style={{ background: '#64748b' }} /> SPY
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#334155' }} /> Peak
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#ef4444', opacity: 0.5 }} /> Drawdown
        </span>
        <span className="legend-item legend-item-regime">
          <span style={{ color: '#64748b' }}>Bands:</span>
          {(['calm_uptrend', 'risk_on_trend', 'choppy', 'risk_off_trend', 'high_vol_panic'] as const).map((regime) => (
            <span key={regime} className="regime-key-chip">
              <span className="regime-key-dot" style={{ background: REGIME_KEY_DOT_COLORS[regime] }} />
              {REGIME_LABELS[regime]}
            </span>
          ))}
        </span>
      </div>

      {/* Monthly returns heatmap strip */}
      {monthlyReturns && monthlyReturns.length > 0 && (
        <div className="monthly-heatmap-wrap">
          <div className="monthly-heatmap-strip">
            {years.map(year => {
              const ym = monthlyByYear.get(year)!;
              return (
                <div key={year} className="heatmap-year-group">
                  <span className="heatmap-year">{year}</span>
                  <div className="heatmap-cells">
                    {Array.from({ length: 12 }, (_, i) => {
                      const mr = ym.get(i + 1);
                      const hasObs = mr && (mr.observations ?? 1) > 0;
                      const val = hasObs ? mr.return_pct : undefined;
                      return (
                        <div
                          key={i}
                          className="heatmap-cell"
                          title={val !== undefined ? `${MONTH_ABBR[i]} ${year}: ${(val * 100).toFixed(2)}%` : `${MONTH_ABBR[i]} ${year}: —`}
                          style={{
                            backgroundColor: val !== undefined ? heatColor(val) : 'rgba(30, 41, 59, 0.5)',
                          }}
                        >
                          {MONTH_ABBR[i][0]}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
