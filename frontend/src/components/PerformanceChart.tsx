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
import { ShadowTimeseries } from '../hooks/useShadowData';
import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  equityData: EquityCurvePoint[];
  drawdownData: DrawdownPoint[];
  monthlyReturns?: MonthlyReturn[];
  timeseries?: TimeseriesPoint[];
  chartMarkers?: ChartMarker[];
  shadow?: ShadowTimeseries | null;
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
  correctedValue: number;
  actualValue: number;
  preHybridValue: number | null;
  optimizedValue: number | null;
  hybridValue: number | null;
  championFrozen: number | null;
  newBrain: number | null;
  incumbent: number | null;
  shadowA: number | null;
  valuePre: number | null;
  newModel: number | null;
  championForward: number | null;
  benchmark: number;
  drawdownPct: number;
  peak: number;
  regimeLabel?: string | null;
}

// PKT-TB-012: the champion line is frozen byte-immutable through this date; the
// New Brain (native two-stage engine) is the primary line forward of it,
// re-anchored C0-continuous to the frozen terminal.
const NEW_BRAIN_BOUNDARY = '2026-06-11';

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
      <div style={{ color: '#94a3b8', marginBottom: 4, fontWeight: 600 }}>
        <span>{dateStr}</span>
      </div>
      <div style={{ color: point.newModel != null ? '#f59e0b' : '#60a5fa' }}>
        {point.newModel != null ? 'New model' : 'Portfolio'}: <span style={{ fontWeight: 600 }}>{formatCurrency(point.newModel ?? point.valuePre ?? point.value)}</span>
      </div>
      {point.championForward !== null && point.championForward !== undefined && (
        <div style={{ color: '#60a5fa', opacity: 0.7 }}>
          Retired champion: <span style={{ fontWeight: 500 }}>{formatCurrency(point.championForward)}</span>
        </div>
      )}
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

const HYBRID_PROMOTION_DATE = '2026-03-28';

// Pre-registered verdict read dates from SHADOW_PREREG.md (the JSON's
// prereg_pointer references that doc, not the dates themselves):
// IC read ≈ 2027-01-27, final utility read ≈ 2027-08-10.
const SHADOW_READ_DATES = '2027-01-27 / 2027-08-10';

function formatShadowStat(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits);
}

export function PerformanceChart({ equityData, drawdownData, monthlyReturns, timeseries = [], chartMarkers, shadow }: Props) {
  // Build drawdown lookup
  const ddMap = new Map(drawdownData.map((d) => [d.date, d.drawdown]));
  const regimeByDate = new Map(timeseries.map((pt) => [pt.date, pt.final_regime_label]));

  // Dual forward shadow (paper books, PKT-TB-007 follow-on). Absent or
  // armed-but-empty payloads render nothing new.
  // The new model (tilt_adapter) = shadow_A, the canon line forward of the
  // boundary. The retired champion (incumbent, no tilt) = shadow_I, the dotted
  // comparison line. shadow_B (the +event-damp variant) is no longer drawn.
  const shadowAByDate = new Map(shadow?.shadow_A ?? []);
  const shadowIByDate = new Map(shadow?.shadow_I ?? []);
  const hasShadowA = shadowAByDate.size > 0;
  const hasChampionForward = shadowIByDate.size > 0;

  // Trim leading flat zone
  const startVal = equityData[0]?.value ?? 0;
  const firstMoveIdx = equityData.findIndex(p => Math.abs(p.value - startVal) / (startVal || 1) > 0.001);
  const trimmedEquity = firstMoveIdx > 1 ? equityData.slice(firstMoveIdx - 1) : equityData;

  // Merge and compute running peak.
  // 2026-05-16 canon promotion: the OPTIMIZED champion line is now the bold
  // primary, rendered SOLID GREEN. The hybrid configuration's counterfactual
  // is preserved in `hybrid_value` and rendered as a DOTTED BLUE comparison.
  // Pre-promotion dates (or dates without optimized_value) fall back to
  // corrected_value / value so the line stays continuous across the canon
  // boundary.
  let peak = 0;
  const allMerged: MergedPoint[] = trimmedEquity.map((point) => {
    const optimizedValue = point.optimized_value ?? null;
    const hybridValue = point.hybrid_value ?? point.corrected_value ?? point.value;
    const correctedValue = point.corrected_value ?? point.value;
    const actualValue = point.actual_value ?? point.value;
    const preHybridValue = point.pre_hybrid_value ?? null;
    // Primary canonical line is optimized when available, else fall back to
    // the corrected/hybrid value so historical periods (pre-2026-05-06) and
    // any rows missing optimized_value stay connected.
    const primaryValue = optimizedValue ?? correctedValue;
    // PKT-TB-012 fields: the frozen champion line (ends at the boundary) and the
    // re-anchored New Brain forward line.
    const championFrozen = (point as { champion_frozen_value?: number | null }).champion_frozen_value ?? null;
    const newBrain = (point as { new_brain_value?: number | null }).new_brain_value ?? null;
    const incumbent = (point as { incumbent_value?: number | null }).incumbent_value ?? null;
    peak = Math.max(peak, primaryValue);
    return {
      date: point.date,
      value: primaryValue,
      correctedValue,
      actualValue,
      preHybridValue,
      optimizedValue,
      hybridValue,
      championFrozen,
      newBrain,
      incumbent,
      // Canon line split at the boundary: solid BLUE = the real champion history
      // through 06-11; solid YELLOW = the new model (tilt_adapter = shadow_A) from
      // 06-11 forward. championForward (dotted BLUE) = the retired champion run
      // forward (shadow_I = incumbent, no tilt) — the comparison line.
      valuePre: point.date <= NEW_BRAIN_BOUNDARY ? primaryValue : null,
      newModel: point.date >= NEW_BRAIN_BOUNDARY ? (shadowAByDate.get(point.date) ?? null) : null,
      championForward: point.date >= NEW_BRAIN_BOUNDARY ? (shadowIByDate.get(point.date) ?? null) : null,
      shadowA: shadowAByDate.get(point.date) ?? null,
      benchmark: point.benchmark,
      drawdownPct: (ddMap.get(point.date) ?? 0) * 100,
      peak,
      regimeLabel: regimeByDate.get(point.date) ?? null,
    };
  });

  const merged = allMerged;

  // PKT-TB-012: "New Brain" attaches ONLY to dates the two-stage engine actually
  // drove (the extender sets new_brain_value only for engine-traded days). Before
  // that, the forward line is the CURRENT (incumbent) algorithm — never branded
  // the rebuild (Attack-5 ruling).
  const hasNewBrain = merged.some((p) => p.newBrain !== null);
  const hasIncumbentForward = merged.some((p) => p.date > NEW_BRAIN_BOUNDARY && p.incumbent !== null);
  const hasChampionFrozen = merged.some((p) => p.championFrozen !== null);
  const newBrainGoLiveDate = merged.find((p) => p.newBrain !== null)?.date;

  // Standing exposure-stripped forecast-rung rent (F-R) + CI, read from the
  // PKT-TB-011 rent ladder the shadow publishes. Carried onto the surface so a
  // viewer sees the brain's selection contribution (a zero-straddling band), not
  // just a market-driven equity curve (Skeptic closing condition 1).
  const organLedger = shadow?.organ_ledger ?? shadow?.stats?.organ_ledger ?? [];
  const forecastRung = organLedger.find((r) => r.component === 'forecast');

  const hybridIdx = merged.findIndex((p) => p.date >= HYBRID_PROMOTION_DATE);
  const hybridDate = hybridIdx >= 0 ? merged[hybridIdx]?.date : undefined;

  // Compute Y domains
  const allValues = merged.flatMap((p) => [p.value, p.benchmark]);
  const yMin = Math.floor(Math.min(...allValues) / 1000) * 1000;
  const yMax = Math.ceil(Math.max(...allValues) / 1000) * 1000;

  const ddMin = Math.min(...merged.map((p) => p.drawdownPct));
  const ddFloor = Math.floor(ddMin / 5) * 5;

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
Background color bands show the detected market regime at each point in time.`}
            label="Performance chart"
          />
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

          {/* Equity fill (under the SOLID PRIMARY optimized line) */}
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

          {/* CANON, part 1 — solid BLUE through the boundary: the real champion
              history up to 2026-06-11 (where the new model was placed). */}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="valuePre"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={false}
            legendType="none"
            connectNulls={false}
          />

          {/* CANON, part 2 — solid YELLOW from the boundary forward: the NEW
              MODEL (tilt_adapter = deterministic rules + small M1 tilt = shadow_A).
              Continuous with the blue history (both meet at the frozen terminal
              on 2026-06-11). The retired two-stage drift line is gone. */}
          {hasShadowA && (
            <Line
              yAxisId="equity"
              type="monotone"
              dataKey="newModel"
              stroke="#f59e0b"
              strokeWidth={2.5}
              dot={false}
              legendType="none"
              connectNulls
              activeDot={{ r: 4, fill: '#f59e0b', stroke: '#0f172a', strokeWidth: 2 }}
            />
          )}

          {/* Retired champion run forward (dotted BLUE) = shadow_I, the
              deterministic rules with NO ML tilt, from the boundary forward.
              Rendered ON TOP of the new model so the dots are visible — the two
              lines overlap to within ~0.03% (the ML tilt adds ~nothing), so this
              shows the dotted champion riding on the yellow new-model line. */}
          {hasChampionForward && (
            <Line
              yAxisId="equity"
              type="monotone"
              dataKey="championForward"
              stroke="#93c5fd"
              strokeWidth={1.5}
              strokeDasharray="2 5"
              dot={false}
              legendType="none"
              connectNulls
            />
          )}
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
          <span className="legend-swatch" style={{ background: '#3b82f6' }} /> Portfolio (champion, through Jun 11)
        </span>
        {hasShadowA && (
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: '#f59e0b' }} /> New model (live, from Jun 11)
          </span>
        )}
        {hasChampionForward && (
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-dashed" style={{ background: '#3b82f6', opacity: 0.6 }} /> Retired champion (comparison)
          </span>
        )}
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-dashed" style={{ background: '#64748b' }} /> SPY
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: '#ef4444', opacity: 0.5 }} /> Drawdown
        </span>
        {shadow && !hasShadowA && (
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: '#f59e0b', opacity: 0.45 }} /> New model: armed — accruing
          </span>
        )}
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

      {/* PKT-TB-012 New Brain surface note. The live primary line CARRIES the
          standing exposure-stripped forecast-rung rent + CI so the line does not
          read as a market-driven equity curve (Skeptic closing condition 1).
          No surface implies a dollar edge: the stripped residual is ~zero. */}
      {(hasNewBrain || hasIncumbentForward || hasChampionFrozen) && (
        <div className="shadow-note">
          {hasNewBrain ? (
            <>
              <strong style={{ color: '#93c5fd' }}>New Brain</strong> (native two-stage engine) — live from {newBrainGoLiveDate}, re-anchored C0-continuous to the frozen champion ($114.9k, Jun 11).
              {' '}
              {forecastRung
                ? <>Forecast-rung rent (exposure-stripped): <strong>{formatShadowStat(forecastRung.stripped_bp_day)}</strong> bp/day
                    {forecastRung.stripped_ci ? ` (95% CI [${forecastRung.stripped_ci.map((v) => v.toFixed(2)).join(', ')}])` : ''}
                    {forecastRung.verdict ? ` — ${forecastRung.verdict}` : ''}.</>
                : <>Forecast-rung rent: accruing (armed).</>}
              {' '}<span style={{ color: '#64748b' }}>Selection edge currently measured at zero; this line measures dollar conversion <em>forward</em> (LIVE_PREREG). Go-live universe <strong>forward_confirmed: false</strong>.</span>
            </>
          ) : (
            <span style={{ color: '#94a3b8' }}>
              <strong style={{ color: '#fbbf24' }}>New Brain not yet live.</strong> The champion line is frozen through Jun 11; the line after it is the <strong>current (incumbent) algorithm</strong>, not the rebuild. The two-stage engine falls back to the incumbent until it writes its first live trades — the New Brain brand attaches only from that day.
            </span>
          )}
        </div>
      )}

      {/* Dual forward shadow accrual note — neutral status only, no verdict
          language before the pre-registered read dates. */}
      {shadow && (
        <div className="shadow-note">
          {hasShadowA ? (
            <>
              New model (live): {shadow.stats.days_accrued} day{shadow.stats.days_accrued === 1 ? '' : 's'} accrued
              {' · '}mean IC {formatShadowStat(shadow.stats.mean_ic, 3)} (t={formatShadowStat(shadow.stats.ic_t, 2)})
              {' · '}utility diff {formatShadowStat(shadow.stats.utility_diff_bp_day)} bp/day
              {shadow.stats.utility_diff_bp_day_ci ?? shadow.stats.ci
                ? ` (95% CI [${(shadow.stats.utility_diff_bp_day_ci ?? shadow.stats.ci)!.map((v) => v.toFixed(2)).join(', ')}])`
                : ''}
              {' — '}accruing; verdict reads pre-registered for {SHADOW_READ_DATES}.
            </>
          ) : (
            <>Shadow (paper): armed — accruing. Verdict reads pre-registered for {SHADOW_READ_DATES}.</>
          )}
        </div>
      )}

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
