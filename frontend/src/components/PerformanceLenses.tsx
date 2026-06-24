import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import { EquityCurvePoint } from '../types';
import { ShadowTimeseries } from '../hooks/useShadowData';
import { InfoTooltip } from './InfoTooltip';

interface Props {
  equityCurve: EquityCurvePoint[];
  shadow?: ShadowTimeseries | null;
}

function formatPct(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
}

function getCurrentModelValue(point: EquityCurvePoint): number {
  return point.optimized_value ?? point.corrected_value ?? point.value;
}

function getPreviousModelValue(point: EquityCurvePoint): number | null {
  return point.hybrid_value ?? point.pre_hybrid_value ?? null;
}

function buildModelComparisonData(equityCurve: EquityCurvePoint[]) {
  const first = equityCurve.find((point) => {
    const current = getCurrentModelValue(point);
    const previous = getPreviousModelValue(point);
    return current > 0 && previous != null && previous > 0;
  });
  if (!first) return [];

  const firstCurrent = getCurrentModelValue(first);
  const firstPrevious = getPreviousModelValue(first);
  if (!firstPrevious) return [];

  return equityCurve
    .map((point) => {
      const current = getCurrentModelValue(point);
      const previous = getPreviousModelValue(point);
      if (current <= 0 || previous == null || previous <= 0) return null;

      const currentModelIndex = (current / firstCurrent) * 100;
      const previousModelIndex = (previous / firstPrevious) * 100;

      return {
        date: point.date,
        dateLabel: format(parseISO(point.date), 'MMM d'),
        currentModelIndex,
        previousModelIndex,
        modelDeltaPct: currentModelIndex - previousModelIndex,
      };
    })
    .filter((point): point is NonNullable<typeof point> => point !== null);
}

// Two-stage (live, canon) vs the tilt — the two models the operator watches.
// The live two-stage book is the canon equity value; the tilt (deterministic
// rules + small M1 nudge) is shadow_A. Both rebased to 100 at the first common
// settled date so their relative paths are directly comparable.
function buildShadowComparisonData(
  equityCurve: EquityCurvePoint[],
  shadow: ShadowTimeseries | null | undefined,
) {
  const tiltPts = shadow?.shadow_A ?? [];
  if (tiltPts.length === 0) return [];
  const tiltByDate = new Map(tiltPts);

  const rows = equityCurve.filter((point) => tiltByDate.has(point.date));
  const first = rows.find((point) => {
    const twoStage = getCurrentModelValue(point);
    const tv = tiltByDate.get(point.date);
    return twoStage > 0 && tv != null && tv > 0;
  });
  if (!first) return [];

  const firstTwoStage = getCurrentModelValue(first);
  const firstTilt = tiltByDate.get(first.date)!;

  return rows
    .map((point) => {
      const twoStage = getCurrentModelValue(point);
      const tv = tiltByDate.get(point.date);
      if (twoStage <= 0 || tv == null || tv <= 0) return null;

      const currentModelIndex = (twoStage / firstTwoStage) * 100;  // two-stage (live)
      const tiltIndex = (tv / firstTilt) * 100;                     // tilt (comparison)

      return {
        date: point.date,
        dateLabel: format(parseISO(point.date), 'MMM d'),
        currentModelIndex,
        previousModelIndex: tiltIndex,
        modelDeltaPct: currentModelIndex - tiltIndex,
      };
    })
    .filter((point): point is NonNullable<typeof point> => point !== null);
}

function buildExcessSpreadData(equityCurve: EquityCurvePoint[]) {
  const first = equityCurve.find((point) => point.value > 0 && point.benchmark > 0);
  if (!first) return [];

  return equityCurve.map((point) => {
    const portfolioReturn = point.value / first.value - 1;
    const benchmarkReturn = point.benchmark / first.benchmark - 1;
    return {
      date: point.date,
      dateLabel: format(parseISO(point.date), 'MMM d'),
      excessSpreadPct: (portfolioReturn - benchmarkReturn) * 100,
    };
  });
}

export function PerformanceLenses({ equityCurve, shadow }: Props) {
  const shadowComparison = buildShadowComparisonData(equityCurve, shadow);
  const usingShadowLens = shadowComparison.length > 0;
  // Shadow JSON present but no settled paper points yet: show the NEW lens
  // explicitly in its armed state rather than silently falling back — the
  // operator should see the panel has changed the day the shadow arms.
  const shadowArmed = !!shadow && !usingShadowLens;
  const modelComparison = usingShadowLens
    ? shadowComparison
    : buildModelComparisonData(equityCurve);
  const excessSpread = buildExcessSpreadData(equityCurve);

  if ((modelComparison.length === 0 && !shadowArmed) || excessSpread.length === 0) {
    return null;
  }

  const lensTitle = usingShadowLens || shadowArmed ? 'Two-stage (live) vs Tilt' : 'Current Model vs Previous';
  const lensSubtitle = usingShadowLens || shadowArmed
    ? 'Rebased to 100 at the boundary — the live two-stage engine vs the tilt (rules + small M1 nudge).'
    : 'Rebased to 100 so the active canon and prior model are directly comparable.';
  const lensNewLabel = usingShadowLens ? 'Tilt' : 'Previous Model';
  const lensDeltaLabel = usingShadowLens ? 'Two-stage − Tilt' : 'Current - Previous';

  return (
    <div className="performance-lenses-section">
      <div className="card-title performance-lenses-title">
        <span>Performance Lenses</span>
        <InfoTooltip
          content="Two quick alternative views of the same portfolio history. Left: current model and previous model rebased to the same starting level. Right: the strategy's running excess return over SPY in percentage points."
          label="Performance lenses"
        />
      </div>

      <div className="performance-lenses-grid">
        <div className="card performance-lens-card">
          <div className="performance-lens-card__header">
            <div>
              <h3>{lensTitle}</h3>
              <p>{lensSubtitle}</p>
            </div>
          </div>
          {shadowArmed ? (
            <div
              style={{
                height: 220,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                color: '#94a3b8',
                fontSize: 13,
              }}
            >
              <div style={{ color: '#f59e0b', fontSize: 15, fontWeight: 600 }}>
                Shadow armed — accruing
              </div>
              <div style={{ textAlign: 'center', maxWidth: 360, lineHeight: 1.5 }}>
                The new brain's paper book starts tonight. Both lines appear here,
                rebased to 100 at the same start, as soon as the first day settles.
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Verdict reads pre-registered: 2027-01-27 (forecast skill) · 2027-08-10 (P&amp;L)
              </div>
            </div>
          ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={modelComparison} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
              <XAxis
                dataKey="dateLabel"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 11 }}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 11 }}
                domain={['auto', 'auto']}
                tickFormatter={(value) => value.toFixed(0)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid rgba(59, 130, 246, 0.25)',
                  borderRadius: '10px',
                }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number, name: string) => [
                  name === 'modelDeltaPct' ? formatPct(value) : value.toFixed(2),
                  name === 'currentModelIndex'
                    ? (usingShadowLens ? 'Two-stage (live)' : 'Current Model')
                    : name === 'previousModelIndex'
                      ? lensNewLabel
                      : lensDeltaLabel,
                ]}
              />
              <ReferenceLine y={100} stroke="rgba(148, 163, 184, 0.35)" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="currentModelIndex"
                stroke={usingShadowLens ? '#f59e0b' : '#34d399'}
                strokeWidth={2.5}
                dot={false}
                name="currentModelIndex"
              />
              <Line
                type="monotone"
                dataKey="previousModelIndex"
                stroke="#60a5fa"
                strokeWidth={1.75}
                strokeDasharray="3 4"
                dot={false}
                name="previousModelIndex"
              />
            </LineChart>
          </ResponsiveContainer>
          )}
        </div>

        <div className="card performance-lens-card">
          <div className="performance-lens-card__header">
            <div>
              <h3>Excess Return Spread</h3>
              <p>Positive means the strategy is ahead of SPY from the same start date.</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={excessSpread} margin={{ top: 8, right: 12, bottom: 8, left: -4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
              <XAxis
                dataKey="dateLabel"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 11 }}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 11 }}
                domain={['auto', 'auto']}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid rgba(16, 185, 129, 0.22)',
                  borderRadius: '10px',
                }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number) => [formatPct(value), 'Excess vs SPY']}
              />
              <ReferenceLine y={0} stroke="rgba(148, 163, 184, 0.35)" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="excessSpreadPct"
                stroke="#34d399"
                strokeWidth={2.5}
                dot={false}
                name="excessSpreadPct"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
