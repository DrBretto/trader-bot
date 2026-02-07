import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea } from 'recharts';
import { TimeseriesPoint } from '../types';

interface Props {
  timeseries: TimeseriesPoint[];
}

const VOL_REGIME_COLORS: Record<string, string> = {
  calm: 'rgba(34, 197, 94, 0.08)',
  unstable_calm: 'rgba(234, 179, 8, 0.12)',
  panic: 'rgba(239, 68, 68, 0.12)',
};

export function VolComplexPanel({ timeseries }: Props) {
  if (timeseries.length === 0) return null;

  const data = timeseries.map(pt => ({
    date: pt.date.slice(5),
    score: pt.vol_uncertainty_score,
    vix_pct: pt.vix_percentile,
    vvix_pct: pt.vvix_percentile,
    regime: pt.vol_regime_label,
  }));

  // Build shaded reference areas for vol regime
  const regimeAreas: { x1: string; x2: string; fill: string }[] = [];
  let currentRegime = '';
  let areaStart = '';
  for (const pt of data) {
    if (pt.regime !== currentRegime) {
      if (currentRegime && areaStart) {
        regimeAreas.push({ x1: areaStart, x2: pt.date, fill: VOL_REGIME_COLORS[currentRegime] || 'transparent' });
      }
      currentRegime = pt.regime;
      areaStart = pt.date;
    }
  }
  if (currentRegime && areaStart) {
    regimeAreas.push({ x1: areaStart, x2: data[data.length - 1].date, fill: VOL_REGIME_COLORS[currentRegime] || 'transparent' });
  }

  return (
    <div className="card">
      <div className="card-title">Volatility Complex</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#64748b' }} interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          {regimeAreas.map((area, i) => (
            <ReferenceArea key={i} x1={area.x1} x2={area.x2} fill={area.fill} />
          ))}
          <Line type="monotone" dataKey="score" stroke="#f97316" strokeWidth={2} dot={false} name="Vol Score" />
          <Line type="monotone" dataKey="vix_pct" stroke="#ef4444" strokeWidth={1} dot={false} name="VIX %ile" strokeDasharray="4 2" />
          <Line type="monotone" dataKey="vvix_pct" stroke="#eab308" strokeWidth={1} dot={false} name="VVIX %ile" strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
