import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TimeseriesPoint } from '../types';

interface Props {
  timeseries: TimeseriesPoint[];
}

export function FragilityPanel({ timeseries }: Props) {
  if (timeseries.length === 0) return null;

  const data = timeseries.map(pt => ({
    date: pt.date.slice(5),
    score: pt.fragility_score,
    correlation: pt.avg_correlation,
    pc1: pt.pc1_explained,
  }));

  return (
    <div className="card">
      <div className="card-title">Cross-Asset Fragility</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#64748b' }} interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <ReferenceLine y={0.75} stroke="#ef4444" strokeDasharray="5 3" label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10, position: 'right' }} />
          <Line type="monotone" dataKey="score" stroke="#a855f7" strokeWidth={2} dot={false} name="Fragility" />
          <Line type="monotone" dataKey="correlation" stroke="#64748b" strokeWidth={1} dot={false} name="Avg Corr" strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
