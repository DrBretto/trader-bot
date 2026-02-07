import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TimeseriesPoint } from '../types';

interface Props {
  timeseries: TimeseriesPoint[];
}

export function EntropyPanel({ timeseries }: Props) {
  if (timeseries.length === 0) return null;

  const data = timeseries.map(pt => ({
    date: pt.date.slice(5),
    score: pt.entropy_score,
    z_score: pt.entropy_z_score,
    shift: pt.entropy_shift_flag ? pt.entropy_score : null,
  }));

  return (
    <div className="card">
      <div className="card-title">Entropy / Distribution Shift</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#64748b' }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="score" stroke="#06b6d4" strokeWidth={2} dot={false} name="Entropy" />
          <Line type="monotone" dataKey="z_score" stroke="#94a3b8" strokeWidth={1} dot={false} name="Z-Score" strokeDasharray="4 2" />
          <Line type="monotone" dataKey="shift" stroke="#ef4444" strokeWidth={0} dot={{ r: 4, fill: '#ef4444' }} name="Shift Event" connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
