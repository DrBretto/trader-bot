import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TimeseriesPoint } from '../types';

interface Props {
  timeseries: TimeseriesPoint[];
}

export function MacroCreditPanel({ timeseries }: Props) {
  if (timeseries.length === 0) return null;

  const data = timeseries.map(pt => ({
    date: pt.date.slice(5), // MM-DD
    score: pt.macro_credit_score,
    slope: pt.yield_slope_10y_3m,
    spread: pt.hy_spread_proxy * 100, // scale for visibility
  }));

  return (
    <div className="card">
      <div className="card-title">Macro / Credit</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#64748b' }} interval="preserveStartEnd" />
          <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} dot={false} name="Macro Score" />
          <Line type="monotone" dataKey="slope" stroke="#22c55e" strokeWidth={1} dot={false} name="Yield Slope %" strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
