import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { DrawdownPoint } from '../types';
import { format, parseISO } from 'date-fns';

interface Props {
  data: DrawdownPoint[];
}

export function DrawdownChart({ data }: Props) {
  const formattedData = data.map((point) => ({
    ...point,
    dateLabel: format(parseISO(point.date), 'MMM yyyy'),
    drawdownPct: point.drawdown * 100,
  }));

  return (
    <div className="card">
      <div className="card-title">Drawdown</div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={formattedData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="dateLabel"
            stroke="#64748b"
            tick={{ fill: '#64748b', fontSize: 12 }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#64748b"
            tick={{ fill: '#64748b', fontSize: 12 }}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            domain={['dataMin', 0]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value: number) => [`${value.toFixed(2)}%`, 'Drawdown']}
          />
          <Area
            type="monotone"
            dataKey="drawdownPct"
            stroke="#ef4444"
            fill="rgba(239, 68, 68, 0.3)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
