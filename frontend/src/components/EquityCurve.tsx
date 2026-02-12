import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { EquityCurvePoint } from '../types';
import { format, parseISO } from 'date-fns';

interface Props {
  data: EquityCurvePoint[];
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function EquityCurve({ data }: Props) {
  const formattedData = data.map((point) => ({
    ...point,
    dateLabel: format(parseISO(point.date), 'MMM yyyy'),
  }));

  return (
    <div className="card">
      <div className="card-title">Portfolio Value</div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formattedData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
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
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value: number) => [formatCurrency(value), '']}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            name="Portfolio"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            name="SPY Total Return"
            stroke="#64748b"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
