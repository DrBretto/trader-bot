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
import { InfoTooltip } from './InfoTooltip';
import { insertGapBreakPoints, sortAndDedupeByDate } from '../utils/timeseries';

interface Props {
  data: DrawdownPoint[];
}

interface DrawdownChartPoint {
  date: string;
  drawdownPct?: number;
}

export function DrawdownChart({ data }: Props) {
  const normalized: DrawdownChartPoint[] = sortAndDedupeByDate(data).map((point) => ({
    date: point.date,
    drawdownPct: point.drawdown * 100,
  }));
  const formattedData = insertGapBreakPoints(normalized, 7, (date) => ({ date }));

  return (
    <div className="card">
      <div className="card-title">
        <span>Drawdown</span>
        <InfoTooltip
          content={`Drawdown is the percent decline from the prior equity peak.
0% means at a new high; more negative values mean deeper underwater periods.
Max drawdown is the minimum value on this series.`}
          label="Drawdown chart"
        />
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={formattedData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="date"
            stroke="#64748b"
            tick={{ fill: '#64748b', fontSize: 12 }}
            interval="preserveStartEnd"
            tickFormatter={(value: string) => format(parseISO(value), 'MMM d')}
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
            formatter={(value: number | string) => {
              const numeric = typeof value === 'number' ? value : Number(value);
              return [Number.isFinite(numeric) ? `${numeric.toFixed(2)}%` : '\u2014', 'Drawdown'];
            }}
            labelFormatter={(value: string) => format(parseISO(value), 'MMM d, yyyy')}
          />
          <Area
            type="monotone"
            dataKey="drawdownPct"
            stroke="#ef4444"
            fill="rgba(239, 68, 68, 0.3)"
            connectNulls={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
