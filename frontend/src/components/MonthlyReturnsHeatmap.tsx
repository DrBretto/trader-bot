import { MonthlyReturn } from '../types';

interface Props {
  data: MonthlyReturn[];
}

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function getColor(value: number): string {
  if (value >= 0.05) return 'rgba(34, 197, 94, 0.9)';
  if (value >= 0.02) return 'rgba(34, 197, 94, 0.6)';
  if (value >= 0) return 'rgba(34, 197, 94, 0.3)';
  if (value >= -0.02) return 'rgba(239, 68, 68, 0.3)';
  if (value >= -0.05) return 'rgba(239, 68, 68, 0.6)';
  return 'rgba(239, 68, 68, 0.9)';
}

export function MonthlyReturnsHeatmap({ data }: Props) {
  const years = [...new Set(data.map((d) => d.year))].sort();

  const dataMap = new Map<string, number>();
  data.forEach((d) => {
    dataMap.set(`${d.year}-${d.month}`, d.return_pct);
  });

  return (
    <div className="card">
      <div className="card-title">Monthly Returns</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ minWidth: '100%' }}>
          <thead>
            <tr>
              <th style={{ width: '60px' }}>Year</th>
              {MONTH_LABELS.map((month) => (
                <th key={month} style={{ textAlign: 'center', padding: '8px 4px' }}>
                  {month}
                </th>
              ))}
              <th style={{ textAlign: 'right' }}>YTD</th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const yearData = data.filter((d) => d.year === year);
              const ytd = yearData.reduce((sum, d) => sum + d.return_pct, 0);

              return (
                <tr key={year}>
                  <td style={{ fontWeight: 600 }}>{year}</td>
                  {Array.from({ length: 12 }, (_, i) => {
                    const value = dataMap.get(`${year}-${i + 1}`);
                    return (
                      <td
                        key={i}
                        style={{
                          textAlign: 'center',
                          padding: '8px 4px',
                          backgroundColor: value !== undefined ? getColor(value) : 'transparent',
                          borderRadius: '4px',
                        }}
                      >
                        {value !== undefined ? `${(value * 100).toFixed(1)}%` : '-'}
                      </td>
                    );
                  })}
                  <td
                    style={{
                      textAlign: 'right',
                      fontWeight: 600,
                      color: ytd >= 0 ? '#22c55e' : '#ef4444',
                    }}
                  >
                    {(ytd * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
