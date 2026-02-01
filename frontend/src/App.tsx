import { format, parseISO } from 'date-fns';
import { useDashboardData } from './hooks/useDashboardData';
import {
  HeroMetrics,
  EquityCurve,
  DrawdownChart,
  MonthlyReturnsHeatmap,
  WeatherReport,
  PortfolioTable,
  CandidatesTable,
  EnsembleStatus,
} from './components';

export function App() {
  const { data, loading, error } = useDashboardData();

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  if (error) {
    return (
      <div className="error">
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>:(</div>
        <div>Failed to load dashboard data</div>
        <div style={{ fontSize: '14px', marginTop: '8px' }}>{error}</div>
      </div>
    );
  }

  if (!data) {
    return <div className="error">No data available</div>;
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Investment Dashboard</h1>
        <span className="last-updated">
          Last updated: {format(parseISO(data.metrics.timestamp), 'MMM d, yyyy h:mm a')}
        </span>
      </header>

      <HeroMetrics metrics={data.metrics} />

      <div className="charts-row">
        <EquityCurve data={data.equity_curve} />
        <DrawdownChart data={data.drawdowns} />
      </div>

      <MonthlyReturnsHeatmap data={data.monthly_returns} />

      <div className="weather-regime-row">
        <WeatherReport weather={data.weather} />
        <div className="card">
          <div className="card-title">Regime Probabilities</div>
          {Object.entries(data.weather.regime.probs).map(([regime, prob]) => (
            <div
              key={regime}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 0',
                borderBottom: '1px solid #334155',
              }}
            >
              <span style={{ textTransform: 'capitalize' }}>{regime.replace(/_/g, ' ')}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div
                  style={{
                    width: '100px',
                    height: '8px',
                    backgroundColor: '#334155',
                    borderRadius: '4px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${prob * 100}%`,
                      height: '100%',
                      backgroundColor:
                        regime === data.weather.regime.regime ? '#3b82f6' : '#64748b',
                    }}
                  />
                </div>
                <span style={{ width: '45px', textAlign: 'right' }}>
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
        <EnsembleStatus ensemble={data.weather.regime.ensemble} />
      </div>

      <div className="tables-row">
        <PortfolioTable holdings={data.holdings} />
        <CandidatesTable candidates={data.candidates} />
      </div>
    </div>
  );
}
