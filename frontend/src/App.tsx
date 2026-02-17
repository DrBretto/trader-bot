import { format, parseISO } from 'date-fns';
import { useDashboardData } from './hooks/useDashboardData';
import { useTimeseriesData } from './hooks/useTimeseriesData';
import {
  HeroMetrics,
  EquityCurve,
  DrawdownChart,
  MonthlyReturnsHeatmap,
  WeatherReport,
  PortfolioTable,
  CandidatesTable,
  EnsembleStatus,
  RegimeStrip,
  MacroCreditPanel,
  VolComplexPanel,
  FragilityPanel,
  EntropyPanel,
  TradeLog,
  InfoTooltip,
} from './components';

export function App() {
  const { data, loading, error } = useDashboardData();
  const { data: timeseries } = useTimeseriesData();

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
        <div className="last-updated" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>
            Last updated: {format(parseISO(data.metrics.timestamp), 'MMM d, yyyy h:mm a')}
          </span>
          <InfoTooltip
            content={`Snapshot ID: ${data.snapshot?.id ?? data.metrics.snapshot_id ?? 'unknown'}
All dashboard panels are computed from this single snapshot to prevent cross-panel timestamp drift.`}
            label="Snapshot timestamp"
            align="right"
          />
        </div>
      </header>

      <HeroMetrics metrics={data.metrics} holdings={data.holdings} equityCurve={data.equity_curve} />

      <div className="charts-row">
        <EquityCurve data={data.equity_curve} />
        <DrawdownChart data={data.drawdowns} />
      </div>

      {/* Market Intelligence Section */}
      <RegimeStrip signals={data.expert_signals} timeseries={timeseries} />

      {timeseries.length > 0 && (
        <div className="expert-panels-grid">
          <MacroCreditPanel timeseries={timeseries} />
          <VolComplexPanel timeseries={timeseries} />
          <FragilityPanel timeseries={timeseries} />
          <EntropyPanel timeseries={timeseries} />
        </div>
      )}

      <div style={{ marginBottom: '24px' }}>
        <MonthlyReturnsHeatmap data={data.monthly_returns} />
      </div>

      <div className="weather-regime-row">
        <WeatherReport weather={data.weather} />
        <div className="card">
          <div className="card-title">
            <span>Regime Probabilities</span>
            <InfoTooltip
              content={`Model probability distribution across regime classes before final expert-rule fusion.
The highest raw probability can still lose if a higher-priority override rule fires.`}
              label="Regime probabilities"
            />
          </div>
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
              title={`${regime.replace(/_/g, ' ')} probability: ${(prob * 100).toFixed(1)}%`}
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
        <EnsembleStatus ensemble={data.weather.regime.ensemble} signals={data.expert_signals} />
      </div>

      <div className="tables-row">
        <PortfolioTable holdings={data.holdings} />
        <CandidatesTable candidates={data.candidates} />
      </div>

      <TradeLog
        trades={data.trades ?? []}
        cumulativeCosts={data.metrics.cumulative_transaction_costs}
        tradeSummary={data.trade_summary}
      />
    </div>
  );
}
