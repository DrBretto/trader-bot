import { useEffect } from 'react';
import { format, parseISO } from 'date-fns';
import { useDashboardData } from './hooks/useDashboardData';
import { useTimeseriesData } from './hooks/useTimeseriesData';
import { isStrictlyIncreasingByDate } from './utils/timeseries';
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

function formatRegimeLabel(regime: string): string {
  return regime.replace(/_/g, ' ');
}

function getArgmaxRegime(probs: Record<string, number>): string | null {
  const entries = Object.entries(probs);
  if (entries.length === 0) return null;
  return entries.reduce((best, current) => (current[1] > best[1] ? current : best))[0];
}

function computeCompoundedYtd(data: { year: number; return_pct: number; observations?: number }[], year: number): number | null {
  const months = data.filter((row) => row.year === year && (row.observations ?? 1) > 0);
  if (months.length === 0) return null;
  return months.reduce((acc, row) => acc * (1 + row.return_pct), 1) - 1;
}

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

  const selectedRegime = data.expert_signals?.final_regime_label ?? data.weather.regime.regime;
  const argmaxRegime = getArgmaxRegime(data.weather.regime.probs);
  const probabilityRows = Object.entries(data.weather.regime.probs)
    .sort((a, b) => b[1] - a[1]);

  useEffect(() => {
    if (!import.meta.env.DEV) return;

    const warnings: string[] = [];
    const overrideReason = data.expert_signals?.override_reason;
    if (argmaxRegime && argmaxRegime !== selectedRegime && !overrideReason) {
      warnings.push(
        `Argmax regime (${argmaxRegime}) differs from selected regime (${selectedRegime}) but no override reason is present.`,
      );
    }

    const zeroObsZeroReturn = data.monthly_returns.filter(
      (row) => (row.observations ?? 1) === 0 && Math.abs(row.return_pct) < 1e-12,
    );
    if (zeroObsZeroReturn.length > 0) {
      warnings.push(
        `Monthly returns include ${zeroObsZeroReturn.length} month(s) with 0.0% and zero observations.`,
      );
    }

    if (!isStrictlyIncreasingByDate(data.drawdowns)) {
      warnings.push('Drawdown series timestamps are not strictly increasing.');
    }

    const snapshotYear = parseISO(data.metrics.timestamp).getFullYear();
    const monthlyYtd = computeCompoundedYtd(data.monthly_returns, snapshotYear);
    if (monthlyYtd != null && Math.abs(monthlyYtd - data.metrics.ytd_return) > 1e-6) {
      warnings.push(
        `YTD mismatch detected (header=${data.metrics.ytd_return.toFixed(6)}, monthly=${monthlyYtd.toFixed(6)}).`,
      );
    }

    warnings.forEach((warning) => {
      // eslint-disable-next-line no-console
      console.warn(`[dashboard-coherence] ${warning}`);
    });
  }, [argmaxRegime, data, selectedRegime]);

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
              content={`Probabilities are raw model outputs before fusion rules.
The highlighted bar is the most likely model regime (argmax probability).
The "Selected" tag marks the post-fusion regime used by the decision engine; it may differ due to overrides/gates.`}
              label="Regime probabilities"
            />
          </div>
          {probabilityRows.map(([regime, prob]) => (
            <div
              key={regime}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 0',
                borderBottom: '1px solid #334155',
              }}
              title={`${formatRegimeLabel(regime)} probability: ${(prob * 100).toFixed(1)}%`}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ textTransform: 'capitalize' }}>{formatRegimeLabel(regime)}</span>
                {regime === selectedRegime && (
                  <span
                    style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      color: '#eab308',
                      border: '1px solid rgba(234, 179, 8, 0.45)',
                      borderRadius: '999px',
                      padding: '1px 6px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.4px',
                    }}
                    title="Final post-fusion regime selected by override/gating rules."
                  >
                    Selected
                  </span>
                )}
              </div>
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
                        regime === argmaxRegime ? '#3b82f6' : '#64748b',
                    }}
                  />
                </div>
                <span style={{ width: '45px', textAlign: 'right' }}>
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
          {argmaxRegime !== selectedRegime && (
            <div style={{ marginTop: 10, fontSize: 11, color: '#94a3b8' }}>
              Most likely model regime differs from selected regime due to fusion overrides.
            </div>
          )}
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
