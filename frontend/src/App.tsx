import { useEffect, useState } from 'react';
import { format, parseISO } from 'date-fns';
import { useDashboardData } from './hooks/useDashboardData';
import { useTimeseriesData } from './hooks/useTimeseriesData';
import { useShadowData } from './hooks/useShadowData';
import { useOptimizerData } from './hooks/useOptimizerData';
import { useIsMobile } from './hooks/useIsMobile';
import { isStrictlyIncreasingByDate } from './utils/timeseries';
import { MobileDashboard } from './components/mobile';
import {
  PortfolioTable,
  CandidatesTable,
  TradeLog,
  InfoTooltip,
  OptimizerStatus,
  OptimizerRunDetail,
  SystemStatusBar,
  TodaysStoryCard,
  SystemBrainPanel,
  PerformanceLenses,
} from './components';
import { PerformanceChart } from './components/PerformanceChart';
import { LearnModeProvider, useLearnMode } from './components/LearnModeProvider';
import { LearnModeOverlay } from './components/LearnModeOverlay';
import { LearnModeNav } from './components/LearnModeNav';

function computeCompoundedYtd(data: { year: number; return_pct: number; observations?: number }[], year: number): number | null {
  const months = data.filter((row) => row.year === year && (row.observations ?? 1) > 0);
  if (months.length === 0) return null;
  return months.reduce((acc, row) => acc * (1 + row.return_pct), 1) - 1;
}

function getArgmaxRegime(probs: Record<string, number>): string | null {
  const entries = Object.entries(probs);
  if (entries.length === 0) return null;
  return entries.reduce((best, current) => (current[1] > best[1] ? current : best))[0];
}

function detectLastFiredSignal(signals: any): string {
  if (!signals) return 'frag';
  if (signals.fragility_score > 0.75) return 'frag';
  if (signals.entropy_shift_flag) return 'entropy';
  if (signals.vol_uncertainty_score > 0.8) return 'vol';
  if (Math.abs(signals.macro_credit_score) > 0.5) return 'macro';
  return 'frag';
}

function formatCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

function formatPct(v: number, digits = 2): string {
  const sign = v >= 0 ? '+' : '';
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

function computeVsSpySpread(equityCurve: { value: number; benchmark: number }[]): number | null {
  if (equityCurve.length < 2) return null;
  const first = equityCurve.find(p => p.value > 0 && p.benchmark > 0);
  const last = [...equityCurve].reverse().find(p => p.value > 0 && p.benchmark > 0);
  if (!first || !last) return null;
  return (last.value / first.value - 1) - (last.benchmark / first.benchmark - 1);
}

export function App() {
  const isMobile = useIsMobile();
  const { data, loading, error } = useDashboardData();
  const { data: timeseries } = useTimeseriesData();
  const { data: shadow } = useShadowData();
  const [selectedOptimizerRun, setSelectedOptimizerRun] = useState<string | undefined>(undefined);
  const [optimizerOpen, setOptimizerOpen] = useState(false);
  const {
    index: optimizerIndex,
    lineage: optimizerLineage,
    candidateBundle: optimizerCandidateBundle,
    detail: optimizerDetail,
    loading: optimizerLoading,
    detailLoading: optimizerDetailLoading,
  } = useOptimizerData(selectedOptimizerRun);

  useEffect(() => {
    if (!import.meta.env.DEV || !data) return;
    const selectedRegime = data.expert_signals?.final_regime_label ?? data.weather.regime.regime;
    const argmaxRegime = getArgmaxRegime(data.weather.regime.probs);
    const warnings: string[] = [];
    if (argmaxRegime && argmaxRegime !== selectedRegime && !data.expert_signals?.override_reason) {
      warnings.push(`Argmax regime (${argmaxRegime}) differs from selected (${selectedRegime}).`);
    }
    if (!isStrictlyIncreasingByDate(data.drawdowns)) {
      warnings.push('Drawdown timestamps not strictly increasing.');
    }
    const snapshotYear = parseISO(data.metrics.timestamp).getFullYear();
    const monthlyYtd = computeCompoundedYtd(data.monthly_returns, snapshotYear);
    if (monthlyYtd != null && Math.abs(monthlyYtd - data.metrics.ytd_return) > 1e-6) {
      warnings.push(`YTD mismatch (header=${data.metrics.ytd_return.toFixed(6)}, monthly=${monthlyYtd.toFixed(6)}).`);
    }
    // eslint-disable-next-line no-console
    warnings.forEach(w => console.warn(`[dashboard-coherence] ${w}`));
  }, [data]);

  if (loading) return <div className="loading">Loading dashboard...</div>;
  if (error) return <div className="error"><div style={{ fontSize: 48, marginBottom: 16 }}>:(</div><div>Failed to load dashboard data</div><div style={{ fontSize: 14, marginTop: 8 }}>{error}</div></div>;
  if (!data) return <div className="error">No data available</div>;

  if (isMobile) {
    return <MobileDashboard data={data} timeseries={timeseries ?? []} shadow={shadow} />;
  }

  const m = data.metrics;
  const vsSpySpread = computeVsSpySpread(data.equity_curve);
  const lastFiredSignal = detectLastFiredSignal(data.expert_signals);
  const isHybridLive = optimizerCandidateBundle?.promotion_status === 'live_active';

  return (
    <LearnModeProvider>
      <div className="dashboard">

        {/* ═══════ ZONE 1: Identity + System + Money (~90px) ═══════ */}
        <div className="zone-1">
          <header className="zone-1-header">
            <h1>Hybrid Ranking System</h1>
            <span className="zone-1-timestamp">
              {format(parseISO(m.timestamp), 'MMM d, yyyy h:mm a')}
              <InfoTooltip content={`Data from snapshot ${data.snapshot?.id ?? m.snapshot_id ?? '?'}. All values on this page reflect this single pipeline run.`} label="Snapshot" align="right" />
              <LearnButton />
            </span>
          </header>

          <LearnModeOverlay paneId="status-bar">
            <SystemStatusBar signals={data.expert_signals} metrics={m} candidateBundle={optimizerCandidateBundle} />
          </LearnModeOverlay>
        </div>

        {/* ═══════ ZONE 2: Chart + Story + Brain ═══════ */}
        <div className="zone-2">
          <div className="zone-2-chart">
            <LearnModeOverlay paneId="perf-summary">
              <div className="card perf-summary-card">
                <div className="perf-metrics-strip">
                  <div className="z1-metric">
                    <span className="z1-metric-value z1-anchor">{formatCurrency(m.total_value)}</span>
                    <span className="z1-metric-label">Total Value</span>
                  </div>
                  <div className="z1-metric">
                    <span className={`z1-metric-value ${m.ytd_return >= 0 ? 'positive' : 'negative'}`}>{formatPct(m.ytd_return)}</span>
                    <span className="z1-metric-label">YTD</span>
                  </div>
                  <div className="z1-metric">
                    <span className="z1-metric-value">{m.sharpe_ratio != null ? m.sharpe_ratio.toFixed(2) : 'N/A'}</span>
                    <span className="z1-metric-label">Sharpe</span>
                  </div>
                  <div className="z1-metric">
                    <span className={`z1-metric-value ${m.max_drawdown >= 0 ? 'positive' : 'negative'}`}>{formatPct(m.max_drawdown)}</span>
                    <span className="z1-metric-label">Max DD</span>
                  </div>
                  <div className="z1-metric">
                    <span className={`z1-metric-value ${(vsSpySpread ?? 0) >= 0 ? 'positive' : 'negative'}`}>{vsSpySpread != null ? formatPct(vsSpySpread) : 'N/A'}</span>
                    <span className="z1-metric-label">vs SPY</span>
                  </div>
                </div>
              </div>
            </LearnModeOverlay>
            <LearnModeOverlay paneId="perf-chart" className="learn-pane-flex-grow">
              <PerformanceChart
                equityData={data.equity_curve}
                drawdownData={data.drawdowns}
                monthlyReturns={data.monthly_returns}
                timeseries={timeseries}
                chartMarkers={data.chart_markers}
                shadow={shadow}
              />
            </LearnModeOverlay>
          </div>
          <div className="zone-2-right">
            <LearnModeOverlay paneId="todays-story">
              <TodaysStoryCard
                signals={data.expert_signals}
                metrics={m}
                candidates={data.candidates}
                fusionRules={data.expert_signals?.fusion_rules}
                weather={data.weather}
              />
            </LearnModeOverlay>
            <LearnModeOverlay paneId="system-brain">
              <SystemBrainPanel
                ensemble={data.weather.regime.ensemble}
                signals={data.expert_signals}
                fusionRules={data.expert_signals?.fusion_rules}
                timeseries={timeseries}
                isHybridLive={isHybridLive}
                mostFiredSignalKey={lastFiredSignal}
              />
            </LearnModeOverlay>
          </div>
        </div>

        {/* ═══════ ZONE 3: Lower Deck ═══════ */}
        <div className="zone-3">
          <div className="lower-deck">
            <div className="lower-deck-slot lower-deck-slot-table">
              <LearnModeOverlay paneId="holdings">
                <PortfolioTable holdings={data.holdings} snapshotTimestamp={m.timestamp} metrics={m} />
              </LearnModeOverlay>
            </div>
            <div className="lower-deck-slot lower-deck-slot-log">
              <LearnModeOverlay paneId="trade-log">
                <TradeLog trades={data.trades ?? []} cumulativeCosts={m.cumulative_transaction_costs} tradeSummary={data.trade_summary} />
              </LearnModeOverlay>
            </div>
            <div className="lower-deck-slot lower-deck-slot-table">
              <LearnModeOverlay paneId="candidates">
                <CandidatesTable candidates={data.candidates} snapshotTimestamp={m.timestamp} />
              </LearnModeOverlay>
            </div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="collapsible-header" onClick={() => setOptimizerOpen(!optimizerOpen)}>
              <div className="card-title" style={{ marginBottom: 0 }}>
                <span>Optimizer</span>
                <InfoTooltip content="Champion-challenger evolution engine. Tests new parameter sets against the live champion via walk-forward backtesting. Promoted candidates become the active scoring model." label="Optimizer" />
              </div>
              <span className={`chevron ${optimizerOpen ? 'open' : ''}`}>▼</span>
            </div>
            {optimizerOpen && (
              <div style={{ marginTop: 12 }}>
                <OptimizerStatus index={optimizerIndex} lineage={optimizerLineage} selectedRunId={selectedOptimizerRun} onSelectRun={runId => setSelectedOptimizerRun(runId)} />
                <OptimizerRunDetail detail={optimizerDetail} loading={optimizerLoading || optimizerDetailLoading} />
              </div>
            )}
          </div>

          <PerformanceLenses equityCurve={data.equity_curve} shadow={shadow} />
        </div>

        <LearnModeNav />
      </div>
    </LearnModeProvider>
  );
}

function LearnButton() {
  const { active, enterLearnMode, exitLearnMode } = useLearnMode();
  return (
    <button className="learn-toggle-btn" onClick={active ? exitLearnMode : enterLearnMode}>
      {active ? 'Exit Learn' : 'Learn'}
    </button>
  );
}
