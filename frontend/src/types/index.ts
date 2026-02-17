export interface PortfolioMetrics {
  total_value: number;
  cash: number;
  invested: number;
  ytd_return: number;
  mtd_return: number;
  sharpe_ratio: number | null;
  sharpe_observations?: number;
  sharpe_min_observations?: number;
  max_drawdown: number;
  current_drawdown: number;
  win_rate: number;
  total_trades: number;
  wins?: number;
  losses?: number;
  breakeven_trades?: number;
  realized_round_trips?: number;
  total_fills?: number;
  cumulative_transaction_costs?: number;
  cash_pct?: number;
  gross_exposure?: number;
  net_exposure?: number;
  top_position_pct?: number;
  beta_proxy?: number | null;
  snapshot_id?: string;
  timestamp: string;
}

export interface Holding {
  symbol: string;
  shares: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  health_score: number;
  vol_bucket: string;
  days_held: number;
}

export interface BuyCandidate {
  symbol: string;
  score: number;
  health_score: number;
  vol_bucket: string;
  behavior: string;
  return_21d: number;
  return_63d: number;
  suggested_size: number;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
  benchmark: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
  observations?: number;
}

export interface ModelPrediction {
  label: string;
  confidence: number;
  probs: Record<string, number>;
}

export interface EnsembleMetrics {
  confidence: number;
  disagreement: number;
  agreement: number;
  position_size_multiplier: number;
  gru_prediction?: ModelPrediction;
  transformer_prediction?: ModelPrediction;
  is_ensemble: boolean;
}

export interface RegimeInfo {
  regime: string;
  description: string;
  risk_level: 'low' | 'medium' | 'high' | 'extreme';
  probs: Record<string, number>;
  ensemble?: EnsembleMetrics;
}

export interface WeatherReport {
  headline: string;
  summary: string;
  regime: RegimeInfo;
  outlook: string;
  risks: string[];
  timestamp: string;
}

export interface ExpertSignals {
  macro_credit_score: number;
  yield_slope_10y_3m: number;
  hy_spread_proxy: number;
  vol_uncertainty_score: number;
  vol_regime_label: 'calm' | 'unstable_calm' | 'panic';
  vix_percentile: number;
  vvix_percentile: number;
  fragility_score: number;
  avg_correlation: number;
  pc1_explained: number;
  entropy_score: number;
  entropy_z_score: number;
  entropy_shift_flag: boolean;
  final_regime_label: string;
  regime_confidence: number;
  position_size_modifier: number;
  risk_throttle_factor: number;
  override_reason?: string | null;
  target_gross_exposure?: number;
  effective_exposure_multiplier?: number;
  throttle_mapping?: string;
  fusion_rules?: FusionRule[];
  ensemble_regime_label?: string;
  panic_prob?: number;
  ensemble_disagreement?: number;
  ensemble_multiplier?: number;
}

export interface FusionRule {
  order: number;
  code: string;
  label: string;
  fired: boolean;
  inputs: string;
  threshold: string;
  effect: string;
}

export interface TimeseriesPoint {
  date: string;
  final_regime_label: string;
  regime_confidence: number;
  trend_risk_on_prob: number;
  panic_prob: number;
  macro_credit_score: number;
  yield_slope_10y_3m: number;
  hy_spread_proxy: number;
  vol_uncertainty_score: number;
  vol_regime_label: string;
  vix_percentile: number;
  vvix_percentile: number;
  skew_value: number;
  fragility_score: number;
  avg_correlation: number;
  pc1_explained: number;
  entropy_score: number;
  entropy_z_score: number;
  entropy_shift_flag: boolean;
  position_size_modifier: number;
  risk_throttle_factor: number;
  spy_close: number;
  portfolio_value: number;
}

export interface Trade {
  timestamp: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'REDUCE';
  shares: number;
  price: number;
  market_price?: number;
  dollars: number;
  transaction_cost_bps?: number;
  reason: string;
  regime: string;
  entry_price?: number;
  pnl?: number;
  pnl_pct?: number;
  days_held?: number;
}

export interface TradeSummary {
  fills_total: number;
  realized_round_trips: number;
  wins: number;
  losses: number;
  breakeven: number;
  win_rate: number;
  unmatched_closing_shares: number;
  cumulative_transaction_costs: number;
}

export interface SnapshotMeta {
  id: string;
  date: string;
  phase: string;
  timestamp: string;
}

export interface PanelSnapshotIds {
  metrics: string;
  equity_curve: string;
  drawdowns: string;
  monthly_returns: string;
  trade_log: string;
  regime: string;
}

export interface DashboardData {
  snapshot?: SnapshotMeta;
  panel_snapshot_ids?: PanelSnapshotIds;
  metrics: PortfolioMetrics;
  holdings: Holding[];
  candidates: BuyCandidate[];
  equity_curve: EquityCurvePoint[];
  drawdowns: DrawdownPoint[];
  monthly_returns: MonthlyReturn[];
  weather: WeatherReport;
  trades?: Trade[];
  trade_summary?: TradeSummary;
  round_trips?: Record<string, unknown>[];
  reset_boundary?: Record<string, unknown> | null;
  expert_signals?: ExpertSignals;
  timeseries_url?: string;
}

export interface OptimizerRunSummary {
  run_id: string;
  started_at: string;
  finished_at: string;
  status: 'promoted' | 'completed' | 'failed' | 'rejected_guardrails' | 'rejected_objective' | string;
  decision: 'promoted' | 'not_promoted' | string;
  champion_version_before?: string;
  challenger_version?: string;
  wf_delta?: number;
  gate_delta?: number;
}

export interface OptimizerRunsIndex {
  updated_at: string;
  active_version: string;
  runs: OptimizerRunSummary[];
}

export interface OptimizerLineageEvent {
  event_type: 'promotion' | 'rollback' | string;
  timestamp: string;
  from_version?: string | null;
  to_version?: string | null;
  run_id?: string | null;
  reason?: string;
  operator?: string;
}

export interface OptimizerLineage {
  updated_at: string;
  active_version: string;
  history: OptimizerLineageEvent[];
}

export interface OptimizerRunDetail {
  run_id: string;
  run_manifest: Record<string, unknown>;
  walk_forward_folds: Record<string, unknown>;
  champion_metrics: Record<string, unknown>;
  challenger_metrics: Record<string, unknown>;
  gate_segment_metrics: Record<string, unknown>;
  guardrail_results: Record<string, unknown>;
  promotion_decision: Record<string, unknown>;
  candidate_params_bundle: Record<string, unknown>;
  generation_log: Record<string, unknown>[];
  run_log_path?: string;
}
