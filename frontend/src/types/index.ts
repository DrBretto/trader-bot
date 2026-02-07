export interface PortfolioMetrics {
  total_value: number;
  cash: number;
  invested: number;
  ytd_return: number;
  mtd_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  current_drawdown: number;
  win_rate: number;
  total_trades: number;
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
  ensemble_regime_label?: string;
  panic_prob?: number;
  ensemble_disagreement?: number;
  ensemble_multiplier?: number;
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

export interface DashboardData {
  metrics: PortfolioMetrics;
  holdings: Holding[];
  candidates: BuyCandidate[];
  equity_curve: EquityCurvePoint[];
  drawdowns: DrawdownPoint[];
  monthly_returns: MonthlyReturn[];
  weather: WeatherReport;
  expert_signals?: ExpertSignals;
  timeseries_url?: string;
}
