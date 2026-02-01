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

export interface DashboardData {
  metrics: PortfolioMetrics;
  holdings: Holding[];
  candidates: BuyCandidate[];
  equity_curve: EquityCurvePoint[];
  drawdowns: DrawdownPoint[];
  monthly_returns: MonthlyReturn[];
  weather: WeatherReport;
}
