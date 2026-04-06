export interface LearnPaneContent {
  headline: string;
  body: string;
  placement: 'below' | 'right' | 'left' | 'above';
  pointers?: string[];
}

export const TOUR_SEQUENCE = [
  'status-bar',
  'perf-summary',
  'perf-chart',
  'todays-story',
  'system-brain',
  'holdings',
  'candidates',
  'trade-log',
];

export const PANE_LABELS: Record<string, string> = {
  'status-bar': 'System Status Bar',
  'perf-summary': 'Performance Summary',
  'perf-chart': 'Performance Chart',
  'todays-story': "Today's Story",
  'system-brain': 'System Brain',
  'holdings': 'Current Holdings',
  'candidates': 'Buy Candidates',
  'trade-log': 'Trade Log',
};

export const LEARN_CONTENT: Record<string, LearnPaneContent> = {
  'status-bar': {
    headline: "Your System's Vital Signs",
    body: "This strip tells you what's running and what the system sees right now. The regime label (the colored dot) is the system's one-word read on the market: calm, trending, choppy, risk-off, or panic. The pipeline diagram below shows the five-stage decision chain that fires every evening.",
    placement: 'below',
    pointers: ['regime-dot', 'pipeline-row'],
  },
  'perf-summary': {
    headline: "How You're Doing at a Glance",
    body: "Five numbers tell the whole performance story. **Total Value** is your current portfolio worth. **YTD Return** is your gain since January 1. **Sharpe Ratio** measures return per unit of risk — above 1.0 is solid. **Max Drawdown** is the worst peak-to-trough drop. **vs SPY** shows whether the system is beating the S&P 500.",
    placement: 'below',
  },
  'perf-chart': {
    headline: 'The Story of Your Capital',
    body: "The blue line is your portfolio value over time. The gray dashed line is SPY — your benchmark. The colored bands show what market regime the system detected each day. Below, the red drawdown strip shows how far you've fallen from your peak. The monthly heatmap shows each month's return.",
    placement: 'right',
  },
  'todays-story': {
    headline: 'What the System Saw Today',
    body: "Every evening, an AI reads the market data and writes this summary. The chips at the top show your cash allocation and position sizing. Red badges mean the system overrode its normal behavior — safety rules protecting your capital in unusual conditions.",
    placement: 'left',
  },
  'system-brain': {
    headline: 'How the System Thinks',
    body: "Two neural networks — a GRU and a Transformer — each predict what kind of market we're in. When they agree, the system acts with confidence. When they disagree, it reduces position sizes. The colored dots below are fusion rules — safety overrides. The signal chart tracks four expert indicators over 90 days.",
    placement: 'left',
    pointers: ['agreement-pct', 'fusion-dots'],
  },
  holdings: {
    headline: 'What You Own Right Now',
    body: "Each row is an open position. **Health Score** is the model's real-time quality assessment — green (70+) is strong, yellow is cautious, red means trouble. **Vol Bucket** classifies volatility, affecting position size. The system sells when health drops below 0.35 or a trailing stop triggers.",
    placement: 'above',
  },
  candidates: {
    headline: 'What the System Wants to Buy Next',
    body: "Assets ranked by a blended score: 65% health model, 35% ranking model. The bar shows this split — green is health, blue is ranking. **Suggested Size** is the dollar amount after all regime, volatility, and safety adjustments.",
    placement: 'above',
  },
  'trade-log': {
    headline: 'Everything the System Has Done',
    body: "Every trade executed, newest first. The system pairs buys with sells into round-trips to measure real P&L. **Win Rate** tells you what percentage made money. Transaction costs are tracked because they add up.",
    placement: 'above',
  },
};
