interface Props {
  active: boolean;
  onClose: () => void;
}

const SECTIONS = [
  {
    zone: 'overview',
    title: 'What is this system?',
    body: 'An autonomous investment system that analyzes market conditions, detects regime changes, and manages a real portfolio through Alpaca. It runs a daily pipeline: collect data, predict the regime, score candidates, size positions, and execute trades.',
  },
  {
    zone: 'status-bar',
    title: 'System Status Bar',
    body: 'Shows the live state of the decision pipeline. The model version, current market regime, cash allocation, and whether the hybrid ranking model is active. The pipeline diagram (Signals → Regime → Fusion → Score → Actions) shows the decision chain — click the (i) button for current values at each stage.',
  },
  {
    zone: 'summary-strip',
    title: 'Performance Summary',
    body: 'Key portfolio metrics at a glance. Total Value is the current mark-to-market portfolio worth. YTD is year-to-date return. Sharpe measures risk-adjusted return (higher is better). Max DD is the worst peak-to-trough drawdown. vs SPY shows cumulative outperformance against the S&P 500 benchmark.',
  },
  {
    zone: 'performance-chart',
    title: 'Performance Chart',
    body: 'The main equity curve. Blue line is the portfolio, gray dashed is SPY (same starting value). Background color bands show the detected market regime. The drawdown strip below shows how far underwater the portfolio has been. Use All / Backtest / Live to filter by era. Monthly returns appear as colored cells at the bottom.',
  },
  {
    zone: 'todays-story',
    title: "Today's Story",
    body: 'A natural-language summary of what the system sees and what it decided. The headline comes from an LLM risk layer analyzing current conditions. Below it: which fusion rules fired (red badges = active overrides), current cash and sizing levels, and top buy candidates.',
  },
  {
    zone: 'system-brain',
    title: 'System Brain',
    body: 'The neural network ensemble and signal layer. Two models (GRU and Transformer) independently predict the market regime. Agreement % shows how aligned they are. Below: fusion rule dots show which risk gates are active. The signal chart tracks four expert inputs (Macro, Vol, Fragility, Entropy) — the bright line is the most relevant signal right now. The regime strip at the bottom shows the 60-day regime history as colored bars.',
  },
  {
    zone: 'lower-deck',
    title: 'Operations Deck',
    body: 'Current Holdings shows open positions with unrealized P&L and health scores. Buy Candidates lists what the system would buy now, ranked by blended score. Trade Log shows executed fills. These update with each pipeline run.',
  },
];

export function LearningGuide({ active, onClose }: Props) {
  if (!active) return null;

  return (
    <div className="learning-guide-overlay" onClick={onClose}>
      <div className="learning-guide-content" onClick={e => e.stopPropagation()}>
        <div className="learning-guide-header">
          <h2 className="learning-guide-title">How This System Works</h2>
          <p className="learning-guide-subtitle">
            A quick guide to reading the dashboard. Each section explains what you're looking at and why it matters.
          </p>
          <button className="learning-guide-close" onClick={onClose}>Close Guide</button>
        </div>
        <div className="learning-guide-sections">
          {SECTIONS.map((section, i) => (
            <div key={section.zone} className="learning-guide-card">
              <div className="learning-guide-number">{i + 1}</div>
              <div className="learning-guide-card-body">
                <h3 className="learning-guide-card-title">{section.title}</h3>
                <p className="learning-guide-card-text">{section.body}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="learning-guide-footer">
          <button className="learning-guide-close" onClick={onClose}>Got it</button>
        </div>
      </div>
    </div>
  );
}
