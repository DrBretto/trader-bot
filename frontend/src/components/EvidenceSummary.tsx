import { InfoTooltip } from './InfoTooltip';
import { CandidateBundleSummary, OptimizerRunsIndex } from '../types';

interface Props {
  optimizerIndex: OptimizerRunsIndex | null;
  candidateBundle: CandidateBundleSummary | null;
}

function deriveLatestDecisionSummary(index: OptimizerRunsIndex | null): string | null {
  if (!index?.runs?.length) return null;
  const latest = index.runs[0];
  const status = latest.status ?? 'unknown';
  const decision = latest.decision ?? 'unknown';
  if (status === 'promoted') return 'Last run promoted a challenger.';
  if (status === 'rejected_guardrails') return `Last run: ${decision} (guardrails not met)`;
  if (status === 'rejected_objective') return `Last run: ${decision} (objective not met)`;
  if (status === 'failed') return `Last run: ${decision} (run failed)`;
  return `Last run: ${decision} (${status})`;
}

function hasAnyPromotion(index: OptimizerRunsIndex | null): boolean {
  return (index?.runs ?? []).some((r) => r.status === 'promoted');
}

export function EvidenceSummary({ optimizerIndex, candidateBundle }: Props) {
  const activeVersion = optimizerIndex?.active_version;
  const candidateVersion = candidateBundle?.version_id;
  const candidateIsStaged = candidateBundle?.promotion_status === 'staged_not_promoted';
  const showCandidate = candidateVersion && candidateIsStaged && !hasAnyPromotion(optimizerIndex);
  const latestDecision = deriveLatestDecisionSummary(optimizerIndex);
  const totalRuns = optimizerIndex?.runs?.length ?? 0;

  return (
    <div className="card evidence-summary-card" style={{ marginBottom: 16 }}>
      <div className="card-title">
        <span>System Evidence Summary</span>
        <InfoTooltip
          content="Current evaluation state of the trading system. Gate evidence is from the most recent honest evaluation window on live-native data. Fold evidence is still limited by historical data quality."
          label="Evidence summary"
        />
      </div>

      <div className="evidence-summary-grid">
        {/* Gate Evidence */}
        <div className="evidence-panel">
          <div className="evidence-panel-label">
            Gate Evidence
            <InfoTooltip
              content="Gate segment: 37 test days on live-native data (2026-01 to 2026-03). This is the strongest honest evidence surface. The system preserved capital during a period when SPY dropped ~18.5%."
              label="Gate evidence"
              align="right"
            />
          </div>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#22c55e' }}>Capital Preserved</div>
          <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
            20 round trips · 50% win rate · SPY −18.5%
          </div>
        </div>

        {/* Model Dependencies */}
        <div className="evidence-panel">
          <div className="evidence-panel-label">
            Model Stack
            <InfoTooltip
              content="Ablation findings: Health model is required for any trading activity. Regime model is the load-bearing component. Expert signal fusion is a tuning seam — currently being evaluated for threshold adjustment."
              label="Model stack"
              align="right"
            />
          </div>
          <div style={{ fontSize: 13, color: '#f8fafc' }}>
            <span style={{ color: '#3b82f6' }}>Health</span>{' '}
            <span style={{ color: '#64748b' }}>→</span>{' '}
            <span style={{ color: '#3b82f6' }}>Regime</span>{' '}
            <span style={{ color: '#64748b' }}>→</span>{' '}
            <span style={{ color: '#eab308' }}>Fusion</span>{' '}
            <span style={{ color: '#64748b' }}>→</span>{' '}
            <span style={{ color: '#94a3b8' }}>Decisions</span>
          </div>
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
            Regime = load-bearing · Fusion = tuning seam
          </div>
        </div>

        {/* Parameter State — now data-driven */}
        <div className="evidence-panel">
          <div className="evidence-panel-label">
            Parameter State
            <InfoTooltip
              content={
                candidateBundle
                  ? `Active: ${activeVersion ?? 'unknown'}\nChallenger: ${candidateVersion ?? 'none'}\n\n${candidateBundle.change_summary}\n\nStatus: ${candidateBundle.promotion_status.replace(/_/g, ' ')}\nRequires: ${candidateBundle.promotion_requires}`
                  : `Active: ${activeVersion ?? 'unknown'}\nNo candidate bundle data available.`
              }
              label="Parameter state"
              align="right"
            />
          </div>
          <div style={{ fontSize: 13, color: '#f8fafc' }}>
            Active: <span style={{ fontFamily: 'monospace', color: '#3b82f6' }}>{activeVersion ?? '—'}</span>
          </div>
          {showCandidate && (
            <div style={{ fontSize: 12, color: '#eab308', marginTop: 2 }}>
              Challenger staged: <span style={{ fontFamily: 'monospace' }}>{candidateVersion}</span>
            </div>
          )}
          {!showCandidate && (
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>No challenger staged</div>
          )}
        </div>

        {/* Optimizer Decision — new panel, derived from index data */}
        <div className="evidence-panel">
          <div className="evidence-panel-label">
            Optimizer
            <InfoTooltip
              content={`Champion–challenger optimizer runs locally.\n${totalRuns} run${totalRuns !== 1 ? 's' : ''} recorded. No promotions to date — all challengers have been rejected by guardrails or objective criteria.\n\nThis is expected in early operation. The optimizer is conservative by design.`}
              label="Optimizer status"
              align="right"
            />
          </div>
          <div style={{ fontSize: 13, color: '#f8fafc' }}>
            {totalRuns} run{totalRuns !== 1 ? 's' : ''} · 0 promotions
          </div>
          {latestDecision && (
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
              {latestDecision}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
