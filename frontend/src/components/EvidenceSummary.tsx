import { InfoTooltip } from './InfoTooltip';

interface Props {
  optimizerActiveVersion?: string;
  optimizerCandidateVersion?: string;
}

export function EvidenceSummary({ optimizerActiveVersion, optimizerCandidateVersion }: Props) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-title">
        <span>System Evidence Summary</span>
        <InfoTooltip
          content="Current evaluation state of the trading system. Gate evidence is from the most recent honest evaluation window on live-native data. Fold evidence is still limited by historical data quality."
          label="Evidence summary"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginTop: 8 }}>
        {/* Gate Evidence */}
        <div style={{ padding: '12px 16px', background: '#0f172a', borderRadius: 8, border: '1px solid #334155' }}>
          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
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
        <div style={{ padding: '12px 16px', background: '#0f172a', borderRadius: 8, border: '1px solid #334155' }}>
          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
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

        {/* Challenger State */}
        <div style={{ padding: '12px 16px', background: '#0f172a', borderRadius: 8, border: '1px solid #334155' }}>
          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
            Parameter State
            <InfoTooltip
              content={`Active: ${optimizerActiveVersion ?? 'unknown'}\nChallenger: ${optimizerCandidateVersion ?? 'none'}\n\nThe challenger adjusts macro_downgrade_threshold from -0.50 to -0.75. Staged but NOT promoted — requires more live evidence before any change goes live.`}
              label="Parameter state"
              align="right"
            />
          </div>
          <div style={{ fontSize: 13, color: '#f8fafc' }}>
            Active: <span style={{ fontFamily: 'monospace', color: '#3b82f6' }}>{optimizerActiveVersion ?? '—'}</span>
          </div>
          {optimizerCandidateVersion && (
            <div style={{ fontSize: 12, color: '#eab308', marginTop: 2 }}>
              Challenger staged: <span style={{ fontFamily: 'monospace' }}>{optimizerCandidateVersion}</span>
            </div>
          )}
          {!optimizerCandidateVersion && (
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>No challenger staged</div>
          )}
        </div>
      </div>
    </div>
  );
}
