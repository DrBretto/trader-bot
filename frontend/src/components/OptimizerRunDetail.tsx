import { InfoTooltip } from './InfoTooltip';
import { OptimizerRunDetail as OptimizerRunDetailType } from '../types';

interface Props {
  detail: OptimizerRunDetailType | null;
  loading: boolean;
}

function readNumber(source: Record<string, unknown>, key: string): number | null {
  const value = source[key];
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return null;
}

function formatPct(value: number | null): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(2)}%`;
}

function formatNum(value: number | null, digits = 3): string {
  if (value == null) return '—';
  return value.toFixed(digits);
}

export function OptimizerRunDetail({ detail, loading }: Props) {
  if (loading) {
    return (
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-title">Optimizer Run Detail</div>
        <div style={{ color: '#94a3b8', fontSize: 14 }}>Loading run detail...</div>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-title">Optimizer Run Detail</div>
        <div style={{ color: '#94a3b8', fontSize: 14 }}>No detail available.</div>
      </div>
    );
  }

  const champion = detail.champion_metrics;
  const challenger = detail.challenger_metrics;
  const gate = detail.gate_segment_metrics;
  const promotion = detail.promotion_decision;
  const candidateBundle = detail.candidate_params_bundle;
  const diffs = (candidateBundle.param_diffs as Record<string, unknown>[] | undefined) ?? [];

  const championWf = readNumber(champion, 'wf_objective');
  const challengerWf = readNumber(challenger, 'wf_objective');
  const championGate = readNumber(gate, 'champion_gate_score');
  const challengerGate = readNumber(gate, 'challenger_gate_score');
  const championGateDd = readNumber(gate, 'champion_gate_max_drawdown');
  const challengerGateDd = readNumber(gate, 'challenger_gate_max_drawdown');

  const guardrailResults = detail.guardrail_results;
  const checks = (guardrailResults.checks as Record<string, unknown>[] | undefined) ?? [];

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <div className="card-title" style={{ justifyContent: 'space-between' }}>
        <span>Optimizer Run Detail</span>
        <InfoTooltip
          label="Run detail"
          content={`Shows champion vs challenger comparison for walk-forward objective and holdout gate.
Guardrails must pass before promotion is allowed.`}
        />
      </div>

      <div className="optimizer-status-grid">
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Run ID</div>
          <div className="optimizer-status-value">{detail.run_id}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Decision</div>
          <div className="optimizer-status-value">{String(promotion.decision ?? '—')}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Champion WF</div>
          <div className="optimizer-status-value">{formatNum(championWf)}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Challenger WF</div>
          <div className="optimizer-status-value">{formatNum(challengerWf)}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Champion Gate Score</div>
          <div className="optimizer-status-value">{formatNum(championGate)}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Challenger Gate Score</div>
          <div className="optimizer-status-value">{formatNum(challengerGate)}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Champion Gate Drawdown</div>
          <div className="optimizer-status-value">{formatPct(championGateDd)}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Challenger Gate Drawdown</div>
          <div className="optimizer-status-value">{formatPct(challengerGateDd)}</div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <div className="card-title" style={{ marginBottom: 8 }}>Guardrail Checks</div>
        <table>
          <thead>
            <tr>
              <th>Check</th>
              <th>Passed</th>
              <th>Value</th>
              <th>Threshold</th>
            </tr>
          </thead>
          <tbody>
            {checks.map((check, index) => (
              <tr key={`${check.name ?? 'check'}-${index}`}>
                <td>{String(check.name ?? 'unknown')}</td>
                <td style={{ color: check.passed ? '#22c55e' : '#ef4444' }}>
                  {String(check.passed ? 'yes' : 'no')}
                </td>
                <td>{check.value != null ? String(check.value) : '—'}</td>
                <td>{check.threshold != null ? String(check.threshold) : '—'}</td>
              </tr>
            ))}
            {checks.length === 0 && (
              <tr>
                <td colSpan={4} style={{ color: '#94a3b8' }}>No guardrail details available.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 18 }}>
        <div className="card-title" style={{ marginBottom: 8 }}>Parameter Diffs vs Active</div>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>From</th>
              <th>To</th>
            </tr>
          </thead>
          <tbody>
            {diffs.slice(0, 40).map((diff, index) => (
              <tr key={`${diff.name ?? 'diff'}-${index}`}>
                <td>{String(diff.name ?? '')}</td>
                <td>{String(diff.from ?? '—')}</td>
                <td>{String(diff.to ?? '—')}</td>
              </tr>
            ))}
            {diffs.length === 0 && (
              <tr>
                <td colSpan={3} style={{ color: '#94a3b8' }}>No parameter differences recorded.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
