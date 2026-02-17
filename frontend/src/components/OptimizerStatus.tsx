import { format, parseISO } from 'date-fns';
import { InfoTooltip } from './InfoTooltip';
import { OptimizerLineage, OptimizerRunsIndex } from '../types';

interface Props {
  index: OptimizerRunsIndex | null;
  lineage: OptimizerLineage | null;
  selectedRunId?: string;
  onSelectRun: (runId: string) => void;
}

function formatDate(value?: string): string {
  if (!value) return '—';
  try {
    return format(parseISO(value), 'MMM d, yyyy h:mm a');
  } catch {
    return value;
  }
}

function formatDelta(value?: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(3)}`;
}

function statusColor(status?: string): string {
  if (!status) return '#94a3b8';
  if (status === 'promoted') return '#22c55e';
  if (status === 'failed') return '#ef4444';
  if (status.startsWith('rejected')) return '#f59e0b';
  return '#94a3b8';
}

export function OptimizerStatus({ index, lineage, selectedRunId, onSelectRun }: Props) {
  const latest = index?.runs?.[0];
  const historyCount = lineage?.history?.length ?? 0;

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <div className="card-title" style={{ justifyContent: 'space-between' }}>
        <span>Optimizer</span>
        <InfoTooltip
          label="Optimizer monitor"
          content={`Champion–challenger optimizer runs locally in one-shot mode.
Promotion only occurs if hard guardrails pass and challenger beats champion on both walk-forward objective and gate metrics.`}
        />
      </div>

      <div className="optimizer-status-grid">
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Active Version</div>
          <div className="optimizer-status-value">{index?.active_version ?? '—'}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Last Run Status</div>
          <div className="optimizer-status-value" style={{ color: statusColor(latest?.status) }}>
            {latest?.status ?? 'no runs'}
          </div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Last Decision</div>
          <div className="optimizer-status-value">{latest?.decision ?? '—'}</div>
        </div>
        <div className="optimizer-status-kv">
          <div className="optimizer-status-label">Lineage Events</div>
          <div className="optimizer-status-value">{historyCount}</div>
        </div>
      </div>

      {latest && (
        <div style={{ marginTop: 16, fontSize: 13, color: '#94a3b8' }}>
          Last run: {formatDate(latest.started_at)} → {formatDate(latest.finished_at)}
        </div>
      )}

      <div style={{ marginTop: 18, overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Decision</th>
              <th>WF Delta</th>
              <th>Gate Delta</th>
              <th>Finished</th>
              <th>View</th>
            </tr>
          </thead>
          <tbody>
            {(index?.runs ?? []).slice(0, 10).map((run) => (
              <tr key={run.run_id}>
                <td>{run.run_id}</td>
                <td style={{ color: statusColor(run.status) }}>{run.status}</td>
                <td>{run.decision}</td>
                <td>{formatDelta(run.wf_delta)}</td>
                <td>{formatDelta(run.gate_delta)}</td>
                <td>{formatDate(run.finished_at)}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => onSelectRun(run.run_id)}
                    style={{
                      backgroundColor: selectedRunId === run.run_id ? '#2563eb' : '#334155',
                      border: 'none',
                      color: '#f8fafc',
                      borderRadius: 6,
                      padding: '6px 10px',
                      cursor: 'pointer',
                      fontSize: 12,
                    }}
                  >
                    {selectedRunId === run.run_id ? 'Selected' : 'View'}
                  </button>
                </td>
              </tr>
            ))}
            {(index?.runs?.length ?? 0) === 0 && (
              <tr>
                <td colSpan={7} style={{ color: '#94a3b8' }}>
                  No optimizer runs recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
