import { OrganLedgerRow, ForecastLeg, ShadowStats } from '../hooks/useShadowData';
import { InfoTooltip } from './InfoTooltip';

// PKT-TB-011 attribution spine. The per-organ rent ledger: one row per
// component of the 5-rung leave-one-organ-out ladder, each shown in TWO
// columns -- GROSS rent and multi-factor EXPOSURE-STRIPPED rent (the
// selection skill) -- with a three-valued verdict and a BH-FDR survival flag.
// An organ either shows rent or is visibly seen NOT to. Nothing here implies
// a dollar edge: the standing exposure-stripped residual is measured at zero.

interface Props {
  organ_ledger?: OrganLedgerRow[];
  forecast_leg?: ForecastLeg;
  stats?: ShadowStats;
}

const COMPONENT_LABEL: Record<string, string> = {
  regime: 'Regime gate',
  forecast: 'Forecast (M1 signal)',
  event: 'Event-damp (M4)',
  universe: 'Universe choice',
};

function verdictStyle(verdict: string): { bg: string; fg: string; label: string } {
  if (verdict === 'positive') return { bg: '#14532d', fg: '#4ade80', label: 'positive' };
  if (verdict.startsWith('zero')) return { bg: '#334155', fg: '#cbd5e1', label: 'zero (measured)' };
  return { bg: '#422006', fg: '#fbbf24', label: 'indeterminate' };
}

function bp(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
}

function ci(c: [number, number] | null | undefined): string {
  return c ? `[${c[0].toFixed(2)}, ${c[1].toFixed(2)}]` : '—';
}

function ForecastSkillCard({ leg }: { leg?: ForecastLeg }) {
  if (!leg) return null;
  return (
    <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 12, marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
        Forecast skill (certified receipt)
        <InfoTooltip content="Realized weekly rank-IC of the M1 forecast. This is the certified skill thing. It is NEVER multiplied by a notional to manufacture a dollar value — conversion is the forecast rung below, reported separately and pre-registered." />
      </div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'baseline' }}>
        <div>
          <span style={{ fontSize: 22, fontWeight: 600 }}>
            {leg.mean_ic === null ? '—' : leg.mean_ic.toFixed(3)}
          </span>
          <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 6 }}>mean IC</span>
        </div>
        <div style={{ fontSize: 13, color: '#cbd5e1' }}>
          t {leg.ic_t === null ? '—' : leg.ic_t.toFixed(2)} · {leg.n_weeks} wk
        </div>
        <div style={{ fontSize: 11, color: leg.certified ? '#4ade80' : '#fbbf24' }}>
          {leg.certified ? 'certified forward' : 'forecast skill — conversion unproven'}
        </div>
      </div>
    </div>
  );
}

export function RentLedger({ organ_ledger, forecast_leg, stats }: Props) {
  const rows = organ_ledger ?? stats?.organ_ledger ?? [];
  const leg = forecast_leg ?? stats?.forecast_leg;
  if (rows.length === 0 && !leg) return null;

  const add = stats?.additivity;
  const div = stats?.divergence;
  const fdr = stats?.fdr;
  const m4 = stats?.m4_subwindow;
  const matBp = stats?.materiality_bp ?? 1.5;

  return (
    <div style={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Per-organ rent ledger</h3>
        <InfoTooltip content="Leave-one-organ-out ladder: each row is one component's marginal rent. GROSS is the raw daily-return difference; STRIPPED removes the realized multi-factor exposure (SPY + duration + commodity), leaving selection skill. A 95% CI fully inside ±1.5 bp/day reads 'zero (measured)'. None of this asserts a dollar edge." />
      </div>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>
        Conversion measured forward, pre-registered, never an in-sample replay. Stripped residual currently ~zero.
      </div>

      <ForecastSkillCard leg={leg} />

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: '#94a3b8', textAlign: 'right', borderBottom: '1px solid #1e293b' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Component</th>
              <th style={{ padding: '6px 8px' }}>Gross bp/day</th>
              <th style={{ padding: '6px 8px' }}>Stripped bp/day</th>
              <th style={{ padding: '6px 8px' }}>95% CI (stripped)</th>
              <th style={{ padding: '6px 8px' }}>t</th>
              <th style={{ padding: '6px 8px' }}>n</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Verdict</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>FDR</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const vs = verdictStyle(r.verdict);
              return (
                <tr key={r.component} style={{ borderBottom: '1px solid #0f172a', textAlign: 'right' }}>
                  <td style={{ textAlign: 'left', padding: '6px 8px' }}>
                    <div style={{ color: '#e2e8f0' }}>
                      {COMPONENT_LABEL[r.component] ?? r.component}
                      {r.badge && (
                        <span style={{ marginLeft: 6, fontSize: 10, color: '#fbbf24' }} title={r.badge}>⚑</span>
                      )}
                    </div>
                    <div style={{ fontSize: 10, color: '#475569' }}>{r.book_pair}</div>
                  </td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{bp(r.gross_bp_day)}</td>
                  <td style={{ padding: '6px 8px', color: '#e2e8f0', fontWeight: 600 }}>{bp(r.stripped_bp_day)}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{ci(r.stripped_ci)}</td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{r.t === null ? '—' : r.t.toFixed(2)}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{r.n_days}</td>
                  <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                    <span style={{ background: vs.bg, color: vs.fg, borderRadius: 4, padding: '2px 6px', fontSize: 10 }}>
                      {vs.label}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '6px 8px', color: r.fdr_survivor ? '#4ade80' : '#475569' }}>
                    {r.fdr_survivor ? '✓' : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {rows.some((r) => r.caveat) && (
        <div style={{ marginTop: 10, fontSize: 10.5, color: '#64748b' }}>
          {rows.filter((r) => r.caveat).map((r) => (
            <div key={r.component} style={{ marginBottom: 2 }}>
              <span style={{ color: '#94a3b8' }}>{COMPONENT_LABEL[r.component] ?? r.component}:</span> {r.caveat}
            </div>
          ))}
        </div>
      )}

      {m4 && (
        <div style={{ marginTop: 10, fontSize: 11, color: '#94a3b8' }}>
          <span style={{ color: '#cbd5e1' }}>M4 sub-window read:</span>{' '}
          {m4.valid
            ? `${bp(m4.mean)} bp/day on ${m4.window?.[0]}…${m4.window?.[1]} (n=${m4.n}, ${m4.verdict})`
            : `not yet valid — ${m4.reason}`}
          {' '}<span style={{ color: '#475569' }}>· in BH-FDR family</span>
        </div>
      )}

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid #0f172a', fontSize: 10.5, color: '#64748b', lineHeight: 1.5 }}>
        {add && (
          <div>
            <span style={{ color: '#94a3b8' }}>Additivity:</span> log-space rungs telescope exactly
            (residual {add.log_residual_bp === null ? '—' : add.log_residual_bp.toFixed(4)} bp).
            bp/day rung-sum is approximate (residual {add.bp_day_residual === null ? '—' : add.bp_day_residual.toFixed(3)} bp/day — the one-signed path-divergence term).
          </div>
        )}
        {fdr && (
          <div>
            <span style={{ color: '#94a3b8' }}>Multiplicity:</span> BH-FDR q={fdr.q}, family of {fdr.n_members}
            {' '}({fdr.family.join(', ')}); {fdr.n_survivors} survivor(s). A lone green line (⚑) is not narratable as a discovery alone.
          </div>
        )}
        {div?.order_dependence_note && (
          <div><span style={{ color: '#94a3b8' }}>Order:</span> {div.order_dependence_note}</div>
        )}
        <div style={{ marginTop: 4, color: '#475569' }}>
          Materiality band ±{matBp} bp/day. Display, not verdict — verdicts are pre-registered ledger-count events (LIVE_PREREG.md).
        </div>
      </div>
    </div>
  );
}
