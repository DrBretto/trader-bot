import { useState, useEffect } from 'react';

// Dual forward shadow (PKT-TB-007 follow-on). Published nightly by
// runs/pkt_tb_007_orthogonal_brain/shadow/shadow_nightly.py to
// dashboard/shadow_timeseries.json. The file may be absent (job not yet
// run) or armed-but-empty (no settled marks yet) — every consumer must
// fail soft and render nothing.
//
// Types live here rather than in types/index.ts because the payload is
// owned by the shadow job, not the main pipeline's dashboard schema.

/** [date, nav] — nav normalized to the live line's NAV at shadow start. */
export type ShadowLinePoint = [string, number];

/** PKT-TB-011: one row of the 5-rung leave-one-organ-out rent ladder. Each
 *  row is one component's marginal (gross AND multi-factor exposure-stripped),
 *  with a three-valued verdict and a BH-FDR survival flag. */
export interface OrganLedgerRow {
  component: string;                       // stable registry id (the ledger join key); valid ids live in the component registry, never hard-coded here
  book_pair: string;                       // e.g. "F-R"
  gross_bp_day: number | null;
  gross_ci: [number, number] | null;
  stripped_bp_day: number | null;          // selection skill (multi-factor strip)
  stripped_ci: [number, number] | null;
  t: number | null;
  n_days: number;
  exposure_bp_day?: number | null;         // realized gross-differential term (C6)
  gross_diff_bp_day?: number | null;
  betas?: Record<string, number>;          // realized SPY/duration/commodity betas
  factors_used?: string[];
  cum_log_rent_bp?: number | null;         // additive log-space rent (C8)
  verdict: 'positive' | 'zero (measured at materiality scale)'
         | 'indeterminate at available power' | string;
  strippable: boolean;
  caveat?: string | null;                  // stripped-column / divergence caveat (C7)
  fdr_survivor?: boolean;
  fdr_p?: number | null;
  badge?: string;                          // "1 of N -- not narratable alone"
}

/** Forecast-skill (IC) card -- the certified thing, NEVER multiplied by a
 *  notional. Conversion is the F-R rung, reported separately. */
export interface ForecastLeg {
  mean_ic: number | null;
  ic_t: number | null;
  n_weeks: number;
  certified: boolean;
  note?: string;
}

export interface ShadowStats {
  days_accrued: number;
  n_weeks_ic: number;
  n_settled?: number;
  mean_ic: number | null;
  ic_t: number | null;
  utility_diff_bp_day: number | null;      // = U-I (deployed New Brain vs incumbent)
  utility_diff_bp_day_ci: [number, number] | null;
  m4a_diff_bp_day?: number | null;         // = E-F (M4 event-damp marginal)
  m4a_diff_bp_day_ci?: [number, number] | null;
  /** Alias of utility_diff_bp_day_ci kept by the publisher. */
  ci?: [number, number] | null;
  materiality_bp?: number;
  organ_ledger?: OrganLedgerRow[];
  forecast_leg?: ForecastLeg;
  m4_subwindow?: {
    valid: boolean; verdict?: string; in_fdr_family?: boolean;
    window?: [string, string]; mean?: number | null; ci?: [number, number] | null;
    t?: number | null; n?: number; reason?: string;
  };
  additivity?: {
    bp_day_total_U_minus_I: number | null; sum_bp_day_rungs: number | null;
    bp_day_residual: number | null; bp_day_note?: string;
    log_total_U_minus_I_bp: number | null; sum_log_rungs_bp: number | null;
    log_residual_bp: number | null; log_note?: string;
  };
  divergence?: {
    order: string[]; order_dependence_note?: string;
    per_rung_path_bp?: Record<string, number>;
  };
  fdr?: { q: number; family: string[]; n_members: number; n_survivors: number };
}

export interface ShadowTimeseries {
  schema: string;
  as_of: string;
  start_date: string | null;
  forward_boundary?: string;
  prereg_pointer?: string;
  live_prereg_pointer?: string;
  live_line: ShadowLinePoint[];
  /** Legacy display aliases (A = forecast rung F, B = event rung E). */
  shadow_A: ShadowLinePoint[];
  shadow_B: ShadowLinePoint[];
  shadow_I?: ShadowLinePoint[];
  /** PKT-TB-011 ladder lines; shadow_U is the deployed New Brain line. */
  shadow_R?: ShadowLinePoint[];
  shadow_F?: ShadowLinePoint[];
  shadow_E?: ShadowLinePoint[];
  shadow_U?: ShadowLinePoint[];
  provisional_date?: string | null;
  ic_series: [string, number, number][];
  organ_ledger?: OrganLedgerRow[];
  forecast_leg?: ForecastLeg;
  stats: ShadowStats;
}

// Challenger (comparison) series in display priority. The bottom model-comparison
// chart renders the FIRST non-empty one, so it stays robust to a varying number of
// challenger series and NEVER assumes a fixed set (e.g. exactly shadow_A + shadow_B):
// a series being absent, or N changing, degrades to "the next available challenger"
// rather than a blank/crashing chart. shadow_A and shadow_F are the same M1 tilt
// (the producer aliases shadow_A := shadow_F), so either renders the tilt line.
export const CHALLENGER_SERIES_KEYS: (keyof ShadowTimeseries)[] = [
  'shadow_A', 'shadow_F', 'shadow_U', 'shadow_E', 'shadow_B', 'shadow_R', 'shadow_I',
];

/** The first non-empty challenger line on the payload, or [] when none exists
 *  (armed-but-empty / absent). Pure + total — never throws on a missing series. */
export function pickChallengerSeries(
  shadow: ShadowTimeseries | null | undefined,
): ShadowLinePoint[] {
  if (!shadow) return [];
  for (const key of CHALLENGER_SERIES_KEYS) {
    const series = shadow[key];
    if (Array.isArray(series) && series.length > 0) {
      return series as ShadowLinePoint[];
    }
  }
  return [];
}

const SHADOW_URL = import.meta.env.VITE_DATA_URL
  ? 'shadow_timeseries.json'
  : './data/shadow_timeseries.json';
const SHADOW_FALLBACK = './data/shadow_timeseries.json';

export function useShadowData() {
  const [data, setData] = useState<ShadowTimeseries | null>(null);

  useEffect(() => {
    async function fetchData() {
      const urls = [SHADOW_URL];
      if (SHADOW_URL !== SHADOW_FALLBACK) urls.push(SHADOW_FALLBACK);

      for (const url of urls) {
        try {
          const cacheBustedUrl = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
          const response = await fetch(cacheBustedUrl, { cache: 'no-store' });
          if (!response.ok) continue;
          const json = await response.json();
          // accept the PKT-TB-011 v2 ladder schema; keep v1 during migration
          if (json?.schema !== 'shadow_timeseries.v2'
              && json?.schema !== 'shadow_timeseries.v1') continue;
          setData(json);
          break;
        } catch {
          continue;
        }
      }
    }
    fetchData();
  }, []);

  return { data };
}
