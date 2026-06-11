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

export interface ShadowStats {
  days_accrued: number;
  n_weeks_ic: number;
  n_settled?: number;
  mean_ic: number | null;
  ic_t: number | null;
  utility_diff_bp_day: number | null;
  utility_diff_bp_day_ci: [number, number] | null;
  m4a_diff_bp_day?: number | null;
  m4a_diff_bp_day_ci?: [number, number] | null;
  /** Alias of utility_diff_bp_day_ci kept by the publisher. */
  ci?: [number, number] | null;
}

export interface ShadowTimeseries {
  schema: string;
  as_of: string;
  start_date: string | null;
  forward_boundary?: string;
  prereg_pointer?: string;
  live_line: ShadowLinePoint[];
  shadow_A: ShadowLinePoint[];
  shadow_B: ShadowLinePoint[];
  shadow_I?: ShadowLinePoint[];
  provisional_date?: string | null;
  ic_series: [string, number, number][];
  stats: ShadowStats;
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
          if (json?.schema !== 'shadow_timeseries.v1') continue;
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
