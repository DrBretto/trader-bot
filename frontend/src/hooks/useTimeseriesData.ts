import { useState, useEffect } from 'react';
import { TimeseriesPoint } from '../types';

const TS_URL = import.meta.env.VITE_DATA_URL
  ? 'timeseries.json'
  : './data/timeseries.json';
const TS_FALLBACK = './data/timeseries.json';

export function useTimeseriesData() {
  const [data, setData] = useState<TimeseriesPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      const urls = [TS_URL];
      if (TS_URL !== TS_FALLBACK) urls.push(TS_FALLBACK);

      for (const url of urls) {
        try {
          const cacheBustedUrl = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
          const response = await fetch(cacheBustedUrl, { cache: 'no-store' });
          if (!response.ok) continue;
          const json = await response.json();
          setData(json);
          break;
        } catch {
          continue;
        }
      }
      setLoading(false);
    }
    fetchData();
  }, []);

  return { data, loading };
}
