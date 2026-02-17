import { useState, useEffect } from 'react';
import { TimeseriesPoint } from '../types';

const TS_URL = import.meta.env.VITE_DATA_URL
  ? 'timeseries.json'
  : './data/timeseries.json';

export function useTimeseriesData() {
  const [data, setData] = useState<TimeseriesPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const cacheBustedUrl = `${TS_URL}${TS_URL.includes('?') ? '&' : '?'}t=${Date.now()}`;
        const response = await fetch(cacheBustedUrl, { cache: 'no-store' });
        if (response.ok) {
          const json = await response.json();
          setData(json);
        }
      } catch {
        // Timeseries is optional; fail silently
      }
      setLoading(false);
    }
    fetchData();
  }, []);

  return { data, loading };
}
