import { useState, useEffect } from 'react';
import { DashboardData } from '../types';

// Production: set VITE_DATA_URL=dashboard.json at build time (see docs/DEPLOY.md).
const DATA_URL = import.meta.env.VITE_DATA_URL || './data/dashboard.json';
const FALLBACK_URL = './data/dashboard.json';

export function useDashboardData() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      const urls = [DATA_URL];
      if (DATA_URL !== FALLBACK_URL) urls.push(FALLBACK_URL);

      for (const url of urls) {
        try {
          const response = await fetch(url);
          if (!response.ok) continue;
          const json = await response.json();
          setData(json);
          setError(null);
          setLoading(false);
          return;
        } catch {
          continue;
        }
      }
      setError('Failed to load dashboard data');
      setLoading(false);
    }

    fetchData();
  }, []);

  return { data, loading, error };
}
