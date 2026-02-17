import { useEffect, useMemo, useState } from 'react';
import { OptimizerLineage, OptimizerRunDetail, OptimizerRunsIndex } from '../types';

const INDEX_URL = './data/optimizer_runs_index.json';
const LINEAGE_URL = './data/active_params_lineage.json';

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const cacheBustedUrl = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
    const response = await fetch(cacheBustedUrl, { cache: 'no-store' });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

export function useOptimizerData(selectedRunId?: string) {
  const [index, setIndex] = useState<OptimizerRunsIndex | null>(null);
  const [lineage, setLineage] = useState<OptimizerLineage | null>(null);
  const [detail, setDetail] = useState<OptimizerRunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    async function loadBase() {
      setLoading(true);
      const [indexData, lineageData] = await Promise.all([
        fetchJson<OptimizerRunsIndex>(INDEX_URL),
        fetchJson<OptimizerLineage>(LINEAGE_URL),
      ]);
      setIndex(indexData);
      setLineage(lineageData);
      setLoading(false);
    }

    loadBase();
  }, []);

  const effectiveRunId = useMemo(() => {
    if (selectedRunId) return selectedRunId;
    return index?.runs?.[0]?.run_id;
  }, [index, selectedRunId]);

  useEffect(() => {
    async function loadDetail(runId: string) {
      setDetailLoading(true);
      const data = await fetchJson<OptimizerRunDetail>(`./data/optimizer_runs/${runId}.json`);
      setDetail(data);
      setDetailLoading(false);
    }

    if (!effectiveRunId) {
      setDetail(null);
      setDetailLoading(false);
      return;
    }

    loadDetail(effectiveRunId);
  }, [effectiveRunId]);

  return {
    index,
    lineage,
    detail,
    loading,
    detailLoading,
  };
}
