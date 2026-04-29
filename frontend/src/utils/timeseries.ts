interface DatedPoint {
  date: string;
}

function toEpochMillis(date: string): number {
  const millis = Date.parse(date);
  return Number.isNaN(millis) ? Number.NaN : millis;
}

export function sortAndDedupeByDate<T extends DatedPoint>(points: T[]): T[] {
  const sorted = [...points].sort((a, b) => {
    const aMillis = toEpochMillis(a.date);
    const bMillis = toEpochMillis(b.date);
    if (Number.isNaN(aMillis) && Number.isNaN(bMillis)) return 0;
    if (Number.isNaN(aMillis)) return 1;
    if (Number.isNaN(bMillis)) return -1;
    return aMillis - bMillis;
  });

  const deduped: T[] = [];
  for (const point of sorted) {
    const last = deduped[deduped.length - 1];
    if (last && last.date === point.date) {
      // Deterministic policy: keep the latest occurrence.
      deduped[deduped.length - 1] = point;
    } else {
      deduped.push(point);
    }
  }
  return deduped;
}

export function isStrictlyIncreasingByDate<T extends DatedPoint>(points: T[]): boolean {
  for (let i = 1; i < points.length; i += 1) {
    const prevMillis = toEpochMillis(points[i - 1].date);
    const currMillis = toEpochMillis(points[i].date);
    if (Number.isNaN(prevMillis) || Number.isNaN(currMillis) || currMillis <= prevMillis) {
      return false;
    }
  }
  return true;
}

export function insertGapBreakPoints<T extends DatedPoint>(
  points: T[],
  maxGapDays: number,
  makeBreakPoint: (date: string) => T,
): T[] {
  if (points.length <= 1) return points;

  const output: T[] = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const curr = points[i];
    const prevMillis = toEpochMillis(prev.date);
    const currMillis = toEpochMillis(curr.date);
    if (!Number.isNaN(prevMillis) && !Number.isNaN(currMillis)) {
      const gapDays = (currMillis - prevMillis) / (1000 * 60 * 60 * 24);
      if (gapDays > maxGapDays) {
        const breakDate = new Date(prevMillis + 24 * 60 * 60 * 1000)
          .toISOString()
          .slice(0, 10);
        output.push(makeBreakPoint(breakDate));
      }
    }
    output.push(curr);
  }
  return output;
}
