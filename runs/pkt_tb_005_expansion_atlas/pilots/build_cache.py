"""PKT-TB-005 pilot data cache builder.

Loads the full optimizer replay dataset (all S3 daily artifacts, no holdout
trimming — holdout discipline is enforced at pilot level, not load level)
plus per-date context rows, and pickles them to the run dir so every pilot
variant replays from identical pinned data (EVIDENCE_PROTOCOL determinism).
"""
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
REPO = RUN_DIR.parents[2]
sys.path.insert(0, str(REPO))

from optimizer.config import load_optimizer_config  # noqa: E402
from optimizer.data_access import load_optimizer_dataset  # noqa: E402
from src.utils.s3_client import S3Client  # noqa: E402

OUT = RUN_DIR / 'data_cache.pkl'


def main() -> None:
    t0 = time.time()
    cfg = load_optimizer_config(str(REPO / 'config/optimizer.committee_20260606.json'))
    cfg.holdout_start = ''  # load everything; pilots split at 2026-03-11 themselves
    cfg.max_days = 900
    ds = load_optimizer_dataset(cfg)
    print(f'snapshots: {len(ds.snapshots)} '
          f'({ds.snapshots[0].date} .. {ds.snapshots[-1].date}) '
          f'in {time.time()-t0:.0f}s', flush=True)

    s3 = S3Client(cfg.bucket, cfg.region)
    ctx_rows = []
    for snap in ds.snapshots:
        try:
            ctx = s3.read_parquet(f'daily/{snap.date}/context.parquet')
            if len(ctx) > 0:
                row = ctx.iloc[0].to_dict()
                row['date'] = snap.date
                ctx_rows.append(row)
        except Exception as exc:  # missing context for a date is recorded, not fatal
            print(f'context missing for {snap.date}: {exc}', flush=True)
    context_df = pd.DataFrame(ctx_rows)
    print(f'context rows: {len(context_df)} in {time.time()-t0:.0f}s', flush=True)

    with open(OUT, 'wb') as f:
        pickle.dump({'dataset': ds, 'context_df': context_df,
                     'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                     'source': 's3://investment-system-data/daily/',
                     'snapshot_range': (ds.snapshots[0].date, ds.snapshots[-1].date)}, f)
    print(f'wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
