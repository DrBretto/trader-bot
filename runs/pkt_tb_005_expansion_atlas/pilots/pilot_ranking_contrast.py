"""PKT-TB-005: formal paired contrast of the active bundle WITH vs WITHOUT its
ranking layer (blend 0.35 vs 0) on the pipeline-faithful harness.

Motivated by the v1/v2 battery discrepancy (holdout +1.6% without ranking vs
-5.9% with) and the rank-IC finding (in-training IC +0.20 collapsing to ~0.00
out-of-window). NOTE the contamination asymmetry: the ranking model was trained
2026-04-29, INSIDE the holdout window — so if anything the holdout read is
biased IN FAVOR of the ranking arm, which makes a negative result stronger,
not weaker. Pre-registered: 1 additional holdout arm-read (blend-0 arm;
blend-0.35 control already read by the battery). Total run looks: 11 of 12.

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_ranking_contrast.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402

SEED = 20260217


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds = cache['dataset']
    bundle = common.load_active_bundle()
    model, norm, blend = common.ranking_setup_from_bundle(bundle)
    segs = common.split_dates([s.date for s in ds.snapshots])

    results = {'arms': {}, 'preregistered_holdout_arm_reads': ['blend_0'],
               'note': 'blend_0.35 control re-uses the battery control read; '
                       'contamination favors the ranking arm (model trained '
                       '2026-04-29, inside the holdout window)'}
    daily = {}
    for arm, (m, n, b) in {'blend_035': (model, norm, blend),
                           'blend_0': (None, None, 0.0)}.items():
        out = {}
        for seg in ('full', 'holdout'):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run = common.run_pilot_replay(ds, segs[seg], bundle, random_seed=SEED,
                                              ranking_model=m, ranking_normalization=n,
                                              ranking_blend=b)
            out[seg] = common.segment_metrics(run)
            daily[(arm, seg)] = common.daily_returns(run['result'])
        results['arms'][arm] = out
        print(arm, json.dumps({s: {k: out[s][k] for k in
              ('total_return', 'sharpe', 'max_drawdown', 'avg_gross_exposure')}
              for s in out}), flush=True)

    for seg in ('full', 'holdout'):
        d_on, d_off = daily[('blend_035', seg)], daily[('blend_0', seg)]
        idx = d_on.index.intersection(d_off.index)
        d = (d_on.loc[idx] - d_off.loc[idx]).astype(float)
        sd = d.std()
        results[f'paired_blend035_minus_blend0_{seg}'] = {
            'n': len(d), 'mean_daily_delta': round(float(d.mean()), 8),
            'sd_daily_delta': round(float(sd), 8),
            't_stat': round(float(d.mean() / (sd / np.sqrt(len(d)))), 3)
                      if sd > 0 and len(d) > 2 else 0.0}

    out_path = RUN_DIR / 'results_ranking_contrast.json'
    out_path.write_text(json.dumps(results, indent=2))
    common.write_manifest(
        'ranking_contrast',
        params={'seed': SEED, 'bundle_version': bundle.get('version_id'),
                'blends': [0.35, 0.0]},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_ranking_contrast.py',
        wall_clock_s=t(), variants_logged=['blend_035/full', 'blend_035/holdout',
                                           'blend_0/full', 'blend_0/holdout'],
        holdout_looks=1,
        notes='Paired daily contrast of the active ranking layer; contamination '
              'asymmetry favors blend_035 on holdout (model trained inside the window).')
    print(json.dumps({k: v for k, v in results.items() if k.startswith('paired')},
                     indent=1), flush=True)
    print(f'wrote {out_path} in {t():.0f}s', flush=True)


if __name__ == '__main__':
    main()
