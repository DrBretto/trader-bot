"""PKT-TB-005 pilot: M-14 ranking model — DOWNGRADED to rank-IC only.

Skeptic precondition check FAILED: the active RankingMLP
(models/ranking_expanded_unconditioned, ranking_blend=0.35 in the active
bundle) was trained 2026-04-29 (file mtimes; commit 30fb365) — AFTER the
2026-03-11 holdout boundary. A holdout replay read of blend-on vs blend-off is
therefore contaminated (the model saw 2026-03-11→04-29 in training) and is NOT
run. Holdout looks consumed by this pilot: 0.

What IS admissible: standalone rank information coefficient (Spearman IC of
model scores vs realized forward 21-day returns) on dates STRICTLY AFTER the
training date (2026-04-30 →), plus the contaminated-window IC labeled as such
for contrast. This measures whether the only real-label model in the stack
carries cross-sectional signal, without spending a holdout look.

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_ranking_ic.py
Outputs: pilots/results_ranking_ic.json + manifests/ranking_ic_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402

TRAINING_DATE = '2026-04-29'
HORIZON = 21  # trading days forward (matches the model's training target)
EXPLORATORY_HORIZONS = [5, 10]  # shorter windows to recover post-training dates


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds = cache['dataset']
    bundle = common.load_active_bundle()
    model, norm, blend = common.ranking_setup_from_bundle(bundle)
    from training.models.ranking_mlp import RANKING_FEATURES

    snaps = ds.snapshots
    dates = [s.date for s in snaps]

    # forward returns: close-to-close over HORIZON snapshot steps, per symbol
    closes = {}
    for s in snaps:
        sub = s.features_df.sort_values('date').groupby('symbol').tail(1)
        closes[s.date] = {r['symbol']: float(r['close']) for _, r in sub.iterrows()
                          if pd.notna(r.get('close'))}

    score_cache = {}

    def ic_series(horizon: int) -> pd.DataFrame:
        rows = []
        for i, s in enumerate(snaps):
            j = i + horizon
            if j >= len(snaps):
                break
            d0, d1 = s.date, snaps[j].date
            c0, c1 = closes[d0], closes[d1]
            if d0 not in score_cache:
                latest = s.features_df.sort_values('date').groupby('symbol').tail(1)
                sc = {}
                for _, row in latest.iterrows():
                    feat = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
                    sc[row['symbol']] = model.predict_scores(feat, norm)
                score_cache[d0] = sc
            sc = score_cache[d0]
            scores, fwd = [], []
            for sym, v in sc.items():
                if sym in c0 and sym in c1 and c0[sym] > 0:
                    scores.append(v)
                    fwd.append(c1[sym] / c0[sym] - 1.0)
            if len(scores) >= 20:
                rho, _ = spearmanr(scores, fwd)
                rows.append({'date': d0, 'ic': float(rho), 'n_symbols': len(scores)})
        return pd.DataFrame(rows)

    df = ic_series(HORIZON)
    def seg_stats(mask, label):
        sub = df[mask]
        if len(sub) < 3:
            return {'label': label, 'n_dates': len(sub), 'note': 'insufficient'}
        m, sd = sub.ic.mean(), sub.ic.std()
        return {'label': label, 'n_dates': len(sub),
                'mean_ic': round(float(m), 4), 'sd_ic': round(float(sd), 4),
                't_stat': round(float(m / (sd / np.sqrt(len(sub)))), 2),
                'pct_positive': round(100 * float((sub.ic > 0).mean()), 1)}

    results = {
        'model': 'models/ranking_expanded_unconditioned (active, blend=0.35)',
        'training_date': TRAINING_DATE,
        'holdout_replay_read': 'NOT RUN — contaminated (model trained 2026-04-29 > holdout start 2026-03-11)',
        'holdout_looks': 0,
        'horizon_days': HORIZON,
        'note': 'IC dates limited by forward-window: last evaluable date is ~21 snapshots before cache end; '
                'post-training segment is small and stated as such.',
        'segments': [
            seg_stats(df.date > TRAINING_DATE, 'post_training_CLEAN'),
            seg_stats((df.date >= common.HOLDOUT_START) & (df.date <= TRAINING_DATE),
                      'holdout_pre_training_CONTAMINATED'),
            seg_stats(df.date < common.HOLDOUT_START, 'pre_holdout_in_training_window'),
            seg_stats(df.date.notna(), 'all_dates'),
        ],
        'exploratory_short_horizons': {},
    }
    for h in EXPLORATORY_HORIZONS:
        dfh = ic_series(h)
        results['exploratory_short_horizons'][f'{h}d'] = {
            'post_training_CLEAN_target_mismatch':
                seg_stats.__call__ if False else None,
        }
        sub = dfh[dfh.date > TRAINING_DATE]
        if len(sub) >= 3:
            m, sd = sub.ic.mean(), sub.ic.std()
            stats = {'n_dates': len(sub), 'mean_ic': round(float(m), 4),
                     'sd_ic': round(float(sd), 4),
                     't_stat': round(float(m / (sd / np.sqrt(len(sub)))), 2),
                     'pct_positive': round(100 * float((sub.ic > 0).mean()), 1)}
        else:
            stats = {'n_dates': len(sub), 'note': 'insufficient'}
        results['exploratory_short_horizons'][f'{h}d'] = {
            'post_training': stats,
            'caveat': 'horizon differs from the 21d training target — exploratory only'}
    out = RUN_DIR / 'results_ranking_ic.json'
    out.write_text(json.dumps(results, indent=2))
    common.write_manifest(
        'ranking_ic',
        params={'horizon': HORIZON, 'model_dir': 'models/ranking_expanded_unconditioned',
                'training_date': TRAINING_DATE},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_ranking_ic.py',
        wall_clock_s=t(),
        variants_logged=['rank_ic_by_segment'],
        holdout_looks=0,
        notes='M-14 downgraded per Skeptic precondition: RankingMLP training date (2026-04-29) '
              'postdates the holdout boundary; replay comparison would be contaminated. '
              'Clean evidence = IC on post-2026-04-29 dates only.')
    print(json.dumps(results['segments'], indent=1), flush=True)
    print(f'wrote {out} in {t():.0f}s', flush=True)


if __name__ == '__main__':
    main()
