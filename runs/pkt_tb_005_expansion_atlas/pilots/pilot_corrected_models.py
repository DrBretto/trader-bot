"""PKT-TB-005 addendum pilot: how would the stretch have gone with the
CORRECTLY TRAINED models?

Context: every deep-era day in the stored record ran short-corpus-mistrained
models (pre-2026-06-06 training-data fallback bug); the corrected 2026-06-06
models (regime GRU 73% / transformer 67% OOS vs 9% before) have exactly one
stored day. This pilot regenerates per-date inference (regime + health) with
the CURRENT production models loaded via the production ModelLoader, from each
date's stored context/features (as-of-date inputs only), and replays.

Production-faithfulness notes:
- ModelLoader.predict_regime TILES the single day's context row 21x as the
  "sequence" — that is what production does (the sequence models never see a
  real temporal sequence at inference). Reproduced exactly, and flagged as a
  finding.
- Ranking layer active per the bundle (blend 0.35), identical in both arms.

In-sample caveat: the corrected models trained on the 11-year corpus through
~2026-02-03 (BEAT_CHAMPION doc; the pkl does not record the range — stated as
an assumption). Replay days ≤ 2026-02-03 are in-sample for them; the segment
dates > 2026-02-03 ("post_training") and the holdout (2026-03-11+) are the
clean reads. Regime/health train on rule pseudo-labels (not outcomes), which
softens but does not remove the in-sample concern.

Pre-registered holdout arm-reads: 1 (corrected_models). Run total: 12 of 12.

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_corrected_models.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402
from src.models.loader import ModelLoader  # noqa: E402
from src.utils.s3_client import S3Client  # noqa: E402

SEED = 20260217
POST_TRAINING_START = '2026-02-04'  # day after historical corpus end (assumption, see header)


def build_corrected_inference(loader: ModelLoader, ds, context_df: pd.DataFrame) -> dict:
    """date -> {'regime': {...}, 'asset_health': [...]} from the corrected models."""
    ctx = context_df.set_index('date')
    out = {}
    buf = io.StringIO()
    for snap in ds.snapshots:
        d = snap.date
        if d not in ctx.index:
            continue
        with contextlib.redirect_stdout(buf):
            r = loader.predict_regime(ctx.loc[d].to_dict())
            latest_date = snap.features_df['date'].max()
            hdf = loader.predict_health(snap.features_df, latest_date)
        regime = {
            'label': r.get('regime_label'), 'regime_label': r.get('regime_label'),
            'probs': dict(r.get('regime_probs', {})),
            'regime_probs': dict(r.get('regime_probs', {})),
            'embedding': list(r.get('regime_embedding', []) or []),
            'confidence': float(r.get('confidence', 1.0)),
            'disagreement': float(r.get('disagreement', 0.0)),
            'agreement': float(r.get('agreement', 1.0)),
            'position_size_multiplier': float(r.get('position_size_multiplier', 1.0)),
            'gru_prediction': r.get('gru_prediction', {}),
            'transformer_prediction': r.get('transformer_prediction', {}),
        }
        health = []
        for _, row in hdf.iterrows():
            health.append({
                'symbol': row['symbol'],
                'health_score': float(row['health_score']),
                'vol_bucket': str(row.get('vol_bucket', 'med')),
                'behavior': str(row.get('behavior', 'mixed')),
                'latent': list(row['latent']) if isinstance(row.get('latent'), (list, np.ndarray)) else [0.0] * 16,
            })
        out[d] = {'regime': regime, 'asset_health': health}
    return out


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds, context_df = cache['dataset'], cache['context_df']
    bundle = common.load_active_bundle()
    ranking = common.ranking_setup_from_bundle(bundle)
    segs = common.split_dates([s.date for s in ds.snapshots])

    loader = ModelLoader(S3Client('investment-system-data', 'us-east-1'))
    versions = loader.load_models()
    print('loaded model versions:', json.dumps(versions), flush=True)
    assert '20260606' in str(versions.get('regime', '')), \
        f'expected corrected 20260606 regime models, got {versions}'

    corrected = build_corrected_inference(loader, ds, context_df)
    print(f'corrected inference built for {len(corrected)} dates [{t():.0f}s]', flush=True)

    # diagnostics: label agreement corrected-vs-stored / corrected-vs-rules
    snap_map = ds.by_date()
    import pilot_baseline_check as pb
    rules = pb.build_prob_series(context_df, [s.date for s in ds.snapshots], None)
    agree_stored = agree_rules = n = 0
    conf, disg = [], []
    for d, inf in corrected.items():
        cl = inf['regime']['label']
        sl = snap_map[d].inference.get('regime', {}).get('label')
        rl = max(rules[d], key=rules[d].get) if rules.get(d) else None
        if cl and sl and rl:
            n += 1
            agree_stored += int(cl == sl)
            agree_rules += int(cl == rl)
        conf.append(inf['regime']['confidence'])
        disg.append(inf['regime']['disagreement'])
    diagnostics = {
        'n_dates': n,
        'corrected_vs_stored_agreement_pct': round(100 * agree_stored / n, 1),
        'corrected_vs_rules_agreement_pct': round(100 * agree_rules / n, 1),
        'mean_confidence': round(float(np.mean(conf)), 3),
        'mean_disagreement': round(float(np.mean(disg)), 3),
    }
    print('diagnostics:', json.dumps(diagnostics), flush=True)

    def transform(date, inference):
        c = corrected.get(date)
        if c is None:
            return inference
        inference['regime'] = c['regime']
        inference['asset_health'] = c['asset_health']
        return inference

    model, norm, blend = ranking

    def replay(dates, tf):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return common.run_pilot_replay(ds, dates, bundle, random_seed=SEED,
                                           inference_transform=tf,
                                           ranking_model=model,
                                           ranking_normalization=norm,
                                           ranking_blend=blend)

    results = {'model_versions': versions, 'diagnostics': diagnostics,
               'post_training_start_assumption': POST_TRAINING_START,
               'preregistered_holdout_arm_reads': ['corrected_models'],
               'arms': {}}
    daily = {}
    for arm, tf in {'stored_control': None, 'corrected_models': transform}.items():
        out = {}
        for seg in ('full', 'holdout'):
            run = replay(segs[seg], tf)
            out[seg] = common.segment_metrics(run)
            daily[(arm, seg)] = common.daily_returns(run['result'])
        results['arms'][arm] = out
        print(arm, json.dumps({s: {k: out[s][k] for k in
              ('total_return', 'sharpe', 'max_drawdown', 'avg_gross_exposure')}
              for s in out}), f'[{t():.0f}s]', flush=True)

    # paired stats: full, holdout, and post-training slice of full
    val_of = {s.date: s.next_date for s in ds.snapshots}
    post_vals = {val_of[d] for d in segs['full'] if d >= POST_TRAINING_START and d in val_of}
    for seg, dates_filter in (('full', None), ('holdout', None),
                              ('full_post_training_only', post_vals)):
        base_seg = 'full' if seg != 'holdout' else 'holdout'
        d_on = daily[('corrected_models', base_seg)]
        d_off = daily[('stored_control', base_seg)]
        idx = d_on.index.intersection(d_off.index)
        if dates_filter is not None:
            idx = [i for i in idx if i in dates_filter]
        d = (d_on.loc[idx] - d_off.loc[idx]).astype(float)
        sd = d.std()
        results[f'paired_corrected_minus_stored_{seg}'] = {
            'n': len(d), 'mean_daily_delta': round(float(d.mean()), 8),
            'sd_daily_delta': round(float(sd), 8),
            't_stat': round(float(d.mean() / (sd / np.sqrt(len(d)))), 3)
                      if sd > 0 and len(d) > 2 else 0.0}

    out_path = RUN_DIR / 'results_corrected_models.json'
    out_path.write_text(json.dumps(results, indent=2))
    common.write_manifest(
        'corrected_models',
        params={'seed': SEED, 'model_versions': versions,
                'post_training_start_assumption': POST_TRAINING_START,
                'bundle_version': bundle.get('version_id'),
                'ranking_blend': blend},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_corrected_models.py',
        wall_clock_s=t(),
        variants_logged=['stored_control/full', 'stored_control/holdout',
                         'corrected_models/full', 'corrected_models/holdout'],
        holdout_looks=1,
        notes='Corrected-2026-06-06-models counterfactual. Production-faithful inference '
              '(incl. the 21x context-row tiling ModelLoader does — the sequence models '
              'never see real sequences at inference). Days <= 2026-02-03 are in-sample '
              'for the corrected models (pseudo-label training softens this); holdout is '
              'the clean read. 12th and final pre-registered holdout arm-read.')
    print(json.dumps({k: v for k, v in results.items() if k.startswith('paired')},
                     indent=1), flush=True)
    print(f'wrote {out_path} in {t():.0f}s', flush=True)


if __name__ == '__main__':
    main()
