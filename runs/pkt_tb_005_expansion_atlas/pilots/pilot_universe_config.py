"""PKT-TB-005 pilot: S-05 (VIXY ejection) + S-10 (duplicate-ticker dedup).

Two one-line universe-config changes, one replay bundle. Skeptic standard:
tie favors removal (rule 4 asymmetry — a noise result is bounded evidence of
harmlessness; the simpler universe wins ties). Pre-registered holdout
arm-reads: 3 (vixy_off, dedup, both).

Pre-check (free, no holdout look): did VIXY ever trade / did duplicates ever
co-hold in the control line? Computed from the control replay's fills.

Dedup sets (keep the most liquid, drop the rest):
  SPY (drop VOO, IVV, VTI) · AGG (drop BND) · VNQ (drop IYR) · SMH (drop SOXX)
  XBI (drop IBB) · EFA (drop VEA) · EEM (drop VWO)

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_universe_config.py
Outputs: pilots/results_universe_config.json + manifests/universe_config_manifest.json
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402

SEEDS_FULL = [20260217, 20260218]
SEEDS_HOLDOUT = [20260217, 20260218, 20260219]
DEDUP_DROP = ['VOO', 'IVV', 'VTI', 'BND', 'IYR', 'SOXX', 'IBB', 'VEA', 'VWO']
ARMS = {
    'control': [],
    'vixy_off': ['VIXY'],
    'dedup': DEDUP_DROP,
    'both': ['VIXY'] + DEDUP_DROP,
}


def replay(ds, dates, bundle, seed, drop_list, ranking):
    model, norm, blend = ranking
    uni = None
    if drop_list:
        uni = ds.universe_df.copy()
        uni.loc[uni['symbol'].isin(drop_list), 'eligible'] = 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return common.run_pilot_replay(
            ds, dates, bundle, random_seed=seed, universe_override=uni,
            ranking_model=model, ranking_normalization=norm, ranking_blend=blend)


def cohold_precheck(fills) -> dict:
    """From control fills: VIXY activity + duplicate-pair co-holding days."""
    pos = defaultdict(float)
    cohold_days = defaultdict(int)
    pairs = [('SPY', 'VOO'), ('SPY', 'IVV'), ('SPY', 'VTI'), ('VOO', 'IVV'),
             ('AGG', 'BND'), ('VNQ', 'IYR'), ('SMH', 'SOXX'), ('XBI', 'IBB'),
             ('EFA', 'VEA'), ('EEM', 'VWO')]
    by_date = defaultdict(list)
    for f in fills:
        by_date[f.get('_trade_date', '')].append(f)
    vixy_fills = sum(1 for f in fills if f.get('symbol') == 'VIXY')
    for date in sorted(by_date):
        for f in by_date[date]:
            sym = f.get('symbol')
            sh = float(f.get('shares', 0) or 0)
            act = str(f.get('action', '')).upper()
            pos[sym] += sh if act == 'BUY' else -sh
        for a, b in pairs:
            if pos[a] > 0 and pos[b] > 0:
                cohold_days[f'{a}+{b}'] += 1
    return {'vixy_fills_control': vixy_fills,
            'duplicate_cohold_days_control': dict(cohold_days)}


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds = cache['dataset']
    bundle = common.load_active_bundle()
    ranking = common.ranking_setup_from_bundle(bundle)
    segs = common.split_dates([s.date for s in ds.snapshots])

    results = {'preregistered_holdout_arm_reads': ['vixy_off', 'dedup', 'both'],
               'skeptic_standard': 'tie favors removal (harmlessness bound)',
               'dedup_dropped': DEDUP_DROP, 'arms': {}}
    daily_store = {}
    for arm, drops in ARMS.items():
        arm_out = {'full': [], 'holdout': []}
        for seed in SEEDS_FULL:
            run = replay(ds, segs['full'], bundle, seed, drops, ranking)
            arm_out['full'].append({'seed': seed, **common.segment_metrics(run)})
            if seed == SEEDS_FULL[0]:
                daily_store[(arm, 'full')] = common.daily_returns(run['result'])
                if arm == 'control':
                    results['precheck'] = cohold_precheck(run['result'].fills)
                    print('precheck:', json.dumps(results['precheck']), flush=True)
        for seed in SEEDS_HOLDOUT:
            run = replay(ds, segs['holdout'], bundle, seed, drops, ranking)
            arm_out['holdout'].append({'seed': seed, **common.segment_metrics(run)})
            if seed == SEEDS_HOLDOUT[0]:
                daily_store[(arm, 'holdout')] = common.daily_returns(run['result'])
        results['arms'][arm] = arm_out
        print(f"{arm}: holdout ret={arm_out['holdout'][0]['total_return']} "
              f"full ret={arm_out['full'][0]['total_return']} [{t():.0f}s]", flush=True)

    for arm in ('vixy_off', 'dedup', 'both'):
        block = {}
        for seg in ('full', 'holdout'):
            d_on, d_off = daily_store[(arm, seg)], daily_store[('control', seg)]
            idx = d_on.index.intersection(d_off.index)
            d = (d_on.loc[idx] - d_off.loc[idx]).astype(float)
            sd = d.std()
            block[f'paired_{seg}'] = {
                'n': len(d), 'mean_daily_delta': round(float(d.mean()), 8),
                'sd_daily_delta': round(float(sd), 8),
                't_stat': round(float(d.mean() / (sd / np.sqrt(len(d)))), 3)
                          if sd > 0 and len(d) > 2 else 0.0}
        results['arms'][arm]['paired_vs_control'] = block

    out = RUN_DIR / 'results_universe_config.json'
    out.write_text(json.dumps(results, indent=2))
    common.write_manifest(
        'universe_config',
        params={'arms': ARMS, 'seeds_full': SEEDS_FULL, 'seeds_holdout': SEEDS_HOLDOUT,
                'bundle_version': bundle.get('version_id', 'active'),
                'ranking_blend': ranking[2]},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_universe_config.py',
        wall_clock_s=t(),
        variants_logged=[f'{a}/{s}' for a in ARMS for s in ('full', 'holdout')],
        holdout_looks=3,
        notes='Tie-favors-removal standard (Skeptic rule 4). VIXY hedge-value half of the '
              'question is NOT testable on this window (no crash day; max drawdown -2.5%) — '
              'only the carry/decay cost side is observable; verdict scoped accordingly and '
              'coordinated with PKT-TB-004 attribution.')
    print(f'wrote {out} in {t():.0f}s', flush=True)


if __name__ == '__main__':
    main()
