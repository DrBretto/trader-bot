#!/usr/bin/env python3
"""Simulate the optimized-champion timeline under each intended risk control to
decide which dead guardrails to enforce. Runs the real replay engine (handles
splits, no look-ahead, real fills) for several configs and reports the risk
profile of each. Read-only against S3.

Configs:
  baseline            - no sector cap (current/buggy behavior)
  sector_0.35         - correlated-cluster cap at 35% (spec 9.1)
  sector_0.25 / 0.50  - cap sensitivity
  sector_0.35+gross   - cap + book-level gross-exposure ceiling

Env: AWS_PROFILE=personal AWS_REGION=us-east-1
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boto3
from src.utils.three_line_replay.replay_engine import S3Cache, load_variant_configs, run_variant
from src.utils.three_line_replay.extender import _build_champion_strategy
from src.steps.decision_engine import _cluster_of, DEFAULT_SECTOR_CLUSTERS

s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
cache = S3Cache(s3, bucket='investment-system-data')
trading_dates = [d for d in cache.list_daily_dates() if d >= '2026-03-11']
universe_df = cache.get_csv('config/universe.csv')
sector_by_symbol = dict(zip(universe_df['symbol'], universe_df['sector']))
base_hybrid, _ = load_variant_configs(cache)


def make_cfg(max_sector_weight=None, reduce=False, gross=False):
    c = copy.deepcopy(base_hybrid)
    if max_sector_weight is None:
        c.decision_params.pop('max_sector_weight', None)
    else:
        c.decision_params['max_sector_weight'] = max_sector_weight
    if not reduce:
        c.decision_params.pop('reduce_health_drop', None)
    # else: keep active.json's reduce_health_drop (0.20)
    deo = dict(c.decision_engine_overrides or {})
    deo['enforce_gross_exposure_cap'] = bool(gross)
    c.decision_engine_overrides = deo
    return c


def metrics(res):
    tl = res['timeline']
    vals = [(r['date'], r['ending_value']) for r in tl]
    # drawdown
    peak = 0.0
    mdd = 0.0
    worst_day = 0.0
    worst_day_date = None
    prev = None
    for d, v in vals:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak if peak > 0 else 0)
        if prev:
            r = v / prev - 1
            if r < worst_day:
                worst_day, worst_day_date = r, d
        prev = v
    # max cluster weight over time + the 06-05 cluster breakdown
    max_cl_w = 0.0
    max_cl = None
    cl0605 = {}
    for r in tl:
        hc = r.get('holdings_at_close', [])
        tot = sum(h['shares'] * (h.get('close_price') or h.get('peak_price') or 0) for h in hc)
        if tot <= 0:
            continue
        cd = {}
        for h in hc:
            mv = h['shares'] * (h.get('close_price') or h.get('peak_price') or 0)
            cl = _cluster_of(h['symbol'], sector_by_symbol, DEFAULT_SECTOR_CLUSTERS)
            cd[cl] = cd.get(cl, 0) + mv
        for cl, mv in cd.items():
            if mv / tot > max_cl_w:
                max_cl_w, max_cl = mv / tot, cl
        if r['date'] == '2026-06-04':
            cl0605 = {cl: round(mv / tot, 3) for cl, mv in sorted(cd.items(), key=lambda x: -x[1])}
    final = vals[-1][1] if vals else None
    # 06-05 single-day return
    r0605 = None
    for i in range(1, len(vals)):
        if vals[i][0] == '2026-06-05':
            r0605 = vals[i][1] / vals[i - 1][1] - 1
    # cash% on 06-04 (did the book de-risk before the drop?)
    cash0604 = None
    for r in tl:
        if r['date'] == '2026-06-04' and r['ending_value']:
            cash0604 = r['ending_cash'] / r['ending_value']
    # action counts by reason
    from collections import Counter
    rc = Counter(a.get('reason', '?').split('_')[0] + '_' + a.get('action', '') for a in res.get('actions', []))
    reduces = sum(1 for a in res.get('actions', []) if a.get('action') == 'REDUCE')
    sells = sum(1 for a in res.get('actions', []) if a.get('action') == 'SELL')
    return dict(final=final, mdd=mdd, worst_day=worst_day, worst_day_date=worst_day_date,
                r0605=r0605, max_cl_w=max_cl_w, max_cl=max_cl, cl0604=cl0605,
                cash0604=cash0604, reduces=reduces, sells=sells)


def peakval(res):
    return max((r['ending_value'] for r in res['timeline']), default=0)

configs = [
    ('baseline (nothing)', make_cfg(None, reduce=False)),
    ('reduce only', make_cfg(None, reduce=True)),
    ('reduce + sector0.50', make_cfg(0.50, reduce=True)),
    ('reduce + sector0.35', make_cfg(0.35, reduce=True)),
]
strat = _build_champion_strategy()
print(f"{'config':20} {'final':>9} {'peak':>9} {'maxDD':>7} {'wDay':>7} {'maxGrowth%':>10} {'REDUCE':>6}")
results = {}
for name, cfg in configs:
    res = run_variant(cache, cfg, strat, trading_dates, universe_df)
    m = metrics(res)
    m['peak'] = peakval(res)
    results[name] = m
    print(f"{name:20} {m['final']:>9.0f} {m['peak']:>9.0f} {m['mdd']*100:>6.2f}% {m['worst_day']*100:>6.2f}% "
          f"{m['max_cl_w']*100:>9.1f}% {m['reduces']:>6}")
print("\nmax cluster reached over the whole run (concentration risk):")
for name, _ in configs:
    print(f"  {name:20}: {results[name]['max_cl']} = {results[name]['max_cl_w']*100:.0f}%")
