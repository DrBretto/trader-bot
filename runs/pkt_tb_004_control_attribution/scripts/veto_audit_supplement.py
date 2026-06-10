#!/usr/bin/env python
"""PKT-TB-004 LLM Veto Auditor — supplementary cuts.

Holdout base rates, downsized-vs-undownsized buy comparison, severity-3 detail,
unassessed-buy dollars. Run: .venv/bin/python runs/pkt_tb_004_control_attribution/scripts/veto_audit_supplement.py
"""
import json
import glob
import os
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

H = '2026-03-11'
ROOT = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(ROOT, 'cache', 'daily')

panel = pd.read_parquet(os.path.join(ROOT, 'cache', 'prices_20260610.parquet'))
panel = panel[['date', 'symbol', 'close']]
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.sort_values(['symbol', 'date'])
series = {s: g.set_index('date')['close'] for s, g in panel.groupby('symbol')}


def fwd(sym, t0, h):
    s = series.get(sym)
    if s is None:
        return None
    i = s.index.searchsorted(pd.Timestamp(t0), side='right') - 1
    if i < 0:
        return None
    e = min(i + h, len(s) - 1)
    if e <= i:
        return None
    return float(s.iloc[e] / s.iloc[i] - 1)


rows = []
spy_dates_holdout = set()
for f in sorted(glob.glob(CACHE + '/*/decisions.json')):
    day = f.split('/')[-2]
    dec = json.load(open(f))
    lr = json.load(open(os.path.dirname(f) + '/llm_risk.json'))
    risks = lr.get('risks', {})
    for a in dec.get('actions', []):
        if a['action'] != 'BUY':
            continue
        adj = float(risks.get(a['symbol'], {}).get('confidence_adjustment', 0) or 0)
        rows.append({'day': day, 'sym': a['symbol'],
                     'assessed': a['symbol'] in risks, 'adj': adj,
                     'f21': fwd(a['symbol'], dec['date'], 21),
                     'dollars': a.get('dollars', 0)})
        if day >= H:
            spy_dates_holdout.add(dec['date'])

b = pd.DataFrame(rows)
out = {}
out['buys_full_n'] = int(len(b))
out['buys_holdout_n'] = int(len(b[b.day >= H]))
for lbl, g in [('full', b), ('holdout', b[b.day >= H])]:
    g2 = g.dropna(subset=['f21'])
    dsd, nds = g2[g2.adj > 0], g2[g2.adj == 0]
    out[lbl] = {
        'all_buys_pct_neg21': round(float((g2.f21 < 0).mean()), 3),
        'all_buys_mean_f21': round(float(g2.f21.mean()), 4),
        'downsized': {'n': int(len(dsd)),
                      'pct_neg21': round(float((dsd.f21 < 0).mean()), 3),
                      'mean_f21': round(float(dsd.f21.mean()), 4)},
        'not_downsized': {'n': int(len(nds)),
                          'pct_neg21': round(float((nds.f21 < 0).mean()), 3),
                          'mean_f21': round(float(nds.f21.mean()), 4)},
    }
out['buys_assessed'] = int(b.assessed.sum())
out['unassessed_buy_dollars'] = round(float(b[~b.assessed].dollars.sum()), 2)
out['total_buy_dollars'] = round(float(b.dollars.sum()), 2)

sev3 = []
for f in sorted(glob.glob(CACHE + '/*/llm_risk.json')):
    lr = json.load(open(f))
    for s, r in lr.get('risks', {}).items():
        if r.get('severity') == 3:
            sev3.append({'day': f.split('/')[-2], 'symbol': s,
                         'veto': r.get('structural_risk_veto'),
                         'adj': r.get('confidence_adjustment'),
                         'rationale': r.get('one_sentence_rationale')})
out['severity3_events'] = sev3

v = [fwd('SPY', dd, 21) for dd in sorted(spy_dates_holdout)]
v = [x for x in v if x is not None]
out['spy_holdout_buydates'] = {'n': len(v),
                               'pct_neg21': round(float(np.mean([x < 0 for x in v])), 3),
                               'mean_f21': round(float(np.mean(v)), 4)}

print(json.dumps(out, indent=1))
with open(os.path.join(ROOT, 'veto_audit_supplement.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
