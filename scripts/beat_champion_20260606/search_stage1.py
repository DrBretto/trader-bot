"""Stage-1 search: regime-ensemble / disagreement-throttle recalibration on the
REAL champion overlay system. Tune on IN-SAMPLE (<=2026-03-10), validate on
HOLDOUT (>=2026-03-11). Never-worse floor = incumbent champion holdout.
"""
import json, itertools, collections
from scripts.beat_champion_20260606.runner import (
    make_cache, champion_strategy, run_real, spy_return)

c = make_cache(); uni = c.get_csv('config/universe.csv')

def valid(dates):
    out = []
    for d in dates:
        try:
            inf = c.get_json(f'daily/{d}/inference.json')
            if isinstance(inf, dict) and 'date' in inf and 'regime' in inf:
                out.append(d)
        except Exception:
            pass
    return out

dates = c.list_daily_dates()
hold = valid([d for d in dates if d >= '2026-03-11'])
ins  = valid([d for d in dates if d <= '2026-03-10'])
champ = champion_strategy()

# incumbent (baked ensemble)
inc_h = run_real(c, uni, hold, '2026-03-11', ensemble_overrides=None, strategy=champ)
inc_i = run_real(c, uni, ins, '2026-01-02', ensemble_overrides=None, strategy=champ)
INC_H = inc_h['return']; INC_I = inc_i['return']
print(f"INCUMBENT  insample {INC_I*100:+.2f}%   holdout {INC_H*100:+.2f}%  (FLOOR)")

def ens(gw, dth, pen, cmin, bfloor=0.5, bspan=0.5, cmax=1.0):
    return {'gru_weight': gw, 'transformer_weight': 1.0 - gw,
            'disagreement_threshold': dth,
            'multiplier': {'base_floor': bfloor, 'base_span': bspan,
                           'disagreement_penalty_scale': pen,
                           'clip_min': cmin, 'clip_max': cmax}}

grid = []
for gw in (0.0, 0.2):
    for dth in (0.3, 0.5, 0.7, 1.0):
        for pen in (0.0, 0.25, 0.5):
            for cmin in (0.5, 0.7, 0.85):
                grid.append((gw, dth, pen, cmin))

rows = []
for (gw, dth, pen, cmin) in grid:
    eo = ens(gw, dth, pen, cmin)
    ri = run_real(c, uni, ins, '2026-01-02', ensemble_overrides=eo, strategy=champ)
    rh = run_real(c, uni, hold, '2026-03-11', ensemble_overrides=eo, strategy=champ)
    rows.append({'gw': gw, 'dth': dth, 'pen': pen, 'cmin': cmin,
                 'ins': ri['return'], 'ins_dd': ri['maxdd'],
                 'hold': rh['return'], 'hold_dd': rh['maxdd']})

# select on IN-SAMPLE only (holdout discipline), tie-break lower in-sample dd
rows_sorted = sorted(rows, key=lambda r: (-r['ins'], r['ins_dd']))
print("\nTOP 12 by IN-SAMPLE (holdout shown for reporting only, NOT used to select):")
print(" gw  dth  pen  cmin |  insample   holdout   hold_dd")
for r in rows_sorted[:12]:
    flag = ' <= beats incumbent holdout' if r['hold'] > INC_H else ''
    print(f" {r['gw']:.1f} {r['dth']:.1f} {r['pen']:.2f} {r['cmin']:.2f} | {r['ins']*100:+7.2f}% {r['hold']*100:+7.2f}% {r['hold_dd']*100:6.2f}%{flag}")

best = rows_sorted[0]
print(f"\nIN-SAMPLE WINNER: gw={best['gw']} dth={best['dth']} pen={best['pen']} cmin={best['cmin']}")
print(f"  in-sample {best['ins']*100:+.2f}% (inc {INC_I*100:+.2f}%)   holdout {best['hold']*100:+.2f}% (inc {INC_H*100:+.2f}%)")
print(f"  beats incumbent holdout? {best['hold'] > INC_H}")

json.dump({'incumbent': {'ins': INC_I, 'hold': INC_H},
           'rows': rows, 'winner': best},
          open('runs/beat_champion_20260606/search_stage1.json', 'w'), indent=1)
print("saved search_stage1.json")
