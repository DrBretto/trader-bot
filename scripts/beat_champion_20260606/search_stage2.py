"""Stage-2: un-nerf the disagreement throttle by patching ONLY
regime.position_size_multiplier (baked regime label/probs kept). Real champion
overlay system. Tune in-sample, validate holdout, never-worse floor.
"""
import json
from scripts.beat_champion_20260606.runner import (
    make_cache, ThrottlePatchCache, champion_strategy, run_real, spy_return)

base = make_cache(); uni = base.get_csv('config/universe.csv')

def valid(dates):
    out = []
    for d in dates:
        try:
            inf = base.get_json(f'daily/{d}/inference.json')
            if isinstance(inf, dict) and 'date' in inf and 'regime' in inf:
                out.append(d)
        except Exception:
            pass
    return out

dates = base.list_daily_dates()
hold = valid([d for d in dates if d >= '2026-03-11'])
ins  = valid([d for d in dates if d <= '2026-03-10'])
champ = champion_strategy()
BENIGN = {'risk_on_trend', 'calm_uptrend'}

inc_h = run_real(base, uni, hold, '2026-03-11', strategy=champ)
inc_i = run_real(base, uni, ins,  '2026-01-02', strategy=champ)
INC_H, INC_I = inc_h['return'], inc_i['return']
print(f"INCUMBENT  ins {INC_I*100:+.2f}%  hold {INC_H*100:+.2f}%  hdd {inc_h['maxdd']*100:.2f}%  (FLOOR)")

# ---- throttle policies (un-nerf) ----
def floor_global(x):
    return lambda m, r: max(m, x)
def floor_benign(x):
    return lambda m, r: (max(m, x) if r.get('label') in BENIGN else m)

policies = []
for x in (0.6, 0.7, 0.8, 0.9, 1.0):
    policies.append((f'floor_global_{x}', floor_global(x)))
for x in (0.7, 0.85, 1.0):
    policies.append((f'floor_benign_{x}', floor_benign(x)))

rows = []
for name, pol in policies:
    pc = ThrottlePatchCache(base, pol)
    ri = run_real(pc, uni, ins,  '2026-01-02', strategy=champ)
    rh = run_real(pc, uni, hold, '2026-03-11', strategy=champ)
    rows.append({'name': name, 'ins': ri['return'], 'ins_dd': ri['maxdd'],
                 'hold': rh['return'], 'hold_dd': rh['maxdd']})
    print(f"  {name:20s} ins {ri['return']*100:+6.2f}%  hold {rh['return']*100:+6.2f}%  hdd {rh['maxdd']*100:6.2f}%"
          + ('  <=BEATS INC HOLD' if rh['return'] > INC_H else ''))

# select on IN-SAMPLE, require also >= incumbent in-sample; report holdout
elig = [r for r in rows if r['ins'] >= INC_I]
sel_pool = elig if elig else rows
best = sorted(sel_pool, key=lambda r: (-r['ins'], r['ins_dd']))[0]
print(f"\nIN-SAMPLE WINNER: {best['name']}  ins {best['ins']*100:+.2f}% (inc {INC_I*100:+.2f}%)  "
      f"hold {best['hold']*100:+.2f}% (inc {INC_H*100:+.2f}%)  beats_inc_hold={best['hold']>INC_H}")
json.dump({'incumbent': {'ins': INC_I, 'hold': INC_H, 'hold_dd': inc_h['maxdd']},
           'rows': rows, 'winner': best},
          open('runs/beat_champion_20260606/search_stage2.json', 'w'), indent=1)
print("saved search_stage2.json")
